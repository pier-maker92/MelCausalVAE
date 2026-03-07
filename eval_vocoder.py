import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import json
import yaml
import math
import time
import shutil
import random
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MelCausalVAE.data.libri_tts import LibriTTS
from MelCausalVAE.data.mls import MLSDataset
from MelCausalVAE.data.librispeech_align_test import LibriSpeechAlignTestDataset
from MelCausalVAE.data.audio_dataset import DataCollator, TestDatasetWrapper

# HuBERT-Large ASR for WER/CER
from transformers import Wav2Vec2Processor, HubertForCTC
from jiwer import wer as compute_wer, cer as compute_cer
try:
    import utmosv2
except ImportError:
    utmosv2 = None

from baseline_vocoder import BaselineVocoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Standalone UTMOS predictor using utmosv2
class UTMOSPredictor:
    def __init__(self, device):
        logger.info("Initializing UTMOSv2 model")
        try:
            self.model = utmosv2.create_model(pretrained=True)
            if hasattr(self.model, "to"):
                self.model.to(device)
            if hasattr(self.model, "float"):
                self.model.float()
            self.device = device
            self.has_utmos = True
            logger.info("UTMOSv2 model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load UTMOSv2 model: {e}")
            self.has_utmos = False

    @torch.no_grad()
    def predict(self, wav_path):
        if not self.has_utmos:
            return None
        try:
            with torch.amp.autocast("cuda", enabled=False):
                mos = self.model.predict(input_path=str(wav_path))
                return float(mos)
        except Exception as e:
            logger.warning(f"UTMOSv2 prediction failed for {wav_path}: {e}")
            return None

    @torch.no_grad()
    def predict_tensors(self, audios: torch.Tensor, sr: int):
        """
        Predict MOS for a batch of audio tensors.
        audios: (Batch, time) or (time,)
        """
        if not self.has_utmos:
            return None
        try:
            with torch.amp.autocast("cuda", enabled=False):
                # model.predict returns a tensor if input is batch tensor
                mos = self.model.predict(data=audios, sr=sr)
                if isinstance(mos, torch.Tensor):
                    return mos.cpu().tolist()
                return mos
        except Exception as e:
            logger.warning(f"Batch tensor UTMOSv2 prediction failed: {e}")
            return None

    @torch.no_grad()
    def predict_batch(self, input_dir):
        if not self.has_utmos:
            return {}
        try:
            logger.info(f"Running batch UTMOSv2 prediction on {input_dir}")
            with torch.amp.autocast("cuda", enabled=False):
                results = self.model.predict(input_dir=str(input_dir))
                mos_map = {
                    Path(r["file_path"]).name: float(r["predicted_mos"])
                    for r in results
                }
                return mos_map
        except Exception as e:
            logger.warning(f"Batch UTMOSv2 prediction failed: {e}")
            return {}


class HuBERTASR:
    """HuBERT-Large-LS960-ft based ASR for WER/CER computation."""
    def __init__(self, device: torch.device, model_name: str = "facebook/hubert-large-ls960-ft"):
        logger.info(f"Loading HuBERT ASR model: {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = HubertForCTC.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def transcribe(self, audio_path: str) -> str:
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        inputs = self.processor(
            audio.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        )
        logits = self.model(inputs.input_values.to(self.device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.lower()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_wav(path: Path, audio: torch.Tensor, sr: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(
        str(path), audio.cpu().to(torch.float32), sample_rate=sr
    )


def evaluate_dataset(
    setting: str,
    filter_librispeech: bool,
    batch_size: int,
    num_samples: int,
    vocoder_model: BaselineVocoder,
    device: torch.device,
    work_dir: Path,
    evaluator: Optional[UTMOSPredictor],
    hubert_asr: HuBERTASR,
    keep_wavs: bool,
    log_images: bool,
    max_images: int,
    original_audio_only: bool,
    skip_ref_metrics: bool,
    run
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"samples": [], "aggregates": {}}

    gt_filepath = "/home/ec2-user/MelCausalVAE/evaluation/GT/metrics.json"
    gt_metrics_map = {}
    if skip_ref_metrics:
        try:
            with open(gt_filepath, "r") as f:
                gt_data = json.load(f)
                for sample in gt_data.get("samples", []):
                    map_id = sample.get("id")
                    if map_id is not None:
                        gt_metrics_map[str(map_id)] = {
                            "ref": sample.get("ref", {}),
                            "ref_transcription": sample.get("ref_transcription", "")
                        }
        except Exception as e:
            logger.warning(f"Could not load pre-computed GT metrics from {gt_filepath}: {e}")

    if setting == "LibriSpeech":
        ds = TestDatasetWrapper(LibriSpeechAlignTestDataset(do_filter=filter_librispeech), "test")
    elif setting == "ID":
        ds = TestDatasetWrapper(LibriTTS(), "test")
    else:
        ds = TestDatasetWrapper(MLSDataset(), "train")

    test_dataloader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=DataCollator()
    )

    per_language = {}
    tmp_audio_dir = work_dir / "tmp_wavs"
    tmp_audio_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    image_count = 0
    
    if num_samples > 0:
        total_samples = min(num_samples, len(ds))
    else:
        total_samples = len(ds)
        
    pbar = tqdm(total=total_samples, desc="Evaluating Vocoder dataset")
    reconstructed_audios = []
    original_audios = []

    for batch in test_dataloader:
        audios_srs = batch["output_audios_srs"]
        lang = batch["language"]
        gt_text = batch["transcription"]
        batch_ids = batch.get("ids", [None] * len(audios_srs))
        
        for idx in range(len(audios_srs)):
            global_idx = processed + idx
            if num_samples > 0 and global_idx >= num_samples:
                break
            
            sample_id = batch_ids[idx] if batch_ids[idx] is not None else global_idx
                
            orig_wav_path = tmp_audio_dir / f"sample_{sample_id}_orig.wav"
            recon_wav_path = tmp_audio_dir / f"sample_{sample_id}_recon.wav"
            prompt_txt = tmp_audio_dir / f"sample_{sample_id}_prompt.txt"

            audio, original_sr = audios_srs[idx]
            original_audio = audio.squeeze().cpu()

            if original_audio_only:
                resampled_audio = original_audio / (original_audio.abs().max() + 1e-8)
                reconstructed_audio = resampled_audio
                vocoder_sr = original_sr
                save_wav(orig_wav_path, resampled_audio, vocoder_sr)
                prompt_txt.write_text(gt_text[idx])
            else:
                reconstructed_audio, out_sr = vocoder_model.reconstruct(audio, original_sr)
                vocoder_sr = out_sr

                # Resample to vocoder sample rate if needed
                if original_sr != vocoder_sr:
                    resampled_audio = torchaudio.functional.resample(original_audio.unsqueeze(0), original_sr, vocoder_sr).squeeze(0)
                else:
                    resampled_audio = original_audio

                # Normalize to prevent clipping safely
                resampled_audio = resampled_audio / (resampled_audio.abs().max() + 1e-8)
                if reconstructed_audio.abs().max() > 0:
                    reconstructed_audio = reconstructed_audio / (reconstructed_audio.abs().max() + 1e-8)
                
                if not skip_ref_metrics:
                    save_wav(orig_wav_path, resampled_audio, vocoder_sr)
                save_wav(recon_wav_path, reconstructed_audio, vocoder_sr)
                prompt_txt.write_text(gt_text[idx])

            # Use HuBERT ASR
            ref_text_normalized = gt_text[idx].lower().strip()
            
            if original_audio_only:
                ref_hyp = hubert_asr.transcribe(str(orig_wav_path))
                recon_hyp = ref_hyp
            else:
                if not skip_ref_metrics:
                    ref_hyp = hubert_asr.transcribe(str(orig_wav_path))
                else:
                    ref_info = gt_metrics_map.get(str(sample_id), {})
                    ref_hyp = ref_info.get("ref_transcription", "")
                recon_hyp = hubert_asr.transcribe(str(recon_wav_path))

            if ref_text_normalized and (original_audio_only or not skip_ref_metrics):
                ref_wer_val = float(compute_wer(ref_text_normalized, ref_hyp))
                ref_cer_val = float(compute_cer(ref_text_normalized, ref_hyp))
            else:
                ref_wer_val = ref_cer_val = float("nan")
                
            if ref_text_normalized and not original_audio_only:
                recon_wer_val = float(compute_wer(ref_text_normalized, recon_hyp))
                recon_cer_val = float(compute_cer(ref_text_normalized, recon_hyp))
            else:
                recon_wer_val = recon_cer_val = float("nan")

            if not original_audio_only and not skip_ref_metrics:
                consistent_wer_val = float(compute_wer(ref_hyp, recon_hyp))
                consistent_cer_val = float(compute_cer(ref_hyp, recon_hyp))
            else:
                consistent_wer_val = consistent_cer_val = float("nan")

            def to_pct(v):
                if v is None or math.isnan(v):
                    return None
                return round(float(v) * 100.0, 2)

            if skip_ref_metrics:
                ref_scores = gt_metrics_map.get(str(sample_id), {}).get("ref", {})
            else:
                ref_scores = {"WER": to_pct(ref_wer_val), "CER": to_pct(ref_cer_val)}
            recon_scores = {"WER": to_pct(recon_wer_val), "CER": to_pct(recon_cer_val)}
            consistency_scores = {
                "WER": to_pct(consistent_wer_val),
                "CER": to_pct(consistent_cer_val),
            }

            results["samples"].append(
                {
                    "index": global_idx,
                    "id": str(sample_id),
                    "language": lang[idx],
                    "gt_text": ref_text_normalized,
                    "ref_transcription": ref_hyp,
                    "recon_transcription": recon_hyp,
                    "ref": ref_scores,
                    "reconstructed": recon_scores,
                    "consistency": consistency_scores,
                    "orig_wav_name": orig_wav_path.name,
                    "recon_wav_name": recon_wav_path.name,
                }
            )

            if log_images and image_count < max_images:
                original_audios.append(
                    wandb.Audio(resampled_audio.numpy(), sample_rate=vocoder_sr)
                )
                if not original_audio_only:
                    reconstructed_audios.append(
                        wandb.Audio(reconstructed_audio.numpy(), sample_rate=vocoder_sr)
                    )
                image_count += 1

            key = lang[idx] if setting == "OOD" else "all"
            if key not in per_language:
                per_language[key] = {}
            for m in ["WER", "CER", "UTMOS"]:
                if m in ref_scores and ref_scores[m] is not None:
                    per_language[key].setdefault(f"ref_{m}", [])
                    per_language[key][f"ref_{m}"].append(float(ref_scores[m]))
                    
            for m in ["WER", "CER"]:
                if m in recon_scores and recon_scores[m] is not None:
                    per_language[key].setdefault(f"recon_{m}", [])
                    per_language[key][f"recon_{m}"].append(float(recon_scores[m]))

            for m in ["WER", "CER"]:
                if consistency_scores[m] is not None:
                    per_language[key].setdefault(f"consistent_{m}", [])
                    per_language[key][f"consistent_{m}"].append(
                        float(consistency_scores[m])
                    )

        processed += len(audios_srs)
        pbar.update(len(audios_srs))
        if num_samples > 0 and processed >= num_samples:
            break
            
    pbar.close()

    # UTMOS Evaluation
    mos_map = {}
    if evaluator is not None and evaluator.has_utmos:
        mos_map = evaluator.predict_batch(tmp_audio_dir)

    for sample in results["samples"]:
        orig_wav_name = sample.pop("orig_wav_name", None)
        recon_wav_name = sample.pop("recon_wav_name", None)

        if mos_map:
            if original_audio_only:
                orig_mos = mos_map.get(orig_wav_name)
                if orig_mos is not None:
                    sample["ref"]["UTMOS"] = round(float(orig_mos), 2)
                    key = sample["language"] if setting == "OOD" else "all"
                    per_language[key].setdefault("ref_UTMOS", []).append(float(orig_mos))
            else:
                orig_mos = mos_map.get(orig_wav_name)
                recon_mos = mos_map.get(recon_wav_name)
                if orig_mos is not None and not skip_ref_metrics:
                    sample["ref"]["UTMOS"] = round(float(orig_mos), 2)
                    key = sample["language"] if setting == "OOD" else "all"
                    per_language[key].setdefault("ref_UTMOS", []).append(float(orig_mos))
                if recon_mos is not None:
                    sample["reconstructed"]["UTMOS"] = round(float(recon_mos), 2)
                    key = sample["language"] if setting == "OOD" else "all"
                    per_language[key].setdefault("recon_UTMOS", []).append(float(recon_mos))

    # Delta
    for sample in results["samples"]:
        delta = {}
        for m in ["WER", "CER", "UTMOS"]:
            rv = sample["ref"].get(m)
            rcv = sample["reconstructed"].get(m)
            if rv is None or rcv is None:
                delta[m] = None
            else:
                d_val = float(rcv - rv)
                delta[m] = round(d_val, 2)
        sample["delta"] = delta

    if not keep_wavs:
        logger.info(f"Cleaning up temporary WAV files in {tmp_audio_dir}")
        import shutil
        try:
            shutil.rmtree(tmp_audio_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup {tmp_audio_dir}: {e}")

    log_payload = {}
    if original_audios:
        log_payload["original_audios"] = original_audios
    if reconstructed_audios:
        log_payload["reconstructed_audios"] = reconstructed_audios

    if run is not None and log_payload:
        run.log(log_payload)

    # Finalize aggregates
    for key, agg in per_language.items():

        def _mean(arr: List[float]) -> Optional[float]:
            arr = [x for x in arr if not math.isnan(x)]
            return float(sum(arr) / len(arr)) if arr else None

        ref_wer = _mean(agg.get("ref_WER", []))
        recon_wer = _mean(agg.get("recon_WER", []))
        ref_cer = _mean(agg.get("ref_CER", []))
        recon_cer = _mean(agg.get("recon_CER", []))
        ref_utmos = _mean(agg.get("ref_UTMOS", []))
        recon_utmos = _mean(agg.get("recon_UTMOS", []))
        consist_wer = _mean(agg.get("consistent_WER", []))
        consist_cer = _mean(agg.get("consistent_CER", []))

        def to_report(v, is_mos=False):
            if v is None or math.isnan(v):
                return None
            return round(float(v), 2)

        agg_ref = {
            "WER": to_report(ref_wer),
            "CER": to_report(ref_cer),
            "UTMOS": to_report(ref_utmos, True),
        }
        agg_recon = {
            "WER": to_report(recon_wer),
            "CER": to_report(recon_cer),
            "UTMOS": to_report(recon_utmos, True),
        }
        agg_consist = {"WER": to_report(consist_wer), "CER": to_report(consist_cer)}

        agg.clear()
        agg.update(
            {"ref": agg_ref, "reconstructed": agg_recon, "consistency": agg_consist}
        )

        discrepancy = {}
        for m in ["WER", "CER", "UTMOS"]:
            rv = agg_ref.get(m)
            rcv = agg_recon.get(m)
            if rv is None or rcv is None:
                discrepancy[m] = None
            else:
                discrepancy[m] = round(float(rcv - rv), 2)

        results["aggregates"][key] = {
            "ref": agg_ref,
            "reconstructed": agg_recon,
            "delta": discrepancy,
        }

    try:
        shutil.rmtree(tmp_audio_dir)
    except Exception:
        pass

    return results

def save_json(output: Dict[str, Any], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved metrics to {out_path}")

def run_single_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Initialize Vocoder only if not just audio evaluation
    vocoder_model = None
    if not args.original_audio_only:
        vocoder_model = BaselineVocoder(
            model_name=args.vocoder_name,
            device=device,
            vocoder_dir=args.vocoder_dir
        )

    # HuBERT-Large ASR for WER/CER
    hubert_asr = HuBERTASR(device)

    # Standalone UTMOS Predictor
    evaluator = UTMOSPredictor(device) if args.UTMOS else None
        
    run = None
    if args.report_to == "wandb":
        run = wandb.init(
            project="MelCausalVAE-eval-vocoder",
            name=args.exp_name,
            config={
                "setting": args.setting,
                "vocoder_name": args.vocoder_name,
                "vocoder_dir": args.vocoder_dir,
            },
        )

    voc_name_safe = args.vocoder_name if not args.original_audio_only else "original_audio"
    work_dir = Path(args.output_dir) / args.exp_name / voc_name_safe
    work_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    try:
        eval_output = evaluate_dataset(
            setting=args.setting,
            filter_librispeech=args.filter_librispeech,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            vocoder_model=vocoder_model,
            device=device,
            work_dir=work_dir,
            evaluator=evaluator,
            hubert_asr=hubert_asr,
            keep_wavs=args.keep_wavs,
            log_images=args.wandb_log_images,
            max_images=args.wandb_max_images,
            original_audio_only=args.original_audio_only,
            skip_ref_metrics=(not args.compute_ref_metrics) and (not args.original_audio_only),
            run=run
        )
        status = "success"
        error = None
    except Exception as e:
        logger.exception("Evaluation failed")
        eval_output = {}
        status = "failed"
        error = str(e)
        
    duration = time.time() - start

    out_path = work_dir / "metrics.json"
    if eval_output:
        save_json(eval_output, out_path)

    if run is not None:
        run.finish()

    return {
        "vocoder": args.vocoder_name if not args.original_audio_only else "original_audio",
        "status": status,
        "duration": duration,
        "metrics_path": str(out_path) if eval_output else None,
        "work_dir": str(work_dir),
        "error": error,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Vocoders using original audio to Mel to Wav workflow")
    parser.add_argument("--vocoder-name", type=str, default="bigvgan", choices=["bigvgan", "vocos", "hifigan"], help="Vocoder architecture to evaluate")
    parser.add_argument("--vocoder-dir", type=str, default=None, help="Optional path to locally cloned model (required for bigvgan)")
    
    parser.add_argument("--original-audio-only", action="store_true", help="Bypass any model and only calculate metrics on the original audio.")
    parser.add_argument("--compute-ref-metrics", action="store_true", help="Compute metrics on the original reference audio instead of using precomputed GT.")
    
    parser.add_argument("--report-to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--setting", type=str, default="ID", choices=["ID", "OOD", "LibriSpeech"])
    parser.add_argument("--wandb-log-images", action="store_true", help="If set, also log audios to W&B")
    parser.add_argument("--wandb-max-images", type=int, default=10, help="Maximum number of audios to log (>1)")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="vocoder_eval")
    parser.add_argument("--output-dir", type=str, default="/home/ec2-user/MelCausalVAE/evaluation")
    parser.add_argument("--UTMOS", action="store_true", help="Enable UTMOS computation")
    parser.add_argument("--keep-wavs", action="store_true", help="Keep temporary WAV files after evaluation")
    parser.add_argument("--filter-librispeech", action="store_true", help="If set, filters the LibriSpeech test set using the E2TTS JSON keys")

    args = parser.parse_args()

    # set default dtype to bfloat16 to match VAE settings from eval.py
    torch.set_default_dtype(torch.bfloat16)

    results = run_single_eval(args)
    
    summary_path = Path(args.output_dir) / args.exp_name / "summary_results.json"
    save_json(results, summary_path)
    
if __name__ == "__main__":
    main()
