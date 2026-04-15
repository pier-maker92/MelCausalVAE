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
import tempfile
from contextlib import nullcontext
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader
from vocos import Vocos
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MelCausalVAE.modules.cfm import DiTConfig
from MelCausalVAE.modules.VAE import VAE, VAEConfig
from MelCausalVAE.modules.Encoder import ConvformerEncoderConfig
from MelCausalVAE.modules.melspecEncoder import MelSpectrogramConfig
from MelCausalVAE.data.libri_tts import LibriTTS
from MelCausalVAE.data.mls import MLSDataset
from MelCausalVAE.data.librispeech_align_test import LibriSpeechAlignTestDataset
from MelCausalVAE.data.librispeech100 import LibriSpeech100h
from MelCausalVAE.data.audio_dataset import DataCollator, TestDatasetWrapper
from MelCausalVAE.baseline_models import BaselineAudioCodec

# HuBERT-Large ASR for WER/CER
from transformers import Wav2Vec2Processor, HubertForCTC
from jiwer import wer as compute_wer, cer as compute_cer

from datasets import load_dataset, concatenate_datasets
import utmosv2


def _no_amp_autocast():
    """Disable autocast on CUDA; no-op context elsewhere (MPS/CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _scalar_to_float(x: Any) -> Optional[float]:
    """utmosv2 may return float, numpy scalar, or torch tensor."""
    if x is None:
        return None
    try:
        if hasattr(x, "item") and callable(getattr(x, "item")):
            return float(x.item())
        return float(x)
    except (TypeError, ValueError):
        return None


def _utmos_results_to_records(results: Any) -> List[Dict[str, Any]]:
    """Normalize utmosv2 ``predict`` return value (list / DataFrame / single dict)."""
    if results is None:
        return []
    if hasattr(results, "to_dict") and callable(getattr(results, "to_dict")):
        try:
            rec = results.to_dict("records")  # pandas DataFrame
            return list(rec) if rec is not None else []
        except Exception:
            pass
    if isinstance(results, dict):
        return [results]
    try:
        return list(results)
    except TypeError:
        return []


def _record_to_basename_mos(r: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    fp = r.get("file_path", r.get("filepath", r.get("path")))
    if fp is None:
        return None
    mos = _scalar_to_float(
        r.get("predicted_mos", r.get("mos", r.get("predicted_MOS")))
    )
    if mos is None:
        return None
    return Path(str(fp)).name, mos


# Standalone UTMOS predictor using utmosv2
class UTMOSPredictor:
    """
    utmosv2 is always run on **CPU** for compatibility: directory batch mode and
    several torch back-ends (notably MPS) often fail or return empty otherwise.
    VAE / HuBERT stay on the eval device passed into ``evaluate_dataset``.
    """

    def __init__(self, _eval_device: torch.device):
        logger.info("Initializing UTMOSv2 model (inference on CPU)")
        self.device_str = "cpu"
        try:
            self.model = utmosv2.create_model(pretrained=True, device=self.device_str)
            if hasattr(self.model, "to"):
                self.model.to(torch.device("cpu"))
            if hasattr(self.model, "float"):
                self.model.float()
            self.has_utmos = True
            logger.info("UTMOSv2 model loaded successfully on CPU.")
        except Exception as e:
            logger.warning("Failed to load UTMOSv2 model: %s", e)
            self.has_utmos = False

    @torch.no_grad()
    def predict(self, wav_path: Path | str) -> Optional[float]:
        if not self.has_utmos:
            return None
        try:
            with _no_amp_autocast():
                mos = self.model.predict(
                    input_path=str(wav_path),
                    device=self.device_str,
                    num_workers=0,
                )
            out = _scalar_to_float(mos)
            if out is None:
                logger.warning(
                    "UTMOSv2 returned non-numeric score for %s: %r", wav_path, mos
                )
            return out
        except Exception as e:
            logger.warning("UTMOSv2 prediction failed for %s: %s", wav_path, e)
            return None

    @torch.no_grad()
    def _predict_dir_per_wav(self, input_dir: Path) -> Dict[str, float]:
        """Slow fallback: one file at a time (reliable when batch API fails)."""
        out: Dict[str, float] = {}
        paths = sorted(input_dir.glob("*.wav"))
        if not paths:
            return out
        logger.info("UTMOSv2 per-file fallback on %s wav(s) under %s", len(paths), input_dir)
        for p in paths:
            s = self.predict(p)
            if s is not None:
                out[p.name] = s
        return out

    @torch.no_grad()
    def predict_batch(self, input_dir: Path | str) -> Dict[str, float]:
        if not self.has_utmos:
            return {}
        input_dir = Path(input_dir)
        try:
            logger.info("Running batch UTMOSv2 prediction on %s", input_dir)
            with _no_amp_autocast():
                results = self.model.predict(
                    input_dir=str(input_dir),
                    device=self.device_str,
                    num_workers=0,
                )
            records = _utmos_results_to_records(results)
            mos_map: Dict[str, float] = {}
            for r in records:
                pair = _record_to_basename_mos(r)
                if pair is not None:
                    bn, score = pair
                    mos_map[bn] = score
                    fp = r.get("file_path", r.get("filepath", r.get("path")))
                    if fp is not None:
                        p = Path(str(fp))
                        mos_map[str(p)] = score
                        mos_map[str(p.resolve())] = score
            logger.info("UTMOSv2 batch parsed %s wav file(s)", len(mos_map))
            if not mos_map:
                mos_map = self._predict_dir_per_wav(input_dir)
            return mos_map
        except Exception as e:
            logger.warning("Batch UTMOSv2 prediction failed: %s; trying per-file.", e)
            return self._predict_dir_per_wav(input_dir)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pre-computed reference metrics (HuBERT ref transcriptions, ref WER/CER/UTMOS) for --skip ref path.
_REPO_ROOT = Path(__file__).resolve().parent
_GT_DIR_DEFAULT = _REPO_ROOT / "evaluation" / "GT"
_GT_DIR_LEGACY = Path("/home/ec2-user/MelCausalVAE/evaluation/GT")


def resolve_gt_metrics_path(gt_filename: str) -> Path:
    """Prefer repo-local ``evaluation/GT``; fall back to legacy EC2 path if present."""
    p = _GT_DIR_DEFAULT / gt_filename
    if p.is_file():
        return p
    legacy = _GT_DIR_LEGACY / gt_filename
    if legacy.is_file():
        return legacy
    return p


class HuBERTASR:
    """HuBERT-Large-LS960-ft based ASR for WER/CER computation."""

    def __init__(
        self, device: torch.device, model_name: str = "facebook/hubert-large-ls960-ft"
    ):
        logger.info(f"Loading HuBERT ASR model: {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = HubertForCTC.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file and return lowercase text."""
        audio, sr = torchaudio.load(audio_path)
        # HuBERT expects 16kHz mono
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


class ECAPASpeakerSimilarity:
    """
    ECAPA-TDNN speaker embeddings (SpeechBrain ``spkrec-ecapa-voxceleb``) on **CPU** for
    stable inference alongside CUDA/MPS eval. Cosine similarity between GT and reconstructed
    waveforms (higher = more similar speaker identity / timbre).
    """

    TARGET_SR = 16000

    def __init__(self):
        self.device = torch.device("cpu")
        self.classifier = None
        self.available = False
        self._savedir = _REPO_ROOT / ".cache" / "speechbrain_spkrec-ecapa-voxceleb"
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            logger.info(
                "Loading ECAPA-TDNN speaker encoder (SpeechBrain, CPU, %s Hz)",
                self.TARGET_SR,
            )
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self._savedir),
                run_opts={"device": "cpu"},
            )
            self.classifier.eval()
            # Global ``torch.set_default_dtype(bfloat16)`` (eval) would leave this model in bf16
            # while waveforms are float32; force fp32 for stable ECAPA inference.
            if hasattr(self.classifier, "float"):
                self.classifier.float()
            self.available = True
            logger.info("ECAPA speaker encoder ready.")
        except Exception as e:
            logger.warning("Failed to load ECAPA speaker encoder: %s", e)

    def _to_mono_2d(self, wave: torch.Tensor) -> torch.Tensor:
        x = wave.detach().float()
        if x.ndim == 2:
            x = x.mean(dim=0)
        else:
            x = x.flatten()
        if x.numel() == 0:
            raise ValueError("empty waveform")
        return x.view(1, -1)

    def _resample(self, wave_2d: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.TARGET_SR:
            return wave_2d
        return torchaudio.functional.resample(
            wave_2d, orig_freq=sr, new_freq=self.TARGET_SR
        )

    @staticmethod
    def _flatten_embedding(t: torch.Tensor) -> torch.Tensor:
        return t.detach().float().reshape(-1)

    @torch.inference_mode()
    def cosine_similarity(
        self,
        wav_gt: torch.Tensor,
        sr_gt: int,
        wav_recon: torch.Tensor,
        sr_recon: int,
    ) -> Optional[float]:
        if not self.available or self.classifier is None:
            return None
        try:
            a = self._resample(self._to_mono_2d(wav_gt), int(sr_gt)).to(
                self.device, dtype=torch.float32
            )
            b = self._resample(self._to_mono_2d(wav_recon), int(sr_recon)).to(
                self.device, dtype=torch.float32
            )
            if a.shape[-1] < 320 or b.shape[-1] < 320:
                logger.warning(
                    "ECAPA: waveform too short (%s / %s samples at 16 kHz); skipping",
                    a.shape[-1],
                    b.shape[-1],
                )
                return None
            # SpeechBrain forward may still allocate bf16 internals if default dtype is bf16.
            _prev_dt = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                ea = self.classifier.encode_batch(a)
                eb = self.classifier.encode_batch(b)
            finally:
                torch.set_default_dtype(_prev_dt)
            ea = self._flatten_embedding(ea).to(torch.float32)
            eb = self._flatten_embedding(eb).to(torch.float32)
            denom = ea.norm() * eb.norm() + 1e-8
            return float((ea @ eb) / denom)
        except Exception as e:
            logger.warning("ECAPA cosine similarity failed: %s", e)
            return None


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_eval_device(gpu_index: int = 0) -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU (matches training-side GPU when possible)."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu_index)}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model_from_config(
    config_dict: Dict[str, Any], checkpoint_path: Optional[str], device: torch.device
) -> Tuple[VAE, torch.nn.Module, str, VAEConfig]:

    encoder_cfg = ConvformerEncoderConfig(**config_dict["convformer"])  # type: ignore
    decoder_cfg = DiTConfig(**config_dict["cfm"])  # type: ignore
    decoder_cfg.expansion_factor = encoder_cfg.compress_factor_C
    mel_cfg = MelSpectrogramConfig(
        use_bigvgan_mel=config_dict["convformer"].get("use_bigvgan_mel", False)
    )  # type: ignore
    use_classic_decoder = config_dict.get("use_classic_decoder", False)
    vae_cfg = VAEConfig(
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
        mel_spec_config=mel_cfg,
        use_classic_decoder=use_classic_decoder,
    )

    model = VAE(vae_cfg, dtype=torch.bfloat16).to(device)
    model.from_pretrained(checkpoint_path)
    model.set_device(device)
    model.set_dtype(torch.bfloat16)
    model.eval()

    if mel_cfg.use_bigvgan_mel:
        try:
            bigvgan_path = (
                "/home/ec2-user/MelCausalVAE/bigvgan/bigvgan_v2_24khz_100band_256x"
            )
            if bigvgan_path not in sys.path:
                sys.path.append(bigvgan_path)
            import bigvgan

            vocoder = bigvgan.BigVGAN.from_pretrained(
                bigvgan_path, use_cuda_kernel=False
            )
            vocoder_type = "bigvgan"
        except Exception as e:
            logger.error(f"Failed to load BigVGAN vocoder: {e}. Falling back to Vocos.")
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            vocoder_type = "vocos"
    else:
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        vocoder_type = "vocos"

    vocoder.to(device)
    return model, vocoder, vocoder_type, vae_cfg


def mel_to_audio(
    vocoder: torch.nn.Module,
    mel: torch.Tensor,
    device: torch.device,
    vocoder_type: str = "vocos",
) -> torch.Tensor:
    # Keep vocoder input dtype aligned with vocoder parameters to avoid
    # conv dtype mismatches (e.g., bf16 input vs fp32 bias).
    if vocoder_type == "bigvgan":
        target_dtype = next(vocoder.parameters()).dtype
    else:
        target_dtype = next(vocoder.backbone.parameters()).dtype
    features = mel.permute(0, 2, 1).to(device=device, dtype=target_dtype)
    if vocoder_type == "bigvgan":
        waveform = vocoder(features)
    else:
        waveform = vocoder.decode(features)  # [1, samples]

    waveform = waveform.float().squeeze().detach().cpu()
    # normalize to prevent clipping
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform.view(-1)


def save_wav(path: Path, audio: torch.Tensor, sr: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        str(path), audio.unsqueeze(0).cpu().to(torch.float32), sample_rate=sr
    )


def get_available_gpus() -> List[int]:
    """Detect available GPUs without initializing CUDA contexts (safe before forking)."""
    try:
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def generate_hyperparam_combinations(
    hyperparam_config: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    if not hyperparam_config:
        return [{}]
    keys = list(hyperparam_config.keys())
    values = list(hyperparam_config.values())
    combos: List[Dict[str, Any]] = []

    # cartesian product
    def _prod(arrs: List[List[Any]], idx: int, cur: Dict[str, Any]):
        if idx == len(arrs):
            combos.append(dict(cur))
            return
        k = keys[idx]
        for v in arrs[idx]:
            cur[k] = v
            _prod(arrs, idx + 1, cur)
            cur.pop(k, None)

    _prod(values, 0, {})
    return combos


def evaluate_dataset(
    setting: str,
    languages: Optional[List[str]],
    num_samples: int,
    batch_size: int,
    model: Optional[VAE],
    vocoder: Optional[torch.nn.Module],
    vocoder_type: str,
    baseline_model: Optional[BaselineAudioCodec],
    n_steps: int,
    temperature: float,
    guidance_scale: float,
    device: torch.device,
    work_dir: Path,
    evaluator: Optional[UTMOSPredictor],
    hubert_asr: HuBERTASR,
    log_images: bool,
    max_images: int,
    keep_wavs: bool,
    filter_librispeech: bool,
    original_audio_only: bool,
    skip_ref_metrics: bool,
    run,
    latent_num_chunks: Optional[int] = None,
    latent_chunk_size: Optional[int] = None,
    spk_embedder: Optional[ECAPASpeakerSimilarity] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"samples": [], "aggregates": {}}

    # Map setting to specific GT metrics file
    gt_filename = "metrics.json"
    if setting == "LibriSpeech":
        gt_filename = "librispeech.json"
    elif setting == "LibriTTS":
        gt_filename = "libritts.json"
    elif setting == "LibriSpeech100":
        gt_filename = "librispeech100.json"

    gt_filepath = resolve_gt_metrics_path(gt_filename)
    gt_metrics_map: Dict[str, Dict[str, Any]] = {}
    if skip_ref_metrics:
        try:
            with open(gt_filepath, "r") as f:
                gt_data = json.load(f)
                for sample in gt_data.get("samples", []):
                    map_id = sample.get("id")
                    if map_id is not None:
                        gt_metrics_map[str(map_id)] = {
                            "ref": sample.get("ref", {}),
                            "ref_transcription": sample.get("ref_transcription", ""),
                        }
            logger.info(
                "Loaded %s pre-computed GT ref entries from %s",
                len(gt_metrics_map),
                gt_filepath,
            )
            if not gt_metrics_map:
                logger.warning(
                    "GT file %s loaded but contains no samples with ids; "
                    "ref metrics from cache will be empty.",
                    gt_filepath,
                )
        except Exception as e:
            logger.warning(
                "Could not load pre-computed GT metrics from %s: %s",
                gt_filepath,
                e,
            )
        if not gt_metrics_map:
            logger.warning(
                "Reference metrics are skipped (--compute-ref-metrics off) but the GT "
                "cache is empty or missing. Ref WER/CER/consistency will be nan. "
                "Expected file: %s",
                gt_filepath,
            )

    if setting == "LibriSpeech":
        ds = TestDatasetWrapper(
            LibriSpeechAlignTestDataset(do_filter=filter_librispeech), "test"
        )
    elif setting == "LibriTTS":
        ds = TestDatasetWrapper(LibriTTS(), "test")
    elif setting == "LibriSpeech100":
        ds = TestDatasetWrapper(LibriSpeech100h(for_evaluation=True), "test")
    else:
        raise NotImplementedError(f"{setting} not implemented")

    test_dataloader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=DataCollator()
    )

    # Aggregation structures
    per_language = {}

    tmp_audio_dir = work_dir / "tmp_wavs"
    tmp_audio_dir.mkdir(parents=True, exist_ok=True)
    img_dir = work_dir / "images"
    if log_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    image_count = 0
    if num_samples > 0:
        total_samples = min(num_samples, len(ds))
    else:
        total_samples = len(ds)
    pbar = tqdm(total=total_samples, desc="Evaluating dataset")
    dtype = torch.bfloat16
    images = []
    reconstructed_audios = []
    original_audios = []
    for batch in test_dataloader:
        audios_srs = batch["output_audios_srs"]
        sr = audios_srs[0][1]
        lang = batch["language"]
        gt_text = batch["transcription"]
        batch_ids = batch.get("ids")  # NO FALLBACK
        audios_srs = [(audio.to(device, dtype=dtype), sr) for audio, sr in audios_srs]

        if original_audio_only:
            original_mel = None
        elif baseline_model is None:
            # --- MelCausalVAE PATH ---
            with torch.no_grad():
                out = model.encode_and_sample(
                    audios_srs=audios_srs,
                    num_steps=n_steps,
                    temperature=temperature,
                    guidance_scale=guidance_scale,
                    generator=None,
                    latent_num_chunks=latent_num_chunks,
                    latent_chunk_size=latent_chunk_size,
                )
            original_mel = out["original_mel"].detach().cpu()
            reconstructed_mel = out["reconstructed_mel"].detach().cpu()
            recon_padding_mask = out["padding_mask"].detach().cpu()
            original_padding_mask = out["original_padding_mask"].detach().cpu()

        # save wavs, prompts, compute metrics
        for idx in range(len(audios_srs)):
            global_idx = processed + idx
            sample_id = batch_ids[idx]

            orig_wav = tmp_audio_dir / f"sample_{sample_id}_orig.wav"
            recon_wav = tmp_audio_dir / f"sample_{sample_id}_recon.wav"
            prompt_txt = tmp_audio_dir / f"sample_{sample_id}_prompt.txt"

            if original_audio_only:
                audio, original_sr = audios_srs[idx]
                original_audio = audio.squeeze().cpu()
                original_audio = original_audio / (original_audio.abs().max() + 1e-8)
                save_wav(orig_wav, original_audio, original_sr)
                prompt_txt.write_text(gt_text[idx])
                reconstructed_audio = original_audio
                ref_sr = recon_sr = int(original_sr)
                cur_original_mel = None
                cur_reconstructed_mel = None
            elif baseline_model is not None:
                # --- BASELINE MODEL PATH ---
                audio, original_sr = audios_srs[idx]
                reconstructed, out_sr = baseline_model.reconstruct(audio, original_sr)

                original_audio = audio.squeeze().cpu()
                reconstructed_audio = reconstructed.cpu()

                # normalize to prevent clipping
                original_audio = original_audio / (original_audio.abs().max() + 1e-8)
                reconstructed_audio = reconstructed_audio / (
                    reconstructed_audio.abs().max() + 1e-8
                )

                if not skip_ref_metrics:
                    save_wav(orig_wav, original_audio, original_sr)
                save_wav(recon_wav, reconstructed_audio, out_sr)
                prompt_txt.write_text(gt_text[idx])
                ref_sr = int(original_sr)
                recon_sr = int(out_sr)
                cur_original_mel = None
                cur_reconstructed_mel = None
            else:
                # Keep per-sample lengths independent. Using a shared min_length can
                # force equal durations for ref/recon and bias metric evaluation.
                cur_original_mel = original_mel[idx]
                cur_reconstructed_mel = reconstructed_mel[idx]
                opm = original_padding_mask[idx]
                rpm = recon_padding_mask[idx]

                to = min(cur_original_mel.size(0), opm.size(0))
                cur_original_mel = cur_original_mel[:to][~opm[:to]]

                tr = min(cur_reconstructed_mel.size(0), rpm.size(0))
                cur_reconstructed_mel = cur_reconstructed_mel[:tr][~rpm[:tr]]

                # Reference metrics should use the true dataset waveform (not vocoded mel).
                source_audio, source_sr = audios_srs[idx]
                original_audio = source_audio.squeeze().float().cpu()
                original_audio = original_audio / (original_audio.abs().max() + 1e-8)

                reconstructed_audio = mel_to_audio(
                    vocoder, cur_reconstructed_mel.unsqueeze(0), device, vocoder_type
                )

                if not skip_ref_metrics:
                    save_wav(orig_wav, original_audio, source_sr)
                save_wav(recon_wav, reconstructed_audio, sr)
                prompt_txt.write_text(gt_text[idx])
                ref_sr = int(source_sr)
                recon_sr = int(sr)

            # --- Compute WER and CER using HuBERT-Large ASR ---
            ref_text_normalized = gt_text[idx].lower().strip()
            if original_audio_only:
                ref_hyp = hubert_asr.transcribe(str(orig_wav))
                recon_hyp = ref_hyp
            else:
                if not skip_ref_metrics:
                    ref_hyp = hubert_asr.transcribe(str(orig_wav))
                else:
                    ref_info = gt_metrics_map.get(str(sample_id)) or {}
                    ref_hyp = ref_info.get("ref_transcription", "")
                recon_hyp = hubert_asr.transcribe(str(recon_wav))

            # WER / CER for original (reference audio)
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

            # Consistency Metrics (Speech Consistency between ref and recon)
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

            if spk_embedder is not None and spk_embedder.available:
                if original_audio_only:
                    recon_scores["ECAPA_cosine"] = 1.0
                else:
                    sim = spk_embedder.cosine_similarity(
                        original_audio,
                        ref_sr,
                        reconstructed_audio,
                        recon_sr,
                    )
                    if sim is not None:
                        recon_scores["ECAPA_cosine"] = round(float(sim), 4)

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
                    "orig_wav_name": orig_wav.name,
                    "recon_wav_name": recon_wav.name,
                }
            )

            # optionally save spectrogram images
            if log_images and image_count < max_images and cur_original_mel is not None:

                def _to_numpy(m: torch.Tensor):
                    m = m.squeeze(0)
                    return m.detach().cpu().float().numpy().T

                fig, axes = plt.subplots(2, 1, figsize=(12, 4))
                im0 = axes[0].imshow(
                    _to_numpy(cur_original_mel), aspect="auto", origin="lower"
                )
                axes[0].set_title(f"Original Mel #{idx}")
                im1 = axes[1].imshow(
                    _to_numpy(cur_reconstructed_mel), aspect="auto", origin="lower"
                )
                axes[1].set_title(f"Reconstructed Mel #{idx}")
                for ax in axes:
                    ax.set_xlabel("Frames")
                    ax.set_ylabel("Bins")
                out_img_path = img_dir / f"sample_{global_idx}.png"
                fig.tight_layout()
                fig.savefig(out_img_path)
                plt.close(fig)
                image_count += 1
                images.append(wandb.Image(fig))

            if log_images and image_count < max_images:
                # also log audio to wandb as audio files
                original_audios.append(
                    wandb.Audio(original_audio.numpy(), sample_rate=sr)
                )
                if not original_audio_only:
                    reconstructed_audios.append(
                        wandb.Audio(reconstructed_audio.numpy(), sample_rate=sr)
                    )

            # aggregate per language (partial)
            if setting == "OOD":
                key = lang[idx]
            else:
                key = "all"
            if key not in per_language:
                per_language[key] = {}
            for m in ["WER", "CER"]:
                if m in ref_scores and ref_scores[m] is not None:
                    per_language[key].setdefault(f"ref_{m}", [])
                    per_language[key][f"ref_{m}"].append(float(ref_scores[m]))
                if m in recon_scores and recon_scores[m] is not None:
                    per_language[key].setdefault(f"recon_{m}", [])
                    per_language[key][f"recon_{m}"].append(float(recon_scores[m]))

            ec = recon_scores.get("ECAPA_cosine")
            if ec is not None:
                per_language[key].setdefault("recon_ECAPA_cosine", []).append(float(ec))

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

    # --- Batch UTMOS Optimization ---
    mos_map = {}
    if evaluator is not None and evaluator.has_utmos:
        mos_map = evaluator.predict_batch(tmp_audio_dir)
        if not mos_map:
            logger.warning(
                "UTMOSv2 batch returned no scores under %s; reconstructed UTMOS will be "
                "missing in metrics (check model.predict / file paths).",
                tmp_audio_dir,
            )

    def _lookup_mos_map(name: Optional[str]) -> Optional[float]:
        if not name or not mos_map:
            return None
        if name in mos_map:
            return float(mos_map[name])
        p = tmp_audio_dir / name
        for k in (str(p), str(p.resolve())):
            if k in mos_map:
                return float(mos_map[k])
        return None

    for sample in results["samples"]:
        orig_wav_name = sample.pop("orig_wav_name", None)
        recon_wav_name = sample.pop("recon_wav_name", None)
        key = sample["language"] if setting == "OOD" else "all"
        sid = sample["id"]
        recon_path = tmp_audio_dir / f"sample_{sid}_recon.wav"
        orig_path = tmp_audio_dir / f"sample_{sid}_orig.wav"

        if mos_map:
            if original_audio_only:
                orig_mos = _lookup_mos_map(orig_wav_name)
                if orig_mos is not None:
                    sample["ref"]["UTMOS"] = round(orig_mos, 2)
            else:
                orig_mos = _lookup_mos_map(orig_wav_name)
                recon_mos = _lookup_mos_map(recon_wav_name)
                # Ref UTMOS from utmos only when we run ref HuBERT/wavs; otherwise ref
                # UTMOS (if any) already came from evaluation/GT/*.json via ref_scores.
                if orig_mos is not None and not skip_ref_metrics:
                    sample["ref"]["UTMOS"] = round(orig_mos, 2)
                if recon_mos is not None:
                    sample["reconstructed"]["UTMOS"] = round(recon_mos, 2)

        # Batch map can be empty or keys may not match; always score recon wav directly.
        if (
            evaluator is not None
            and evaluator.has_utmos
            and not original_audio_only
            and recon_path.is_file()
            and sample["reconstructed"].get("UTMOS") is None
        ):
            fv = evaluator.predict(recon_path)
            if fv is not None:
                sample["reconstructed"]["UTMOS"] = round(float(fv), 2)
            else:
                logger.warning(
                    "UTMOSv2 per-file recon failed for %s (basename %s)",
                    recon_path,
                    recon_wav_name,
                )

        if (
            evaluator is not None
            and evaluator.has_utmos
            and not skip_ref_metrics
            and orig_path.is_file()
            and sample["ref"].get("UTMOS") is None
        ):
            fv = evaluator.predict(orig_path)
            if fv is not None:
                sample["ref"]["UTMOS"] = round(float(fv), 2)

        # Always aggregate UTMOS from the final per-sample values.
        # This covers both computed scores and reference scores loaded from GT.
        ref_utmos = sample.get("ref", {}).get("UTMOS")
        recon_utmos = sample.get("reconstructed", {}).get("UTMOS")
        if ref_utmos is not None:
            per_language[key].setdefault("ref_UTMOS", []).append(float(ref_utmos))
        if recon_utmos is not None:
            per_language[key].setdefault("recon_UTMOS", []).append(float(recon_utmos))

    # Calculate final delta for all samples
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

    # Cleanup temporary WAVs if requested
    if not keep_wavs:
        logger.info(f"Cleaning up temporary WAV files in {tmp_audio_dir}")
        import shutil

        try:
            shutil.rmtree(tmp_audio_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup {tmp_audio_dir}: {e}")

    log_payload = {}
    if images:
        log_payload["images"] = images
    if original_audios:
        log_payload["original_audios"] = original_audios
    if reconstructed_audios:
        log_payload["reconstructed_audios"] = reconstructed_audios

    if run is not None and log_payload:
        run.log(log_payload)

    # finalize aggregates
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
        recon_ecapa = _mean(agg.get("recon_ECAPA_cosine", []))
        consist_wer = _mean(agg.get("consistent_WER", []))
        consist_cer = _mean(agg.get("consistent_CER", []))

        def to_report(v, is_mos=False):
            if v is None or math.isnan(v):
                return None
            return round(float(v), 2)

        def ecapa_report(v: Optional[float]) -> Optional[float]:
            if v is None or math.isnan(v):
                return None
            return round(float(v), 4)

        agg_ref = {
            "WER": to_report(ref_wer),
            "CER": to_report(ref_cer),
            "UTMOS": to_report(ref_utmos, True),
        }
        agg_recon = {
            "WER": to_report(recon_wer),
            "CER": to_report(recon_cer),
            "UTMOS": to_report(recon_utmos, True),
            "ECAPA_cosine": ecapa_report(recon_ecapa),
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

    return results


def save_json(output: Dict[str, Any], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved metrics to {out_path}")


def run_single_eval(
    checkpoint: str,
    gpu_index: int,
    base_args: Dict[str, Any],
    hyperparams: Dict[str, Any],
    timeout: int,
    run,
) -> Dict[str, Any]:
    device = pick_eval_device(gpu_index)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    logger.info("Using eval compute device: %s", device)
    set_seed(base_args.get("seed", 42))

    baseline_model_name = base_args.get("baseline_model")
    original_audio_only = base_args.get("original_audio_only", False)
    model = None
    vocoder = None
    vocoder_type = "vocos"
    baseline_model = None

    if original_audio_only:
        pass  # Skip loading everything
    elif baseline_model_name:
        baseline_hz = base_args.get("baseline_hz")
        baseline_tau = base_args.get("baseline_tau")
        logger.info(f"Using baseline model: {baseline_model_name}")
        baseline_model = BaselineAudioCodec(
            model_name=baseline_model_name,
            device=device,
            baseline_hz=baseline_hz,
            baseline_tau=baseline_tau,
        )
    else:
        # build model
        model, vocoder, vocoder_type, vae_cfg = build_model_from_config(
            base_args["config_dict"], checkpoint, device
        )

    # HuBERT-Large ASR for WER/CER
    hubert_asr = HuBERTASR(device)

    # Standalone UTMOS Predictor
    evaluator = None
    if base_args.get("UTMOS"):
        evaluator = UTMOSPredictor(device)
        if not evaluator.has_utmos:
            logger.error(
                "--utmos was set but utmosv2 did not load; reconstructed UTMOS will be absent. "
                "See warning above for the exception."
            )

    spk_embedder: Optional[ECAPASpeakerSimilarity] = None
    if base_args.get("ecapa_speaker_similarity"):
        spk_embedder = ECAPASpeakerSimilarity()
        if not spk_embedder.available:
            logger.error(
                "--ecapa-speaker-similarity was set but the ECAPA model did not load; "
                "ECAPA_cosine will be absent. See warning above."
            )

    # resolve args
    setting = base_args["setting"]
    languages = base_args.get("languages")
    num_samples = base_args["num_samples"]
    batch_size = base_args["batch_size"]
    n_steps = int(hyperparams.get("n_steps", base_args["n_steps"]))
    temperature = float(hyperparams.get("temperature", base_args["temperature"]))
    guidance_scale = float(
        hyperparams.get("guidance_scale", base_args["guidance_scale"])
    )
    log_images = bool(base_args.get("wandb_log_images", False))
    wandb_max_images = int(base_args.get("wandb_max_images", 10))

    if original_audio_only:
        work_dir = (
            Path(base_args["output_dir"]) / base_args["exp_name"] / "original_audio"
        )
    elif baseline_model_name:
        # Reorganized baseline structure: evaluation/baseline_run/<exp_name>/[param_disambiguation]
        work_dir = (
            Path(base_args["output_dir"]) / "baseline_run" / base_args["exp_name"]
        )
        if baseline_model_name == "dualcodec" and base_args.get("baseline_hz"):
            work_dir = work_dir / f"hz{base_args['baseline_hz']}"
        elif (
            baseline_model_name == "flexicodec"
            and base_args.get("baseline_tau") is not None
        ):
            work_dir = work_dir / f"tau{base_args['baseline_tau']}"
    else:
        ckpt_path = Path(checkpoint)
        ckpt_dir_name = (
            ckpt_path.parent.name
            if ckpt_path.name == "model.safetensors"
            else ckpt_path.name
        )
        work_dir = (
            Path(base_args["output_dir"])
            / base_args["exp_name"]
            / ckpt_dir_name
            / f"gpu{gpu_index}_n{n_steps}_t{temperature}_g{guidance_scale}"
        )
        ln = base_args.get("latent_num_chunks")
        if ln is not None:
            work_dir = work_dir / f"latent_nchunks{int(ln)}"
    work_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    try:
        eval_output = evaluate_dataset(
            setting=setting,
            languages=languages,
            num_samples=num_samples,
            batch_size=batch_size,
            model=model,
            vocoder=vocoder,
            vocoder_type=vocoder_type,
            baseline_model=baseline_model,
            n_steps=n_steps,
            temperature=temperature,
            guidance_scale=guidance_scale,
            device=device,
            work_dir=work_dir,
            evaluator=evaluator,
            hubert_asr=hubert_asr,
            log_images=log_images,
            max_images=wandb_max_images,
            keep_wavs=base_args.get("keep_wavs", False),
            filter_librispeech=base_args.get("filter_librispeech", False),
            original_audio_only=original_audio_only,
            skip_ref_metrics=(not base_args.get("compute_ref_metrics", False))
            and (not original_audio_only),
            run=run,
            latent_num_chunks=base_args.get("latent_num_chunks"),
            latent_chunk_size=base_args.get("latent_chunk_size"),
            spk_embedder=spk_embedder,
        )
        status = "success"
        error = None
    except Exception as e:
        logger.exception("Evaluation failed")
        eval_output = {}
        status = "failed"
        error = str(e)
    duration = time.time() - start

    # save json for this run
    out_path = work_dir / "metrics.json"
    if eval_output:
        save_json(eval_output, out_path)

    return {
        "checkpoint": checkpoint,
        "gpu_index": gpu_index,
        "hyperparams": {
            "n_steps": n_steps,
            "temperature": temperature,
            "guidance_scale": guidance_scale,
            "latent_num_chunks": base_args.get("latent_num_chunks"),
            "latent_chunk_size": base_args.get("latent_chunk_size"),
            "ecapa_speaker_similarity": base_args.get("ecapa_speaker_similarity", False),
        },
        "status": status,
        "duration": duration,
        "metrics_path": str(out_path) if eval_output else None,
        "work_dir": str(work_dir),
        "error": error,
    }


def _log_wandb_latent_chunk_audio_table(
    *,
    run: Any,
    checkpoint: str,
    combo: Dict[str, Any],
    base_args: Dict[str, Any],
    gpu_index: int,
) -> None:
    """
    Log a list of ``wandb.Audio`` (with captions) under ``latent_chunk_audio_gallery`` so
    the W&B **Media** panel shows playable players. Tables with audio cells often surface
    only as JSON downloads instead of an in-app gallery.
    """
    if run is None:
        return
    chunk_counts: List[int] = base_args["wandb_latent_chunk_counts"]
    if not chunk_counts:
        return

    config_dict = base_args["config_dict"]
    assert config_dict is not None
    cf = config_dict["convformer"]
    latent_dim = int(cf["latent_dim"])
    chunk_sz = (
        int(base_args["latent_chunk_size"])
        if base_args.get("latent_chunk_size") is not None
        else int(cf.get("latent_chunk_size", 2))
    )
    if latent_dim % chunk_sz != 0:
        raise ValueError(
            f"latent_dim {latent_dim} must be divisible by latent_chunk_size {chunk_sz}"
        )
    max_chunks = latent_dim // chunk_sz
    for n in chunk_counts:
        if n < 1 or n > max_chunks:
            raise ValueError(
                f"--wandb-latent-chunk-counts must be within [1, {max_chunks}] "
                f"for this config; got {n}"
            )

    n_steps = int(combo.get("n_steps", base_args["n_steps"]))
    temperature = float(combo.get("temperature", base_args["temperature"]))
    guidance_scale = float(combo.get("guidance_scale", base_args["guidance_scale"]))
    latent_chunk_size_opt: Optional[int] = base_args.get("latent_chunk_size")

    device = pick_eval_device(gpu_index)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    set_seed(base_args.get("seed", 42))

    model, vocoder, vocoder_type, _ = build_model_from_config(
        config_dict, checkpoint, device
    )
    dtype = torch.bfloat16

    setting = base_args["setting"]
    filter_ls = base_args.get("filter_librispeech", False)
    if setting == "LibriSpeech":
        ds = TestDatasetWrapper(
            LibriSpeechAlignTestDataset(do_filter=filter_ls), "test"
        )
    elif setting == "LibriTTS":
        ds = TestDatasetWrapper(LibriTTS(), "test")
    elif setting == "LibriSpeech100":
        ds = TestDatasetWrapper(LibriSpeech100h(for_evaluation=True), "test")
    else:
        raise NotImplementedError(
            f"--wandb-latent-chunk-audio-table: setting {setting} not implemented"
        )

    num_samples_cfg = int(base_args.get("num_samples", 0))
    cap = int(base_args.get("wandb_latent_chunk_table_max_samples", 24))
    if num_samples_cfg > 0:
        total_samples = min(num_samples_cfg, len(ds))
    else:
        total_samples = min(cap, len(ds))

    batch_size = min(int(base_args.get("batch_size", 4)), max(total_samples, 1))
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=DataCollator()
    )

    gallery: List[wandb.Audio] = []

    processed = 0
    logger.info(
        "Building W&B latent-chunk audio gallery (%s samples × %s chunk settings)",
        total_samples,
        len(chunk_counts),
    )
    for batch in loader:
        audios_srs = batch["output_audios_srs"]
        sr = int(audios_srs[0][1])
        batch_ids = batch.get("ids")
        audios_srs_bf16 = [
            (audio.to(device, dtype=dtype), s_r) for audio, s_r in audios_srs
        ]
        bsz = len(audios_srs_bf16)
        for idx in range(bsz):
            if processed >= total_samples:
                break
            sample_id = batch_ids[idx] if batch_ids is not None else processed
            one = [(audios_srs_bf16[idx][0], audios_srs_bf16[idx][1])]

            source_audio = one[0][0].squeeze().float().cpu()
            source_audio = source_audio / (source_audio.abs().max() + 1e-8)
            sid = str(sample_id)
            gt_np = source_audio.numpy().astype("float32", copy=False)
            gallery.append(
                wandb.Audio(
                    gt_np,
                    sample_rate=sr,
                    caption=f"{sid} · ground truth",
                )
            )

            for n_chunks in chunk_counts:
                with torch.no_grad():
                    out = model.encode_and_sample(
                        audios_srs=one,
                        num_steps=n_steps,
                        temperature=temperature,
                        guidance_scale=guidance_scale,
                        generator=None,
                        latent_num_chunks=n_chunks,
                        latent_chunk_size=latent_chunk_size_opt,
                    )
                rec_mel = out["reconstructed_mel"][0]
                rpm = out["padding_mask"][0]
                tr = min(rec_mel.size(0), rpm.size(0))
                cur_rec = rec_mel[:tr][~rpm[:tr]]
                wav = mel_to_audio(
                    vocoder, cur_rec.unsqueeze(0), device, vocoder_type
                )
                w_np = wav.numpy().astype("float32", copy=False)
                gallery.append(
                    wandb.Audio(
                        w_np,
                        sample_rate=sr,
                        caption=f"{sid} · encoded {n_chunks} chunk(s)",
                    )
                )
            processed += 1
        if processed >= total_samples:
            break

    # Single log: Media panel → pick ``latent_chunk_audio_gallery`` for playable clips.
    run.log({"latent_chunk_audio_gallery": gallery})
    logger.info(
        "Logged %s wandb.Audio clip(s) as latent_chunk_audio_gallery (see W&B Media tab)",
        len(gallery),
    )


def orchestrate(
    checkpoints: List[str],
    base_args: Dict[str, Any],
    hyperparam_combinations: Optional[List[Dict[str, Any]]],
    max_workers: Optional[int],
    timeout: int,
    multi_gpu: bool,
) -> List[Dict[str, Any]]:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # generate task list
    if hyperparam_combinations is None:
        hyperparam_combinations = [{}]
    tasks: List[Tuple[str, Dict[str, Any]]] = []
    for ckpt in checkpoints:
        for combo in hyperparam_combinations:
            tasks.append((ckpt, combo))

    gpus = get_available_gpus()

    use_wandb = base_args.get("report_to") == "wandb"
    run = None
    if use_wandb:
        wb_config: Dict[str, Any] = {
            "setting": base_args["setting"],
            "languages": base_args["languages"],
        }
        if base_args.get("wandb_latent_chunk_audio_table"):
            wb_config["wandb_latent_chunk_counts"] = base_args.get(
                "wandb_latent_chunk_counts", []
            )
        run = wandb.init(
            project="MelCausalVAE-eval",
            name=base_args["exp_name"],
            config=wb_config,
        )

    # Single-process path (default): run sequentially on first GPU (or CPU if none)
    if not multi_gpu:
        if gpus:
            gpu_index = gpus[0]
        else:
            gpu_index = 0  # CUDA index only; run_single_eval may still use MPS or CPU
        dev = pick_eval_device(gpu_index)
        logger.info(
            f"Running in single-process mode on gpu_index={gpu_index} "
            f"(cuda_gpus={gpus}, resolved_device={dev}) with {len(tasks)} task(s)"
        )
        results: List[Dict[str, Any]] = []
        for i, (ckpt, combo) in enumerate(tasks):
            logger.info(f"Running task {i+1}/{len(tasks)}: ckpt={ckpt}, combo={combo}")
            res = run_single_eval(ckpt, gpu_index, base_args, combo, timeout, run)
            logger.info(
                f"Task completed: {res.get('checkpoint')} on GPU {res.get('gpu_index')} status={res.get('status')}"
            )
            results.append(res)
        if (
            run is not None
            and base_args.get("wandb_latent_chunk_audio_table")
            and not base_args.get("baseline_model")
            and not base_args.get("original_audio_only")
        ):
            if len(tasks) > 1:
                logger.warning(
                    "--wandb-latent-chunk-audio-table: multiple eval tasks; "
                    "table uses the first checkpoint and first hyperparam combo only."
                )
            try:
                ck0, combo0 = tasks[0]
                _log_wandb_latent_chunk_audio_table(
                    run=run,
                    checkpoint=ck0,
                    combo=combo0,
                    base_args=base_args,
                    gpu_index=gpu_index,
                )
            except Exception:
                logger.exception("Failed to log W&B latent-chunk audio table")
        return results

    # Multi-GPU parallel path
    if base_args.get("wandb_latent_chunk_audio_table"):
        logger.warning(
            "--wandb-latent-chunk-audio-table is ignored in --multi-gpu mode "
            "(omit --multi-gpu to log the table)."
        )
    if not gpus:
        raise RuntimeError("No available GPUs found for --multi-gpu mode")
    max_workers = max_workers or len(gpus)

    # Use 'spawn' to avoid CUDA fork-related deadlocks
    ctx = mp.get_context("spawn")

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = {}
        for i, (ckpt, combo) in enumerate(tasks):
            gpu_index = gpus[i % len(gpus)]
            logger.info(f"Submitting task: ckpt={ckpt}, gpu={gpu_index}, combo={combo}")
            futures[
                ex.submit(
                    run_single_eval, ckpt, gpu_index, base_args, combo, timeout, run
                )
            ] = (
                ckpt,
                combo,
                gpu_index,
            )
        for fut in as_completed(futures):
            res = fut.result()
            logger.info(
                f"Task completed: {res.get('checkpoint')} on GPU {res.get('gpu_index')} status={res.get('status')}"
            )
            results.append(res)

    if run is not None:
        run.finish()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VAE reconstructions with CER and UTMOS, with GPU orchestration"
    )
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument(
        "--report-to", type=str, default="none", choices=["none", "wandb"]
    )
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--guidance-scale", type=float, default=1.3)
    parser.add_argument(
        "--setting",
        type=str,
        default="LibriSpeech",
        choices=["LibriTTS", "OOD", "LibriSpeech", "LibriSpeech100"],
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Languages for OOD setting (e.g., english german spanish)",
    )
    parser.add_argument(
        "--wandb-log-images",
        action="store_true",
        help="If set, also log spectrogram images to W&B",
    )
    parser.add_argument(
        "--wandb-max-images",
        type=int,
        default=10,
        help="Maximum number of images to log (>1)",
    )

    # Baseline arguments
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=None,
        choices=["encodec", "mimi", "dualcodec", "xytokenizer", "flexicodec"],
        help="If specified, bypasses the internal VAE and tests one of the baseline audio codecs.",
    )
    parser.add_argument(
        "--baseline-hz",
        type=int,
        default=None,
        help="Frame rate for dualcodec (12 or 25)",
    )
    parser.add_argument(
        "--baseline-tau",
        type=float,
        default=None,
        help="Tau merging_threshold for flexicodec (e.g. 1.0, 0.91, 0.867)",
    )

    # orchestration inputs
    parser.add_argument(
        "--checkpoints", nargs="+", help="List of checkpoints to evaluate"
    )
    parser.add_argument(
        "--checkpoints-file", type=str, help="File with one checkpoint path per line"
    )

    parser.add_argument(
        "--original-audio-only",
        action="store_true",
        help="Bypass model entirely and only evaluate original audio metrics.",
    )
    parser.add_argument(
        "--compute-ref-metrics",
        action="store_true",
        help="Compute reference metrics (off by default to speed up evaluation).",
    )
    parser.add_argument(
        "--hyperparam-sweep",
        type=str,
        help="JSON path for hyperparam sweep with keys: n_steps, temperature, guidance_scale",
    )
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="default_exp")
    parser.add_argument(
        "--output-dir", type=str, default="/home/ec2-user/MelCausalVAE/evaluation"
    )
    parser.add_argument(
        "--multi-gpu", action="store_true", help="Enable multi-GPU parallel inference"
    )
    parser.add_argument(
        "--utmos",
        "--UTMOS",
        action="store_true",
        dest="utmos",
        help="Run utmosv2 on temp wavs to score **reconstructed** audio (and reference wavs "
        "when --compute-ref-metrics is set). Without this flag, ref UTMOS may still appear "
        "from evaluation/GT/*.json when ref metrics are skipped.",
    )
    parser.add_argument(
        "--ecapa-speaker-similarity",
        action="store_true",
        help="Cosine similarity between ECAPA-TDNN embeddings (SpeechBrain spkrec-ecapa-voxceleb) "
        "of ground-truth vs reconstructed audio (CPU, resampled to 16 kHz). Stored as "
        "reconstructed.ECAPA_cosine per sample and in aggregates (1.0 for --original-audio-only).",
    )
    parser.add_argument(
        "--keep-wavs",
        action="store_true",
        help="Keep temporary WAV files after evaluation",
    )
    parser.add_argument(
        "--filter-librispeech",
        action="store_true",
        help="If set, filters the LibriSpeech test set using the E2TTS JSON keys",
    )
    parser.add_argument(
        "--latent-num-chunks",
        type=int,
        default=None,
        help="Bottleneck ablation: decode using only the first N latent channel chunks "
        "(1 .. latent_dim/chunk_size); other channels zeroed (mu). Omit for full z.",
    )
    parser.add_argument(
        "--latent-chunk-size",
        type=int,
        default=None,
        help="Channels per chunk for --latent-num-chunks (default: convformer latent_chunk_size).",
    )
    parser.add_argument(
        "--wandb-latent-chunk-audio-table",
        action="store_true",
        help="Log latent_chunk_audio_gallery: a list of wandb.Audio (with captions) for the W&B "
        "**Media** panel — GT plus one clip per N in --wandb-latent-chunk-counts (first N latent "
        "chunks at decode). Requires --report-to wandb. Runs after eval in single-process mode; "
        "uses the first checkpoint and first hyperparam combo if several tasks are queued.",
    )
    parser.add_argument(
        "--wandb-latent-chunk-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="Latent chunk counts for extra encoded clips in --wandb-latent-chunk-audio-table gallery.",
    )
    parser.add_argument(
        "--wandb-latent-chunk-table-max-samples",
        type=int,
        default=24,
        help="Max samples in the W&B chunk gallery per run when --num-samples is 0.",
    )

    args = parser.parse_args()

    if args.utmos:
        logger.info(
            "Neural MOS (utmosv2) requested: will score wavs under each run's tmp_wavs/."
        )
    if args.ecapa_speaker_similarity:
        logger.info(
            "ECAPA speaker similarity requested: install speechbrain if missing; runs on CPU."
        )

    # set default dtype to bfloat16
    torch.set_default_dtype(torch.bfloat16)

    # resolve checkpoints (not required for baselines)
    checkpoints: List[str] = []
    if args.checkpoints_file:
        with open(args.checkpoints_file, "r") as f:
            checkpoints = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if args.checkpoints:
        checkpoints.extend(args.checkpoints)

    if args.baseline_model is None and not checkpoints and not args.original_audio_only:
        raise ValueError(
            "No checkpoints provided. Use --checkpoints or --checkpoints-file"
        )

    if (
        args.baseline_model is not None or args.original_audio_only
    ) and not checkpoints:
        # Dummy checkpoint for iteration
        checkpoints = ["baseline_run" if args.baseline_model else "original_audio"]

    # load config dict once and pass down
    config_dict = None
    if args.config_path:
        config_dict = load_config(args.config_path)
    elif args.baseline_model is None and not args.original_audio_only:
        raise ValueError("Internal model evaluation requires --config-path")

    if args.latent_num_chunks is not None:
        if args.baseline_model is not None or args.original_audio_only:
            raise ValueError(
                "--latent-num-chunks applies only to internal VAE evaluation."
            )
        assert config_dict is not None
        cf = config_dict["convformer"]
        latent_dim = int(cf["latent_dim"])
        chunk_sz = (
            int(args.latent_chunk_size)
            if args.latent_chunk_size is not None
            else int(cf.get("latent_chunk_size", 2))
        )
        if latent_dim % chunk_sz != 0:
            raise ValueError(
                f"latent_dim {latent_dim} must be divisible by latent_chunk_size {chunk_sz}"
            )
        max_chunks = latent_dim // chunk_sz
        if args.latent_num_chunks < 1 or args.latent_num_chunks > max_chunks:
            raise ValueError(
                f"--latent-num-chunks must be in [1, {max_chunks}] for this config, "
                f"got {args.latent_num_chunks}"
            )

    wandb_latent_chunk_counts = list(dict.fromkeys(args.wandb_latent_chunk_counts))
    if args.wandb_latent_chunk_audio_table:
        if args.report_to != "wandb":
            raise ValueError(
                "--wandb-latent-chunk-audio-table requires --report-to wandb"
            )
        if args.baseline_model is not None or args.original_audio_only:
            raise ValueError(
                "--wandb-latent-chunk-audio-table applies only to internal VAE evaluation."
            )
        assert config_dict is not None
        cf = config_dict["convformer"]
        latent_dim = int(cf["latent_dim"])
        chunk_sz = (
            int(args.latent_chunk_size)
            if args.latent_chunk_size is not None
            else int(cf.get("latent_chunk_size", 2))
        )
        if latent_dim % chunk_sz != 0:
            raise ValueError(
                f"latent_dim {latent_dim} must be divisible by latent_chunk_size {chunk_sz}"
            )
        max_chunks = latent_dim // chunk_sz
        for n in wandb_latent_chunk_counts:
            if n < 1 or n > max_chunks:
                raise ValueError(
                    f"--wandb-latent-chunk-counts must be within [1, {max_chunks}] "
                    f"for this config; got {n}"
                )

    # hyperparam combinations
    if args.hyperparam_sweep:
        with open(args.hyperparam_sweep, "r") as f:
            sweep_cfg = json.load(f)
        hyperparam_combinations = generate_hyperparam_combinations(sweep_cfg)
    else:
        hyperparam_combinations = generate_hyperparam_combinations(
            {
                "n_steps": [args.n_steps],
                "temperature": [args.temperature],
                "guidance_scale": [args.guidance_scale],
            }
        )
        # collapse if singular defaults
        if len(hyperparam_combinations) == 1:
            hyperparam_combinations = None

    # validate wandb-max-images
    if args.wandb_max_images is not None and args.wandb_max_images <= 1:
        raise ValueError("--wandb-max-images must be greater than 1")

    base_args: Dict[str, Any] = {
        "config_dict": config_dict,
        "report_to": args.report_to,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "temperature": args.temperature,
        "guidance_scale": args.guidance_scale,
        "setting": args.setting,
        "languages": args.languages,
        "seed": args.seed,
        "exp_name": args.exp_name,
        "output_dir": args.output_dir,
        "wandb_log_images": args.wandb_log_images,
        "wandb_max_images": args.wandb_max_images,
        "baseline_model": args.baseline_model,
        "baseline_hz": args.baseline_hz,
        "baseline_tau": args.baseline_tau,
        "UTMOS": args.utmos,
        "ecapa_speaker_similarity": args.ecapa_speaker_similarity,
        "keep_wavs": args.keep_wavs,
        "filter_librispeech": args.filter_librispeech,
        "original_audio_only": args.original_audio_only,
        "compute_ref_metrics": args.compute_ref_metrics,
        "latent_num_chunks": args.latent_num_chunks,
        "latent_chunk_size": args.latent_chunk_size,
        "wandb_latent_chunk_audio_table": args.wandb_latent_chunk_audio_table,
        "wandb_latent_chunk_counts": wandb_latent_chunk_counts,
        "wandb_latent_chunk_table_max_samples": args.wandb_latent_chunk_table_max_samples,
    }

    # orchestrate one job per GPU
    results = orchestrate(
        checkpoints=checkpoints,
        base_args=base_args,
        hyperparam_combinations=hyperparam_combinations,
        max_workers=args.max_workers,
        timeout=args.timeout,
        multi_gpu=args.multi_gpu,
    )

    # # wandb: upload json metrics, and optionally images
    # if args.report_to == "wandb":
    #     try:
    #         import wandb

    #         wandb.save(str(summary_path))
    #         if args.wandb_log_images:
    #             # collect image files across runs up to the cap
    #             image_files: List[str] = []
    #             for res in results:
    #                 work_dir = res.get("work_dir")
    #                 if not work_dir:
    #                     continue
    #                 img_dir = Path(work_dir) / "images"
    #                 if img_dir.exists():
    #                     for p in sorted(img_dir.glob("*.png")):
    #                         image_files.append(str(p))
    #                         if len(image_files) >= args.wandb_max_images:
    #                             break
    #                 if len(image_files) >= args.wandb_max_images:
    #                     break
    #             if image_files:
    #                 wandb.log({"mel_spectrograms": [wandb.Image(p) for p in image_files]})
    #         run.finish()
    #     except Exception:
    #         logger.warning("W&B logging failed; continuing without upload")


if __name__ == "__main__":
    main()

# python eval.py \
#   --config-path /home/ec2-user/MelCausalVAE/configs/settings/setting1.yaml \
#   --checkpoints /home/ec2-user/checkpoints/setting1/checkpoint-44000/model.safetensors \
#   --setting ID \
#   --num-samples 24 --batch-size 4 \
#   --n-steps 4 --temperature 0.2 --guidance-scale 1.3 \
#   --exp-name ID_eval \
#   --output-dir /home/ec2-user/MelCausalVAE/evaluation \
#   --report-to wandb \
#   --wandb-log-images \
#   --wandb-max-images 24

# LibriSpeech test set evaluation with HuBERT ASR:
# python eval.py \
#   --config-path configs/settings/exps/1d.yaml \
#   --checkpoints checkpoints/exps/1d-8x-NAR-AE/model.safetensors \
#   --setting LibriSpeech100-eval \
#   --num-samples 250 --batch-size 8 \
#   --n-steps 6 --temperature 0.4 --guidance-scale 1.3 \
#   --exp-name LibriSpeech100-eval \
#   --output-dir evaluation

# latent factorized eval

# latent factorized eval
# python eval.py \
#   --config-path configs/settings/bottleneck-ablations/dropout_eval.yaml \
#   --checkpoints checkpoints/ablations/dropout-5-epochs/model.safetensors \
#   --setting LibriSpeech \
#   --filter-librispeech \
#   --num-samples 10 --batch-size 8 \
#   --n-steps 6 --temperature 0.4 --guidance-scale 1.3 \
#   --exp-name dropout_eval \
#   --output-dir evaluation/bottleneck-ablations/LibriSpeech \
#   --UTMOS \
#   --latent-num-chunks 1
