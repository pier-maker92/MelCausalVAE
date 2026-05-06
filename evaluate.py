import os
import json
import torch
import torchaudio
import wandb
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from jiwer import wer as compute_wer, cer as compute_cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

logger = logging.getLogger(__name__)


class UTMOSPredictor:
    """Standalone UTMOS predictor using utmosv2."""

    def __init__(self, device: torch.device):
        logger.info("Initializing UTMOSv2 model")
        import utmosv2

        self.model = utmosv2.create_model(pretrained=True, device=str(device))
        self.device = device
        logger.info("UTMOSv2 model loaded successfully.")

    @torch.no_grad()
    def predict(self, wav_path: str) -> Optional[float]:
        # Disable utmosv2 multiprocessing to avoid issues
        mos = self.model.predict(
            input_path=str(wav_path),
            device=str(self.device),
            num_workers=0,
        )
        return float(mos)


class WhisperASR:
    """Whisper based ASR for WER/CER computation."""

    def __init__(
        self, device: torch.device, model_name: str = "openai/whisper-large-v3"
    ):
        logger.info(f"Loading Whisper ASR model: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(
            device
        )
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def transcribe(self, audio: torch.Tensor, sr: int) -> str:
        """Transcribe audio tensor and return lowercase text."""
        # Whisper expects 16kHz mono
        if audio.dim() > 1:
            audio = audio.mean(dim=0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        input_features = self.processor(
            audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device).to(self.model.dtype)

        # Generate transcription
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return transcription.lower().strip()


def run_evaluation(
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    vocoder_type: str,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    dataset_name: str,
    num_samples: int = 100,
    run_id: str = "default_run",
) -> Dict[str, float]:
    """
    Perform evaluation on 100 samples from the test set.
    Calculates UTMOS, WER, CER for ground truth and reconstructed audio.
    Logs to wandb and saves to local CSV.
    """
    model.eval()
    eval_dir = Path("evaluation")
    gt_cache_path = eval_dir / f"{dataset_name}_ground_truth.json"
    
    # Create a unique directory for this run's validation results
    csv_dir = eval_dir / "validation_training" / run_id
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Load or initialize GT cache
    gt_cache = {}
    if gt_cache_path.exists():
        try:
            with open(gt_cache_path, "r") as f:
                gt_cache = json.load(f)
            logger.info(f"Loaded ground truth cache from {gt_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load ground truth cache (it might be corrupted or being written to): {e}")
            gt_cache = {}

    # Initialize predictors
    utmos_predictor = UTMOSPredictor(device)
    asr_model = WhisperASR(device)

    samples_metrics = []
    processed_count = 0

    # Temporary directory for UTMOS if needed
    temp_wav_dir = eval_dir / "temp_wavs" / run_id
    temp_wav_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting evaluation on {num_samples} samples...")

    for batch in eval_dataloader:
        if processed_count >= num_samples:
            break

        audios_srs = batch["output_audios_srs"]  # List of (tensor, sr)
        gt_texts = [t.lower().strip() for t in batch["transcription"]]
        sample_ids = batch.get(
            "ids", [f"sample_{processed_count + i}" for i in range(len(audios_srs))]
        )

        # Prepare device-ready audio for model
        model_audios_srs = [
            (audio.to(device).to(model.dtype), sr) for audio, sr in audios_srs
        ]

        # Reconstruct
        with torch.no_grad():
            reconstruction_results = model.encode_decode(
                audios_srs=model_audios_srs,
                num_steps=16,
                temperature=0.2,
                guidance_scale=1.3,
                phoneme_alignments=batch.get("phoneme_alignments", None),
            )

            # reconstructed_mel is in results["decoder_output"].audio_features
            reconstructed_mels = reconstruction_results["decoder_output"].audio_features
            reconstructed_masks = reconstruction_results["decoder_output"].padding_mask

        # Process each sample in batch
        for i in range(len(audios_srs)):
            if processed_count >= num_samples:
                break

            sid = str(sample_ids[i])
            gt_text = gt_texts[i]
            orig_audio, orig_sr = audios_srs[i]

            # 1. Get Ground Truth Metrics
            if sid in gt_cache:
                gt_metrics = gt_cache[sid]
            else:
                # Compute GT metrics once
                # UTMOS needs a file path
                gt_wav_path = temp_wav_dir / f"{sid}_gt.wav"
                torchaudio.save(str(gt_wav_path), orig_audio.unsqueeze(0).cpu(), orig_sr)

                gt_utmos = utmos_predictor.predict(str(gt_wav_path))
                gt_transcription = asr_model.transcribe(orig_audio, orig_sr)
                gt_wer = compute_wer(gt_text, gt_transcription)
                gt_cer = compute_cer(gt_text, gt_transcription)

                gt_metrics = {
                    "UTMOS": gt_utmos,
                    "WER": gt_wer,
                    "CER": gt_cer,
                    "transcription": gt_transcription,
                }
                gt_cache[sid] = gt_metrics

            # 2. Reconstruct audio from mel
            mel = reconstructed_mels[i]
            mask = reconstructed_masks[i]
            mel = (
                mel[~mask].unsqueeze(0).permute(0, 2, 1).float().to(device)
            )  # [1, F, T]

            # Use Vocos/BigVGAN to decode
            with torch.no_grad():
                if vocoder_type == "bigvgan":
                    recon_audio = vocoder(mel).cpu().squeeze()
                else:
                    recon_audio = vocoder.decode(mel).cpu().squeeze()

            # Normalize recon_audio
            recon_audio = recon_audio / (recon_audio.abs().max() + 1e-8)

            # 3. Compute Reconstructed Metrics
            recon_wav_path = temp_wav_dir / f"{sid}_recon.wav"
            torchaudio.save(str(recon_wav_path), recon_audio.unsqueeze(0).cpu(), orig_sr)

            recon_utmos = utmos_predictor.predict(str(recon_wav_path))
            recon_transcription = asr_model.transcribe(recon_audio, orig_sr)
            recon_wer = compute_wer(gt_text, recon_transcription)
            recon_cer = compute_cer(gt_text, recon_transcription)

            # 4. Calculate Deltas
            sample_res = {
                "id": sid,
                "step": step,
                "gt_text": gt_text,
                "gt_UTMOS": gt_metrics["UTMOS"],
                "gt_WER": gt_metrics["WER"],
                "gt_CER": gt_metrics["CER"],
                "recon_UTMOS": recon_utmos,
                "recon_WER": recon_wer,
                "recon_CER": recon_cer,
                "dUTMOS": (
                    (recon_utmos - gt_metrics["UTMOS"])
                    if (recon_utmos is not None and gt_metrics["UTMOS"] is not None)
                    else None
                ),
                "dWER": (recon_wer - gt_metrics["WER"]),
                "dCER": (recon_cer - gt_metrics["CER"]),
            }
            samples_metrics.append(sample_res)
            processed_count += 1

            # Cleanup temp files
            if recon_wav_path.exists():
                recon_wav_path.unlink()
            gt_wav_tmp = temp_wav_dir / f"{sid}_gt.wav"
            if gt_wav_tmp.exists():
                gt_wav_tmp.unlink()

    # Save GT cache robustly
    try:
        temp_gt_cache_path = gt_cache_path.with_suffix(".tmp")
        with open(temp_gt_cache_path, "w") as f:
            json.dump(gt_cache, f, indent=4)
        temp_gt_cache_path.replace(gt_cache_path)
    except Exception as e:
        logger.warning(f"Failed to save ground truth cache: {e}")

    # Aggregate results
    df = pd.DataFrame(samples_metrics)
    csv_path = csv_dir / f"val_step_{step}.csv"
    df.to_csv(csv_path, index=False)

    # Calculate means for logging
    summary_metrics = {
        "avg_UTMOS": df["recon_UTMOS"].mean(),
        "avg_WER": df["recon_WER"].mean(),
        "avg_CER": df["recon_CER"].mean(),
        "avg_dUTMOS": df["dUTMOS"].mean(),
        "avg_dWER": df["dWER"].mean(),
        "avg_dCER": df["dCER"].mean(),
    }

    # Log to WandB
    if wandb.run is not None:
        logger.info(f"Logging metrics table to WandB (rows: {len(df)})")
        table = wandb.Table(dataframe=df)
        wandb.log({"eval/metrics_table": table})
    else:
        logger.warning("wandb.run is None, metrics table not logged to WandB")

    logger.info(f"Evaluation complete. Results saved to {csv_path}")

    # Cleanup temp directory
    try:
        if temp_wav_dir.exists():
            # Remove all files just in case some were left behind
            for f in temp_wav_dir.glob("*"):
                f.unlink()
            temp_wav_dir.rmdir()
    except Exception as e:
        logger.warning(f"Failed to cleanup temp_wav_dir {temp_wav_dir}: {e}")

    return summary_metrics
