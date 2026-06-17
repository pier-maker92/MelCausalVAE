"""
Individual Chunk Isolation sweep for MelCausalVAE models.
Evaluates the performance of each latent chunk by zeroing out all others.

Usage:
  python evaluation/run_chunk_isolation_sweep.py --checkpoint checkpoints/paper/setting30/model.safetensors --chunks 1 2 3 4 --chunk-size 4 --upload-wandb --save-audio
"""

import os
import sys
import io
import json
import math
import time
import logging
import argparse
import tempfile
from pathlib import Path

# ── dataclasses monkey-patch for fairseq/hydra on Python 3.11 ──────────────
import dataclasses
_orig_get_field = dataclasses._get_field
def _patched_get_field(cls, a_name, a_type, *args, **kwargs):
    try:
        return _orig_get_field(cls, a_name, a_type, *args, **kwargs)
    except ValueError as e:
        if "mutable default" in str(e):
            default = getattr(cls, a_name, dataclasses.MISSING)
            actual_default = default.default if isinstance(default, dataclasses.Field) else default
            if actual_default is not dataclasses.MISSING:
                default_cls = actual_default.__class__
                orig_hash = getattr(default_cls, "__hash__", None)
                try:
                    default_cls.__hash__ = lambda self: id(self)
                except TypeError:
                    pass
                try:
                    return _orig_get_field(cls, a_name, a_type, *args, **kwargs)
                finally:
                    try:
                        if orig_hash is None:
                            default_cls.__hash__ = None
                        else:
                            default_cls.__hash__ = orig_hash
                    except TypeError:
                        pass
        raise
dataclasses._get_field = _patched_get_field
# ────────────────────────────────────────────────────────────────────────────

# patch torch.load for weights_only compatibility
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jiwer import wer as compute_wer, cer as compute_cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from vocos import Vocos

# add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

try:
    from MelCausalVAE.modules.builder import build_model
except ModuleNotFoundError:
    from modules.builder import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
PARQUET = (
    Path.home()
    / ".cache/huggingface/hub"
    / "datasets--gilkeyio--librispeech-alignments"
    / "snapshots"
    / "0daa1eb43dda38ee6ce752e785555380e5628f5c"
    / "data"
    / "test_clean-00000-of-00001.parquet"
)
GT_JSON = repo_root / "evaluation/GT/librispeech.json"

TARGET_SR = 24000

# Reference metrics from GT JSON (pre-computed over 2620 samples)
REF_WER = 2.49    # %
REF_CER = 0.67    # %
REF_UTMOS = 3.20  # raw score


# ── Data loading ──────────────────────────────────────────────────────────

def load_samples(n: int):
    """Load first n samples from the test-clean parquet, decode audio."""
    logger.info(f"Loading {n} samples from {PARQUET}")
    df = pd.read_parquet(PARQUET).head(n)

    samples = []
    for _, row in df.iterrows():
        audio_bytes = row["audio"]["bytes"]
        audio_np, sr = sf.read(io.BytesIO(audio_bytes))
        audio_t = torch.FloatTensor(audio_np)
        if audio_t.dim() > 1:
            audio_t = audio_t.mean(dim=-1)
        if sr != TARGET_SR:
            audio_t = torchaudio.functional.resample(audio_t, sr, TARGET_SR)
            sr = TARGET_SR
        audio_t = audio_t / (audio_t.abs().max() + 1e-8)
        samples.append(
            {
                "id": str(row["id"]),
                "audio": audio_t,      # [T]
                "sr": sr,
                "transcript": str(row["transcript"]).lower().strip(),
            }
        )
    return samples


# ── Model loading ──────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        config_path = repo_root / "checkpoints/paper" / checkpoint_path.parent.name / "config.json"
    
    with open(config_path) as f:
        config_dict = json.load(f)

    logger.info(f"Building model from {config_path}…")
    model = build_model(config_dict)
    model.from_pretrained(str(checkpoint_path))
    model.eval()
    model.to(device)

    logger.info("Loading Vocos vocoder…")
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocoder.eval()
    vocoder.to(device)

    return model, vocoder


# ── ASR ────────────────────────────────────────────────────────────────────

class WhisperASR:
    def __init__(self, device, model_name="openai/whisper-large-v3"):
        logger.info(f"Loading Whisper: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def transcribe(self, audio: torch.Tensor, sr: int) -> str:
        if audio.dim() > 1:
            audio = audio.mean(dim=0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        feats = (
            self.processor(audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
            .input_features.to(self.device)
            .to(self.model.dtype)
        )
        ids = self.model.generate(feats)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].lower().strip()


# ── UTMOS ─────────────────────────────────────────────────────────────────

class UTMOSPredictor:
    def __init__(self, device):
        import utmos
        logger.info("Loading UTMOS model…")
        torch.set_default_dtype(torch.float32)
        self.model = utmos.Score()
        self.device = device
        torch.set_default_dtype(torch.float32)

    @torch.no_grad()
    def predict(self, wav_path: str) -> float:
        with torch.autocast(device_type=self.device.type, enabled=False):
            return float(self.model.calculate_wav_file(str(wav_path)))


# ── Encode-decode with chunk isolation ──────────────────────────────────────

@torch.no_grad()
def encode_decode_single(model, vocoder, sample, device, chunk: int, chunk_size: int,
                         n_steps=16, temperature=0.2, guidance_scale=1.3):
    """
    Process one sample: encode → ISOLATE chunk → decode.
    Returns reconstructed audio tensor [T] at TARGET_SR.
    """
    dtype = torch.float32
    audios_srs = [(sample["audio"].to(device, dtype=dtype), sample["sr"])]

    enc_features, enc_padding_mask, _, _, _ = model.extract_features(audios_srs)
    encoder_output = model.encode(enc_features, enc_padding_mask)

    z = encoder_output.z.clone()
    
    # Isolation Logic: Keep ONLY the specified chunk (1-indexed)
    start = (chunk - 1) * chunk_size
    end = chunk * chunk_size
    
    new_z = torch.zeros_like(z)
    if start < z.shape[-1]:
        real_end = min(end, z.shape[-1])
        new_z[..., start:real_end] = z[..., start:real_end]
    z = new_z

    recon_mel, masks = model.sample(
        num_steps=n_steps,
        temperature=temperature,
        guidance_scale=guidance_scale,
        z=z,
        padding_mask=encoder_output.padding_mask,
    )

    mel_valid = recon_mel[0][~masks[0]].unsqueeze(0)
    vocoder_dtype = next(vocoder.backbone.parameters()).dtype
    mel_input = mel_valid.permute(0, 2, 1).to(device=device, dtype=vocoder_dtype)
    wav = vocoder.decode(mel_input).float().squeeze().detach().cpu()
    wav = wav / (wav.abs().max() + 1e-8)
    return wav


# ── Per-chunk evaluation ──────────────────────────────────────────────────

def evaluate_chunk(model, vocoder, samples, device, chunk, chunk_size,
                   asr, utmos_predictor, tmp_dir: Path):
    """Run encode-decode with chunk isolation, compute metrics, return aggregated dict."""
    wers, cers, utmos_scores = [], [], []
    for sample in samples:
        recon_audio = encode_decode_single(
            model, vocoder, sample, device, chunk=chunk, chunk_size=chunk_size
        )
        gt_text = sample["transcript"]
        recon_hyp = asr.transcribe(recon_audio, TARGET_SR)
        wer_val = float(compute_wer(gt_text, recon_hyp)) * 100.0
        cer_val = float(compute_cer(gt_text, recon_hyp)) * 100.0

        wav_path = tmp_dir / f"recon_{sample['id']}_isolate_chunk{chunk}.wav"
        torchaudio.save(str(wav_path), recon_audio.unsqueeze(0), TARGET_SR)
        mos = utmos_predictor.predict(str(wav_path))
        wav_path.unlink(missing_ok=True)

        wers.append(wer_val)
        cers.append(cer_val)
        utmos_scores.append(mos)

        logger.info(
            f"  [Isolate] chunk={chunk} id={sample['id']} WER={wer_val:.1f}% CER={cer_val:.1f}% UTMOS={mos:.2f}"
        )

    return {
        "chunk": chunk,
        "WER": float(np.mean(wers)),
        "CER": float(np.mean(cers)),
        "UTMOS": float(np.mean(utmos_scores)),
        "WER_samples": wers,
        "CER_samples": cers,
        "UTMOS_samples": utmos_scores,
    }


# ── Plot ──────────────────────────────────────────────────────────────────

def make_plot(results, out_path: Path, title: str):
    chunks = [r["chunk"] for r in results]
    wers   = [r["WER"]   for r in results]
    cers   = [r["CER"]   for r in results]
    utmos  = [r["UTMOS"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(chunks, wers,  marker="o", linewidth=2, color="#1f77b4", label="WER (isolated)")
    ax1.plot(chunks, cers,  marker="s", linewidth=2, color="#ff7f0e", label="CER (isolated)")
    ax1.axhline(REF_WER,  color="#1f77b4", linestyle="--", linewidth=1.5, label="WER (reference)")
    ax1.axhline(REF_CER,  color="#ff7f0e", linestyle="--", linewidth=1.5, label="CER (reference)")
    ax1.set_xlabel("Isolated Chunk Index", fontsize=12)
    ax1.set_ylabel("WER / CER (%)", fontsize=12)
    ax1.set_xticks(chunks)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(chunks, utmos, marker="^", linewidth=2, color="#2ca02c", label="UTMOS (isolated)")
    ax2.axhline(REF_UTMOS, color="#2ca02c", linestyle="--", linewidth=1.5, label="UTMOS (reference)")
    ax2.set_ylabel("UTMOS", fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax1.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved to {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chunk isolation sweep evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 3, 4], help="Chunk indices to isolate")
    parser.add_argument("--chunk-size", type=int, default=4, help="Dimensions per chunk")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--upload-wandb", action="store_true", help="Upload results to wandb")
    parser.add_argument("--save-audio", action="store_true", help="Save audio samples locally and to wandb")
    parser.add_argument("--wandb-project", type=str, default="MelCausalVAE-eval")
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--guidance-scale", type=float, default=1.3)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_dtype(torch.bfloat16)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_default_dtype(torch.float32)
    else:
        raise RuntimeError("No device available.")

    ckpt_path = Path(args.checkpoint)
    setting_name = ckpt_path.parent.name
    exp_name = f"{setting_name}-chunk-isolation"

    out_dir = Path(__file__).parent / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp_wavs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if args.upload_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config={
                "type": "chunk-isolation",
                "checkpoint": str(ckpt_path),
                "chunks": args.chunks,
                "chunk_size": args.chunk_size,
                "num_samples": args.num_samples,
            },
        )

    samples = load_samples(args.num_samples)
    model, vocoder = load_model(args.checkpoint, device)
    asr = WhisperASR(device)
    utmos_predictor = UTMOSPredictor(device)

    all_results = []
    for chunk in args.chunks:
        logger.info(f"\n── Isolating chunk {chunk} (dims {(chunk-1)*args.chunk_size} to {chunk*args.chunk_size}) ──")
        res = evaluate_chunk(
            model=model,
            vocoder=vocoder,
            samples=samples,
            device=device,
            chunk=chunk,
            chunk_size=args.chunk_size,
            asr=asr,
            utmos_predictor=utmos_predictor,
            tmp_dir=tmp_dir,
        )
        all_results.append(res)

        if run is not None:
            log_dict = {
                f"isolate_chunk_{chunk}/WER": res["WER"],
                f"isolate_chunk_{chunk}/CER": res["CER"],
                f"isolate_chunk_{chunk}/UTMOS": res["UTMOS"],
                "chunk_idx": chunk,
            }
            if args.save_audio:
                chunk_wav_dir = out_dir / f"wavs_isolate_chunk_{chunk}"
                chunk_wav_dir.mkdir(parents=True, exist_ok=True)
                for i in range(min(3, len(samples))):
                    s = samples[i]
                    recon_audio = encode_decode_single(model, vocoder, s, device, chunk, args.chunk_size)
                    wav_path = chunk_wav_dir / f"{s['id']}_isolate.wav"
                    torchaudio.save(str(wav_path), recon_audio.unsqueeze(0), TARGET_SR)
                    log_dict[f"audio/isolate_chunk_{chunk}_sample_{i}"] = wandb.Audio(str(wav_path), caption=f"Isolate Chunk {chunk} - {s['id']}", sample_rate=TARGET_SR)
            run.log(log_dict)

    plot_path = out_dir / "chunk_isolation_plot.png"
    make_plot(all_results, plot_path, title=f"Chunk Isolation — {setting_name}")

    if run is not None:
        run.log({"isolation_plot": wandb.Image(str(plot_path))})
        run.finish()

    logger.info("Done.")

if __name__ == "__main__":
    main()
