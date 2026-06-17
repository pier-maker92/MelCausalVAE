"""
General Bottleneck sweep for MelCausalVAE models.

Usage:
  python evaluation/run_tail_sweep.py --checkpoint checkpoints/paper/setting30/model.safetensors --all-chunks --chunk-size 4 --num-samples 10
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
            actual_default = (
                default.default if isinstance(default, dataclasses.Field) else default
            )
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
if "SLURM_TMPDIR" in os.environ:
    _base_dir = Path(os.environ["SLURM_TMPDIR"]) / "datasets" / "librispeech-aligned"
else:
    _base_dir = Path(os.environ.get("SCRATCH", "")) / "datasets" / "librispeech-aligned"

PARQUET = _base_dir / "test_clean" / "test_clean-00000-of-00001.parquet"

if not PARQUET.exists():
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
REF_WER = 2.49  # %
REF_CER = 0.67  # %
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
                "audio": audio_t,  # [T]
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
        # Fallback if config.json is not in the same folder
        config_path = (
            repo_root
            / "checkpoints/paper"
            / checkpoint_path.parent.name
            / "config.json"
        )

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
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(
            device
        )
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def transcribe(self, audio: torch.Tensor, sr: int) -> str:
        if audio.dim() > 1:
            audio = audio.mean(dim=0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        feats = (
            self.processor(
                audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
            )
            .input_features.to(self.device)
            .to(self.model.dtype)
        )
        ids = self.model.generate(feats)
        return (
            self.processor.batch_decode(ids, skip_special_tokens=True)[0]
            .lower()
            .strip()
        )


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


# ── Encode-decode with chunk masking ──────────────────────────────────────


@torch.no_grad()
def encode_decode_single(
    model,
    vocoder,
    sample,
    device,
    chunk: int,
    chunk_size: int,
    n_steps=16,
    temperature=0.2,
    guidance_scale=1.3,
):
    """
    Process one sample: encode → mask z → decode.
    Returns reconstructed audio tensor [T] at TARGET_SR.
    Processes each sample individually to avoid padding-induced NaN in batched mode.
    """
    dtype = next(model.parameters()).dtype
    audios_srs = [(sample["audio"].to(device, dtype=dtype), sample["sr"])]

    enc_features, enc_padding_mask, _, _, _ = model.extract_features(audios_srs)
    encoder_output = model.encode(enc_features, enc_padding_mask)

    z = encoder_output.z.clone()
    keep_dims = chunk * chunk_size
    if keep_dims < z.shape[-1]:
        z[..., keep_dims:] = 0.0

    # sample() returns (mel, padding_mask), mel already denormalized
    recon_mel, masks = model.sample(
        num_steps=n_steps,
        temperature=temperature,
        guidance_scale=guidance_scale,
        z=z,
        padding_mask=encoder_output.padding_mask,
    )

    mel_valid = recon_mel[0][~masks[0]].unsqueeze(0)  # [1, T_valid, mel_dim]
    vocoder_dtype = next(vocoder.backbone.parameters()).dtype
    mel_input = mel_valid.permute(0, 2, 1).to(device=device, dtype=vocoder_dtype)
    wav = vocoder.decode(mel_input).float().squeeze().detach().cpu()
    wav = wav / (wav.abs().max() + 1e-8)
    return wav


# ── Per-chunk evaluation ──────────────────────────────────────────────────


def evaluate_chunk(
    model,
    vocoder,
    samples,
    device,
    chunk,
    chunk_size,
    asr,
    utmos_predictor,
    tmp_dir: Path,
):
    """Run encode-decode with chunk masking, compute metrics, return aggregated dict."""
    wers, cers, utmos_scores = [], [], []
    for sample in samples:
        recon_audio = encode_decode_single(
            model, vocoder, sample, device, chunk=chunk, chunk_size=chunk_size
        )
        gt_text = sample["transcript"]
        recon_hyp = asr.transcribe(recon_audio, TARGET_SR)
        wer_val = float(compute_wer(gt_text, recon_hyp)) * 100.0
        cer_val = float(compute_cer(gt_text, recon_hyp)) * 100.0

        # UTMOS
        wav_path = tmp_dir / f"recon_{sample['id']}_chunk{chunk}.wav"
        torchaudio.save(str(wav_path), recon_audio.unsqueeze(0), TARGET_SR)
        mos = utmos_predictor.predict(str(wav_path))
        wav_path.unlink(missing_ok=True)

        wers.append(wer_val)
        cers.append(cer_val)
        utmos_scores.append(mos)

        logger.info(
            f"  chunk={chunk} id={sample['id']} WER={wer_val:.1f}% CER={cer_val:.1f}% UTMOS={mos:.2f}"
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
    wers = [r["WER"] for r in results]
    cers = [r["CER"] for r in results]
    utmos = [r["UTMOS"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # WER / CER on left axis
    ax1.plot(
        chunks,
        wers,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="WER (reconstructed)",
    )
    ax1.plot(
        chunks,
        cers,
        marker="s",
        linewidth=2,
        color="#ff7f0e",
        label="CER (reconstructed)",
    )
    ax1.axhline(
        REF_WER, color="#1f77b4", linestyle="--", linewidth=1.5, label="WER (reference)"
    )
    ax1.axhline(
        REF_CER, color="#ff7f0e", linestyle="--", linewidth=1.5, label="CER (reference)"
    )
    ax1.set_xlabel("Latent chunks (N)", fontsize=12)
    ax1.set_ylabel("WER / CER (%)", fontsize=12)
    # ax1.set_xscale("log", base=2)
    ax1.set_xticks(chunks)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.grid(True, linestyle="--", alpha=0.5)

    # UTMOS on right axis
    ax2.plot(
        chunks,
        utmos,
        marker="^",
        linewidth=2,
        color="#2ca02c",
        label="UTMOS (reconstructed)",
    )
    ax2.axhline(
        REF_UTMOS,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label="UTMOS (reference)",
    )
    ax2.set_ylabel("UTMOS", fontsize=12)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax1.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved to {out_path}")
    return out_path


# ── Report ────────────────────────────────────────────────────────────────


def print_report(results, chunk_size):
    print("\n" + "=" * 70)
    print(
        f"{'Chunks (N)':>12} {'chunk_size':>12} {'WER (%)':>10} {'CER (%)':>10} {'UTMOS':>10}"
    )
    print("-" * 70)
    print(
        f"{'reference':>12} {'—':>12} {REF_WER:>10.2f} {REF_CER:>10.2f} {REF_UTMOS:>10.2f}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['chunk']:>12} {chunk_size:>12} {r['WER']:>10.2f} {r['CER']:>10.2f} {r['UTMOS']:>10.2f}"
        )
    print("=" * 70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Tail chunk sweep evaluation")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Chunk counts to evaluate",
    )
    parser.add_argument(
        "--all-chunks",
        action="store_true",
        help="Evaluate all possible chunks from 1 to latent_dim/chunk_size (overrides --chunks)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=4, help="Dimensions per chunk"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--upload-wandb", action="store_true", help="Upload results to wandb"
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save audio samples locally and to wandb",
    )
    parser.add_argument("--wandb-project", type=str, default="MelCausalVAE-eval")
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--guidance-scale", type=float, default=1.3)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_dtype(torch.bfloat16)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_default_dtype(torch.float32)
    else:
        raise RuntimeError("No CUDA or MPS device available.")
    logger.info(f"Using device: {device}")

    # Experiment Name derived from checkpoint parent directory
    ckpt_path = Path(args.checkpoint)
    setting_name = ckpt_path.parent.name
    exp_name = f"{setting_name}-tail-sweep"

    # Output dir
    out_dir = Path(__file__).parent / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp_wavs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Optional wandb
    run = None
    if args.upload_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config={
                "checkpoint": str(ckpt_path),
                "chunks": args.chunks,
                "chunk_size": args.chunk_size,
                "num_samples": args.num_samples,
                "n_steps": args.n_steps,
                "temperature": args.temperature,
                "guidance_scale": args.guidance_scale,
                "setting": "LibriSpeech-test-clean",
            },
        )

    # Load data
    samples = load_samples(args.num_samples)
    logger.info(f"Loaded {len(samples)} samples")

    # Load model
    model, vocoder = load_model(args.checkpoint, device)

    if args.all_chunks:
        max_chunks = (
            getattr(model.config.encoder_config, "latent_dim", 64) // args.chunk_size
        )
        args.chunks = list(range(1, max_chunks + 1))
        logger.info(
            f"--all-chunks specified. Evaluating all chunks from 1 to {max_chunks}"
        )

    # Load ASR + UTMOS
    asr = WhisperASR(device)
    utmos_predictor = UTMOSPredictor(device)

    # Run sweep
    all_results = []
    for chunk in args.chunks:
        logger.info(
            f"\n── Evaluating chunk={chunk} (keeps {chunk * args.chunk_size}/{model.config.encoder_config.latent_dim} dims) ──"
        )
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
                f"chunk_{chunk}/WER": res["WER"],
                f"chunk_{chunk}/CER": res["CER"],
                f"chunk_{chunk}/UTMOS": res["UTMOS"],
                "chunk": chunk,
                "WER": res["WER"],
                "CER": res["CER"],
                "UTMOS": res["UTMOS"],
            }

            if args.save_audio:
                chunk_wav_dir = out_dir / f"wavs_chunk_{chunk}"
                chunk_wav_dir.mkdir(parents=True, exist_ok=True)
                # Log first 3 samples as audio to wandb
                for i in range(min(3, len(samples))):
                    s = samples[i]
                    recon_audio = encode_decode_single(
                        model,
                        vocoder,
                        s,
                        device,
                        chunk=chunk,
                        chunk_size=args.chunk_size,
                    )
                    wav_path = chunk_wav_dir / f"{s['id']}_recon.wav"
                    torchaudio.save(str(wav_path), recon_audio.unsqueeze(0), TARGET_SR)
                    log_dict[f"audio/chunk_{chunk}_sample_{i}"] = wandb.Audio(
                        str(wav_path),
                        caption=f"Chunk {chunk} - {s['id']}",
                        sample_rate=TARGET_SR,
                    )

            run.log(log_dict)
        elif args.save_audio:
            # Still save audio locally if requested but no wandb
            chunk_wav_dir = out_dir / f"wavs_chunk_{chunk}"
            chunk_wav_dir.mkdir(parents=True, exist_ok=True)
            for i in range(min(3, len(samples))):
                s = samples[i]
                recon_audio = encode_decode_single(
                    model, vocoder, s, device, chunk=chunk, chunk_size=args.chunk_size
                )
                wav_path = chunk_wav_dir / f"{s['id']}_recon.wav"
                torchaudio.save(str(wav_path), recon_audio.unsqueeze(0), TARGET_SR)

    # Save JSON
    report = {
        "exp_name": exp_name,
        "checkpoint": str(ckpt_path),
        "chunk_size": args.chunk_size,
        "num_samples": args.num_samples,
        "reference": {"WER": REF_WER, "CER": REF_CER, "UTMOS": REF_UTMOS},
        "results": [
            {k: v for k, v in r.items() if not k.endswith("_samples")}
            for r in all_results
        ],
        "results_per_sample": [
            {
                "chunk": r["chunk"],
                "WER_samples": r["WER_samples"],
                "CER_samples": r["CER_samples"],
                "UTMOS_samples": r["UTMOS_samples"],
            }
            for r in all_results
        ],
    }
    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Saved metrics to {json_path}")

    # Print report
    print_report(all_results, args.chunk_size)

    # Plot
    plot_path = out_dir / "tail_sweep_plot.png"
    make_plot(
        all_results,
        plot_path,
        title=f"Metrics vs latent bottleneck — {setting_name} ({args.num_samples} samples)",
    )

    # Upload to wandb
    if run is not None:
        wandb_table = wandb.Table(
            columns=["chunk", "WER (%)", "CER (%)", "UTMOS"],
            data=[[r["chunk"], r["WER"], r["CER"], r["UTMOS"]] for r in all_results],
        )
        run.log(
            {
                "tail_sweep/table": wandb_table,
                "tail_sweep/plot": wandb.Image(str(plot_path)),
                "ref/WER": REF_WER,
                "ref/CER": REF_CER,
                "ref/UTMOS": REF_UTMOS,
            }
        )
        run.finish()
        logger.info("Results uploaded to wandb.")

    # Final cleanup of tmp_dir
    try:
        if tmp_dir.exists():
            for f in tmp_dir.glob("*.wav"):
                f.unlink()
            tmp_dir.rmdir()
    except:
        pass

    logger.info("Done.")


if __name__ == "__main__":
    main()
