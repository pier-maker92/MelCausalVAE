#!/usr/bin/env python3
"""Decode male.wav low-pass latents for several physical cutoff frequencies."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.dont_write_bytecode = True


DEFAULT_CHECKPOINT_DIR = "/Users/software/Research/MelCausalVAE/checkpoints/set15-64"
DEFAULT_AUDIO_PATH = "/Users/software/Research/MelCausalVAE/demopage/audio/male.wav"
DEFAULT_OUTPUT_DIR = "/Users/software/Research/MelCausalVAE/lab/outputs/male_lowpass_cutoffs"
DEFAULT_HF_HOME = "/Volumes/Crucial X6/HF_HOME"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode latent attribute variants from male.wav for multiple low-pass cutoffs."
    )
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--audio-path", default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--cutoffs", nargs="+", type=float, default=[1.0, 4.0, 8.0, 12.0])
    parser.add_argument("--latent-sample-rate", type=float, default=None)
    parser.add_argument("--orders", nargs="+", type=int, default=[20])
    parser.add_argument(
        "--component",
        choices=("pros", "sem", "full"),
        default="pros",
        help="Latent component to decode: pros=z_pros+z_mean, sem=z_sem+z_mean, full=z_pros+z_sem+z_mean.",
    )
    parser.add_argument(
        "--exclude-mean",
        action="store_true",
        help="Decode the requested latent component without adding z_mean.",
    )
    parser.add_argument(
        "--zero-speaker",
        action="store_true",
        help="Zero the speaker embedding before decoding.",
    )
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _set_caches(hf_home: str) -> None:
    tmp_root = Path(os.environ.get("TMPDIR", "/private/tmp"))
    for child in ("melcausalvae-mpl", "melcausalvae-xdg", "melcausalvae-torch"):
        (tmp_root / child).mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(hf_home) / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_home) / "transformers")
    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
    os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
    os.environ["MPLCONFIGDIR"] = str(tmp_root / "melcausalvae-mpl")
    os.environ["XDG_CACHE_HOME"] = str(tmp_root / "melcausalvae-xdg")
    os.environ["TORCH_HOME"] = str(tmp_root / "melcausalvae-torch")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model(checkpoint_dir: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(_repo_root()))
    from modules.builder import build_model

    with (checkpoint_dir / "config.json").open() as config_file:
        cfg_dict = json.load(config_file)

    model = build_model(cfg_dict)
    model.from_pretrained(str(checkpoint_dir))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_wav_mono(path: Path, target_sample_rate: int) -> torch.Tensor:
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


def _save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    import torchaudio

    audio = audio.detach().cpu()
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak.clamp_min(1e-8)
    torchaudio.save(str(path), audio, sample_rate)


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    _set_caches(args.hf_home)

    device = _resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    audio_path = Path(args.audio_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(_repo_root()))
    from modules.lp_filter import LowPassFilter

    model = _load_model(checkpoint_dir, device=device)
    latent_sample_rate = (
        args.latent_sample_rate
        if args.latent_sample_rate is not None
        else model.config.lowpass_filter_config.sample_rate
    )
    if latent_sample_rate is None:
        latent_sample_rate = (
            model.config.sample_rate
            / model.config.mel_spectrogram_config.hop_length
            / model.config.compress_factor
        )

    sample_rate = int(model.config.sample_rate)
    wav = _load_wav_mono(audio_path, sample_rate).to(device)
    audios_srs = [(wav, sample_rate)]

    features = model.extract_features(audios_srs)
    enc_features, enc_padding_mask = features[0], features[1]
    encoded = model.encode(enc_features, enc_padding_mask)
    speaker_embedding = model.extract_speaker_embedding(audios_srs)
    if args.zero_speaker and speaker_embedding is not None:
        speaker_embedding = torch.zeros_like(speaker_embedding)

    _save_audio(output_dir / "original_male.wav", wav.unsqueeze(0), sample_rate)

    manifest = []
    for cutoff_hz in args.cutoffs:
        for order in args.orders:
            nyquist = latent_sample_rate / 2
            effective_cutoff_hz = cutoff_hz
            if cutoff_hz >= nyquist:
                effective_cutoff_hz = math.nextafter(nyquist, 0.0)

            model.lowpass_filter = LowPassFilter(
                cutoff_hz=effective_cutoff_hz,
                sample_rate=latent_sample_rate,
                order=order,
            ).to(device)
            attrs = model.encode_attributes(encoded.z, padding_mask=encoded.padding_mask)
            if args.component == "pros":
                z_variant = attrs.z_pros
                component_name = "z_pros"
            elif args.component == "sem":
                z_variant = attrs.z_sem
                component_name = "z_sem"
            else:
                z_variant = attrs.z_pros + attrs.z_sem
                component_name = "z_pros_plus_z_sem"

            if not args.exclude_mean:
                z_variant = z_variant + attrs.z_mean
                component_name = f"{component_name}_plus_mean"

            if args.zero_speaker:
                component_name = f"{component_name}_zero_speaker"

            generator = torch.Generator(device=device).manual_seed(args.seed)
            mel, mel_mask = model.sample(
                num_steps=args.num_steps,
                temperature=args.temperature,
                guidance_scale=args.guidance_scale,
                z=z_variant,
                generator=generator,
                padding_mask=encoded.padding_mask,
                speaker_embedding=speaker_embedding,
            )
            features = mel[0, ~mel_mask[0]].unsqueeze(0).permute(0, 2, 1)
            audio = model.vocoder.decode(features)

            cutoff_tag = f"{cutoff_hz:g}hz".replace(".", "p")
            output_path = (
                output_dir
                / f"male_{component_name}_cutoff_{cutoff_tag}_order_{order}.wav"
            )
            _save_audio(output_path, audio, sample_rate)
            manifest.append(
                {
                    "component": args.component,
                    "cutoff_hz": cutoff_hz,
                    "effective_cutoff_hz": effective_cutoff_hz,
                    "path": str(output_path),
                    "latent_sample_rate": latent_sample_rate,
                    "nyquist_hz": nyquist,
                    "order": order,
                    "exclude_mean": args.exclude_mean,
                    "zero_speaker": args.zero_speaker,
                }
            )

    summary = {
        "audio_path": str(audio_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(device),
        "num_steps": args.num_steps,
        "latent_shape": list(encoded.z.shape),
        "outputs": manifest,
    }
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
