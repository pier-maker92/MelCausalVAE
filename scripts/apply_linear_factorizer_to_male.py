#!/usr/bin/env python3
"""Apply a trained linear factorizer to male.wav and decode factor combinations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True


DEFAULT_HF_HOME = "/Volumes/Crucial X6/HF_HOME"
DEFAULT_CHECKPOINT_DIR = (
    "/Users/software/Research/MelCausalVAE/checkpoints/paper/baseline/"
    "18-denc128-novq"
)
DEFAULT_FACTORIZER_PATH = (
    "/Volumes/Crucial X6/Research/Datasets/libritts-r-dicodec-factorizer/"
    "shared_shift_factorizer.pt"
)
DEFAULT_AUDIO_PATH = "/Users/software/Research/MelCausalVAE/audios/male.wav"
DEFAULT_OUTPUT_DIR = "/Users/software/Research/MelCausalVAE/lab/outputs/male_linear_factorizer"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode male.wav with z_pros/z_sem from the trained linear factorizer."
    )
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--factorizer-path", default=DEFAULT_FACTORIZER_PATH)
    parser.add_argument("--audio-path", default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _set_external_caches(hf_home: str) -> Path:
    hf_home_path = Path(hf_home).expanduser()
    for child in (
        "hub",
        "assets",
        "transformers",
        "xdg",
        "torch",
        "tmp",
        "matplotlib",
    ):
        (hf_home_path / child).mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home_path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home_path / "hub")
    os.environ["HF_ASSETS_CACHE"] = str(hf_home_path / "assets")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home_path / "transformers")
    os.environ["XDG_CACHE_HOME"] = str(hf_home_path / "xdg")
    os.environ["TORCH_HOME"] = str(hf_home_path / "torch")
    os.environ["TMPDIR"] = str(hf_home_path / "tmp")
    os.environ["TEMP"] = str(hf_home_path / "tmp")
    os.environ["TMP"] = str(hf_home_path / "tmp")
    os.environ["MPLCONFIGDIR"] = str(hf_home_path / "matplotlib")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return hf_home_path


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

    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")

    with config_path.open() as config_file:
        cfg_dict = json.load(config_file)

    model = build_model(cfg_dict)
    model.from_pretrained(str(checkpoint_dir))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_factorizer(path: Path, device: torch.device) -> tuple[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Factorizer not found: {path}\n"
            "Train it first or pass --factorizer-path to the factorizer .pt file."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    factorizer_type = payload.get("factorizer_type")
    if factorizer_type == "shared_shift" or "B" in payload or "B0" in payload:
        if "B" in payload:
            matrix = payload["B"]
        elif "B0" in payload and "delta" in payload:
            matrix = payload["B0"] + payload["delta"]
        else:
            raise KeyError(f"Shared-shift payload has no B or B0+delta keys: {path}")
        return "shared_shift", matrix.to(device=device, dtype=torch.float32)

    if factorizer_type == "projection" or "A" in payload or "A0" in payload:
        if "A" in payload:
            matrix = payload["A"]
        elif "A0" in payload and "delta" in payload:
            matrix = payload["A0"] + payload["delta"]
        else:
            raise KeyError(f"Projection payload has no A or A0+delta keys: {path}")
        return "projection", matrix.to(device=device, dtype=torch.float32)
    else:
        raise KeyError(f"Unknown factorizer payload format: {path}")


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
    hf_home = _set_external_caches(args.hf_home)
    device = _resolve_device(args.device)

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    factorizer_path = Path(args.factorizer_path).expanduser()
    audio_path = Path(args.audio_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(checkpoint_dir, device=device)
    factorizer_type, matrix = _load_factorizer(factorizer_path, device=device)
    from vocos import Vocos

    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()

    wav = _load_wav_mono(audio_path, int(model.config.sample_rate)).to(device)
    audios_srs = [(wav, int(model.config.sample_rate))]

    enc_features, enc_padding_mask, _, _, _ = model.extract_features(audios_srs)
    encoded = model.encode(enc_features, enc_padding_mask)
    attrs = model.encode_attributes(encoded.z, padding_mask=encoded.padding_mask)

    z_mean = attrs.z_mean
    if factorizer_type == "projection":
        x = encoded.z - z_mean
        z_pros = x @ matrix
        z_sem = x - z_pros
    else:
        shared = attrs.z_pros @ matrix
        z_pros = attrs.z_pros - shared
        z_sem = attrs.z_sem + shared

    if encoded.padding_mask is not None:
        valid_mask = (~encoded.padding_mask).to(device=device, dtype=z_pros.dtype).unsqueeze(-1)
        z_pros = z_pros * valid_mask
        z_sem = z_sem * valid_mask

    valid = ~encoded.padding_mask[0]
    np.save(output_dir / "z_pros.npy", z_pros[0, valid, :].float().cpu().numpy())
    np.save(output_dir / "z_sem.npy", z_sem[0, valid, :].float().cpu().numpy())
    np.save(output_dir / "z_mean.npy", z_mean[0].squeeze(0).float().cpu().numpy())

    speaker_embedding = model.extract_speaker_embedding(audios_srs)
    variants = {
        "z_pros_plus_z_mean": z_pros + z_mean,
        "z_sem_plus_z_mean": z_sem + z_mean,
        "z_pros_plus_z_sem_plus_z_mean": z_pros + z_sem + z_mean,
    }

    _save_audio(output_dir / "original_male.wav", wav.detach().cpu().unsqueeze(0), int(model.config.sample_rate))
    manifest = []
    for name, z_variant in variants.items():
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
        audio = vocoder.decode(features)
        output_path = output_dir / f"{name}.wav"
        _save_audio(output_path, audio, int(model.config.sample_rate))
        manifest.append({"variant": name, "path": str(output_path)})

    summary = {
        "audio_path": str(audio_path),
        "checkpoint_dir": str(checkpoint_dir),
        "factorizer_path": str(factorizer_path),
        "factorizer_type": factorizer_type,
        "hf_home": str(hf_home),
        "device": str(device),
        "num_steps": args.num_steps,
        "latent_shape": list(encoded.z.shape),
        "valid_frames": int(valid.sum().item()),
        "outputs": manifest,
    }
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
