#!/usr/bin/env python3
"""Decode male.wav with native z_sem replaced by a trained KMeans codebook."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.dont_write_bytecode = True


DEFAULT_HF_HOME = "/Volumes/Crucial X6/HF_HOME"
DEFAULT_CHECKPOINT_DIR = (
    "/Users/software/Research/MelCausalVAE/checkpoints/paper/baseline/"
    "18-denc128-novq"
)
DEFAULT_AUDIO_PATH = "/Users/software/Research/MelCausalVAE/audios/male.wav"
DEFAULT_KMEANS_PATH = (
    "/Volumes/Crucial X6/Research/Datasets/libritts-r-dicodec-zsem-kmeans/"
    "original-zsem-512-it300/encoder_kmeans.pt"
)
DEFAULT_FACTORIZER_PATH = (
    "/Volumes/Crucial X6/Research/Datasets/libritts-r-dicodec-factorizer/"
    "shared_shift_factorizer.pt"
)
DEFAULT_OUTPUT_DIR = "/Users/software/Research/MelCausalVAE/lab/outputs/male_zsem_kmeans"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode audio with KMeans-quantized native z_sem."
    )
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--audio-path", default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--kmeans-path", default=DEFAULT_KMEANS_PATH)
    parser.add_argument("--factorizer-path", default=DEFAULT_FACTORIZER_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--kmeans-chunk-size", type=int, default=16384)
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


def _load_kmeans(path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "centroids" not in payload:
        raise ValueError(f"Invalid KMeans codebook: {path}")
    payload["centroids"] = payload["centroids"].to(device=device, dtype=torch.float32)
    return payload


def _kmeans_latent_source(kmeans_codebook: dict) -> str:
    selection = kmeans_codebook.get("latent_selection") or {}
    return str(selection.get("source") or kmeans_codebook.get("source") or "z_sem")


def _load_shared_shift_matrix(path: Path, device: torch.device) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(
            f"Factorizer not found: {path}\n"
            "Pass --factorizer-path for KMeans trained on z_sem_new."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    factorizer_type = payload.get("factorizer_type")
    if factorizer_type != "shared_shift" and "B" not in payload and "B0" not in payload:
        raise ValueError(f"Expected a shared_shift factorizer, got: {path}")
    if "B" in payload:
        matrix = payload["B"]
    elif "B0" in payload and "delta" in payload:
        matrix = payload["B0"] + payload["delta"]
    else:
        raise KeyError(f"Shared-shift payload has no B or B0+delta keys: {path}")
    return matrix.to(device=device, dtype=torch.float32)


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    hf_home = _set_external_caches(args.hf_home)
    device = _resolve_device(args.device)

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    audio_path = Path(args.audio_path).expanduser()
    kmeans_path = Path(args.kmeans_path).expanduser()
    factorizer_path = Path(args.factorizer_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(checkpoint_dir, device=device)
    from vocos import Vocos

    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    kmeans_codebook = _load_kmeans(kmeans_path, device=device)

    wav = _load_wav_mono(audio_path, int(model.config.sample_rate)).to(device)
    audios_srs = [(wav, int(model.config.sample_rate))]

    enc_features, enc_padding_mask, _, _, _ = model.extract_features(audios_srs)
    encoded = model.encode(enc_features, enc_padding_mask)
    attrs = model.encode_attributes(encoded.z, padding_mask=encoded.padding_mask)
    latent_source = _kmeans_latent_source(kmeans_codebook)
    z_pros_for_full = attrs.z_pros
    z_sem_for_kmeans = attrs.z_sem
    factorizer_used = None
    if latent_source == "z_sem_new":
        matrix = _load_shared_shift_matrix(factorizer_path, device=device)
        shared = attrs.z_pros @ matrix
        z_sem_for_kmeans = attrs.z_sem + shared
        z_pros_for_full = attrs.z_pros - shared
        factorizer_used = str(factorizer_path)
    elif latent_source != "z_sem":
        raise ValueError(f"Unsupported KMeans latent source: {latent_source}")

    kmeans_encoded = model._kmeans_encode(
        z_sem_for_kmeans,
        padding_mask=encoded.padding_mask,
        kmeans_codebook=kmeans_codebook,
        chunk_size=args.kmeans_chunk_size,
    )
    z_sem_kmeans = kmeans_encoded["quantized"]

    speaker_embedding = model.extract_speaker_embedding(audios_srs)
    variants = {
        "native_z_sem_plus_z_pros_plus_z_mean": (
            z_sem_for_kmeans + z_pros_for_full + attrs.z_mean
        ),
        "kmeans_z_sem_only": z_sem_kmeans,
        "kmeans_z_sem_plus_z_pros_plus_z_mean": (
            z_sem_kmeans + z_pros_for_full + attrs.z_mean
        ),
    }

    _save_audio(
        output_dir / "original_male.wav",
        wav.detach().cpu().unsqueeze(0),
        int(model.config.sample_rate),
    )

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

    valid = ~encoded.padding_mask[0]
    valid_indices = kmeans_encoded["indices"][0, valid]
    summary = {
        "audio_path": str(audio_path),
        "checkpoint_dir": str(checkpoint_dir),
        "hf_home": str(hf_home),
        "device": str(device),
        "kmeans_path": str(kmeans_path),
        "kmeans_latent_source": latent_source,
        "factorizer_path": factorizer_used,
        "num_steps": args.num_steps,
        "latent_shape": list(encoded.z.shape),
        "valid_frames": int(valid.sum().item()),
        "unique_kmeans_codes": int(valid_indices.unique().numel()),
        "outputs": manifest,
    }
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
