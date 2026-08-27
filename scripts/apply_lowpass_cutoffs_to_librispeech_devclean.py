#!/usr/bin/env python3
"""Decode low-pass attribute variants for the first LibriSpeech dev-clean samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.dont_write_bytecode = True


DEFAULT_HF_HOME = "/Volumes/Crucial X6/HF_HOME"
DEFAULT_DATASET_DIR = "/Users/software/Research/datasets/librispeech-aligned/dev_clean"
DEFAULT_CHECKPOINT_DIR = (
    "/Users/software/Research/MelCausalVAE/checkpoints/paper/baseline/"
    "18-denc128-novq"
)
DEFAULT_OUTPUT_DIR = (
    "/Users/software/Research/MelCausalVAE/lab/outputs/"
    "librispeech_devclean_lowpass_first3"
)
DEFAULT_SPLIT = "validation"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode low-pass attribute variants for the first dev-clean samples."
    )
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--cutoff-hz", type=float, default=1.5)
    parser.add_argument("--order", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _set_external_caches(hf_home: str) -> Path:
    hf_home_path = Path(hf_home).expanduser()
    for child in (
        "datasets",
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
    os.environ["HF_DATASETS_CACHE"] = str(hf_home_path / "datasets")
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

    with (checkpoint_dir / "config.json").open() as config_file:
        cfg_dict = json.load(config_file)

    model = build_model(cfg_dict)
    model.from_pretrained(str(checkpoint_dir))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _audio_tensor(row: dict[str, Any]) -> tuple[torch.Tensor, int]:
    audio = row["audio"]
    waveform = torch.as_tensor(audio["array"], dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform, int(audio["sampling_rate"])


def _sample_id(row: dict[str, Any], index: int) -> str:
    return Path(str(row.get("id") or f"sample_{index:08d}")).stem


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

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(_repo_root()))
    from datasets import load_dataset
    from modules.lp_filter import LowPassFilter

    dataset = load_dataset("parquet", data_dir=args.dataset_dir, split=args.split)
    indices = args.indices if args.indices is not None else list(range(args.num_samples))
    rows = [dataset[index] for index in indices]

    model = _load_model(Path(args.checkpoint_dir).expanduser(), device=device)
    model.lowpass_filter = LowPassFilter(
        cutoff_hz=args.cutoff_hz,
        sample_rate=model.config.lowpass_filter_config.sample_rate,
        order=args.order,
    ).to(device)

    audios_srs = []
    sample_ids = []
    for index, row in zip(indices, rows):
        waveform, sample_rate = _audio_tensor(row)
        audios_srs.append((waveform.to(device), sample_rate))
        sample_id = _sample_id(row, index)
        sample_ids.append(sample_id)
        _save_audio(output_dir / f"{sample_id}_original.wav", waveform.unsqueeze(0), sample_rate)

    enc_features, enc_padding_mask, _, _ = model.extract_features(audios_srs)
    encoded = model.encode(enc_features, enc_padding_mask)
    attrs = model.encode_attributes(encoded.z, padding_mask=encoded.padding_mask)
    speaker_embedding = model.extract_speaker_embedding(audios_srs)

    variants = {
        "z_sem_plus_mean": attrs.z_sem + attrs.z_mean,
        "z_pros_plus_mean": attrs.z_pros + attrs.z_mean,
        "z_pros_plus_z_sem_plus_mean": attrs.z_pros + attrs.z_sem + attrs.z_mean,
    }

    manifest = []
    for variant_name, z_variant in variants.items():
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
        for row_idx, sample_id in enumerate(sample_ids):
            features = mel[row_idx, ~mel_mask[row_idx]].unsqueeze(0).permute(0, 2, 1)
            audio = model.vocoder.decode(features)
            output_path = output_dir / f"{sample_id}_{variant_name}.wav"
            _save_audio(output_path, audio, int(model.config.sample_rate))
            manifest.append(
                {
                    "sample_id": sample_id,
                    "variant": variant_name,
                    "path": str(output_path),
                }
            )

    summary = {
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "checkpoint_dir": args.checkpoint_dir,
        "hf_home": str(hf_home),
        "device": str(device),
        "num_samples": args.num_samples,
        "indices": indices,
        "cutoff_hz": args.cutoff_hz,
        "order": args.order,
        "sample_ids": sample_ids,
        "outputs": manifest,
    }
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
