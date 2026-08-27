#!/usr/bin/env python3
"""Download LibriTTS-R train-clean-100 to an external HF cache and encode attributes.

Example:
    export HF_HOME="/Volumes/Crucial X6/HF_HOME"
    python scripts/encode_libritts_r_dicodec_attributes.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


DEFAULT_HF_HOME = "/Volumes/Crucial X6/HF_HOME"
DEFAULT_CHECKPOINT_DIR = (
    "/Users/software/Research/MelCausalVAE/checkpoints/paper/baseline/"
    "18-denc128-novq"
)
DEFAULT_OUTPUT_DIR = (
    "/Volumes/Crucial X6/Research/Datasets/libritts-r-dicodec-attributes"
)
DEFAULT_DATASET = "parler-tts/libritts_r_filtered"
DEFAULT_CONFIG = "clean"
DEFAULT_SPLIT = "train.clean.100"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode z_sem, z_pros and z_mean for the LibriTTS-R "
            "train-clean-100 split."
        )
    )
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=(
            "Dataset split. Aliases train-clean-100 and train_clean_100 are "
            "normalized to Hugging Face's train.clean.100."
        ),
    )
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute samples whose three .npy files already exist.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Fall back to CPU if MPS is unavailable.",
    )
    return parser.parse_args()


def _normalize_split(split: str) -> str:
    aliases = {
        "train-clean-100": DEFAULT_SPLIT,
        "train_clean_100": DEFAULT_SPLIT,
        "train.clean.100": DEFAULT_SPLIT,
    }
    return aliases.get(split, split)


def _set_external_caches(hf_home: str) -> Path:
    hf_home_path = Path(hf_home).expanduser()
    hf_home_path.mkdir(parents=True, exist_ok=True)
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

    # Some ops still fall back from MPS internally; keep the process alive if so.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return hf_home_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_model(checkpoint_dir: Path, device: "torch.device") -> "torch.nn.Module":
    import torch

    sys.path.insert(0, str(_repo_root()))
    from modules.builder import build_model

    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")

    with config_path.open() as config_file:
        model_config = json.load(config_file)

    model = build_model(model_config)
    model.from_pretrained(str(checkpoint_dir))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _sample_id(row: dict[str, Any], index: int) -> str:
    value = row.get("id") or row.get("path") or f"sample_{index:08d}"
    return Path(str(value)).stem


def _audio_tensor(row: dict[str, Any]) -> tuple["torch.Tensor", int]:
    import torch

    audio = row["audio"]
    waveform = torch.as_tensor(audio["array"], dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform, int(audio["sampling_rate"])


def _collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def _complete(sample_dir: Path) -> bool:
    return all((sample_dir / name).is_file() for name in ("z_sem.npy", "z_pros.npy", "z_mean.npy"))


def _save_array(path: Path, tensor: "torch.Tensor") -> None:
    import numpy as np

    array = tensor.detach().float().cpu().numpy()
    np.save(path, array)


def main() -> None:
    args = _parse_args()
    args.split = _normalize_split(args.split)
    hf_home = _set_external_caches(args.hf_home)

    import numpy as np  # noqa: F401  # Ensures NumPy import errors happen early.
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("MPS is not available. Re-run with --allow-cpu to use CPU.")

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}")
    print(f"Loading {args.dataset_name}/{args.dataset_config} split={args.split}")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        cache_dir=str(hf_home / "datasets"),
    )

    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Loading model from {checkpoint_dir} on {device}")
    model = _load_model(checkpoint_dir, device=device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )

    written = 0
    skipped = 0
    seen = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding train-clean-100", unit="batch"):
            audios_srs = []
            sample_dirs = []
            sample_ids = []
            rows_to_encode = []

            for row in batch:
                absolute_index = args.start_index + seen
                seen += 1
                sample_id = _sample_id(row, absolute_index)
                sample_dir = output_dir / sample_id
                if _complete(sample_dir) and not args.overwrite:
                    skipped += 1
                    continue
                waveform, sample_rate = _audio_tensor(row)
                audios_srs.append((waveform.to(device), sample_rate))
                sample_dirs.append(sample_dir)
                sample_ids.append(sample_id)
                rows_to_encode.append((row, absolute_index))

            if not audios_srs:
                continue

            features, padding_mask, _, _, _ = model.extract_features(audios_srs)
            encoded = model.encode(features, padding_mask)
            attrs = model.encode_attributes(encoded.z, padding_mask=encoded.padding_mask)

            for row_idx, sample_dir in enumerate(sample_dirs):
                valid = ~encoded.padding_mask[row_idx]
                if not valid.any():
                    raise RuntimeError(f"No valid latent frames for {sample_ids[row_idx]}")

                sample_dir.mkdir(parents=True, exist_ok=True)
                _save_array(sample_dir / "z_sem.npy", attrs.z_sem[row_idx, valid, :])
                _save_array(sample_dir / "z_pros.npy", attrs.z_pros[row_idx, valid, :])
                _save_array(sample_dir / "z_mean.npy", attrs.z_mean[row_idx].squeeze(0))

                row, absolute_index = rows_to_encode[row_idx]
                metadata = {
                    "dataset_name": args.dataset_name,
                    "dataset_config": args.dataset_config,
                    "split": args.split,
                    "index": absolute_index,
                    "id": sample_ids[row_idx],
                    "source_path": row.get("path"),
                    "speaker_id": row.get("speaker_id"),
                    "chapter_id": row.get("chapter_id"),
                    "checkpoint_dir": str(checkpoint_dir),
                    "hf_home": str(hf_home),
                    "shape": {
                        "z_sem": list(attrs.z_sem[row_idx, valid, :].shape),
                        "z_pros": list(attrs.z_pros[row_idx, valid, :].shape),
                        "z_mean": list(attrs.z_mean[row_idx].squeeze(0).shape),
                    },
                }
                with (sample_dir / "metadata.json").open("w") as metadata_file:
                    json.dump(metadata, metadata_file, indent=2, sort_keys=True)
                written += 1

    print(
        f"Done. written={written} skipped={skipped} output_dir={output_dir} "
        f"split={args.split}"
    )


if __name__ == "__main__":
    main()
