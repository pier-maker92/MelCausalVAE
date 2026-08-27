#!/usr/bin/env python3
"""Batch-encode Dicodec latents z from one of the project datasets.

This script only runs feature extraction + encoder. It does not decode and it
does not compute z_sem/z_pros/z_mean attributes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


DEFAULT_CHECKPOINT_DIR = None
DEFAULT_OUTPUT_DIR = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode only z latents from mls, librispeech, or libritts-r."
    )
    parser.add_argument(
        "--dataset-name",
        choices=("mls", "librispeech", "libritts-r"),
        required=True,
    )
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--mls-languages",
        nargs="+",
        default=None,
        help="Optional MLS languages: french german spanish english.",
    )
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional explicit model config JSON. Defaults to checkpoint-dir/config.json.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-audio-len", type=float, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_device(requested: str) -> "torch.device":
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")
    return torch.device(requested)


def _load_model(
    checkpoint_dir: Path,
    config_json: Path | None,
    device: "torch.device",
) -> "torch.nn.Module":
    import torch

    sys.path.insert(0, str(_repo_root()))
    from modules.builder import build_model

    config_path = config_json or checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with config_path.open() as config_file:
        model_config = json.load(config_file)

    model = build_model(model_config)
    model.from_pretrained(str(checkpoint_dir))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_dataset(dataset_name: str, mls_languages: list[str] | None):
    sys.path.insert(0, str(_repo_root()))
    if dataset_name == "mls":
        from data.mls import MLSDataset

        return MLSDataset(languages=mls_languages)
    if dataset_name == "librispeech":
        from data.librispeech import LibriSpeechDataset

        return LibriSpeechDataset()
    if dataset_name == "libritts-r":
        from data.libri_tts_r import LibriTTSR

        return LibriTTSR()
    raise ValueError(f"Unsupported dataset: {dataset_name}")


class IndexedDataset:
    def __init__(self, dataset, start_index: int, max_samples: int | None):
        self.dataset = dataset
        self.start_index = start_index
        remaining = max(0, len(dataset) - start_index)
        self.length = remaining if max_samples is None else min(max_samples, remaining)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self.start_index + index
        item = dict(self.dataset[source_index])
        item["_encode_index"] = source_index
        return item


def _collate(instances: list[dict[str, Any]]) -> dict[str, Any]:
    from data.audio_dataset import DataCollator

    batch = DataCollator()(instances)
    batch["indices"] = [instance["_encode_index"] for instance in instances]
    return batch


def _safe_id(value: Any, index: int) -> str:
    if value is None:
        return f"sample_{index:08d}"
    safe = Path(str(value)).stem
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in safe)
    return safe or f"sample_{index:08d}"


def _validate_padding_mask(
    z: "torch.Tensor",
    padding_mask: "torch.BoolTensor",
    sample_ids: list[str],
) -> list["torch.Tensor"]:
    if padding_mask.dtype != torch.bool:
        raise TypeError(f"padding_mask must be bool, got {padding_mask.dtype}")
    if padding_mask.shape != z.shape[:2]:
        raise ValueError(
            "padding_mask must match z batch/time dimensions, got "
            f"mask={tuple(padding_mask.shape)} z={tuple(z.shape)}"
        )

    valid_rows = []
    for row, sample_id in enumerate(sample_ids):
        valid = ~padding_mask[row]
        if not valid.any():
            raise RuntimeError(f"No valid latent frames for {sample_id}")
        first_padded = padding_mask[row].float().argmax().item()
        if padding_mask[row].any() and (~padding_mask[row, first_padded:]).any():
            raise RuntimeError(f"Non-contiguous padding mask for {sample_id}")
        valid_rows.append(valid)
    return valid_rows


def _save_tensor(path: Path, tensor: "torch.Tensor") -> None:
    import numpy as np

    np.save(path, tensor.detach().float().cpu().numpy())


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from data.audio_dataset import TrainDatasetWrapper, TestDatasetWrapper

    device = _resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    config_json = Path(args.config_json).expanduser() if args.config_json else None
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dataset = _load_dataset(args.dataset_name, args.mls_languages)
    wrapper_cls = TrainDatasetWrapper if args.split == "train" else TestDatasetWrapper
    dataset = wrapper_cls(raw_dataset, args.split, max_audio_len=args.max_audio_len)
    dataset = IndexedDataset(dataset, args.start_index, args.max_samples)

    model = _load_model(checkpoint_dir, config_json, device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )

    written = 0
    skipped = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding z/{args.dataset_name}/{args.split}"):
            if "audio_input_srs" not in batch:
                raise KeyError("Batch is missing audio_input_srs.")
            if "16k_audio" not in batch:
                raise KeyError(
                    "Batch is missing 16k_audio; WavLM needs explicit 16 kHz audio."
                )

            sample_ids = [
                _safe_id(sample_id, index)
                for sample_id, index in zip(batch.get("ids", []), batch["indices"])
            ]
            sample_dirs = [output_dir / sample_id for sample_id in sample_ids]
            keep_rows = [
                args.overwrite or not (sample_dir / "z.npy").is_file()
                for sample_dir in sample_dirs
            ]
            if not any(keep_rows):
                skipped += len(sample_dirs)
                continue

            audios_srs = [
                (audio.to(device), sample_rate)
                for audio, sample_rate in batch["audio_input_srs"]
            ]
            audio_16khz = [audio.to(device) for audio in batch["16k_audio"]]

            features, padding_mask, _, _, _ = model.extract_features(
                audios_srs,
                audio_16khz=audio_16khz,
            )
            encoded = model.encode(features, padding_mask)
            valid_rows = _validate_padding_mask(
                encoded.z,
                encoded.padding_mask,
                sample_ids,
            )

            for row, sample_dir in enumerate(sample_dirs):
                if not keep_rows[row]:
                    skipped += 1
                    continue

                valid = valid_rows[row]
                z_valid = encoded.z[row, valid, :]
                sample_dir.mkdir(parents=True, exist_ok=True)
                _save_tensor(sample_dir / "z.npy", z_valid)
                _save_tensor(
                    sample_dir / "padding_mask.npy",
                    encoded.padding_mask[row],
                )
                metadata = {
                    "dataset_name": args.dataset_name,
                    "split": args.split,
                    "index": batch["indices"][row],
                    "id": sample_ids[row],
                    "checkpoint_dir": str(checkpoint_dir),
                    "config_json": str(config_json or checkpoint_dir / "config.json"),
                    "shape": {
                        "z": list(z_valid.shape),
                        "padding_mask": list(encoded.padding_mask[row].shape),
                    },
                    "valid_frames": int(valid.sum().item()),
                    "padded_frames": int(encoded.padding_mask[row].sum().item()),
                }
                with (sample_dir / "metadata.json").open("w") as metadata_file:
                    json.dump(metadata, metadata_file, indent=2, sort_keys=True)
                written += 1

    print(
        f"Done. written={written} skipped={skipped} "
        f"dataset={args.dataset_name} split={args.split} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
