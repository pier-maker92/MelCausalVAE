#!/usr/bin/env python3
"""Test attribute reconstructions and encode throughput on LibriSpeech dev-clean."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


DEFAULT_HF_HOME = "/Volumes/Crucial X6/HF_HOME"
DEFAULT_DATASET_DIR = "/Users/software/Research/datasets/librispeech-aligned/dev_clean"
DEFAULT_CHECKPOINT_DIR = (
    "/Users/software/Research/MelCausalVAE/checkpoints/paper/baseline/"
    "18-denc128-novq"
)
DEFAULT_OUTPUT_DIR = "/Users/software/Research/MelCausalVAE/lab/outputs/devclean_attribute_tests"
DEFAULT_SPLIT = "validation"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--benchmark-samples", type=int, default=32)
    parser.add_argument("--benchmark-batch-sizes", default="1,2,4,8")
    parser.add_argument("--max-duration-seconds", type=float, default=12.0)
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


def _load_model(checkpoint_dir: Path, device: "torch.device") -> "torch.nn.Module":
    sys.path.insert(0, str(_repo_root()))
    from modules.builder import build_model

    with (checkpoint_dir / "config.json").open() as config_file:
        model_config = json.load(config_file)

    model = build_model(model_config)
    model.from_pretrained(str(checkpoint_dir))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _sample_id(row: dict[str, Any], index: int) -> str:
    return Path(str(row.get("id") or row.get("path") or f"sample_{index:08d}")).stem


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


def _sync_mps(device: "torch.device") -> None:
    import torch

    if device.type == "mps":
        torch.mps.synchronize()


def _pick_indices(dataset, count: int, max_duration_seconds: float) -> list[int]:
    indices = []
    for idx, row in enumerate(dataset):
        audio = row["audio"]
        duration = len(audio["array"]) / int(audio["sampling_rate"])
        if duration <= max_duration_seconds:
            indices.append(idx)
        if len(indices) >= count:
            return indices
    raise RuntimeError(
        f"Found only {len(indices)} samples <= {max_duration_seconds}s; need {count}."
    )


def _decode_variants(
    model,
    vocoder,
    rows: list[dict[str, Any]],
    indices: list[int],
    output_dir: Path,
    label: str,
    num_steps: int,
    temperature: float,
    guidance_scale: float,
    seed: int,
) -> list[dict[str, Any]]:
    import torch
    import torchaudio

    audios_srs = []
    sample_ids = []
    for row, index in zip(rows, indices):
        waveform, sample_rate = _audio_tensor(row)
        audios_srs.append((waveform.to(model.device), sample_rate))
        sample_ids.append(_sample_id(row, index))

        original_path = output_dir / f"{label}_{_sample_id(row, index)}_original.wav"
        torchaudio.save(str(original_path), waveform.unsqueeze(0), sample_rate)

    with torch.no_grad():
        enc_features, enc_padding_mask, _, _, _ = model.extract_features(audios_srs)
        encoded = model.encode(enc_features, enc_padding_mask)
        attrs = model.encode_attributes(encoded.z, padding_mask=encoded.padding_mask)
        speaker_embedding = model.extract_speaker_embedding(audios_srs)

    variants = {
        "z_sem_plus_z_mean": attrs.z_sem + attrs.z_mean,
        "z_sem_plus_z_mean_plus_z_pros": attrs.z_sem + attrs.z_mean + attrs.z_pros,
        "z_pros_only": attrs.z_pros,
    }
    manifest = []
    for variant_name, z_variant in variants.items():
        with torch.no_grad():
            generator = torch.Generator(device=model.device).manual_seed(seed)
            mel, mel_mask = model.sample(
                num_steps=num_steps,
                temperature=temperature,
                guidance_scale=guidance_scale,
                z=z_variant,
                generator=generator,
                padding_mask=encoded.padding_mask,
                speaker_embedding=speaker_embedding,
            )
        for row_idx, sample_id in enumerate(sample_ids):
            features = mel[row_idx, ~mel_mask[row_idx]].unsqueeze(0).permute(0, 2, 1)
            with torch.no_grad():
                audio = vocoder.decode(features).detach().cpu()
            peak = audio.abs().max()
            if peak > 0:
                audio = audio / peak.clamp_min(1e-8)
            output_path = output_dir / f"{label}_{sample_id}_{variant_name}.wav"
            torchaudio.save(str(output_path), audio, int(model.config.sample_rate))
            manifest.append(
                {
                    "batch_label": label,
                    "sample_id": sample_id,
                    "variant": variant_name,
                    "path": str(output_path),
                }
            )
    return manifest


def _benchmark_encode(
    model,
    dataset,
    indices: list[int],
    batch_sizes: list[int],
    device: "torch.device",
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    rows = [dataset[index] for index in indices]
    results = []
    for batch_size in batch_sizes:
        loader = DataLoader(
            rows,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        )
        encoded_samples = 0
        _sync_mps(device)
        started = time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                audios_srs = []
                for row in batch:
                    waveform, sample_rate = _audio_tensor(row)
                    audios_srs.append((waveform.to(device), sample_rate))
                features, padding_mask, _, _, _ = model.extract_features(audios_srs)
                _ = model.encode(features, padding_mask)
                encoded_samples += len(batch)
        _sync_mps(device)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "batch_size": batch_size,
                "samples": encoded_samples,
                "seconds": elapsed,
                "samples_per_second": encoded_samples / elapsed,
            }
        )
    return results


def main() -> None:
    args = _parse_args()
    hf_home = _set_external_caches(args.hf_home)

    import torch
    from datasets import load_dataset
    from vocos import Vocos

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available.")
    device = torch.device("mps")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("parquet", data_dir=args.dataset_dir, split=args.split)
    selected = _pick_indices(
        dataset,
        count=max(2, args.benchmark_samples),
        max_duration_seconds=args.max_duration_seconds,
    )

    model = _load_model(Path(args.checkpoint_dir).expanduser(), device=device)
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()

    manifest = []
    manifest.extend(
        _decode_variants(
            model=model,
            vocoder=vocoder,
            rows=[dataset[selected[0]]],
            indices=[selected[0]],
            output_dir=output_dir,
            label="batch1",
            num_steps=args.num_steps,
            temperature=args.temperature,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    )
    manifest.extend(
        _decode_variants(
            model=model,
            vocoder=vocoder,
            rows=[dataset[selected[0]], dataset[selected[1]]],
            indices=[selected[0], selected[1]],
            output_dir=output_dir,
            label="batch2",
            num_steps=args.num_steps,
            temperature=args.temperature,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    )

    batch_sizes = [int(item) for item in args.benchmark_batch_sizes.split(",") if item]
    benchmark = _benchmark_encode(
        model=model,
        dataset=dataset,
        indices=selected[: args.benchmark_samples],
        batch_sizes=batch_sizes,
        device=device,
    )

    payload = {
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "checkpoint_dir": args.checkpoint_dir,
        "hf_home": str(hf_home),
        "device": str(device),
        "selected_indices": selected[: max(2, args.benchmark_samples)],
        "num_steps": args.num_steps,
        "audio_manifest": manifest,
        "benchmark": benchmark,
    }
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(payload, summary_file, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
