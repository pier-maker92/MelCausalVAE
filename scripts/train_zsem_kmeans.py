#!/usr/bin/env python3
"""Train a KMeans codebook on saved z_sem attributes."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.dont_write_bytecode = True


DEFAULT_ATTRIBUTES_DIR = (
    "/Volumes/Crucial X6/Research/Datasets/libritts-r-dicodec-attributes"
)
DEFAULT_OUTPUT_DIR = (
    "/Volumes/Crucial X6/Research/Datasets/libritts-r-dicodec-zsem-kmeans/512"
)
DEFAULT_FACTORIZER_PATH = (
    "/Volumes/Crucial X6/Research/Datasets/libritts-r-dicodec-factorizer/"
    "shared_shift_factorizer.pt"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KMeans on saved z_sem attributes.")
    parser.add_argument("--attributes-dir", default=DEFAULT_ATTRIBUTES_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--latent-source",
        default="z_sem",
        choices=("z_sem", "z_sem_new"),
        help=(
            "Use native z_sem.npy, or compute shared-shift z_sem_new = "
            "z_sem + z_pros @ B from --factorizer-path."
        ),
    )
    parser.add_argument("--factorizer-path", default=DEFAULT_FACTORIZER_PATH)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--num-clusters", type=int, default=512)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500_000,
        help="Maximum total z_sem frames to train on. Use 0 for all frames.",
    )
    parser.add_argument(
        "--max-frames-per-sample",
        type=int,
        default=512,
        help="Maximum frames sampled from each utterance. Use 0 for all frames.",
    )
    parser.add_argument("--collect-batch-size", type=int, default=128)
    parser.add_argument("--kmeans-chunk-size", type=int, default=16384)
    return parser.parse_args()


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


def _attribute_dirs(attributes_dir: Path) -> list[Path]:
    dirs = [
        item
        for item in attributes_dir.iterdir()
        if item.is_dir() and (item / "z_sem.npy").is_file()
    ]
    dirs.sort()
    if not dirs:
        raise FileNotFoundError(f"No z_sem.npy files found under {attributes_dir}")
    return dirs


def _load_shared_shift_matrix(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(
            f"Factorizer not found: {path}\n"
            "Train the shared-shift factorizer first or pass --factorizer-path."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    factorizer_type = payload.get("factorizer_type")
    if factorizer_type != "shared_shift" and "B" not in payload and "B0" not in payload:
        raise ValueError(
            f"--latent-source z_sem_new requires a shared_shift factorizer, got: {path}"
        )
    if "B" in payload:
        matrix = payload["B"]
    elif "B0" in payload and "delta" in payload:
        matrix = payload["B0"] + payload["delta"]
    else:
        raise KeyError(f"Shared-shift payload has no B or B0+delta keys: {path}")
    return matrix.detach().cpu().to(dtype=torch.float32)


def _choose_frames(
    num_frames: int,
    max_frames_per_sample: int,
    rng: random.Random,
) -> np.ndarray | None:
    if max_frames_per_sample <= 0 or num_frames <= max_frames_per_sample:
        return None
    return np.array(rng.sample(range(num_frames), max_frames_per_sample), dtype=np.int64)


def _collect_zsem_frames(
    sample_dirs: list[Path],
    max_frames: int | None,
    max_frames_per_sample: int,
    collect_batch_size: int,
    seed: int,
    latent_source: str,
    shared_shift_matrix: torch.Tensor | None,
) -> torch.Tensor:
    rng = random.Random(seed)
    chunks: list[torch.Tensor] = []
    collected = 0
    shared_shift_array = (
        shared_shift_matrix.numpy() if shared_shift_matrix is not None else None
    )

    progress = tqdm(sample_dirs, desc=f"Collecting {latent_source} frames", unit="sample")
    pending: list[torch.Tensor] = []
    for sample_dir in progress:
        z_sem = np.load(sample_dir / "z_sem.npy", mmap_mode="r")
        frame_indices = _choose_frames(
            int(z_sem.shape[0]),
            max_frames_per_sample=max_frames_per_sample,
            rng=rng,
        )
        if frame_indices is None:
            array = np.array(z_sem, dtype=np.float32, copy=True)
        else:
            array = np.array(z_sem[frame_indices], dtype=np.float32, copy=True)

        if latent_source == "z_sem_new":
            if shared_shift_matrix is None or shared_shift_array is None:
                raise ValueError("shared_shift_matrix is required for z_sem_new.")
            z_pros = np.load(sample_dir / "z_pros.npy", mmap_mode="r")
            if z_pros.shape != z_sem.shape:
                raise ValueError(
                    f"Shape mismatch in {sample_dir}: "
                    f"z_pros={z_pros.shape}, z_sem={z_sem.shape}"
                )
            if frame_indices is None:
                z_pros_array = np.array(z_pros, dtype=np.float32, copy=True)
            else:
                z_pros_array = np.array(
                    z_pros[frame_indices],
                    dtype=np.float32,
                    copy=True,
                )
            if z_pros_array.shape[-1] != shared_shift_matrix.shape[0]:
                raise ValueError(
                    f"Factorizer dim {shared_shift_matrix.shape[0]} is incompatible "
                    f"with z_pros dim {z_pros_array.shape[-1]} in {sample_dir}"
                )
            array += z_pros_array @ shared_shift_array

        if max_frames is not None:
            remaining = max_frames - collected
            if remaining <= 0:
                break
            if array.shape[0] > remaining:
                array = array[:remaining]

        tensor = torch.from_numpy(array)
        pending.append(tensor)
        collected += int(tensor.shape[0])

        if len(pending) >= collect_batch_size:
            chunks.append(torch.cat(pending, dim=0))
            pending.clear()

        progress.set_postfix(frames=collected)
        if max_frames is not None and collected >= max_frames:
            break

    if pending:
        chunks.append(torch.cat(pending, dim=0))
    if not chunks:
        raise RuntimeError(f"No {latent_source} frames were collected.")

    points = torch.cat(chunks, dim=0).contiguous()
    if points.ndim != 2:
        raise ValueError(
            f"Expected [frames, dim] {latent_source} points, got {tuple(points.shape)}"
        )
    return points


def _assign(points: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(points, centroids).square()
    return distances.min(dim=1)


def _kmeans(
    points: torch.Tensor,
    num_clusters: int,
    max_iterations: int,
    tolerance: float,
    seed: int,
    chunk_size: int,
    latent_source: str,
) -> tuple[torch.Tensor, float, int]:
    if points.shape[0] < num_clusters:
        raise ValueError(
            f"Need at least {num_clusters} frames, collected only {points.shape[0]}."
        )

    generator = torch.Generator(device=points.device).manual_seed(seed)
    initial_indices = torch.randperm(
        points.shape[0],
        generator=generator,
        device=points.device,
    )[:num_clusters]
    centroids = points[initial_indices].clone()
    inertia = float("nan")
    num_chunks = math.ceil(points.shape[0] / chunk_size)

    progress = tqdm(
        total=max_iterations * num_chunks,
        desc=f"KMeans {latent_source}",
        unit="chunk",
    )
    try:
        for iteration in range(1, max_iterations + 1):
            sums = torch.zeros_like(centroids)
            counts = torch.zeros(num_clusters, dtype=torch.long, device=points.device)
            inertia = 0.0

            for point_chunk in points.split(chunk_size):
                min_distances, assignments = _assign(point_chunk, centroids)
                sums.index_add_(0, assignments, point_chunk)
                counts += torch.bincount(assignments, minlength=num_clusters)
                inertia += float(min_distances.sum().detach().cpu())
                progress.update(1)
                progress.set_postfix(
                    iteration=f"{iteration}/{max_iterations}",
                    inertia=f"{inertia:.2f}",
                )

            nonempty = counts > 0
            updated_centroids = centroids.clone()
            updated_centroids[nonempty] = (
                sums[nonempty] / counts[nonempty].unsqueeze(1).to(sums.dtype)
            )
            if (~nonempty).any():
                replacement_indices = torch.randint(
                    points.shape[0],
                    ((~nonempty).sum().item(),),
                    generator=generator,
                    device=points.device,
                )
                updated_centroids[~nonempty] = points[replacement_indices]

            centroid_shift = (updated_centroids - centroids).norm(dim=1).max().item()
            centroids = updated_centroids
            progress.set_postfix(
                iteration=f"{iteration}/{max_iterations}",
                inertia=f"{inertia:.2f}",
                shift=f"{centroid_shift:.2e}",
            )
            if centroid_shift <= tolerance:
                progress.total = progress.n
                progress.refresh()
                return centroids, inertia, iteration
    finally:
        progress.close()

    return centroids, inertia, max_iterations


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    attributes_dir = Path(args.attributes_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    sample_dirs = _attribute_dirs(attributes_dir)
    if args.max_samples is not None:
        rng = random.Random(args.seed)
        rng.shuffle(sample_dirs)
        sample_dirs = sample_dirs[: args.max_samples]
        sample_dirs.sort()

    shared_shift_matrix = None
    factorizer_path = None
    if args.latent_source == "z_sem_new":
        factorizer_path = Path(args.factorizer_path).expanduser()
        shared_shift_matrix = _load_shared_shift_matrix(factorizer_path)

    max_frames = None if args.max_frames <= 0 else args.max_frames
    print(f"Found {len(sample_dirs)} attribute samples in {attributes_dir}")
    print(f"Collecting up to {max_frames or 'all'} frames on CPU")
    points = _collect_zsem_frames(
        sample_dirs=sample_dirs,
        max_frames=max_frames,
        max_frames_per_sample=args.max_frames_per_sample,
        collect_batch_size=args.collect_batch_size,
        seed=args.seed,
        latent_source=args.latent_source,
        shared_shift_matrix=shared_shift_matrix,
    )
    print(f"Training KMeans on {tuple(points.shape)} using device={device}")
    points = points.to(device=device, dtype=torch.float32)

    centroids, inertia, iterations = _kmeans(
        points=points,
        num_clusters=args.num_clusters,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        seed=args.seed,
        chunk_size=args.kmeans_chunk_size,
        latent_source=args.latent_source,
    )
    centroids_cpu = centroids.detach().cpu()

    latent_selection: dict[str, Any] = {
        "indices": None,
        "start": 0,
        "end": int(points.shape[-1]),
        "num_dims": int(points.shape[-1]),
        "source": args.latent_source,
    }
    payload = {
        "centroids": centroids_cpu,
        "feature_dims": int(points.shape[-1]),
        "latent_selection": latent_selection,
        "num_clusters": args.num_clusters,
        "source": f"saved_{args.latent_source}",
        "attributes_dir": str(attributes_dir),
        "factorizer_path": str(factorizer_path) if factorizer_path is not None else None,
    }
    torch.save(payload, output_dir / "encoder_kmeans.pt")
    np.save(output_dir / "centroids.npy", centroids_cpu.numpy())

    summary = {
        "attributes_dir": str(attributes_dir),
        "output_dir": str(output_dir),
        "source": f"saved_{args.latent_source}",
        "factorizer_path": str(factorizer_path) if factorizer_path is not None else None,
        "num_attribute_samples": len(sample_dirs),
        "num_frames": int(points.shape[0]),
        "feature_dims": int(points.shape[-1]),
        "latent_selection": latent_selection,
        "num_clusters": args.num_clusters,
        "iterations": iterations,
        "inertia": inertia,
        "config": vars(args),
    }
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
