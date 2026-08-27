import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm


INPUT_DIR = Path("/Volumes/Crucial X6/Research/dicodec-attributes")
OUTPUT_DIR = INPUT_DIR / "kmeans"
ATTRIBUTE = "z_sem"
N_FRAMES = 500_000
K_VALUES = (128, 512, 1024)
SEED = 1234
SHUFFLE = True


def find_attribute_files(input_dir: Path, attribute: str) -> List[Path]:
    filename = f"{attribute}.npy"
    paths = [
        path
        for path in input_dir.glob(f"*/{filename}")
        if path.parent.name != "kmeans"
    ]
    if SHUFFLE:
        random.Random(SEED).shuffle(paths)
    return paths


def sample_frames(paths: Iterable[Path], n_frames: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    collected = 0

    for path in tqdm(paths, desc="Sampling frames", unit="file"):
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.shape[0] == 0:
            continue

        remaining = n_frames - collected
        if remaining <= 0:
            break

        take = min(array.shape[0], remaining)
        if take == array.shape[0]:
            indices = np.arange(array.shape[0])
            rng.shuffle(indices)
        else:
            indices = rng.choice(array.shape[0], size=take, replace=False)

        chunks.append(np.asarray(array[indices], dtype=np.float32))
        collected += take

    if collected < n_frames:
        raise RuntimeError(
            f"Only collected {collected:,} frames, requested {n_frames:,}."
        )

    frames = np.concatenate(chunks, axis=0)
    if frames.shape[0] > n_frames:
        frames = frames[:n_frames]
    rng.shuffle(frames)
    return np.ascontiguousarray(frames, dtype=np.float32)


def train_kmeans(
    frames: np.ndarray,
    k: int,
    seed: int,
    epochs: int,
    batch_size: int,
) -> MiniBatchKMeans:
    batch_size = min(batch_size, frames.shape[0])
    rng = np.random.default_rng(seed + k)
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        init="random",
        batch_size=batch_size,
        random_state=seed,
        n_init=1,
        max_iter=1,
        reassignment_ratio=0.01,
        init_size=max(batch_size, 3 * k),
        verbose=0,
    )

    steps_per_epoch = int(np.ceil(frames.shape[0] / batch_size))
    with tqdm(
        total=epochs * steps_per_epoch,
        desc=f"KMeans k={k}",
        unit="batch",
        dynamic_ncols=True,
    ) as progress:
        for _ in range(epochs):
            order = rng.permutation(frames.shape[0])
            for start in range(0, frames.shape[0], batch_size):
                batch_indices = order[start : start + batch_size]
                kmeans.partial_fit(frames[batch_indices])
                progress.update(1)
    return kmeans


def save_kmeans(
    kmeans: MiniBatchKMeans,
    output_dir: Path,
    attribute: str,
    k: int,
    n_frames: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    base = output_dir / f"{attribute}_kmeans_{k}"

    torch.save(
        {
            "centroids": torch.from_numpy(centroids),
            "attribute": attribute,
            "n_clusters": k,
            "n_frames": n_frames,
            "seed": seed,
            "inertia": float(kmeans.inertia_),
        },
        base.with_suffix(".pt"),
    )
    np.save(base.with_suffix(".npy"), centroids)
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(
            {
                "attribute": attribute,
                "n_clusters": k,
                "n_frames": n_frames,
                "seed": seed,
                "inertia": float(kmeans.inertia_),
                "centroids_shape": list(centroids.shape),
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MiniBatchKMeans codebooks from saved Dicodec attribute npy files."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--attribute", default=ATTRIBUTE)
    parser.add_argument("--n-frames", type=int, default=N_FRAMES)
    parser.add_argument("--k", type=int, nargs="+", default=list(K_VALUES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"SHUFFLE = {SHUFFLE}")
    paths = find_attribute_files(args.input_dir, args.attribute)
    if not paths:
        raise RuntimeError(
            f"No {args.attribute}.npy files found under {args.input_dir}."
        )
    print(f"Found {len(paths):,} {args.attribute}.npy files.")

    frames = sample_frames(paths, args.n_frames, args.seed)
    print(f"Sampled frames: {frames.shape}, dtype={frames.dtype}")

    for k in args.k:
        print(f"Training kmeans k={k}...")
        kmeans = train_kmeans(
            frames=frames,
            k=k,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        save_kmeans(kmeans, args.output_dir, args.attribute, k, frames.shape[0], args.seed)
        print(
            f"Saved k={k} centroids under {args.output_dir} "
            f"(inertia={kmeans.inertia_:.4f})."
        )


if __name__ == "__main__":
    main()
