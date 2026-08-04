import json
import os
from pathlib import Path
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.audio_dataset import DataCollator, TrainDatasetWrapper
from data.libri_tts_r import LibriTTSR
from modules.builder import build_model


def _load_frozen_model(checkpoint_dir: str, device: torch.device) -> torch.nn.Module:
    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")

    with config_path.open() as config_file:
        model_config = json.load(config_file)

    model = build_model(model_config)
    model.from_pretrained(checkpoint_dir)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def _collect_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    feature_dims: int,
    max_frames: int,
    max_batches: int | None,
) -> torch.Tensor:
    chunks = []
    collected_frames = 0
    total_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    progress_bar = tqdm(
        dataloader,
        total=total_batches,
        desc="Encoding LibriTTS-R",
        unit="batch",
    )

    for batch_index, batch in enumerate(progress_bar):
        if max_batches is not None and batch_index >= max_batches:
            break

        (
            encoder_features,
            encoder_padding_mask,
            _,
            _,
            _,
        ) = model.extract_features(batch["output_audios_srs"])
        encoder_output = model.encode(encoder_features, encoder_padding_mask)
        features = encoder_output.mu[~encoder_output.padding_mask, :feature_dims]
        features = features.float().cpu()

        remaining_frames = max_frames - collected_frames
        if features.shape[0] > remaining_frames:
            features = features[:remaining_frames]
        chunks.append(features)
        collected_frames += features.shape[0]
        progress_bar.set_postfix(frames=collected_frames)

        if collected_frames >= max_frames:
            break

    if not chunks:
        raise RuntimeError("No valid encoder frames were collected from LibriTTS-R.")

    return torch.cat(chunks, dim=0)


def _assign(points: torch.Tensor, centroids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(points, centroids).square()
    return distances.min(dim=1)


def _kmeans(
    points: torch.Tensor,
    num_clusters: int,
    max_iterations: int,
    tolerance: float,
    seed: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, float, int]:
    if points.shape[0] < num_clusters:
        raise ValueError(
            f"Need at least {num_clusters} frames, collected only {points.shape[0]}."
        )

    generator = torch.Generator(device=points.device).manual_seed(seed)
    initial_indices = torch.randperm(
        points.shape[0], generator=generator, device=points.device
    )[:num_clusters]
    centroids = points[initial_indices].clone()
    inertia = float("nan")

    progress_bar = tqdm(
        range(1, max_iterations + 1),
        desc="K-means",
        unit="iteration",
    )
    for iteration in progress_bar:
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(num_clusters, dtype=torch.long)
        inertia = 0.0

        for point_chunk in points.split(chunk_size):
            min_distances, assignments = _assign(point_chunk, centroids)
            sums.index_add_(0, assignments, point_chunk)
            counts += torch.bincount(assignments, minlength=num_clusters)
            inertia += min_distances.sum().item()

        nonempty = counts > 0
        updated_centroids = centroids.clone()
        updated_centroids[nonempty] = (
            sums[nonempty] / counts[nonempty].unsqueeze(1)
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
        progress_bar.set_postfix(inertia=f"{inertia:.2f}", shift=f"{centroid_shift:.2e}")
        if centroid_shift <= tolerance:
            return centroids, inertia, iteration

    return centroids, inertia, max_iterations


@hydra.main(version_base=None, config_path="configs", config_name="kmeans_encoder")
def main(cfg: DictConfig) -> None:
    checkpoint_dir = os.path.expandvars(cfg.checkpoint_dir)
    output_dir = Path(os.path.expandvars(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    dataset = TrainDatasetWrapper(
        LibriTTSR(),
        "test",
        max_audio_len=cfg.max_audio_seconds,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=DataCollator(),
    )

    model = _load_frozen_model(checkpoint_dir, device)
    features = _collect_features(
        model=model,
        dataloader=dataloader,
        feature_dims=cfg.feature_dims,
        max_frames=cfg.max_frames,
        max_batches=cfg.max_batches,
    )
    centroids, inertia, iterations = _kmeans(
        points=features,
        num_clusters=cfg.num_clusters,
        max_iterations=cfg.max_iterations,
        tolerance=cfg.tolerance,
        seed=cfg.seed,
        chunk_size=cfg.kmeans_chunk_size,
    )

    torch.save(
        {
            "centroids": centroids,
            "feature_dims": cfg.feature_dims,
            "num_clusters": cfg.num_clusters,
            "checkpoint_dir": checkpoint_dir,
        },
        output_dir / "encoder_kmeans.pt",
    )
    summary = {
        "checkpoint_dir": checkpoint_dir,
        "num_frames": features.shape[0],
        "feature_dims": cfg.feature_dims,
        "num_clusters": cfg.num_clusters,
        "iterations": iterations,
        "inertia": inertia,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)


if __name__ == "__main__":
    main()