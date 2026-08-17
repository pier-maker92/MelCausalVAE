import json
import math
import os
from pathlib import Path
from typing import Any, Tuple

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.audio_dataset import DataCollator, TrainDatasetWrapper
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


def _build_base_dataset(dataset_name: str):
    if dataset_name == "mls":
        from data.mls import MLSDataset

        return MLSDataset()
    if dataset_name == "libritts":
        from data.libri_tts import LibriTTS

        return LibriTTS()
    if dataset_name in ["librispeech_aligned", "librispeech-aligned"]:
        from data.librispeech_align import LibriSpeechAlignDataset

        return LibriSpeechAlignDataset()
    if dataset_name in ["libritts-r", "libritts_r"]:
        from data.libri_tts_r import LibriTTSR

        return LibriTTSR()
    raise ValueError(f"Dataset {dataset_name} not supported")


def _as_int_list(values: Any) -> list[int] | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple, ListConfig)):
        return [int(value) for value in values]
    raise TypeError("latent_indices must be a list of integer dimensions.")


def _resolve_latent_selection(cfg: DictConfig) -> dict[str, Any]:
    latent_indices = _as_int_list(cfg.get("latent_indices", None))
    if latent_indices is not None:
        if not latent_indices:
            raise ValueError("latent_indices cannot be empty.")
        if min(latent_indices) < 0:
            raise ValueError("latent_indices cannot contain negative dimensions.")
        return {
            "indices": latent_indices,
            "start": None,
            "end": None,
            "num_dims": len(latent_indices),
        }

    latent_start = int(cfg.get("latent_start", 0) or 0)
    if latent_start < 0:
        raise ValueError("latent_start cannot be negative.")

    latent_end = cfg.get("latent_end", None)
    if latent_end is None:
        feature_dims = cfg.get("feature_dims", None)
        if feature_dims is None:
            raise ValueError("Set feature_dims, latent_end, or latent_indices.")
        latent_end = latent_start + int(feature_dims)
    else:
        latent_end = int(latent_end)

    if latent_end <= latent_start:
        raise ValueError(
            f"Invalid latent slice [{latent_start}:{latent_end}]. "
            "latent_end must be greater than latent_start."
        )

    return {
        "indices": None,
        "start": latent_start,
        "end": latent_end,
        "num_dims": latent_end - latent_start,
    }


def _select_latent(features: torch.Tensor, selection: dict[str, Any]) -> torch.Tensor:
    indices = selection["indices"]
    if indices is not None:
        max_index = max(indices)
        if max_index >= features.shape[-1]:
            raise ValueError(
                f"latent_indices contains dimension {max_index}, "
                f"but encoder latent has only {features.shape[-1]} dimensions."
            )
        index_tensor = torch.as_tensor(indices, device=features.device, dtype=torch.long)
        return features.index_select(dim=-1, index=index_tensor)

    start = selection["start"]
    end = selection["end"]
    if end > features.shape[-1]:
        raise ValueError(
            f"Requested latent slice [{start}:{end}], "
            f"but encoder latent has only {features.shape[-1]} dimensions."
        )
    return features[:, start:end]


@torch.no_grad()
def _collect_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    latent_selection: dict[str, Any],
    max_frames: int | None,
    max_batches: int | None,
    dataset_name: str,
    progress_bar: tqdm,
) -> torch.Tensor:
    chunks = []
    collected_frames = 0
    progress_bar.set_description(f"Encoding {dataset_name}")

    for batch_index, batch in enumerate(dataloader):
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
        valid_features = encoder_output.mu[~encoder_output.padding_mask]
        features = _select_latent(valid_features, latent_selection)
        features = features.float().cpu()

        if max_frames is not None:
            remaining_frames = max_frames - collected_frames
            if features.shape[0] > remaining_frames:
                features = features[:remaining_frames]
        chunks.append(features)
        collected_frames += features.shape[0]
        progress_bar.update(1)
        progress_bar.set_postfix(phase="encode", frames=collected_frames)

        if max_frames is not None and collected_frames >= max_frames:
            break

    if not chunks:
        raise RuntimeError(f"No valid encoder frames were collected from {dataset_name}.")

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
    progress_bar: tqdm,
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
    num_chunks = math.ceil(points.shape[0] / chunk_size)
    progress_bar.total = progress_bar.n + max_iterations * num_chunks
    progress_bar.set_description("K-means")
    progress_bar.refresh()

    for iteration in range(1, max_iterations + 1):
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(num_clusters, dtype=torch.long)
        inertia = 0.0

        for point_chunk in points.split(chunk_size):
            min_distances, assignments = _assign(point_chunk, centroids)
            sums.index_add_(0, assignments, point_chunk)
            counts += torch.bincount(assignments, minlength=num_clusters)
            inertia += min_distances.sum().item()
            progress_bar.update(1)
            progress_bar.set_postfix(
                phase="kmeans",
                iteration=f"{iteration}/{max_iterations}",
                inertia=f"{inertia:.2f}",
            )

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
        progress_bar.set_postfix(
            phase="kmeans",
            iteration=f"{iteration}/{max_iterations}",
            inertia=f"{inertia:.2f}",
            shift=f"{centroid_shift:.2e}",
        )
        if centroid_shift <= tolerance:
            progress_bar.total = progress_bar.n
            progress_bar.refresh()
            return centroids, inertia, iteration

    return centroids, inertia, max_iterations


@hydra.main(version_base=None, config_path="configs", config_name="kmeans_encoder")
def main(cfg: DictConfig) -> None:
    checkpoint_dir = os.path.expandvars(cfg.checkpoint_dir)
    output_dir = Path(os.path.expandvars(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = OmegaConf.to_container(cfg.get("training", {}), resolve=True) or {}
    dataset_name = cfg.get("dataset_name", None) or training_cfg.get("dataset_name")
    if dataset_name is None:
        raise ValueError("Set dataset_name or training.dataset_name.")
    dataset_split = cfg.get("dataset_split", "test")
    latent_selection = _resolve_latent_selection(cfg)
    max_frames = cfg.get("max_frames", None)
    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError("max_frames must be positive, or null to use all frames.")
    shuffle_dataset = bool(cfg.get("shuffle_dataset", False))

    device = torch.device(cfg.device)
    dataset = TrainDatasetWrapper(
        _build_base_dataset(dataset_name),
        dataset_split,
        max_audio_len=cfg.max_audio_seconds,
        enable_perturbed_audio=bool(training_cfg.get("enable_perturbed_audio", False)),
        perturbed_pitch_shift_max_semitones=float(
            training_cfg.get("perturbed_pitch_shift_max_semitones", 8.0)
        ),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle_dataset,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=DataCollator(),
        generator=torch.Generator().manual_seed(int(cfg.seed)) if shuffle_dataset else None,
    )

    model = _load_frozen_model(checkpoint_dir, device)
    total_batches = len(dataloader) if cfg.max_batches is None else min(len(dataloader), cfg.max_batches)
    progress_bar = tqdm(
        total=total_batches,
        desc="K-means encoder",
        unit="step",
    )
    try:
        features = _collect_features(
            model=model,
            dataloader=dataloader,
            latent_selection=latent_selection,
            max_frames=max_frames,
            max_batches=cfg.max_batches,
            dataset_name=dataset_name,
            progress_bar=progress_bar,
        )
        centroids, inertia, iterations = _kmeans(
            points=features,
            num_clusters=cfg.num_clusters,
            max_iterations=cfg.max_iterations,
            tolerance=cfg.tolerance,
            seed=cfg.seed,
            chunk_size=cfg.kmeans_chunk_size,
            progress_bar=progress_bar,
        )
    finally:
        progress_bar.close()

    torch.save(
        {
            "centroids": centroids,
            "feature_dims": latent_selection["num_dims"],
            "latent_selection": latent_selection,
            "num_clusters": cfg.num_clusters,
            "checkpoint_dir": checkpoint_dir,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
        },
        output_dir / "encoder_kmeans.pt",
    )
    summary = {
        "checkpoint_dir": checkpoint_dir,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "num_frames": features.shape[0],
        "feature_dims": latent_selection["num_dims"],
        "latent_selection": latent_selection,
        "num_clusters": cfg.num_clusters,
        "iterations": iterations,
        "inertia": inertia,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)


if __name__ == "__main__":
    main()
