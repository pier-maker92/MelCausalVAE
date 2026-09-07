"""Dataset boundary: each item is a [T, D] tensor or a mapping containing it."""

from importlib import import_module
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .configs import DataConfig


class LatentDataset(Dataset):
    def __init__(self, path: str):
        root = Path(path)
        self.files = [root] if root.is_file() else sorted(root.rglob("*.pt"))
        if not self.files:
            raise ValueError(f"No latent .pt files found in {root}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return torch.load(self.files[index], map_location="cpu", weights_only=True)


class LatentCollator:
    def __init__(self, latent_dim: int, max_frames: int, key: str):
        self.latent_dim = latent_dim
        self.max_frames = max_frames
        self.key = key

    def prepare(self, item) -> torch.Tensor:
        z = item[self.key] if isinstance(item, dict) else item
        if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Each dataset item must contain latents [T, {self.latent_dim}].")
        if z.shape[0] < 2:
            raise ValueError("Each training sequence must contain at least two frames.")
        if not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError("Latents must be finite floating point tensors.")
        return z[:self.max_frames].detach().to(device="cpu", dtype=torch.float32)

    def __call__(self, items) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [self.prepare(item) for item in items]
        z = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([len(item) for item in sequences])
        valid = torch.arange(z.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        return z, valid


def build_dataset(config: DataConfig, validation: bool = False) -> Dataset | None:
    if config.factory:
        kwargs = config.validation_kwargs if validation else config.train_kwargs
        if kwargs is None:
            return None
        module, name = config.factory.rsplit(":", 1)
        dataset = getattr(import_module(module), name)(**kwargs)
    else:
        path = config.validation_path if validation else config.train_path
        if path is None:
            return None
        dataset = LatentDataset(path)
    if len(dataset) == 0:
        raise ValueError("Dataset cannot be empty.")
    return dataset
