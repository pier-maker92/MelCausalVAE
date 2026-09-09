"""Aligned latent pairs from Parquet exports, PT files or custom datasets."""

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .configs import ASRConfig, DataConfig
from .asr import TextTokenizer
from .output_dataclasses import LatentBatch


class LatentDataset(Dataset):
    def __init__(self, path: str):
        root = Path(path)
        self.files = [root] if root.is_file() else sorted(
            p for p in root.rglob("*.pt") if not p.name.startswith(".")
        )
        if not self.files:
            raise ValueError(f"No latent .pt files found in {root}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return torch.load(self.files[index], map_location="cpu", weights_only=True)


class LatentCollator:
    def __init__(self, latent_dim: int, max_frames: int, key: str = "z",
                 input: str = "z", target: str = "z", asr_config: ASRConfig | None = None,
                 text_key: str = "transcript", language_modeling: bool = True):
        self.latent_dim = latent_dim
        self.max_frames = max_frames
        self.key = key
        self.input = input
        self.target = target
        self.tokenizer = TextTokenizer(asr_config) if asr_config is not None and asr_config.enabled else None
        self.text_key = text_key
        self.minimum_frames = 2 if language_modeling else 1

    def select(self, item, source: str):
        if isinstance(item, torch.Tensor):
            if self.input != self.target:
                raise ValueError("Cross-source training requires a mapping with both latent sources.")
            return item
        if not isinstance(item, Mapping):
            raise TypeError("Dataset items must be tensors or mappings.")
        if source == "z":
            return item[self.key]
        if "attributes" in item:
            return item["attributes"]["z_sem"]
        return item["z_sem"]

    def prepare(self, value) -> torch.Tensor:
        z = torch.as_tensor(value)
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Each source must contain latents [T, {self.latent_dim}].")
        if z.shape[0] < self.minimum_frames:
            raise ValueError(f"Each training sequence must contain at least {self.minimum_frames} frames.")
        if not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError("Latents must be finite floating point tensors.")
        return z.detach().to(device="cpu", dtype=torch.float32)

    def prepare_pair(self, item) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.prepare(self.select(item, self.input))
        targets = inputs if self.input == self.target else self.prepare(self.select(item, self.target))
        if len(inputs) != len(targets):
            raise ValueError("Input and target must have identical frame counts before truncation.")
        if self.tokenizer is not None and len(inputs) > self.max_frames:
            raise ValueError("ASR cannot truncate latents with a full transcript. Increase max_frames or use aligned chunks.")
        return inputs[:self.max_frames], targets[:self.max_frames]

    def __call__(self, items) -> LatentBatch:
        pairs = [self.prepare_pair(item) for item in items]
        inputs, targets = zip(*pairs)
        lengths = torch.tensor([len(item) for item in inputs])
        inputs = pad_sequence(inputs, batch_first=True)
        targets = pad_sequence(targets, batch_first=True)
        valid = torch.arange(inputs.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        text_targets = text_lengths = None
        if self.tokenizer is not None:
            if any(not isinstance(item, Mapping) or self.text_key not in item for item in items):
                raise ValueError(f"ASR requires the '{self.text_key}' field in every dataset item.")
            texts = [self.tokenizer.encode(item[self.text_key]) for item in items]
            text_targets = pad_sequence(texts, batch_first=True, padding_value=-1)
            text_lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)
        return LatentBatch(inputs, targets, valid, text_targets, text_lengths)


def parquet_files(path: str, partitions: list[str]) -> list[str]:
    root = Path(path)
    if root.is_file():
        if partitions or root.suffix != ".parquet" or root.name.startswith("."):
            raise ValueError("Expected a Parquet file without partition selectors.")
        return [str(root)]
    # Read explicit partitions or direct children; never recursively mix splits.
    directories = [root / name for name in partitions] if partitions else [root]
    files = []
    for directory in directories:
        shards = sorted(p for p in directory.glob("*.parquet") if p.is_file() and not p.name.startswith("."))
        if not shards:
            raise FileNotFoundError(f"No Parquet shards in {directory}; select partitions for a dataset root.")
        files.extend(str(p) for p in shards)
    return files


def load_parquet(config: DataConfig, path: str, partitions: list[str], asr_enabled: bool = False):
    from datasets import load_dataset

    columns = []
    if "z" in (config.input, config.target):
        columns.append(config.key)
    if "z_sem" in (config.input, config.target):
        columns.append("attributes")
    if asr_enabled:
        columns.append(config.text_key)
    return load_dataset(
        "parquet", data_files={"train": parquet_files(path, partitions)}, split="train",
        columns=columns, cache_dir=config.cache_dir, keep_in_memory=False,
    )


def build_dataset(config: DataConfig, validation: bool = False, asr_enabled: bool = False) -> Dataset | None:
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
        if config.format == "parquet":
            partitions = config.validation_partitions if validation else config.train_partitions
            dataset = load_parquet(config, path, partitions, asr_enabled)
        else:
            dataset = LatentDataset(path)
    if len(dataset) == 0:
        raise ValueError("Dataset cannot be empty.")
    return dataset
