"""Serializable configuration for latent speech language modeling."""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class QuantizerConfig:
    dim: int = 64
    codebook_size: int = 1024
    decay: float = 0.99
    eps: float = 1e-5
    reset_dead_codes: bool = True
    reset_every_forward: int = 10


@dataclass
class TransformerConfig:
    dim: int = 256
    heads: int = 4
    layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    max_length: int = 2048


@dataclass
class DiffusionConfig:
    hidden_dim: int = 256
    layers: int = 3
    time_dim: int = 64
    sigma_min: float = 0.0
    sampling_steps: int = 20
    temperature: float = 1.0


@dataclass
class LossConfig:
    flow: float = 1.0
    reconstruction_l1: float = 1.0
    reconstruction_l2: float = 1.0
    commitment: float = 0.25


@dataclass
class ModelConfig:
    latent_dim: int = 64
    projection_hidden_dim: int = 256
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self):
        positive = [self.latent_dim, self.projection_hidden_dim, self.quantizer.dim,
                    self.quantizer.codebook_size, self.transformer.dim,
                    self.transformer.heads, self.transformer.layers,
                    self.transformer.ff_dim, self.transformer.max_length,
                    self.diffusion.hidden_dim, self.diffusion.layers,
                    self.diffusion.time_dim, self.diffusion.sampling_steps]
        if any(value <= 0 for value in positive):
            raise ValueError("Model dimensions, depths and sampling_steps must be positive.")
        if self.transformer.dim % self.transformer.heads:
            raise ValueError("transformer.dim must be divisible by heads.")
        if self.diffusion.time_dim % 2:
            raise ValueError("diffusion.time_dim must be even.")
        if not 0 <= self.transformer.dropout < 1:
            raise ValueError("transformer.dropout must be in [0, 1).")
        if not 0 <= self.diffusion.sigma_min < 1 or self.diffusion.temperature < 0:
            raise ValueError("Invalid diffusion sigma_min or temperature.")
        if any(getattr(self.loss, f.name) < 0 for f in fields(self.loss)):
            raise ValueError("Loss weights must be nonnegative.")


@dataclass
class DataConfig:
    # Default: one [T, D] tensor (or {'z': tensor}) per .pt file.
    train_path: str = "data/latents/train"
    validation_path: str | None = None
    key: str = "z"
    max_frames: int = 2048
    # Optional importable callable returning a map-style Dataset.
    factory: str | None = None
    train_kwargs: dict = field(default_factory=dict)
    validation_kwargs: dict | None = None


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/sm_quantizer"
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    log_every: int = 10
    max_steps: int | None = None
    resume: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        t = self.training
        if min(t.batch_size, t.epochs, t.log_every, self.data.max_frames) <= 0:
            raise ValueError("Batch size, epochs, log_every and max_frames must be positive.")
        if self.data.max_frames < 2 or self.data.max_frames > self.model.transformer.max_length:
            raise ValueError("data.max_frames must be in [2, transformer.max_length].")
        if t.num_workers < 0 or t.learning_rate <= 0 or t.weight_decay < 0 or t.grad_clip < 0:
            raise ValueError("Invalid optimizer or worker settings.")
        if t.max_steps is not None and t.max_steps <= 0:
            raise ValueError("max_steps must be positive or null.")


def from_dict(cls, values: dict):
    """Build nested dataclasses, rejecting misspelled YAML keys."""
    if not isinstance(values, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}.")
    defaults = cls()
    unknown = values.keys() - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        default = getattr(defaults, name)
        kwargs[name] = from_dict(type(default), value) if is_dataclass(default) else value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with Path(path).open() as handle:
        return from_dict(Config, yaml.safe_load(handle) or {})
