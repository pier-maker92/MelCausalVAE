"""Serializable configuration for latent speech language modeling."""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

from .fsq_levels import FSQ_LEVELS


@dataclass
class QuantizerConfig:
    type: str = "vq_ema"
    dim: int = 64
    codebook_size: int = 1024
    decay: float = 0.99
    eps: float = 1e-5
    reset_dead_codes: bool = True
    reset_every_forward: int = 10
    entropy_temperature: float = 100.0

    def __post_init__(self):
        if self.type not in {"vq_ema", "bsq", "fsq"}:
            raise ValueError("quantizer.type must be vq_ema, bsq or fsq.")
        if self.entropy_temperature <= 0:
            raise ValueError("entropy_temperature must be positive.")
        if self.type == "bsq" and (self.codebook_size < 2 or self.codebook_size & (self.codebook_size - 1)):
            raise ValueError("BSQ codebook_size must be a power of two >= 2.")
        if self.type == "fsq" and self.codebook_size not in FSQ_LEVELS:
            raise ValueError(f"FSQ codebook_size must be one of {list(FSQ_LEVELS)}.")

    @property
    def resolved_dim(self) -> int:
        if self.type == "bsq":
            return self.codebook_size.bit_length() - 1
        if self.type == "fsq":
            return len(FSQ_LEVELS[self.codebook_size])
        return self.dim


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
class EncoderConfig:
    type: str = "mlp"
    hidden_dim: int | None = None  # null preserves projection_hidden_dim for old checkpoints
    layers: int = 1
    kernel_size: int = 3
    dropout: float = 0.0

    def __post_init__(self):
        if self.type not in {"mlp", "causal_conv"}:
            raise ValueError("encoder.type must be mlp or causal_conv.")
        if self.layers < 1 or self.kernel_size < 1 or (self.hidden_dim is not None and self.hidden_dim < 1):
            raise ValueError("Encoder dimensions and depths must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("encoder.dropout must be in [0, 1).")


@dataclass
class ASRConfig:
    enabled: bool = False
    curriculum: bool = False
    curriculum_start_pct: float = 50.0
    curriculum_end_pct: float = 0.0
    hidden_size: int = 512
    layers: int = 2
    dropout: float = 0.1
    embedding_dim: int = 256
    upsample_factor: int = 1
    tokenizer: str = "char"  # char or sentencepiece (BPE/unigram model)
    characters: str = "abcdefghijklmnopqrstuvwxyz' "
    tokenizer_model: str | None = None
    vocab_size: int = 1000  # SentencePiece vocabulary, excluding the added CTC blank

    @property
    def num_tokens(self) -> int:
        return len(self.characters) if self.tokenizer == "char" else self.vocab_size

    def __post_init__(self):
        if min(self.hidden_size, self.layers, self.embedding_dim, self.upsample_factor, self.vocab_size) < 1:
            raise ValueError("ASR dimensions and upsample_factor must be positive.")
        if not 0 <= self.curriculum_start_pct <= 100 or not 0 <= self.curriculum_end_pct <= 100:
            raise ValueError("ASR curriculum percentages must be in [0, 100].")
        if not 0 <= self.dropout < 1 or self.tokenizer not in {"char", "sentencepiece"}:
            raise ValueError("Invalid ASR dropout or tokenizer.")
        if not self.characters or len(set(self.characters)) != len(self.characters):
            raise ValueError("ASR characters must be nonempty and unique.")
        if self.enabled and self.tokenizer == "sentencepiece" and not self.tokenizer_model:
            raise ValueError("ASR sentencepiece requires tokenizer_model.")


@dataclass
class LossConfig:
    flow: float = 1.0
    reconstruction_l1: float = 1.0
    reconstruction_l2: float = 1.0
    commitment: float = 0.25
    bsq_regularization: float = 0.1
    asr: float = 1.0


@dataclass
class ModelConfig:
    latent_dim: int = 64
    projection_hidden_dim: int = 256
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    language_modeling: bool = True
    simple_reconstruction: bool = False
    asr: ASRConfig = field(default_factory=ASRConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self):
        if self.simple_reconstruction:
            self.language_modeling = False
            self.asr.enabled = False
            self.loss.flow = 0.0
            self.loss.asr = 0.0
            self.loss.commitment = 0.0
            self.loss.bsq_regularization = 0.0
        positive = [
            self.latent_dim,
            self.projection_hidden_dim,
            self.quantizer.resolved_dim,
            self.quantizer.codebook_size,
            self.transformer.dim,
            self.transformer.heads,
            self.transformer.layers,
            self.transformer.ff_dim,
            self.transformer.max_length,
            self.diffusion.hidden_dim,
            self.diffusion.layers,
            self.diffusion.time_dim,
            self.diffusion.sampling_steps,
        ]
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
    format: str = "pt"
    input: str = "z"
    target: str = "z"
    text_key: str = "transcript"
    train_partitions: list[str] = field(default_factory=list)
    validation_partitions: list[str] = field(default_factory=list)
    cache_dir: str | None = None
    max_frames: int = 2048
    # Optional importable callable returning a map-style Dataset.
    factory: str | None = None
    train_kwargs: dict = field(default_factory=dict)
    validation_kwargs: dict | None = None

    def __post_init__(self):
        if self.format not in {"pt", "parquet"}:
            raise ValueError("data.format must be pt or parquet.")
        if self.input not in {"z", "z_sem"} or self.target not in {"z", "z_sem"}:
            raise ValueError("data.input and data.target must be z or z_sem.")
        for partitions in (self.train_partitions, self.validation_partitions):
            if not isinstance(partitions, list) or any(not isinstance(p, str) or not p or p in {".", ".."} or Path(p).name != p for p in partitions):
                raise ValueError("Partitions must be a list of directory names.")
            if len(set(partitions)) != len(partitions):
                raise ValueError("Partitions must not contain duplicates.")


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
    save_every_steps: int | None = None
    max_save_limit: int = 3
    max_steps: int | None = None
    resume: str | None = None
    wandb_mode: str = "disabled"
    wandb_project: str = "dicodec-sm-quantizer"
    wandb_run_name: str | None = None
    wandb_id: str | None = None

    def __post_init__(self):
        if type(self.max_save_limit) is not int or self.max_save_limit < 1:
            raise ValueError("max_save_limit must be a positive integer.")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline or disabled.")
        if self.save_every_steps is not None and (type(self.save_every_steps) is not int or self.save_every_steps <= 0):
            raise ValueError("save_every_steps must be a positive integer or null.")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        t = self.training
        if min(t.batch_size, t.epochs, t.log_every, self.data.max_frames) <= 0:
            raise ValueError("Batch size, epochs, log_every and max_frames must be positive.")
        if self.model.language_modeling and (self.data.max_frames < 2 or self.data.max_frames > self.model.transformer.max_length):
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
