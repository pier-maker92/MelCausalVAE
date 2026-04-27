from typing import Optional
from dataclasses import dataclass


@dataclass
class SigmaVAEencoderConfig:
    logvar_layer: bool = True
    kl_loss_weight: float = 1e-3
    target_std: Optional[float] = None
    kl_loss_warmup_steps: int = 1000
    latent_dim: int = 64


@dataclass
class RegularizationConfig:
    chunk_size: int = 2


@dataclass
class DropoutConfig(RegularizationConfig):
    dropout_start: float = 0.0
    dropout_end: float = 0.8
    dropout_hierarchical: bool = False  # independent dropout for each chunk


@dataclass
class KLChunkRegularizer(RegularizationConfig):
    kl_weight_start: float = 1e-10
    kl_weight_end: float = 1e-2


@dataclass
class VQConfig:
    num_embeddings: int = 128
    commitment_weight: float = 0.25
    reset_dead_codes: bool = True
    reset_max_per_step: Optional[int] = None
    reset_every_forward: int = 1
    use_ema_codebook: bool = False
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5
    dim_to_quantize: Optional[int] = None


@dataclass
class EncoderConfig(SigmaVAEencoderConfig):
    mel_dim: int = 100
    d_model: int = 512
    compress_factor_C: int = 8
    tf_heads: int = 8
    tf_layers: int = 4
    drop_p: float = 0.1
    n_residual_blocks: int = 3
    freeze_encoder_before_latent_heads: bool = False
    # Optional Modules
    vq_config: Optional[VQConfig] = None
    dropout_regularizer_config: Optional[DropoutConfig] = None
    kl_chunk_regularizer_config: Optional[KLChunkRegularizer] = None
