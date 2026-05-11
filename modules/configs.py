from typing import Optional
from dataclasses import dataclass, field, asdict


#########################
#        encoder        #
#########################


@dataclass
class SigmaVAEEncoderConfig:
    latent_dim: int = 64
    target_std: float = 1.0
    logvar_layer: bool = False
    kl_loss_weight: float = 1e-5
    kl_loss_warmup_steps: Optional[int] = None
    kl_loss_warmup_ratio: Optional[float] = None
    use_softplus: bool = False
    use_slt: bool = False


@dataclass
class RegularizationConfig:
    chunk_size: int = 2


@dataclass
class DropoutConfig(RegularizationConfig):
    dropout_start: float = 0.0
    dropout_end: float = 0.8
    dropout_hierarchical: bool = False  # independent dropout for each chunk
    strategy: str = "linear"  # linear | sigmoid
    k: float = 1.0
    x0: float = 0.0


@dataclass
class KLChunkRegularizer(RegularizationConfig):
    kl_weight_start: float = 1e-10
    kl_weight_end: float = 1e-4
    strategy: str = "linear"  # linear | sigmoid
    k: float = 1.0
    x0: float = 0.0


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
    residual_and_tail_dropout_p: float = 0.1
    add_vq_residual_to_stoch: bool = False


@dataclass
class EncoderConfig(SigmaVAEEncoderConfig):
    mel_dim: int = 100
    d_model: int = 512
    compress_factor_C: int = 4
    tf_heads: int = 8
    tf_layers: int = 4
    drop_p: float = 0.1
    n_residual_blocks: int = 3
    freeze_encoder_before_latent_heads: bool = False
    # Optional Modules
    vq_config: Optional[VQConfig] = None
    dropout_regularizer_config: Optional[DropoutConfig] = None
    kl_chunk_regularizer_config: Optional[KLChunkRegularizer] = None


#########################
#        decoder        #
#########################


@dataclass
class DiTConfig:
    audio_latent_dim: int = 64
    dit_dim: int = 768
    dit_depth: int = 6
    dit_heads: int = 8
    dit_dropout_rate: float = 0.1
    use_conv_layer: bool = False
    sigma: float = 1e-5
    expansion_factor: int = 4  # ~25Hz
    mel_dim: int = 100
    uncond_prob: float = 0.0
    is_causal: bool = True
    use_window_attention: bool = True
    window_attention_seconds: float = 3.0
    use_group_bidirectional: bool = False


#########################
#      spectrogram      #
#########################


@dataclass
class MelSpectrogramConfig:
    mel_channels: int = 100
    sampling_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    padding: str = "center"
    normalize: bool = True
    use_bigvgan_mel: bool = False


#########################
#          VAE          #
#########################


@dataclass(kw_only=True)
class VAEConfig:
    mel_dim: int
    latent_dim: int
    sample_rate: int
    compress_factor: int
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_config: DiTConfig = field(default_factory=DiTConfig)
    mel_spectrogram_config: MelSpectrogramConfig = field(
        default_factory=MelSpectrogramConfig
    )

    def __post_init__(self):
        self.mel_spectrogram_config.n_mels = self.mel_dim
        self.mel_spectrogram_config.sampling_rate = self.sample_rate

        self.encoder_config.mel_dim = self.mel_dim
        self.encoder_config.latent_dim = self.latent_dim
        self.encoder_config.compress_factor_C = self.compress_factor

        self.decoder_config.mel_dim = self.mel_dim
        self.decoder_config.audio_latent_dim = self.latent_dim
        self.decoder_config.expansion_factor = self.compress_factor

    @property
    def hidden_size(self) -> int:
        """Return hidden dimension for DeepSpeed compatibility"""
        return max(
            getattr(self.encoder_config, "d_model"),
            getattr(self.decoder_config, "dit_dim"),
        )

    def to_dict(self):
        """Convert config to dict for W&B logging compatibility"""
        d = asdict(self)
        d["model_type"] = "VAE"
        return d
