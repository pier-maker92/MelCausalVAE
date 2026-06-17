from typing import Optional, List
from dataclasses import dataclass, field, asdict
import json


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
    use_reparameterization_trick: bool = False
    use_instance_norm: bool = False


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
    pre_quantization: bool = False


@dataclass
class KLChunkRegularizer(RegularizationConfig):
    kl_weight_start: float = 1e-10
    kl_weight_end: float = 1e-4
    strategy: str = "linear"  # linear | sigmoid
    k: float = 1.0
    x0: float = 0.0


@dataclass
class NoiseConfig(RegularizationConfig):
    noise_start: float = 0.0
    noise_end: float = 1.0
    strategy: str = "linear"  # linear | sigmoid
    k: float = 1.0
    x0: float = 0.0
    noise_type: str = "additive"  # additive (mu + sigma*eps) | interpolate (mu*(1-t) + eps*t)
    sigma_type: str = "fixed"  # fixed | stochastic
    use_softplus: bool = False
    pre_quantization: bool = False


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
    fsq_levels: Optional[List[int]] = None


@dataclass
class SemanticDistillationConfig:
    wavlm_layer: int = 18
    cosine_loss_weight: float = 1.0
    ortho_loss_weight: float = 1.0


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
    noise_regularizer_config: Optional[NoiseConfig] = None
    semantic_distillation_config: Optional[SemanticDistillationConfig] = None


#########################
#        decoder        #
#########################


@dataclass
class DiTConfig:
    audio_latent_dim: int
    dit_dim: int
    dit_depth: int
    dit_heads: int
    dit_dropout_rate: float
    use_conv_layer: bool
    expansion_factor: int
    mel_dim: int
    uncond_prob: float
    is_causal: bool
    use_window_attention: bool
    window_attention_seconds: float
    kernel_size: int
    causal_convolution: bool
    upsample: str
    sigma: float = 1e-5
    use_group_bidirectional: bool = False
    speaker_cond_dim: Optional[int] = None


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

@dataclass
class WavLMConfig:
    pretrained_model_name: str = "microsoft/wavlm-large"
    layer: int = 6
    sampling_rate: int = 16000
    normalize: bool = True



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
    wavlm_config: Optional[WavLMConfig] = None

    def __post_init__(self):
        self.mel_spectrogram_config.n_mels = self.mel_dim
        self.mel_spectrogram_config.sampling_rate = self.sample_rate

        if self.wavlm_config is not None:
            self.encoder_config.mel_dim = 1024
        else:
            self.encoder_config.mel_dim = self.mel_dim

        self.encoder_config.latent_dim = self.latent_dim
        self.encoder_config.compress_factor_C = self.compress_factor

        self.decoder_config.mel_dim = self.mel_dim
        self.decoder_config.audio_latent_dim = self.latent_dim
        if self.wavlm_config is not None:
            self.decoder_config.expansion_factor = 2 * self.compress_factor
        else:
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

    def to_json_string(self):
        """Convert config to JSON string for Hugging Face Trainer compatibility"""
        return json.dumps(self.to_dict(), indent=2)
