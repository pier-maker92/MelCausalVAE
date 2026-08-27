from typing import Optional, List, Union
from dataclasses import dataclass, asdict, field
import json
import warnings




#########################
#        encoder        #
#########################


@dataclass(kw_only=True)
class SigmaVAEEncoderConfig:
    latent_dim: int
    target_std: float
    logvar_layer: bool
    kl_loss_weight: float
    kl_loss_warmup_steps: Optional[int] = None
    kl_loss_warmup_ratio: Optional[float] = None
    use_softplus: bool
    use_slt: bool
    use_reparameterization_trick: bool
    use_std_sweep: bool



@dataclass(kw_only=True)
class RegularizationConfig:
    chunk_size: int


@dataclass(kw_only=True)
class DropoutConfig(RegularizationConfig):
    dropout_start: float
    dropout_end: float
    dropout_hierarchical: bool  # independent dropout for each chunk
    strategy: str  # linear | sigmoid
    k: float
    x0: float
    pre_quantization: bool



@dataclass(kw_only=True)
class KLChunkRegularizer(RegularizationConfig):
    kl_weight_start: float
    kl_weight_end: float
    strategy: str  # linear | sigmoid
    k: float
    x0: float


@dataclass(kw_only=True)
class NoiseConfig(RegularizationConfig):
    noise_start: float
    noise_end: float
    strategy: str  # linear | sigmoid
    k: float
    x0: float
    noise_type: str  # additive (mu + sigma*eps) | interpolate (mu*(1-t) + eps*t)
    sigma_type: str  # fixed | stochastic
    use_softplus: bool
    pre_quantization: bool


@dataclass(kw_only=True)
class VQConfig:
    num_embeddings: int
    add_residual:bool
    add_residual_p:float
    drop_acoustic_p:float
    vq_type: str # "bsq" or "fsq" or "vq" or "vq_ema"
    vq_dim: Optional[int] = None
    commitment_weight: float
    ema_decay: float
    ema_eps: float
    reset_dead_codes: bool
    reset_every_forward: int
    # BSQ-only: per-bit entropy regularization to prevent codebook collapse.
    entropy_loss_weight: float
    entropy_temperature: float
    dim_to_quantize: Optional[Union[int, str]] = None
    recon_weight: Optional[float] = None

    def __post_init__(self):
        if self.vq_type not in {"bsq", "fsq", "vq", "vq_ema"}:
            raise ValueError("vq_type must be one of: bsq, fsq, vq, vq_ema.")
        if not 0.0 <= self.add_residual_p <= 1.0:
            raise ValueError("add_residual_p must be in [0, 1].")
        if self.entropy_loss_weight < 0.0:
            raise ValueError("entropy_loss_weight must be >= 0.")
        if self.entropy_temperature <= 0.0:
            raise ValueError("entropy_temperature must be > 0.")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1).")
        if self.ema_eps <= 0.0:
            raise ValueError("ema_eps must be > 0.")
        if self.reset_every_forward <= 0:
            raise ValueError("reset_every_forward must be > 0.")


@dataclass(kw_only=True)
class EncoderConfig(SigmaVAEEncoderConfig):
    mel_dim: int
    d_model: int
    compress_factor_C: int
    tf_heads: int
    tf_layers: int
    drop_p: float
    n_residual_blocks: int
    # Optional Modules
    dropout_regularizer_config: Optional[DropoutConfig] = None
    kl_chunk_regularizer_config: Optional[KLChunkRegularizer] = None
    noise_regularizer_config: Optional[NoiseConfig] = None


#########################
#        decoder        #
#########################


@dataclass(kw_only=True)
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
    sigma: float
    use_group_bidirectional: bool
    speaker_cond_dim: Optional[int] = None
    local_speaker_conditioning: bool
    normalize_context_vector: bool


#########################
#      spectrogram      #
#########################


@dataclass(kw_only=True)
class MelSpectrogramConfig:
    sampling_rate: int
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    padding: str = "center"
    normalize: bool = True


@dataclass(kw_only=True)
class WavLMConfig:
    layer: int
    sampling_rate: int
    normalize: bool


@dataclass(kw_only=True)
class SpeakerEncoderConfig:
    sampling_rate: int
    embedding_dim: int
    wavlm_layers: Optional[list[int]] = None
    wavlm_layer_weights: Optional[list[float]] = None
    wavlm_layer_combine: str
    wavlm_pooling: str
    wavlm_normalize_features: bool
    wavlm_freeze: bool
    wavlm_attention_channels: int

    def __post_init__(self):
        if self.wavlm_layers is None:
            raise ValueError("speaker_encoder_config.wavlm_layers must be explicit.")
        if self.wavlm_layers is not None and len(self.wavlm_layers) == 0:
            raise ValueError("speaker_encoder_config.wavlm_layers cannot be empty.")
        if self.wavlm_layer_weights is not None and self.wavlm_layers is None:
            raise ValueError(
                "speaker_encoder_config.wavlm_layer_weights requires wavlm_layers."
            )


@dataclass(kw_only=True)
class WavLMModuleConfig:
    pretrained_model_name: str
    feature_extractor_config: Optional[WavLMConfig] = None
    speaker_encoder_config: Optional[SpeakerEncoderConfig] = None

#########################
#          Dicodec          #
#########################


@dataclass(kw_only=True)
class LowPassFilterConfig:
    cutoff_hz: float = 1.42
    sample_rate: Optional[float] = None
    order: int = 56

    def __post_init__(self):
        if self.order <= 0:
            raise ValueError("lowpass_filter_config.order must be > 0.")
        if self.order % 2 != 0:
            raise ValueError(
                "lowpass_filter_config.order must be even so that "
                "kernel_size = order + 1 is odd."
            )
        if self.sample_rate is None:
            return
        if not 0.0 < self.cutoff_hz < self.sample_rate / 2:
            raise ValueError(
                "lowpass_filter_config.cutoff_hz must be between 0 and Nyquist "
                f"({self.sample_rate / 2:.2f} Hz)."
            )


@dataclass(kw_only=True)
class DicodecConfig:
    mel_dim: int
    latent_dim: int
    sample_rate: int
    compress_factor: int
    encoder_config: EncoderConfig
    decoder_config: DiTConfig
    mel_spectrogram_config: MelSpectrogramConfig
    wavlm_module_config: Optional[WavLMModuleConfig] = None
    lowpass_filter_config: LowPassFilterConfig = field(
        default_factory=LowPassFilterConfig
    )

    def __post_init__(self):
        self.mel_spectrogram_config.n_mels = self.mel_dim
        self.mel_spectrogram_config.sampling_rate = self.sample_rate
        if self.lowpass_filter_config.sample_rate is None:
            self.lowpass_filter_config.sample_rate = (
                self.sample_rate
                / self.mel_spectrogram_config.hop_length
                / self.compress_factor
            )
            self.lowpass_filter_config.__post_init__()

        if self.wavlm_module_config is not None and self.wavlm_module_config.feature_extractor_config is not None:
            self.encoder_config.mel_dim = 1024
        else:
            self.encoder_config.mel_dim = self.mel_dim

        self.encoder_config.latent_dim = self.latent_dim
        self.encoder_config.compress_factor_C = self.compress_factor

        self.decoder_config.mel_dim = self.mel_dim
        self.decoder_config.audio_latent_dim = self.latent_dim
        self.decoder_config.expansion_factor = self.compress_factor

        if self.wavlm_module_config is not None and self.wavlm_module_config.speaker_encoder_config is not None:
            speaker_cond_dim = self.wavlm_module_config.speaker_encoder_config.embedding_dim
            configured_dim = self.decoder_config.speaker_cond_dim
            if configured_dim is not None and configured_dim != speaker_cond_dim:
                raise ValueError(
                    "decoder.speaker_cond_dim must match "
                    "speaker_encoder_config.embedding_dim when speaker conditioning "
                    "is enabled"
                )
            self.decoder_config.speaker_cond_dim = speaker_cond_dim

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
        d["model_type"] = "Dicodec"
        return d

    def to_json_string(self):
        """Convert config to JSON string for Hugging Face Trainer compatibility"""
        return json.dumps(self.to_dict(), indent=2)
