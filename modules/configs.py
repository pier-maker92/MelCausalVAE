from typing import Optional, List, Union
from dataclasses import dataclass, field, asdict
import json
import warnings


class DeprecatedConfigMixin:
    @property
    def deprecated_attributes(self):
        """Returns a dict mapping deprecated attribute names to a tuple of (default_value, warning_message)."""
        return {}

    def check_deprecated(self):
        for attr, (default_val, msg) in self.deprecated_attributes.items():
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val != default_val:
                    warnings.warn(
                        f"Deprecated feature '{attr}' used: {msg}",
                        DeprecationWarning,
                        stacklevel=2,
                    )


#########################
#        encoder        #
#########################


@dataclass
class SigmaVAEEncoderConfig(DeprecatedConfigMixin):
    latent_dim: int = 64
    target_std: float = 1.0
    logvar_layer: bool = False
    kl_loss_weight: float = 1e-5
    kl_loss_warmup_steps: Optional[int] = None
    kl_loss_warmup_ratio: Optional[float] = None
    use_softplus: bool = False
    use_slt: bool = False
    use_reparameterization_trick: bool = False
    use_std_sweep: bool = False
    use_instance_norm: bool = False

    @property
    def deprecated_attributes(self):
        return {
            "use_slt": (False, "use_slt is no longer supported and will be ignored.")
        }

    def __post_init__(self):
        self.check_deprecated()


@dataclass
class RegularizationConfig:
    chunk_size: int = 2


@dataclass
class DropoutConfig(RegularizationConfig, DeprecatedConfigMixin):
    dropout_start: float = 0.0
    dropout_end: float = 0.8
    dropout_hierarchical: bool = False  # independent dropout for each chunk
    strategy: str = "linear"  # linear | sigmoid
    k: float = 1.0
    x0: float = 0.0
    pre_quantization: bool = False

    @property
    def deprecated_attributes(self):
        return {
            "pre_quantization": (
                False,
                "pre_quantization is deprecated. Dropout is now always applied before quantization.",
            )
        }

    def __post_init__(self):
        self.check_deprecated()


@dataclass
class KLChunkRegularizer(RegularizationConfig):
    kl_weight_start: float = 1e-10
    kl_weight_end: float = 1e-4
    strategy: str = "linear"  # linear | sigmoid
    k: float = 1.0
    x0: float = 0.0


@dataclass
class NoiseConfig(RegularizationConfig, DeprecatedConfigMixin):
    noise_start: float = 0.0
    noise_end: float = 1.0
    strategy: str = "linear"  # linear | sigmoid
    k: float = 1.0
    x0: float = 0.0
    noise_type: str = (
        "additive"  # additive (mu + sigma*eps) | interpolate (mu*(1-t) + eps*t)
    )
    sigma_type: str = "fixed"  # fixed | stochastic
    use_softplus: bool = False
    pre_quantization: bool = False

    @property
    def deprecated_attributes(self):
        return {
            "pre_quantization": (
                False,
                "pre_quantization is deprecated. Noise is now always applied before quantization.",
            )
        }

    def __post_init__(self):
        self.check_deprecated()


@dataclass
class VQConfig:
    num_embeddings: int
    drop_acoustic_p: float
    vq_type: str  # "bsq" or "fsq" or "vq" or "vq_ema"
    vq_dim: Optional[int] = None
    commitment_weight: float = 0.25
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    reset_dead_codes: bool = False
    reset_every_forward: int = 10
    # BSQ-only: per-bit entropy regularization to prevent codebook collapse.
    entropy_loss_weight: float = 0.0
    entropy_temperature: float = 1.0

    def __post_init__(self):
        if self.vq_type not in {"bsq", "fsq", "vq", "vq_ema"}:
            raise ValueError("vq_type must be one of: bsq, fsq, vq, vq_ema.")


@dataclass
class SemanticDistillationConfig:
    wavlm_layer: int = 18
    cosine_loss_weight: float = 1.0
    ortho_loss_weight: float = 1.0


@dataclass
class PitchLossConfig:
    fmin: float = 50.0
    fmax: float = 550.0
    torchcrepe_model: str = "tiny"
    torchcrepe_decoder: str = "weighted_argmax"
    torchcrepe_batch_size: int = 128
    torchcrepe_hop_length: Optional[int] = None
    periodicity_threshold: float = 0.5
    acoustic_loss_weight: float = 1.0
    semantic_adv_loss_weight: float = 0.1
    voiced_loss_weight: float = 0.1
    contour_loss_weight: float = 0.0
    acoustic_semantic_adv_loss_weight: float = 0.0
    grl_lambda: float = 1.0
    grl_warmup_steps: int = 0
    acoustic_semantic_grl_lambda: float = 1.0
    acoustic_semantic_grl_warmup_steps: int = 0

    def __post_init__(self):
        if self.fmin <= 0.0 or self.fmax <= self.fmin:
            raise ValueError("PitchLossConfig requires 0 < fmin < fmax.")
        if self.torchcrepe_model not in {"tiny", "full"}:
            raise ValueError("torchcrepe_model must be 'tiny' or 'full'.")
        if self.torchcrepe_decoder not in {"weighted_argmax", "argmax", "viterbi"}:
            raise ValueError(
                "torchcrepe_decoder must be 'weighted_argmax', 'argmax', or 'viterbi'."
            )
        if self.torchcrepe_batch_size <= 0:
            raise ValueError("torchcrepe_batch_size must be > 0.")
        if self.torchcrepe_hop_length is not None and self.torchcrepe_hop_length <= 0:
            raise ValueError("torchcrepe_hop_length must be > 0 when set.")
        if not 0.0 <= self.periodicity_threshold <= 1.0:
            raise ValueError("periodicity_threshold must be in [0, 1].")
        if self.grl_lambda < 0.0:
            raise ValueError("grl_lambda must be >= 0.")
        if self.grl_warmup_steps < 0:
            raise ValueError("grl_warmup_steps must be >= 0.")
        if self.acoustic_semantic_adv_loss_weight < 0.0:
            raise ValueError("acoustic_semantic_adv_loss_weight must be >= 0.")
        if self.acoustic_semantic_grl_lambda < 0.0:
            raise ValueError("acoustic_semantic_grl_lambda must be >= 0.")
        if self.acoustic_semantic_grl_warmup_steps < 0:
            raise ValueError("acoustic_semantic_grl_warmup_steps must be >= 0.")


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
    semantic_downsample_factor: int = 1
    semantic_dim: Optional[int] = None
    acoustic_dim: Optional[int] = None
    vq_acoustic_config: Optional[VQConfig] = None
    acoustic_logvar: bool = False
    # Optional Modules
    vq_config: Optional[VQConfig] = None
    dropout_regularizer_config: Optional[DropoutConfig] = None
    kl_chunk_regularizer_config: Optional[KLChunkRegularizer] = None
    noise_regularizer_config: Optional[NoiseConfig] = None
    semantic_distillation_config: Optional[SemanticDistillationConfig] = None
    pitch_loss_config: Optional[PitchLossConfig] = None


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
    local_speaker_conditioning: bool = True
    normalize_context_vector: bool = False


#########################
#   standard decoder    #
#########################


@dataclass
class StandardDecoderConfig:
    audio_latent_dim: int
    mel_dim: int
    compress_factor: int
    d_model: int = 512
    tf_heads: int = 8
    tf_layers: int = 4
    n_residual_blocks: int = 3
    drop_p: float = 0.1


@dataclass
class DiscriminatorConfig:
    recon_loss_weight: float = 45.0  # L1 reconstruction loss scale (HiFi-GAN style)
    adv_loss_weight: float = 1.0
    fm_loss_weight: float = 2.0
    discrim_lr: float = 2e-4


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


@dataclass
class SpeakerEncoderConfig:
    encoder_type: str = "ecapa"
    pretrained_model_name: str = "speechbrain/spkrec-ecapa-voxceleb"
    sampling_rate: int = 16000
    embedding_dim: int = 192
    wavlm_layers: Optional[Union[list[int], str]] = None
    wavlm_layer_weights: Optional[list[float]] = None
    wavlm_layer_combine: str = "weighted_sum"
    wavlm_pooling: str = "mean_std"
    wavlm_normalize_features: bool = True
    wavlm_freeze: bool = True
    wavlm_attention_channels: int = 128

    def __post_init__(self):
        if self.encoder_type not in {"ecapa", "wavlm"}:
            raise ValueError(
                "speaker_encoder_config.encoder_type must be ecapa or wavlm."
            )
        if (
            self.encoder_type == "wavlm"
            and self.pretrained_model_name == "speechbrain/spkrec-ecapa-voxceleb"
        ):
            self.pretrained_model_name = "microsoft/wavlm-large"
        all_layers = self.wavlm_layers == "*" or self.wavlm_layers == ["*"]
        if (
            self.wavlm_layers is not None
            and not all_layers
            and len(self.wavlm_layers) == 0
        ):
            raise ValueError("speaker_encoder_config.wavlm_layers cannot be empty.")
        if all_layers and self.wavlm_layer_weights is not None:
            raise ValueError(
                "speaker_encoder_config.wavlm_layer_weights cannot be used with "
                "wavlm_layers='*'."
            )
        if self.wavlm_layer_weights is not None and self.wavlm_layers is None:
            raise ValueError(
                "speaker_encoder_config.wavlm_layer_weights requires wavlm_layers."
            )


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
    speaker_encoder_config: Optional[SpeakerEncoderConfig] = None

    def __post_init__(self):
        self.mel_spectrogram_config.n_mels = self.mel_dim
        self.mel_spectrogram_config.sampling_rate = self.sample_rate

        if self.wavlm_config is not None:
            self.encoder_config.mel_dim = 1024
        else:
            self.encoder_config.mel_dim = self.mel_dim

        self.encoder_config.latent_dim = self.latent_dim
        self.encoder_config.compress_factor_C = self.compress_factor
        self._set_encoder_branch_dims()

        self.decoder_config.mel_dim = self.mel_dim
        self.decoder_config.audio_latent_dim = self.latent_dim
        self.decoder_config.expansion_factor = self.compress_factor

        if self.speaker_encoder_config is not None:
            speaker_cond_dim = self.speaker_encoder_config.embedding_dim
            configured_dim = self.decoder_config.speaker_cond_dim
            if configured_dim is not None and configured_dim != speaker_cond_dim:
                raise ValueError(
                    "decoder.speaker_cond_dim must match "
                    "speaker_encoder_config.embedding_dim when speaker conditioning "
                    "is enabled"
                )
            self.decoder_config.speaker_cond_dim = speaker_cond_dim
        elif getattr(self.encoder_config, "use_instance_norm", False):
            spk_dim = self.latent_dim * 2
            self.decoder_config.speaker_cond_dim = spk_dim

    @property
    def hidden_size(self) -> int:
        """Return hidden dimension for DeepSpeed compatibility"""
        return max(
            getattr(self.encoder_config, "d_model"),
            getattr(self.decoder_config, "dit_dim"),
        )

    def _set_encoder_branch_dims(self):
        semantic_dim = self.encoder_config.semantic_dim
        acoustic_dim = self.encoder_config.acoustic_dim
        if semantic_dim is None and acoustic_dim is None:
            self.encoder_config.semantic_dim = self.latent_dim // 2
            self.encoder_config.acoustic_dim = (
                self.latent_dim - self.encoder_config.semantic_dim
            )
        elif semantic_dim is None:
            self.encoder_config.semantic_dim = self.latent_dim - acoustic_dim
        elif acoustic_dim is None:
            self.encoder_config.acoustic_dim = self.latent_dim - semantic_dim

        semantic_dim = self.encoder_config.semantic_dim
        acoustic_dim = self.encoder_config.acoustic_dim
        if semantic_dim <= 0 or acoustic_dim <= 0:
            raise ValueError("semantic_dim and acoustic_dim must both be > 0.")

    def to_dict(self):
        """Convert config to dict for W&B logging compatibility"""
        d = asdict(self)
        d["model_type"] = "VAE"
        return d

    def to_json_string(self):
        """Convert config to JSON string for Hugging Face Trainer compatibility"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass(kw_only=True)
class VAEStandardConfig:
    """Config for VAEWithStandardDecoder (CNN decoder + GAN discriminator)."""

    mel_dim: int
    latent_dim: int
    sample_rate: int
    compress_factor: int
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_config: StandardDecoderConfig = field(
        default_factory=lambda: StandardDecoderConfig(
            audio_latent_dim=64, mel_dim=100, compress_factor=8
        )
    )
    discriminator_config: DiscriminatorConfig = field(
        default_factory=DiscriminatorConfig
    )
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
        self._set_encoder_branch_dims()

        self.decoder_config.mel_dim = self.mel_dim
        self.decoder_config.audio_latent_dim = self.latent_dim
        self.decoder_config.compress_factor = self.compress_factor

    @property
    def hidden_size(self) -> int:
        return max(self.encoder_config.d_model, self.decoder_config.d_model)

    def _set_encoder_branch_dims(self):
        semantic_dim = self.encoder_config.semantic_dim
        acoustic_dim = self.encoder_config.acoustic_dim
        if semantic_dim is None and acoustic_dim is None:
            self.encoder_config.semantic_dim = self.latent_dim // 2
            self.encoder_config.acoustic_dim = (
                self.latent_dim - self.encoder_config.semantic_dim
            )
        elif semantic_dim is None:
            self.encoder_config.semantic_dim = self.latent_dim - acoustic_dim
        elif acoustic_dim is None:
            self.encoder_config.acoustic_dim = self.latent_dim - semantic_dim

        semantic_dim = self.encoder_config.semantic_dim
        acoustic_dim = self.encoder_config.acoustic_dim
        if semantic_dim <= 0 or acoustic_dim <= 0:
            raise ValueError("semantic_dim and acoustic_dim must both be > 0.")
        if semantic_dim + acoustic_dim != self.latent_dim:
            raise ValueError(
                "semantic_dim + acoustic_dim must match latent_dim "
                f"({semantic_dim} + {acoustic_dim} != {self.latent_dim})."
            )

    def to_dict(self):
        d = asdict(self)
        d["model_type"] = "VAEStandard"
        return d

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)
