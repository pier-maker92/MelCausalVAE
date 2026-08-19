import os
import json
from dataclasses import fields
from typing import Dict, Any

from .VAE import VAE
from .configs import (
    VAEConfig,
    VAEStandardConfig,
    EncoderConfig,
    DiTConfig,
    StandardDecoderConfig,
    DiscriminatorConfig,
    MelSpectrogramConfig,
    VQConfig,
    DropoutConfig,
    KLChunkRegularizer,
    SemanticDistillationConfig,
    PitchLossConfig,
    NoiseConfig,
    SpeakerEncoderConfig,
)


def _filter_dataclass_kwargs(config_cls, values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {field.name for field in fields(config_cls)}
    return {key: value for key, value in values.items() if key in allowed}


def _load_vq_config(encoder_cfg: Dict[str, Any]):
    """Load VQ/BSQ config from encoder config without broad normalization hacks."""
    vq_dict = encoder_cfg.pop("vq_config", None)
    bsq_dict = encoder_cfg.pop("bsq_config", None)

    if vq_dict is not None:
        vq_dict = dict(vq_dict)
        if "vq_dim" not in vq_dict and "dim_to_quantize" in vq_dict:
            vq_dict["vq_dim"] = vq_dict.pop("dim_to_quantize")
        if vq_dict.pop("use_ema_codebook", False):
            vq_dict["vq_type"] = "vq_ema"
        vq_dict.setdefault("vq_type", "vq")
        vq_dict = _filter_dataclass_kwargs(VQConfig, vq_dict)
        return VQConfig(**vq_dict)

    if bsq_dict is not None:
        bsq_dict = dict(bsq_dict)
        num_embeddings = bsq_dict["codebook_size"]
        return VQConfig(
            num_embeddings=num_embeddings,
            add_residual=False,
            add_residual_p=0.0,
            vq_type="bsq",
        )

    return None


def _load_named_vq_config(encoder_cfg: Dict[str, Any], key: str):
    vq_dict = encoder_cfg.pop(key, None)
    if vq_dict is None:
        return None
    vq_dict = dict(vq_dict)
    if "vq_dim" not in vq_dict and "dim_to_quantize" in vq_dict:
        vq_dict["vq_dim"] = vq_dict.pop("dim_to_quantize")
    if vq_dict.pop("use_ema_codebook", False):
        vq_dict["vq_type"] = "vq_ema"
    vq_dict.setdefault("vq_type", "vq")
    return VQConfig(**_filter_dataclass_kwargs(VQConfig, vq_dict))


def build_model(cfg_dict: Dict[str, Any]) -> VAE:
    """Builds a VAE model from a configuration dictionary."""
    # Handle both hydra config (encoder) and checkpoint config (encoder_config)
    encoder_cfg = cfg_dict.get("encoder_config", cfg_dict.get("encoder", {})).copy()
    decoder_cfg = cfg_dict.get("decoder_config", cfg_dict.get("decoder", {})).copy()
    mel_spec_cfg = cfg_dict.get("mel_spectrogram_config", {}).copy()

    decoder_cfg.setdefault("mel_dim", cfg_dict.get("mel_dim"))
    decoder_cfg.setdefault("audio_latent_dim", cfg_dict.get("latent_dim"))
    decoder_cfg.setdefault("expansion_factor", cfg_dict.get("compress_factor"))
    decoder_cfg.setdefault("upsample", cfg_dict.get("upsample"))

    decoder_config = DiTConfig(**_filter_dataclass_kwargs(DiTConfig, decoder_cfg))

    vq_config = _load_vq_config(encoder_cfg)
    vq_acoustic_config = _load_named_vq_config(encoder_cfg, "vq_acoustic_config")

    dropout_dict = encoder_cfg.pop("dropout_regularizer_config", None)
    dropout_config = (
        DropoutConfig(**_filter_dataclass_kwargs(DropoutConfig, dropout_dict))
        if dropout_dict
        else None
    )

    kl_dict = encoder_cfg.pop("kl_chunk_regularizer_config", None)
    kl_config = (
        KLChunkRegularizer(**_filter_dataclass_kwargs(KLChunkRegularizer, kl_dict))
        if kl_dict
        else None
    )

    noise_dict = encoder_cfg.pop("noise_regularizer_config", None)
    noise_config = (
        NoiseConfig(**_filter_dataclass_kwargs(NoiseConfig, noise_dict))
        if noise_dict
        else None
    )

    distill_dict = encoder_cfg.pop("semantic_distillation_config", None)
    distill_config = (
        SemanticDistillationConfig(
            **_filter_dataclass_kwargs(SemanticDistillationConfig, distill_dict)
        )
        if distill_dict
        else None
    )

    pitch_dict = encoder_cfg.pop("pitch_loss_config", None)
    pitch_config = (
        PitchLossConfig(**_filter_dataclass_kwargs(PitchLossConfig, pitch_dict))
        if pitch_dict
        else None
    )

    encoder_config = EncoderConfig(
        vq_config=vq_config,
        vq_acoustic_config=vq_acoustic_config,
        dropout_regularizer_config=dropout_config,
        kl_chunk_regularizer_config=kl_config,
        noise_regularizer_config=noise_config,
        semantic_distillation_config=distill_config,
        pitch_loss_config=pitch_config,
        **_filter_dataclass_kwargs(EncoderConfig, encoder_cfg),
    )

    mel_spec_cfg["use_bigvgan_mel"] = cfg_dict.get(
        "use_bigvgan_mel", mel_spec_cfg.get("use_bigvgan_mel", False)
    )
    mel_spec_config = MelSpectrogramConfig(
        **_filter_dataclass_kwargs(MelSpectrogramConfig, mel_spec_cfg)
    )

    from .configs import WavLMConfig

    wavlm_dict = cfg_dict.get("wavlm_config", None)
    wavlm_config = (
        WavLMConfig(**_filter_dataclass_kwargs(WavLMConfig, wavlm_dict))
        if wavlm_dict
        else None
    )

    speaker_encoder_dict = cfg_dict.get("speaker_encoder_config", None)
    speaker_encoder_config = (
        SpeakerEncoderConfig(
            **_filter_dataclass_kwargs(SpeakerEncoderConfig, speaker_encoder_dict)
        )
        if speaker_encoder_dict
        else None
    )

    vae_config = VAEConfig(
        mel_dim=cfg_dict.get("mel_dim"),
        latent_dim=cfg_dict.get("latent_dim"),
        sample_rate=cfg_dict.get("sample_rate"),
        compress_factor=cfg_dict.get("compress_factor"),
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        mel_spectrogram_config=mel_spec_config,
        wavlm_config=wavlm_config,
        speaker_encoder_config=speaker_encoder_config,
    )

    training_cfg = cfg_dict.get("training", {}) or {}
    return VAE(
        config=vae_config,
        train_only_vq=training_cfg.get("train_only_vq", False),
        train_only_vq_and_decoder=training_cfg.get("train_only_vq_and_decoder", False),
    )


def build_standard_model(cfg_dict: Dict[str, Any]):
    """Builds a VAEWithStandardDecoder from a configuration dictionary."""
    from .VAE_standard import VAEWithStandardDecoder

    encoder_cfg = cfg_dict.get("encoder_config", cfg_dict.get("encoder", {})).copy()
    decoder_cfg = cfg_dict.get("decoder_config", cfg_dict.get("decoder", {})).copy()
    mel_spec_cfg = cfg_dict.get("mel_spectrogram_config", {}).copy()

    # Pop discriminator sub-keys from decoder config
    discrim_keys = (
        "recon_loss_weight",
        "adv_loss_weight",
        "fm_loss_weight",
        "discrim_lr",
    )
    discrim_kwargs = {k: decoder_cfg.pop(k) for k in discrim_keys if k in decoder_cfg}
    discrim_cfg = DiscriminatorConfig(**discrim_kwargs)

    # Remove keys not in StandardDecoderConfig
    for k in (
        "decoder_type",
        "dit_dim",
        "dit_depth",
        "dit_heads",
        "dit_dropout_rate",
        "use_conv_layer",
        "sigma",
        "uncond_prob",
        "is_causal",
        "use_window_attention",
        "window_attention_seconds",
        "use_group_bidirectional",
        "speaker_cond_dim",
        "local_speaker_conditioning",
        "kernel_size",
        "causal_convolution",
        "upsample",
        "expansion_factor",
        "mel_dim",
        "audio_latent_dim",
        "compress_factor",
    ):
        decoder_cfg.pop(k, None)

    decoder_config = StandardDecoderConfig(
        audio_latent_dim=cfg_dict.get("latent_dim"),
        mel_dim=cfg_dict.get("mel_dim"),
        compress_factor=cfg_dict.get("compress_factor"),
        **decoder_cfg,
    )

    vq_config = _load_vq_config(encoder_cfg)
    vq_acoustic_config = _load_named_vq_config(encoder_cfg, "vq_acoustic_config")

    dropout_dict = encoder_cfg.pop("dropout_regularizer_config", None)
    dropout_config = DropoutConfig(**dropout_dict) if dropout_dict else None

    kl_dict = encoder_cfg.pop("kl_chunk_regularizer_config", None)
    kl_config = KLChunkRegularizer(**kl_dict) if kl_dict else None

    noise_dict = encoder_cfg.pop("noise_regularizer_config", None)
    noise_config = NoiseConfig(**noise_dict) if noise_dict else None

    distill_dict = encoder_cfg.pop("semantic_distillation_config", None)
    distill_config = (
        SemanticDistillationConfig(**distill_dict) if distill_dict else None
    )

    pitch_dict = encoder_cfg.pop("pitch_loss_config", None)
    pitch_config = PitchLossConfig(**pitch_dict) if pitch_dict else None

    encoder_config = EncoderConfig(
        vq_config=vq_config,
        vq_acoustic_config=vq_acoustic_config,
        dropout_regularizer_config=dropout_config,
        kl_chunk_regularizer_config=kl_config,
        noise_regularizer_config=noise_config,
        semantic_distillation_config=distill_config,
        pitch_loss_config=pitch_config,
        **encoder_cfg,
    )

    mel_spec_cfg["use_bigvgan_mel"] = cfg_dict.get(
        "use_bigvgan_mel", mel_spec_cfg.get("use_bigvgan_mel", False)
    )
    mel_spec_config = MelSpectrogramConfig(**mel_spec_cfg)

    from .configs import WavLMConfig

    wavlm_dict = cfg_dict.get("wavlm_config", None)
    wavlm_config = WavLMConfig(**wavlm_dict) if wavlm_dict else None

    vae_config = VAEStandardConfig(
        mel_dim=cfg_dict.get("mel_dim"),
        latent_dim=cfg_dict.get("latent_dim"),
        sample_rate=cfg_dict.get("sample_rate"),
        compress_factor=cfg_dict.get("compress_factor"),
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        discriminator_config=discrim_cfg,
        mel_spectrogram_config=mel_spec_config,
        wavlm_config=wavlm_config,
    )

    return VAEWithStandardDecoder(config=vae_config)
