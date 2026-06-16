import os
import json
from typing import Dict, Any

from .VAE import VAE
from .configs import (
    VAEConfig,
    EncoderConfig,
    DiTConfig,
    MelSpectrogramConfig,
    VQConfig,
    DropoutConfig,
    KLChunkRegularizer,
    SemanticDistillationConfig,
    NoiseConfig,
)


def build_model(cfg_dict: Dict[str, Any]) -> VAE:
    """Builds a VAE model from a configuration dictionary."""
    # Handle both hydra config (encoder) and checkpoint config (encoder_config)
    encoder_cfg = cfg_dict.get("encoder_config", cfg_dict.get("encoder", {})).copy()
    decoder_cfg = cfg_dict.get("decoder_config", cfg_dict.get("decoder", {})).copy()
    mel_spec_cfg = cfg_dict.get("mel_spectrogram_config", {}).copy()
    
    decoder_cfg.setdefault("mel_dim", cfg_dict.get("mel_dim"))
    decoder_cfg.setdefault("audio_latent_dim", cfg_dict.get("latent_dim"))
    decoder_cfg.setdefault("expansion_factor", cfg_dict.get("compress_factor"))
    decoder_cfg.setdefault("upsample", cfg_dict.get("upsample", "conv"))

    decoder_config = DiTConfig(**decoder_cfg)

    vq_dict = encoder_cfg.pop("vq_config", None)
    vq_config = VQConfig(**vq_dict) if vq_dict else None

    dropout_dict = encoder_cfg.pop("dropout_regularizer_config", None)
    dropout_config = DropoutConfig(**dropout_dict) if dropout_dict else None

    kl_dict = encoder_cfg.pop("kl_chunk_regularizer_config", None)
    kl_config = KLChunkRegularizer(**kl_dict) if kl_dict else None

    noise_dict = encoder_cfg.pop("noise_regularizer_config", None)
    noise_config = NoiseConfig(**noise_dict) if noise_dict else None

    distill_dict = encoder_cfg.pop("semantic_distillation_config", None)
    distill_config = SemanticDistillationConfig(**distill_dict) if distill_dict else None

    encoder_config = EncoderConfig(
        vq_config=vq_config,
        dropout_regularizer_config=dropout_config,
        kl_chunk_regularizer_config=kl_config,
        noise_regularizer_config=noise_config,
        semantic_distillation_config=distill_config,
        **encoder_cfg,
    )

    mel_spec_cfg["use_bigvgan_mel"] = cfg_dict.get(
        "use_bigvgan_mel", mel_spec_cfg.get("use_bigvgan_mel", False)
    )
    mel_spec_config = MelSpectrogramConfig(**mel_spec_cfg)

    from .configs import WavLMConfig
    wavlm_dict = cfg_dict.get("wavlm_config", None)
    wavlm_config = WavLMConfig(**wavlm_dict) if wavlm_dict else None

    vae_config = VAEConfig(
        mel_dim=cfg_dict.get("mel_dim"),
        latent_dim=cfg_dict.get("latent_dim"),
        sample_rate=cfg_dict.get("sample_rate"),
        compress_factor=cfg_dict.get("compress_factor"),
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        mel_spectrogram_config=mel_spec_config,
        wavlm_config=wavlm_config,
    )

    return VAE(config=vae_config)
