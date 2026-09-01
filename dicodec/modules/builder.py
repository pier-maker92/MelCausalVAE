import os
import json
from pathlib import Path
from dataclasses import fields
from typing import Dict, Any

from .dicodec import Dicodec
from .configs import (
    DicodecConfig,
    EncoderConfig,
    DiTConfig,
    MelSpectrogramConfig,
    DropoutConfig,
    KLChunkRegularizer,
    NoiseConfig,
    SpeakerEncoderConfig,
    LowPassFilterConfig,
    FocalQuantizerConfig,
    VQConfig,
    SemanticQuantizerConfig,
)


def _filter_dataclass_kwargs(config_cls, values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {field.name for field in fields(config_cls)}
    return {key: value for key, value in values.items() if key in allowed}


def _load_default_model_config() -> Dict[str, Any]:
    from omegaconf import OmegaConf

    defaults_path = Path(__file__).resolve().parents[2] / "configs" / "defaults" / "model.yaml"
    return OmegaConf.to_container(OmegaConf.load(defaults_path), resolve=True)


def _build_vq_config(vq_dict: Dict[str, Any] | None) -> VQConfig | None:
    if not vq_dict:
        return None
    vq_dict = vq_dict.copy()
    focal_encoder_dict = vq_dict.pop("focal_encoder_config", None)
    focal_decoder_dict = vq_dict.pop("focal_decoder_config", None)
    focal_encoder_config = (
        FocalQuantizerConfig(
            **_filter_dataclass_kwargs(FocalQuantizerConfig, focal_encoder_dict)
        )
        if focal_encoder_dict
        else None
    )
    focal_decoder_config = (
        FocalQuantizerConfig(
            **_filter_dataclass_kwargs(FocalQuantizerConfig, focal_decoder_dict)
        )
        if focal_decoder_dict
        else None
    )
    return VQConfig(
        focal_encoder_config=focal_encoder_config,
        focal_decoder_config=focal_decoder_config,
        **_filter_dataclass_kwargs(VQConfig, vq_dict),
    )


def build_model(cfg_dict: Dict[str, Any]) -> Dicodec:
    """Builds a Dicodec model from a configuration dictionary."""
    # Handle both hydra config (encoder) and checkpoint config (encoder_config)
    encoder_cfg = cfg_dict.get("encoder_config", cfg_dict.get("encoder", {})).copy()
    decoder_cfg = cfg_dict.get("decoder_config", cfg_dict.get("decoder", {})).copy()
    mel_spec_cfg = cfg_dict.get("mel_spectrogram_config", {}).copy()

    encoder_cfg.setdefault("use_reparameterization_trick", False)
    encoder_cfg.setdefault("use_std_sweep", False)

    decoder_cfg.setdefault("mel_dim", cfg_dict.get("mel_dim"))
    decoder_cfg.setdefault("audio_latent_dim", cfg_dict.get("latent_dim"))
    decoder_cfg.setdefault("expansion_factor", cfg_dict.get("compress_factor"))
    decoder_cfg.setdefault("upsample", cfg_dict.get("upsample"))
    decoder_cfg.setdefault("local_speaker_conditioning", True)
    decoder_cfg.setdefault("normalize_context_vector", False)

    decoder_config = DiTConfig(**_filter_dataclass_kwargs(DiTConfig, decoder_cfg))

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

    encoder_config = EncoderConfig(
        dropout_regularizer_config=dropout_config,
        kl_chunk_regularizer_config=kl_config,
        noise_regularizer_config=noise_config,
        **_filter_dataclass_kwargs(EncoderConfig, encoder_cfg),
    )


    mel_spec_config = MelSpectrogramConfig(
        **_filter_dataclass_kwargs(MelSpectrogramConfig, mel_spec_cfg)
    )

    from .configs import WavLMConfig, WavLMModuleConfig

    wavlm_module_dict = cfg_dict.get("wavlm_module_config", None)
    
    if wavlm_module_dict:
        feature_extractor_dict = wavlm_module_dict.get("feature_extractor_config", None)
        feature_extractor_config = (
            WavLMConfig(**_filter_dataclass_kwargs(WavLMConfig, feature_extractor_dict))
            if feature_extractor_dict else None
        )
        speaker_encoder_dict = wavlm_module_dict.get("speaker_encoder_config", None)
        speaker_encoder_config = (
            SpeakerEncoderConfig(**_filter_dataclass_kwargs(SpeakerEncoderConfig, speaker_encoder_dict))
            if speaker_encoder_dict else None
        )
        wavlm_module_config = WavLMModuleConfig(
            pretrained_model_name=wavlm_module_dict.get("pretrained_model_name"),
            feature_extractor_config=feature_extractor_config,
            speaker_encoder_config=speaker_encoder_config
        )
    else:
        wavlm_module_config = None

    lowpass_filter_dict = cfg_dict.get(
        "lowpass_filter_config", cfg_dict.get("lowpass_filter", None)
    )
    if lowpass_filter_dict and "kernel_size" in lowpass_filter_dict:
        lowpass_filter_dict = lowpass_filter_dict.copy()
        lowpass_filter_dict.setdefault("order", lowpass_filter_dict["kernel_size"] - 1)
        lowpass_filter_dict.pop("kernel_size", None)
    lowpass_filter_config = (
        LowPassFilterConfig(
            **_filter_dataclass_kwargs(LowPassFilterConfig, lowpass_filter_dict)
        )
        if lowpass_filter_dict
        else LowPassFilterConfig()
    )
    semantic_quantizer_dict = cfg_dict.get("semantic_quantizer_config", None)
    if semantic_quantizer_dict:
        semantic_quantizer_dict = semantic_quantizer_dict.copy()
        semantic_quantizer_dict["vq_config"] = _build_vq_config(
            semantic_quantizer_dict.get("vq_config")
        )
    semantic_quantizer_config = (
        SemanticQuantizerConfig(
            **_filter_dataclass_kwargs(
                SemanticQuantizerConfig,
                semantic_quantizer_dict,
            )
        )
        if semantic_quantizer_dict
        else SemanticQuantizerConfig()
    )

    dicodec_config = DicodecConfig(
        mel_dim=cfg_dict.get("mel_dim"),
        latent_dim=cfg_dict.get("latent_dim"),
        sample_rate=cfg_dict.get("sample_rate"),
        compress_factor=cfg_dict.get("compress_factor"),
        mix_attributes_strategy=cfg_dict.get("mix_attributes_strategy", "add"),
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        mel_spectrogram_config=mel_spec_config,
        wavlm_module_config=wavlm_module_config,
        lowpass_filter_config=lowpass_filter_config,
        semantic_quantizer_config=semantic_quantizer_config,
    )

    training_cfg = cfg_dict.get("training", {}) or {}
    return Dicodec(
        config=dicodec_config,
        train_only_vq=training_cfg.get("train_only_vq", False),
        train_only_vq_and_decoder=training_cfg.get("train_only_vq_and_decoder", False),
    )


def load_pretrained_model(checkpoint_dir: str):

    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    default_model_cfg = _load_default_model_config()
    for key in ("mix_attributes_strategy", "semantic_quantizer_config"):
        cfg_dict.setdefault(key, default_model_cfg[key])

    model = build_model(cfg_dict)

    checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(checkpoint_path):
        model.from_pretrained(checkpoint_path)
    else:
        # try without model.safetensors in case checkpoint_dir itself is the file
        model.from_pretrained(checkpoint_dir)

    model.eval()

    return model
