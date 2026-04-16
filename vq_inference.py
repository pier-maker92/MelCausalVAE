#!/usr/bin/env python3
"""
Minimal VQ Mel-VAE inference API for external use.

Usage pattern::

    model = MelVAEInference.load("configs/inference/vq_inference_full.yaml", "checkpoint.safetensors")
    wav = model.load_wav_mono_resampled("in.wav")
    latent = model.encode(wav, model.sample_rate)
    audio = model.sample(latent)  # waveform [T] float32 CPU
    # Ablations: pass ``None`` for a slot to zero it (``...`` keeps the encoded value).
    audio_q = model.sample(latent, residual=None, tail=None)

``encode`` returns ``VQLatentParts`` with ``vq_ids``, ``residual``, ``tail``, and masks for the decoder.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torchaudio
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from modules.cfm import DiTConfig
from modules.Encoder import ConvformerEncoderConfig
from modules.melspecEncoder import MelSpectrogramConfig
from modules.VAE import VAE, VAEConfig
from vocos import Vocos

logger = logging.getLogger(__name__)


def default_inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class VQLatentParts:
    """Latent decomposition at one encoder time resolution (batch size 1)."""

    vq_ids: torch.LongTensor
    residual: torch.Tensor
    tail: torch.Tensor
    padding_mask: Optional[torch.BoolTensor]
    durations: Optional[torch.LongTensor]


class _LatentPaddingView:
    __slots__ = ("padding_mask", "durations")

    def __init__(self, padding_mask, durations):
        self.padding_mask = padding_mask
        self.durations = durations


def _reconstructed_mel_padding_mask(
    model: VAE, conv_out, reconstructed_mel: torch.Tensor
) -> torch.Tensor:
    B, Tm, _ = reconstructed_mel.shape
    device = reconstructed_mel.device
    ef = model._encoder_temporal_upsample_factor()
    mask = torch.zeros((B, Tm), device=device, dtype=torch.bool)
    if conv_out.durations is not None:
        expanded = conv_out.durations.long() * ef
        valid_lengths = expanded.sum(dim=1).long()
        for b in range(B):
            valid_len = min(int(valid_lengths[b].item()), Tm)
            mask[b, valid_len:] = True
    else:
        pm = conv_out.padding_mask.repeat_interleave(ef, dim=1)
        if pm.shape[1] < Tm:
            ext = torch.zeros(B, Tm - pm.shape[1], device=device, dtype=torch.bool)
            pm = torch.cat([pm, ext], dim=1)
        else:
            pm = pm[:, :Tm]
        mask = pm
    return mask


def _mel_to_audio(
    vocoder: torch.nn.Module,
    mel: torch.Tensor,
    device: torch.device,
    vocoder_type: str,
) -> torch.Tensor:
    if vocoder_type == "bigvgan":
        target_dtype = next(vocoder.parameters()).dtype
    else:
        target_dtype = next(vocoder.backbone.parameters()).dtype
    features = mel.permute(0, 2, 1).to(device=device, dtype=target_dtype)
    if vocoder_type == "bigvgan":
        waveform = vocoder(features)
    else:
        waveform = vocoder.decode(features)
    waveform = waveform.float().squeeze().detach().cpu()
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform.view(-1)


class MelVAEInference:
    def __init__(
        self,
        vae: VAE,
        vocoder: torch.nn.Module,
        vocoder_type: str,
        device: torch.device,
    ):
        self.vae = vae
        self._vocoder = vocoder
        self._vocoder_type = vocoder_type
        self.device = device
        self.sample_rate = int(vae.wav2mel.sampling_rate)
        self._qd = int(vae.config.encoder_config.vq_quant_dim)
        if not getattr(vae.config.encoder_config, "use_vq", False):
            raise ValueError("MelVAEInference expects convformer.use_vq=true.")

    def eval(self) -> None:
        """Set VAE and vocoder to eval mode (disables ``nn.Dropout`` and train-only behavior)."""
        self.vae.eval()
        self._vocoder.eval()

    @classmethod
    def load(
        cls,
        config_path: Union[str, Path],
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MelVAEInference":
        config_path = Path(config_path)
        checkpoint_path = Path(checkpoint_path)
        if device is None:
            device = default_inference_device()
        logger.info("Inference device: %s", device)

        if device.type in ("mps", "cpu") and dtype == torch.bfloat16:
            logger.info(
                "Using float32 on %s (avoid bfloat16 for this inference device).",
                device.type,
            )
            dtype = torch.float32

        with open(config_path, "r") as f:
            config_dict: Dict[str, Any] = yaml.safe_load(f)

        encoder_cfg = ConvformerEncoderConfig(**config_dict["convformer"])  # type: ignore[arg-type]
        decoder_cfg = DiTConfig(**config_dict["cfm"])  # type: ignore[arg-type]
        decoder_cfg.expansion_factor = encoder_cfg.compress_factor_C
        mel_cfg = MelSpectrogramConfig(
            use_bigvgan_mel=config_dict["convformer"].get("use_bigvgan_mel", False)
        )
        vae_cfg = VAEConfig(
            encoder_config=encoder_cfg,
            decoder_config=decoder_cfg,
            mel_spec_config=mel_cfg,
            use_classic_decoder=config_dict.get("use_classic_decoder", False),
        )

        model = VAE(vae_cfg, dtype=dtype).to(device)
        model.from_pretrained(str(checkpoint_path))
        model.set_device(device)
        model.set_dtype(dtype)
        model.eval()

        if mel_cfg.use_bigvgan_mel:
            try:
                bigvgan_path = "/home/ec2-user/MelCausalVAE/bigvgan/bigvgan_v2_24khz_100band_256x"
                if bigvgan_path not in sys.path:
                    sys.path.append(bigvgan_path)
                import bigvgan

                vocoder = bigvgan.BigVGAN.from_pretrained(
                    bigvgan_path, use_cuda_kernel=False
                )
                vocoder_type = "bigvgan"
            except Exception as e:
                logger.error(
                    "Failed to load BigVGAN vocoder: %s. Falling back to Vocos.", e
                )
                vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
                vocoder_type = "vocos"
        else:
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            vocoder_type = "vocos"

        vocoder.to(device)
        vocoder.eval()
        return cls(model, vocoder, vocoder_type, device)

    def load_wav_mono_resampled(self, path: Union[str, Path]) -> torch.Tensor:
        """Load a wav file as mono float32 on ``self.device`` at ``self.sample_rate``."""
        path = Path(path)
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        if sr != self.sample_rate:
            logger.info("Resampling %s: %s Hz -> %s Hz", path.name, sr, self.sample_rate)
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, self.sample_rate
            ).squeeze(0)
        return wav.to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def encode(self, audio_tensor: torch.Tensor, sample_rate: int) -> VQLatentParts:
        if int(sample_rate) != self.sample_rate:
            raise ValueError(
                f"sample_rate={sample_rate} but model expects {self.sample_rate} Hz."
            )
        self.eval()
        audios_srs = [(audio_tensor, sample_rate)]
        encoded = self.vae.wav2mel(audios_srs)
        mel = encoded.audio_features.to(self.vae.dtype)
        pad = encoded.padding_mask
        conv_out = self.vae.encoder(x=mel, padding_mask=pad, step=None)

        if conv_out.vq_latent_residual is None or conv_out.vq_indices is None:
            raise ValueError("Encoder did not return VQ fields (vq_indices / vq_latent_residual).")

        tail = conv_out.mu[..., self._qd :].to(self.vae.dtype)
        return VQLatentParts(
            vq_ids=conv_out.vq_indices,
            residual=conv_out.vq_latent_residual.to(self.vae.dtype),
            tail=tail,
            padding_mask=conv_out.padding_mask,
            durations=conv_out.durations,
        )

    def _vq_embed(self, vq_ids: torch.LongTensor) -> torch.Tensor:
        return self.vae.encoder.vq.codebook(vq_ids)

    def _build_z(
        self,
        latent: VQLatentParts,
        *,
        vq_ids,
        residual,
        tail,
    ) -> torch.Tensor:
        B, T, _ = latent.tail.shape
        device = self.device
        dtype = self.vae.dtype

        if vq_ids is ...:
            q_emb = self._vq_embed(latent.vq_ids)
        elif vq_ids is None:
            q_emb = torch.zeros((B, T, self._qd), device=device, dtype=dtype)
        else:
            q_emb = self._vq_embed(vq_ids)

        if residual is ...:
            res = latent.residual.to(dtype=dtype)
        elif residual is None:
            res = torch.zeros((B, T, self._qd), device=device, dtype=dtype)
        else:
            res = residual.to(device=device, dtype=dtype)

        if tail is ...:
            tail_t = latent.tail.to(dtype=dtype)
        elif tail is None:
            tail_t = torch.zeros_like(latent.tail, device=device, dtype=dtype)
        else:
            tail_t = tail.to(device=device, dtype=dtype)

        head = q_emb + res
        return torch.cat([head, tail_t], dim=-1).to(dtype)

    @torch.no_grad()
    def sample(
        self,
        latent: VQLatentParts,
        *,
        vq_ids=...,
        residual=...,
        tail=...,
        num_steps: int = 16,
        temperature: float = 0.2,
        guidance_scale: float = 1.3,
    ) -> torch.Tensor:
        """
        Decode mel from latent parts and vocode to a waveform.

        Use the Ellipsis default (``...``) for a slot to keep the value from ``latent``.
        Pass ``None`` to zero that component (ablation).
        """
        self.eval()
        z = self._build_z(latent, vq_ids=vq_ids, residual=residual, tail=tail)
        conv_for_recon = _LatentPaddingView(latent.padding_mask, latent.durations)
        recon = self.vae.sample(
            num_steps=num_steps,
            z=z,
            padding_mask=conv_for_recon.padding_mask,
            durations=conv_for_recon.durations,
            temperature=temperature,
            guidance_scale=guidance_scale,
        )
        rmask = _reconstructed_mel_padding_mask(self.vae, conv_for_recon, recon)
        T = min(recon.shape[1], rmask.shape[1])
        recon = recon[:, :T]
        rmask = rmask[:, :T]
        mel_valid = recon[:, ~rmask[0]]
        return _mel_to_audio(
            self._vocoder, mel_valid, self.device, self._vocoder_type
        )

    def decode(self, *args, **kwargs) -> torch.Tensor:
        """Alias of :meth:`sample` for naming preference."""
        return self.sample(*args, **kwargs)
