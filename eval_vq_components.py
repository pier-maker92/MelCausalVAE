#!/usr/bin/env python3
"""
Evaluate VQ latent components on one (or two) audios.

Decoding modes on the VQ head (first ``vq_quant_dim`` channels):
- both: quantized + residual
- quantized_only: quantized only
- residual_only: residual only
- tail_only: zero VQ head, keep only non-quantized tail

If ``--target-audio`` is provided, swap behavior is controlled by ``--swap-part``:
- quantized (default): swap only quantized embeddings on VQ head; ignore head residual
- tail: swap only non-quantized tail; keep source VQ head

This script always runs the model in eval mode so dropout is disabled.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

ENCODER_INPUT_SR_HZ = 24_000

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model_from_config(
    config_dict: Dict[str, Any], checkpoint_path: str, device: torch.device
) -> Tuple[VAE, torch.nn.Module, str]:
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

    model = VAE(vae_cfg, dtype=torch.bfloat16).to(device)
    model.from_pretrained(checkpoint_path)
    model.set_device(device)
    model.set_dtype(torch.bfloat16)
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
    return model, vocoder, vocoder_type


def load_wav_mono_resampled(
    path: Path, target_sr: int, device: torch.device
) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != target_sr:
        logger.info("Resampling %s: %s Hz -> %s Hz", path.name, sr, target_sr)
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0), sr, target_sr
        ).squeeze(0)
    return wav.to(device=device, dtype=torch.float32)


def mel_to_audio(
    vocoder: torch.nn.Module,
    mel: torch.Tensor,
    device: torch.device,
    vocoder_type: str = "vocos",
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


def save_wav(path: Path, audio: torch.Tensor, sr: int = ENCODER_INPUT_SR_HZ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        str(path), audio.unsqueeze(0).cpu().to(torch.float32), sample_rate=sr
    )


class _LatentPaddingView:
    __slots__ = ("padding_mask", "durations")

    def __init__(self, padding_mask, durations):
        self.padding_mask = padding_mask
        self.durations = durations


def reconstructed_mel_padding_mask(
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


@torch.no_grad()
def encode_vq_components(
    model: VAE, audio_tensor: torch.Tensor, target_sr: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor, Optional[torch.Tensor]]:
    """
    Returns:
      q_head: [B,T,qd] quantized embeddings
      r_head: [B,T,qd] residual used in VQ head
      tail: [B,T,latent_dim-qd] deterministic tail (from mu)
      padding_mask, durations
    """
    # Hard force eval to keep all dropout disabled.
    model.eval()

    audios_srs = [(audio_tensor, target_sr)]
    encoded = model.wav2mel(audios_srs)
    mel = encoded.audio_features.to(model.dtype)
    pad = encoded.padding_mask
    conv_out = model.encoder(x=mel, padding_mask=pad, step=None)

    qd = int(model.config.encoder_config.vq_quant_dim)
    if not getattr(model.config.encoder_config, "use_vq", False):
        raise ValueError("This script expects convformer.use_vq=true.")
    if conv_out.vq_latent_residual is None:
        raise ValueError("Encoder did not return vq_latent_residual.")

    # In eval, residual dropout is disabled. Encoder builds:
    # z_head = mu_q + residual
    # residual = conv_out.vq_latent_residual
    z_head = conv_out.z[..., :qd]
    r_head = conv_out.vq_latent_residual.to(z_head.dtype)
    q_head = z_head - r_head
    tail = conv_out.mu[..., qd:].to(z_head.dtype)
    return q_head, r_head, tail, conv_out.padding_mask, conv_out.durations


@torch.no_grad()
def run(
    *,
    model: VAE,
    vocoder: torch.nn.Module,
    vocoder_type: str,
    audio_tensor: torch.Tensor,
    target_sr: int,
    mode: str,
    n_steps: int,
    temperature: float,
    guidance_scale: float,
    target_audio_tensor: Optional[torch.Tensor] = None,
    swap_part: str = "quantized",
    mean_tail: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    q_src, r_src, tail_src, pad_src, dur_src = encode_vq_components(
        model, audio_tensor, target_sr
    )
    q = q_src
    tail = tail_src
    if target_audio_tensor is not None:
        q_tgt, _, tail_tgt, _, _ = encode_vq_components(model, target_audio_tensor, target_sr)
        t_use = min(q.shape[1], q_tgt.shape[1])
        if q.shape[1] != q_tgt.shape[1]:
            logger.warning(
                "Latent time mismatch source=%s target=%s; swapping quantized on min length=%s.",
                q.shape[1],
                q_tgt.shape[1],
                t_use,
            )
        if swap_part == "quantized":
            q = q.clone()
            q[:, :t_use, :] = q_tgt[:, :t_use, :]
        elif swap_part == "tail":
            tail = tail.clone()
            tail[:, :t_use, :] = tail_tgt[:, :t_use, :]
        else:
            raise ValueError(f"Unknown swap_part: {swap_part}")

    if mean_tail:
        tail_mean = tail.mean(dim=1, keepdim=True)
        tail = tail_mean.repeat(1, tail.shape[1], 1)

    # In swap mode, ignore VQ residual on the head entirely:
    # use only swapped quantized embeddings for the VQ head.
    if target_audio_tensor is not None and swap_part == "quantized":
        head = q
        tail = tail_src if mode == "both" else torch.zeros_like(tail_src)
    elif mode == "both":
        head = q + r_src
        tail = tail
    elif mode == "quantized_only":
        head = q
        tail = torch.zeros_like(tail_src)
    elif mode == "residual_only":
        head = r_src
        tail = torch.zeros_like(tail_src)
    elif mode == "tail_only":
        head = torch.zeros_like(q_src)
        tail = tail
    else:
        raise ValueError(f"Unknown mode: {mode}")

    z = torch.cat([head, tail], dim=-1).to(model.dtype)
    conv_for_recon = _LatentPaddingView(pad_src, dur_src)
    recon = model.sample(
        num_steps=n_steps,
        z=z,
        padding_mask=conv_for_recon.padding_mask,
        durations=conv_for_recon.durations,
        temperature=temperature,
        guidance_scale=guidance_scale,
    )
    rmask = reconstructed_mel_padding_mask(model, conv_for_recon, recon)
    T = min(recon.shape[1], rmask.shape[1])
    recon = recon[:, :T]
    rmask = rmask[:, :T]
    valid = recon[:, ~rmask[0]]
    return recon, valid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate VQ components decoding modes.")
    p.add_argument("--config-path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input-audio", type=Path, required=True)
    p.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "quantized_only", "residual_only", "tail_only"],
        help="Decode with quantized+residual, quantized only, or residual only.",
    )
    p.add_argument(
        "--target-audio",
        type=Path,
        default=None,
        help="If set: swap from target according to --swap-part.",
    )
    p.add_argument(
        "--swap-part",
        type=str,
        default="quantized",
        choices=["quantized", "tail"],
        help="Which latent part to swap from target: quantized head or non-quantized tail.",
    )
    p.add_argument(
        "--mean",
        action="store_true",
        help="Replace all tail frames with their temporal mean.",
    )
    p.add_argument("--n-steps", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--guidance-scale", type=float, default=1.3)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--output-name", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(args.seed)

    if not args.input_audio.is_file():
        raise FileNotFoundError(f"Input audio not found: {args.input_audio}")
    if args.target_audio is not None and not args.target_audio.is_file():
        raise FileNotFoundError(f"Target audio not found: {args.target_audio}")

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    config_dict = load_config(args.config_path)
    model, vocoder, vocoder_type = build_model_from_config(
        config_dict, args.checkpoint, device
    )

    mel_sr_model = int(model.wav2mel.sampling_rate)
    if mel_sr_model != ENCODER_INPUT_SR_HZ:
        raise RuntimeError(
            f"Script resamples audio to {ENCODER_INPUT_SR_HZ} Hz, but model uses "
            f"MelSpectrogramEncoder.sampling_rate={mel_sr_model}."
        )

    wav = load_wav_mono_resampled(args.input_audio, ENCODER_INPUT_SR_HZ, device)
    wav_tgt = None
    if args.target_audio is not None:
        wav_tgt = load_wav_mono_resampled(args.target_audio, ENCODER_INPUT_SR_HZ, device)

    _, mel_valid = run(
        model=model,
        vocoder=vocoder,
        vocoder_type=vocoder_type,
        audio_tensor=wav,
        target_sr=ENCODER_INPUT_SR_HZ,
        mode=args.mode,
        n_steps=args.n_steps,
        temperature=args.temperature,
        guidance_scale=args.guidance_scale,
        target_audio_tensor=wav_tgt,
        swap_part=args.swap_part,
        mean_tail=args.mean,
    )

    waveform = mel_to_audio(vocoder, mel_valid, device, vocoder_type)

    out_dir = args.output_dir or args.input_audio.parent
    stem = args.output_name or args.input_audio.stem
    swap_tag = (
        f"_swap{args.swap_part.capitalize()}" if args.target_audio is not None else ""
    )
    mean_tag = "_meanTail" if args.mean else ""
    out_name = (
        f"{stem}{swap_tag}{mean_tag}_{args.mode}_"
        f"steps{args.n_steps}_t{args.temperature}_g{args.guidance_scale}.wav"
    )
    out_path = out_dir / out_name
    save_wav(out_path, waveform, sr=ENCODER_INPUT_SR_HZ)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
