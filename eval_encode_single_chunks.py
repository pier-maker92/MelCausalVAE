#!/usr/bin/env python3
"""
Encode one audio file through the VAE, keep a range of latent channel chunks
(``latent_chunk_size`` from config, default 2), zero the rest, then decode.

* ``--num-chunks N`` — keep chunks ``0 .. N-1`` (cannot mix with start/end).
* ``--start-chunk S`` / ``--end-chunk E`` — keep chunks ``S .. E`` inclusive
  (defaults: ``S=0``, ``E=last`` when both omitted).

* ``--target-audio`` — enables **swap**: latent channels for chunks ``S..E`` (or
  ``0..N-1`` with ``--num-chunks``) are taken from the target encoding and
  written into the same positions; remaining channels stay from the input.
  Without ``--mean``, source and target are cropped to the **shorter** length
  (samples). ``--mean`` uses full lengths (input drives output time).

Ingresso WAV (e target in swap): **sempre** ricampionati a **24 kHz** mono prima
del mel; il checkpoint deve avere ``MelSpectrogramEncoder.sampling_rate == 24000``
(default di ``MelSpectrogramConfig``).

Con ``--mean`` e ``--target-audio``: media temporale dei chunk del target e
``repeat_interleave`` su tutti i frame del latent dell'input; **nessun** crop
al più corto tra le due tracce (comanda la lunghezza dell'input).

Per default il decode usa ``mu`` (latente deterministico), non il campione ``z``
da reparameterize: con ``logvar_layer: false`` il vecchio comportamento aggiunge
rumore casuale e può degradare molto la qualità. Usa ``--stochastic-latent``
per forzare ``z`` come in training.

Example:
  python eval_encode_single_chunks.py \\
    --config-path path/to/exp.yaml \\
    --checkpoint path/to/model.safetensors \\
    --input-audio path/to/utterance.wav \\
    --start-chunk 4 --end-chunk 12 \\
    --n-steps 16 --temperature 0.2 --guidance-scale 1.3
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

from modules.decoder.cfm import DiTConfig
from modules.Encoder import ConvformerEncoderConfig
from modules.feature_extractor import MelSpectrogramConfig
from modules.VAE import VAE, VAEConfig
from vocos import Vocos

# Tutti i file audio passati al mel devono essere a questo sample rate (Hz).
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
) -> Tuple[VAE, torch.nn.Module, str, VAEConfig]:
    encoder_cfg = ConvformerEncoderConfig(**config_dict["convformer"])  # type: ignore[arg-type]
    decoder_cfg = DiTConfig(**config_dict["cfm"])  # type: ignore[arg-type]
    decoder_cfg.expansion_factor = encoder_cfg.compress_factor_C
    mel_cfg = MelSpectrogramConfig(
        use_bigvgan_mel=config_dict["convformer"].get("use_bigvgan_mel", False)
    )
    use_classic_decoder = config_dict.get("use_classic_decoder", False)
    vae_cfg = VAEConfig(
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
        mel_spec_config=mel_cfg,
        use_classic_decoder=use_classic_decoder,
    )

    model = VAE(vae_cfg, dtype=torch.bfloat16).to(device)
    model.from_pretrained(checkpoint_path)
    model.set_device(device)
    model.set_dtype(torch.bfloat16)
    model.eval()

    if mel_cfg.use_bigvgan_mel:
        try:
            bigvgan_path = (
                "/home/ec2-user/MelCausalVAE/bigvgan/bigvgan_v2_24khz_100band_256x"
            )
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
    return model, vocoder, vocoder_type, vae_cfg


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


def load_wav_mono_resampled(
    path: Path, target_sr: int, device: torch.device
) -> torch.Tensor:
    """
    Carica mono e ricampiona a ``target_sr`` (qui sempre ``ENCODER_INPUT_SR_HZ`` = 24 kHz)
    prima di ``wav2mel`` / encoder.
    """
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != target_sr:
        logger.info("Resampling %s: %s Hz → %s Hz", path.name, sr, target_sr)
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
    return wav.to(device=device, dtype=torch.float32)


def mask_latent_chunk_range(
    z: torch.Tensor,
    *,
    start_chunk: int,
    end_chunk: int,
    chunk_size: int,
    latent_dim: int,
) -> torch.Tensor:
    """Keep latent channels for chunks ``start_chunk .. end_chunk`` (inclusive); zero the rest."""
    max_chunks = latent_dim // chunk_size
    if latent_dim % chunk_size != 0:
        logger.warning(
            "latent_dim=%s not divisible by chunk_size=%s; max_chunks=%s (floor)",
            latent_dim,
            chunk_size,
            max_chunks,
        )
    if start_chunk < 0 or end_chunk >= max_chunks or start_chunk > end_chunk:
        raise ValueError(
            f"Invalid chunk range [{start_chunk}, {end_chunk}] "
            f"(valid chunk indices: 0 .. {max_chunks - 1}, inclusive)"
        )
    lo = start_chunk * chunk_size
    hi = (end_chunk + 1) * chunk_size
    out = z.clone()
    if lo > 0:
        out[:, :, :lo] = 0
    if hi < latent_dim:
        out[:, :, hi:] = 0
    return out


def resolve_chunk_range(
    *,
    num_chunks: int | None,
    start_chunk: int | None,
    end_chunk: int | None,
    max_chunks: int,
) -> tuple[int, int]:
    """
    Return ``(start_chunk, end_chunk)`` inclusive.

    * If ``num_chunks`` is set: must not set ``start_chunk`` / ``end_chunk``; range is ``0 .. num_chunks-1``.
    * Otherwise: ``start_chunk`` defaults to ``0``, ``end_chunk`` to ``max_chunks - 1``.
    """
    used_range = start_chunk is not None or end_chunk is not None
    if num_chunks is not None and used_range:
        raise ValueError(
            "Use either --num-chunks or (--start-chunk / --end-chunk), not both."
        )
    if num_chunks is not None:
        if num_chunks < 1 or num_chunks > max_chunks:
            raise ValueError(
                f"--num-chunks must be in [1, {max_chunks}], got {num_chunks}"
            )
        return 0, num_chunks - 1

    s = 0 if start_chunk is None else start_chunk
    e = max_chunks - 1 if end_chunk is None else end_chunk
    if s < 0 or e >= max_chunks or s > e:
        raise ValueError(
            f"Invalid chunk range: start={s}, end={e} "
            f"(valid indices 0 .. {max_chunks - 1}, inclusive)"
        )
    return s, e


def swap_latent_chunk_range(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    *,
    start_chunk: int,
    end_chunk: int,
    chunk_size: int,
    latent_dim: int,
) -> torch.Tensor:
    """
    Start from ``z_src`` and replace channels
    ``[start_chunk * cs : (end_chunk + 1) * cs)`` with the same slice from ``z_tgt``.
    """
    max_chunks = latent_dim // chunk_size
    if start_chunk < 0 or end_chunk >= max_chunks or start_chunk > end_chunk:
        raise ValueError(
            f"Invalid chunk range [{start_chunk}, {end_chunk}] "
            f"(valid chunk indices: 0 .. {max_chunks - 1})"
        )
    lo = start_chunk * chunk_size
    hi = (end_chunk + 1) * chunk_size
    out = z_src.clone()
    out[:, :, lo:hi] = z_tgt[:, :, lo:hi]
    return out


def mean_interleave_chunk_swap(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    *,
    start_chunk: int,
    end_chunk: int,
    chunk_size: int,
    latent_dim: int,
) -> torch.Tensor:
    """
    Media del target lungo il tempo sui canali dei chunk ``[lo:hi)``;
    ``repeat_interleave`` su ``dim=1`` fino a ``T_src`` e sostituzione su ``z_src``.
    """
    max_chunks = latent_dim // chunk_size
    if start_chunk < 0 or end_chunk >= max_chunks or start_chunk > end_chunk:
        raise ValueError(
            f"Invalid chunk range [{start_chunk}, {end_chunk}] "
            f"(valid chunk indices: 0 .. {max_chunks - 1})"
        )
    lo = start_chunk * chunk_size
    hi = (end_chunk + 1) * chunk_size
    mu = z_tgt[:, :, lo:hi].mean(dim=1, keepdim=True)
    tiled = mu.repeat_interleave(z_src.shape[1], dim=1)
    out = z_src.clone()
    out[:, :, lo:hi] = tiled.to(out.dtype)
    return out


class _LatentPaddingView:
    """Solo ``padding_mask`` / ``durations`` per ``reconstructed_mel_padding_mask``."""

    __slots__ = ("padding_mask", "durations")

    def __init__(self, padding_mask, durations):
        self.padding_mask = padding_mask
        self.durations = durations


def _encode_audio_to_latent(
    model: VAE, audio_tensor: torch.Tensor, target_sr: int, *, use_sample_z: bool
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns ``latent``, ``padding_mask``, ``durations``.

    ``latent`` is ``conv_out.z`` (stochastic) if ``use_sample_z`` else ``mu``
    (deterministic; migliore per reconstruction in eval).
    """
    audios_srs = [(audio_tensor, target_sr)]
    encoded = model.wav2mel(audios_srs)
    mel = encoded.audio_features.to(model.dtype)
    pad = encoded.padding_mask
    conv_out = model.encoder(x=mel, padding_mask=pad, step=None)
    latent = conv_out.z if use_sample_z else conv_out.mu
    return latent, conv_out.padding_mask, conv_out.durations


def reconstructed_mel_padding_mask(
    model: VAE, conv_out, reconstructed_mel: torch.Tensor
) -> torch.Tensor:
    """Match ``VAE.encode_and_sample`` mask construction for trimming audio."""
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
def run(
    *,
    model: VAE,
    vocoder: torch.nn.Module,
    vocoder_type: str,
    audio_tensor: torch.Tensor,
    target_sr: int,
    start_chunk: int,
    end_chunk: int,
    chunk_size: int,
    n_steps: int,
    temperature: float,
    guidance_scale: float,
    device: torch.device,
    target_audio_tensor: Optional[torch.Tensor] = None,
    mean_fill: bool = False,
    stochastic_latent: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (full_mel_bf16 [1,T,F], trimmed_mel for vocoder [1,T_valid,F]).

    If ``target_audio_tensor`` is set, **swap** mode: same chunk range is copied
    from the target latent into the source latent (same time/channel positions);
    other channels stay from the source. Padding / durations for decoding come
    from the source after time alignment.

    If ``mean_fill`` (con swap): media del target sui chunk selezionati lungo ``T``,
    poi ``repeat_interleave`` su tutti i frame del latent di input; maschere/durate
    restano quelle dell'input (lunghezza comandata dall'input).

    Se ``stochastic_latent`` è False (default), usa ``mu`` invece di ``z`` dopo
    l'encoder (reconstruction più pulita).
    """
    latent_dim = model.config.encoder_config.latent_dim
    use_z = stochastic_latent

    if target_audio_tensor is not None:
        z_src, pad_src, dur_src = _encode_audio_to_latent(
            model, audio_tensor, target_sr, use_sample_z=use_z
        )
        z_tgt, _, _ = _encode_audio_to_latent(
            model, target_audio_tensor, target_sr, use_sample_z=use_z
        )

        if mean_fill:
            z = mean_interleave_chunk_swap(
                z_src,
                z_tgt,
                start_chunk=start_chunk,
                end_chunk=end_chunk,
                chunk_size=chunk_size,
                latent_dim=latent_dim,
            )
            conv_for_recon = _LatentPaddingView(pad_src, dur_src)
        else:
            t_src, t_tgt = z_src.shape[1], z_tgt.shape[1]
            if t_src != t_tgt:
                logger.warning(
                    "Latent time steps differ (source=%s, target=%s); using min length for swap.",
                    t_src,
                    t_tgt,
                )
            t_use = min(t_src, t_tgt)
            z_src = z_src[:, :t_use]
            z_tgt = z_tgt[:, :t_use]
            pad_src = pad_src[:, :t_use]
            if dur_src is not None:
                dur_src = dur_src[:, :t_use]

            z = swap_latent_chunk_range(
                z_src,
                z_tgt,
                start_chunk=start_chunk,
                end_chunk=end_chunk,
                chunk_size=chunk_size,
                latent_dim=latent_dim,
            )
            conv_for_recon = _LatentPaddingView(pad_src, dur_src)
    else:
        z_src, pad_src, dur_src = _encode_audio_to_latent(
            model, audio_tensor, target_sr, use_sample_z=use_z
        )
        z = mask_latent_chunk_range(
            z_src,
            start_chunk=start_chunk,
            end_chunk=end_chunk,
            chunk_size=chunk_size,
            latent_dim=latent_dim,
        )
        conv_for_recon = _LatentPaddingView(pad_src, dur_src)

    recon = model.sample(
        num_steps=n_steps,
        z=z,
        padding_mask=conv_for_recon.padding_mask,
        durations=conv_for_recon.durations,
        temperature=temperature,
        guidance_scale=guidance_scale,
    )
    # ``VAE.sample`` denormalizza già se ``mel_spec_config.normalize`` (non ripetere qui).

    rmask = reconstructed_mel_padding_mask(model, conv_for_recon, recon)
    T = min(recon.shape[1], rmask.shape[1])
    recon = recon[:, :T]
    rmask = rmask[:, :T]
    valid = recon[:, ~rmask[0]]
    # valid [1, T_valid, F]
    return recon, valid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode one WAV, keep a latent chunk range (or first K chunks), decode."
    )
    p.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="YAML with convformer / cfm (and optional use_classic_decoder), like eval.py",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model weights (.safetensors)",
    )
    p.add_argument(
        "--input-audio",
        type=Path,
        required=True,
        help="Path to input .wav (any path; not hardcoded)",
    )
    p.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Keep chunks 0 .. N-1 (cannot combine with --start-chunk / --end-chunk). "
        "Default with no range args: all chunks.",
    )
    p.add_argument(
        "--start-chunk",
        type=int,
        default=None,
        help="First chunk index to keep (inclusive). Default 0 unless --num-chunks is set.",
    )
    p.add_argument(
        "--end-chunk",
        type=int,
        default=None,
        help="Last chunk index to keep (inclusive). Default: last chunk unless --num-chunks is set.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Latent channels per chunk (default: convformer latent_chunk_size, else 2)",
    )
    p.add_argument("--n-steps", type=int, default=16, help="Decoder ODE / flow steps")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--guidance-scale", type=float, default=1.3)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output WAV (default: next to input audio)",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename stem (default: derived from input + chunk range)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cuda:0 / cpu (default: cuda if available else cpu)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--target-audio",
        type=Path,
        default=None,
        help="Second WAV: enables swap — latent chunks in the selected range are "
        "taken from the target and pasted into the same positions over the input "
        "latent; other channels stay from the input. Without --mean, input and target "
        "are cropped to the shorter waveform length (samples) before mel/encode.",
    )
    p.add_argument(
        "--mean",
        action="store_true",
        help="Requires --target-audio. Per i chunk selezionati: media del target "
        "lungo il tempo, poi repeat_interleave su tutti i frame del latent dell'input. "
        "La lunghezza temporale segue solo l'input (nessun crop source/target al minimo).",
    )
    p.add_argument(
        "--stochastic-latent",
        action="store_true",
        help="Usa il campione z (reparameterize) invece di mu. Default: mu, per "
        "reconstruction più stabile (importante con logvar_layer false / Sigma-VAE).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(args.seed)

    if not args.input_audio.is_file():
        raise FileNotFoundError(f"Input audio not found: {args.input_audio}")
    if args.target_audio is not None and not args.target_audio.is_file():
        raise FileNotFoundError(f"Target audio not found: {args.target_audio}")
    if args.mean and args.target_audio is None:
        raise ValueError("--mean richiede --target-audio (modalità swap).")

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    config_dict = load_config(args.config_path)
    model, vocoder, vocoder_type, vae_cfg = build_model_from_config(
        config_dict, args.checkpoint, device
    )

    mel_sr_model = int(model.wav2mel.sampling_rate)
    if mel_sr_model != ENCODER_INPUT_SR_HZ:
        raise RuntimeError(
            f"Questo script ricampiona sempre l'audio a {ENCODER_INPUT_SR_HZ} Hz prima dell'encoder, "
            f"ma il MelSpectrogramEncoder caricato usa sampling_rate={mel_sr_model} Hz. "
            "Allinea il config (MelSpectrogramConfig.sampling_rate = 24000) al checkpoint, "
            "oppure adatta lo script se usi un frontend mel diverso."
        )

    enc_cfg = model.config.encoder_config
    latent_dim = enc_cfg.latent_dim
    chunk_size = args.chunk_size
    if chunk_size is None:
        chunk_size = getattr(enc_cfg, "latent_chunk_size", 2)
    max_chunks = latent_dim // chunk_size
    if latent_dim % chunk_size != 0:
        raise ValueError(
            f"latent_dim {latent_dim} must be divisible by chunk_size {chunk_size}"
        )

    start_c, end_c = resolve_chunk_range(
        num_chunks=args.num_chunks,
        start_chunk=args.start_chunk,
        end_chunk=args.end_chunk,
        max_chunks=max_chunks,
    )

    encoder_sr = ENCODER_INPUT_SR_HZ
    wav = load_wav_mono_resampled(args.input_audio, encoder_sr, device)
    wav_tgt: Optional[torch.Tensor] = None
    if args.target_audio is not None:
        wav_tgt = load_wav_mono_resampled(args.target_audio, encoder_sr, device)
        if not args.mean:
            n_in = int(wav.shape[0])
            n_tgt = int(wav_tgt.shape[0])
            if n_in != n_tgt:
                n_crop = min(n_in, n_tgt)
                logger.info(
                    "Swap: cropping source and target to %s samples (shorter of %s vs %s).",
                    n_crop,
                    n_in,
                    n_tgt,
                )
                wav = wav[:n_crop]
                wav_tgt = wav_tgt[:n_crop]

    _, mel_valid = run(
        model=model,
        vocoder=vocoder,
        vocoder_type=vocoder_type,
        audio_tensor=wav,
        target_sr=encoder_sr,
        start_chunk=start_c,
        end_chunk=end_c,
        chunk_size=chunk_size,
        n_steps=args.n_steps,
        temperature=args.temperature,
        guidance_scale=args.guidance_scale,
        device=device,
        target_audio_tensor=wav_tgt,
        mean_fill=args.mean,
        stochastic_latent=args.stochastic_latent,
    )

    waveform = mel_to_audio(vocoder, mel_valid, device, vocoder_type)

    out_dir = args.output_dir or args.input_audio.parent
    stem = args.output_name or args.input_audio.stem
    mode_parts = []
    if args.target_audio is not None:
        mode_parts.append("swap")
    if args.mean:
        mode_parts.append("mean")
    if args.stochastic_latent:
        mode_parts.append("stochz")
    mode_tag = ("_" + "_".join(mode_parts)) if mode_parts else ""
    out_name = (
        f"{stem}{mode_tag}_chunks{start_c}-{end_c}_of{max_chunks}_"
        f"steps{args.n_steps}_t{args.temperature}_g{args.guidance_scale}.wav"
    )
    out_path = out_dir / out_name
    save_wav(out_path, waveform, sr=encoder_sr)
    logger.info(
        "Saved %s (swap=%s, mean_fill=%s, stochastic_latent=%s, chunks [%s, %s] / %s, chunk_size=%s, latent_dim=%s)",
        out_path,
        args.target_audio is not None,
        args.mean,
        args.stochastic_latent,
        start_c,
        end_c,
        max_chunks,
        chunk_size,
        latent_dim,
    )


if __name__ == "__main__":
    main()

# single audio

# python eval_encode_single_chunks.py \
#     --config-path configs/settings/bottleneck-ablations/dropout_eval.yaml \
#     --checkpoint checkpoints/ablations/dropout-5-epochs/model.safetensors \
#     --input-audio cond_male.wav \
#     --start-chunk 0 --end-chunk 31 \
#     --n-steps 16 --temperature 0.2 --guidance-scale 1.3

# swap with target audio
# python eval_encode_single_chunks.py \
#     --config-path configs/settings/bottleneck-ablations/dropout.yaml \
#     --checkpoint checkpoints/ablations/dropout-5-epochs/model.safetensors \
#     --input-audio cond_male.wav \
#     --target-audio female.wav \
#     --start-chunk 6 --end-chunk 31 \
#     --n-steps 16 --temperature 0.3 --guidance-scale 1.9
