"""Same-frame SLM quantization and DiCodec waveform reconstruction."""

import argparse
from pathlib import Path

import torch

from .configs import ModelConfig, from_dict
from .model import SMQuantizer


ROOT = Path(__file__).resolve().parents[3]
DICODEC_CHECKPOINT = ROOT / "checkpoints/paper/v2/ls/25"
QUANTIZER_ROOT = ROOT / "checkpoints/paper/slm/25"
QUANTIZER_CHECKPOINT = QUANTIZER_ROOT / "fsq/last.pt"


def load_quantizer(path: Path, device: torch.device, expected_type: str | None = None) -> SMQuantizer:
    if not path.is_file():
        raise FileNotFoundError(f"Quantizer checkpoint not found: {path}. Check that its copy has completed.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    data = checkpoint["config"]["data"]
    if data.get("input") != "z_sem" or data.get("target") != "z_sem":
        raise ValueError("This inference requires a z_sem -> z_sem quantizer checkpoint.")
    config = from_dict(ModelConfig, checkpoint["config"]["model"])
    if expected_type is not None and config.quantizer.type != expected_type:
        raise ValueError(f"Requested {expected_type}, but checkpoint contains {config.quantizer.type}.")
    model = SMQuantizer(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def load_audio(path: Path, sample_rate: int, device: torch.device) -> torch.Tensor:
    import torchaudio

    waveform, rate = torchaudio.load(str(path))
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    waveform = waveform.mean(dim=0)
    if rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, rate, sample_rate)
    waveform = waveform / waveform.abs().max().clamp_min(1e-8)
    return waveform.to(device)


def select_latents(z: torch.Tensor, q_sem: torch.Tensor, attributes, mode: str) -> torch.Tensor:
    if mode == "quantized":
        return q_sem
    if mode == "residual":
        return z - q_sem
    if mode == "full":
        return q_sem + attributes.z_pros + attributes.z_mean
    raise ValueError(f"Unknown reconstruction mode: {mode}")


@torch.inference_mode()
def reconstruct(model, quantizer: SMQuantizer, waveform: torch.Tensor, target_waveform: torch.Tensor | None = None, mode: str = "full") -> torch.Tensor:
    if quantizer.reconstruction_head is None:
        raise ValueError("This checkpoint has no reconstruction head: audio reconstruction requires a reconstruction-trained model.")
    sample_rate = model.config.sample_rate
    audios = [(waveform, sample_rate)]
    features, padding, _, _ = model.extract_features(audios)
    encoded = model.encode(features, padding, compute_attributes=True)
    valid = torch.ones(encoded.z.shape[:2], dtype=torch.bool, device=encoded.z.device) if encoded.padding_mask is None else ~encoded.padding_mask
    # Reconstruct same-frame semantic latents;
    codes = quantizer.encode(encoded.attributes.z_sem, valid).codes
    q_sem = quantizer.reconstruction_head(codes).masked_fill(~valid.unsqueeze(-1), 0)
    latents = select_latents(encoded.z, q_sem, encoded.attributes, mode)
    latents = latents.masked_fill(~valid.unsqueeze(-1), 0)
    speaker_audio = waveform if target_waveform is None else target_waveform
    speaker = model.extract_speaker_embedding([(speaker_audio, sample_rate)])
    if speaker is None:
        raise RuntimeError("The selected DiCodec checkpoint must provide a speaker embedding.")
    mel, mel_padding = model.sample(z=latents, padding_mask=encoded.padding_mask, speaker_embedding=speaker, num_steps=12, temperature=0.2, guidance_scale=1.5)
    if mel_padding is not None:
        mel = mel[:, ~mel_padding[0]]  # CLI reconstructs one utterance at a time.
    audio = model.vocoder.decode(mel.permute(0, 2, 1))
    return audio / audio.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Reconstruct audio with the trained SLM semantic quantizer.")
    parser.add_argument("-i", type=Path, default=ROOT / "audio_assets/male.wav", help="Input audio")
    parser.add_argument("-ta", type=Path, default=None, help="Audio providing the decoder speaker embedding")
    parser.add_argument("--type", choices=("vq_ema", "fsq", "bsq"), default="fsq", help="Quantizer checkpoint type (default: fsq)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("-q", action="store_true", help="Decode only q_sem")
    modes.add_argument("-r", action="store_true", help="Decode only the full residual z - q_sem")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Keep --help independent of the full DiCodec inference dependencies.
    import torchaudio
    from ..builder import load_pretrained_model

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = QUANTIZER_ROOT / args.type / "last.pt"
    print(f"Loading SLM quantizer: {checkpoint}", flush=True)
    quantizer = load_quantizer(checkpoint, device, expected_type=args.type)
    print(f"Loading DiCodec: {DICODEC_CHECKPOINT} ({device})", flush=True)
    model = load_pretrained_model(str(DICODEC_CHECKPOINT)).to(device).eval()
    if quantizer.config.latent_dim != model.config.latent_dim:
        raise ValueError("DiCodec and quantizer latent dimensions do not match.")
    waveform = load_audio(args.i, model.config.sample_rate, device)
    target = None if args.ta is None else load_audio(args.ta, model.config.sample_rate, device)
    mode = "quantized" if args.q else "residual" if args.r else "full"
    audio = reconstruct(model, quantizer, waveform, target, mode)
    speaker_suffix = "" if args.ta is None else f"_speaker_{args.ta.stem}"
    output = args.i.with_name(f"{args.i.stem}_slm_{args.type}_{mode}{speaker_suffix}.wav")
    torchaudio.save(str(output), audio.cpu(), model.config.sample_rate)
    print(f"Saved {mode}: {output.resolve()}", flush=True)
