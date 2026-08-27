import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HF_HOME = Path("/Volumes/Crucial X6/HF_HOME")
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torchaudio
import torchaudio.transforms as T

from dicodec.modules.builder import load_pretrained_model
from dicodec.modules.configs import FocalQuantizerConfig, VQConfig
from dicodec.modules.quantizer.vq import VectorQuantizer


CHECKPOINT_DIR = Path(
    "/Users/software/Research/MelCausalVAE/checkpoints/paper/baseline/18-denc128-novq"
)
QUANTIZER_DIR = CHECKPOINT_DIR / "quantizers" / "z_sem_vq_ema_1024"
AUDIO_PATH = Path("/Users/software/Research/MelCausalVAE/demopage/audio/male.wav")
OUTPUT_PATH = Path("/Users/software/Research/MelCausalVAE/demopage/audio/male_zsem_vq_ema_1024.wav")
SEED = 1234


def get_device(device: str) -> torch.device:
    if device != "auto":
        if device == "mps":
            try:
                torch.zeros(1, device="mps")
                return torch.device("mps")
            except Exception:
                print("MPS unavailable in this runtime, falling back to CPU.")
                return torch.device("cpu")
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def load_wav_mono_resampled(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    wav = wav / (wav.abs().max() + 1e-8)
    return wav.squeeze(0)


def _maybe_focal_config(config_dict: dict | None) -> FocalQuantizerConfig | None:
    if not config_dict:
        return None
    return FocalQuantizerConfig(**config_dict)


def load_quantizer(quantizer_dir: Path, device: torch.device) -> VectorQuantizer:
    with open(quantizer_dir / "config.json", "r") as f:
        payload = json.load(f)

    vq_config_dict = dict(payload["vq_config"])
    vq_config_dict["focal_encoder_config"] = _maybe_focal_config(
        vq_config_dict.get("focal_encoder_config")
    )
    vq_config_dict["focal_decoder_config"] = _maybe_focal_config(
        vq_config_dict.get("focal_decoder_config")
    )
    feature_dim = int(payload["feature_dim"])
    vq_config = VQConfig(**vq_config_dict)
    quantizer = VectorQuantizer(config=vq_config, dim=feature_dim).to(device)

    checkpoint = torch.load(quantizer_dir / "model.pt", map_location="cpu")
    quantizer.load_state_dict(checkpoint["state_dict"], strict=True)
    quantizer.eval()
    return quantizer


def quantize_semantic(model, quantizer, z, padding_mask):
    attrs = model.encode_attributes(z, padding_mask=padding_mask)
    vq_out = quantizer(attrs.z_sem, padding_mask=padding_mask)
    z_reconstructed = vq_out.quantized + attrs.z_pros + attrs.z_mean
    return z_reconstructed


def residual_semantic(model, quantizer, z, padding_mask):
    attrs = model.encode_attributes(z, padding_mask=padding_mask)
    vq_out = quantizer(attrs.z_sem, padding_mask=padding_mask)
    z_reconstructed = (attrs.z_sem - vq_out.quantized) + attrs.z_pros + attrs.z_mean
    return z_reconstructed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate male.wav using a trained z_sem quantizer checkpoint."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--quantizer-dir", type=Path, default=QUANTIZER_DIR)
    parser.add_argument("--audio-path", type=Path, default=AUDIO_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--guidance-scale", type=float, default=1.3)
    parser.add_argument("--mode", choices=["quantized", "residual"], default="quantized")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = get_device(args.device)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from {args.checkpoint_dir} on {device}...")
    model = load_pretrained_model(str(args.checkpoint_dir)).to(device).eval()
    print(f"Loading quantizer from {args.quantizer_dir}...")
    quantizer = load_quantizer(args.quantizer_dir, device=device)

    wav = load_wav_mono_resampled(args.audio_path, model.config.sample_rate).to(device)
    audios_srs = [(wav, model.config.sample_rate)]

    enc_features, enc_padding_mask, _, _ = model.extract_features(
        audios_srs,
        target_audios_srs=audios_srs,
    )
    encoder_output = model.encode(enc_features, enc_padding_mask)
    speaker_embedding = model.extract_speaker_embedding(audios_srs)

    if args.mode == "quantized":
        z = quantize_semantic(
            model=model,
            quantizer=quantizer,
            z=encoder_output.z,
            padding_mask=encoder_output.padding_mask,
        )
    else:
        z = residual_semantic(
            model=model,
            quantizer=quantizer,
            z=encoder_output.z,
            padding_mask=encoder_output.padding_mask,
        )
    reconstructed_mel, _ = model.sample(
        num_steps=args.num_steps,
        temperature=args.temperature,
        guidance_scale=args.guidance_scale,
        z=z,
        padding_mask=encoder_output.padding_mask,
        speaker_embedding=speaker_embedding,
        generator=torch.Generator(device=device).manual_seed(SEED),
    )
    audio = model.vocoder.decode(reconstructed_mel.permute(0, 2, 1))
    audio = audio / (audio.abs().max(dim=-1, keepdim=True)[0] + 1e-8)
    torchaudio.save(str(args.output_path), audio.detach().cpu(), model.config.sample_rate)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
