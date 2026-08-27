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


CHECKPOINT_DIR = Path(
    "/Users/software/Research/MelCausalVAE/checkpoints/paper/baseline/18-denc128-novq"
)
AUDIO_PATH = Path("/Users/software/Research/MelCausalVAE/demopage/audio/male.wav")
OUTPUT_DIR = Path("/Users/software/Research/MelCausalVAE/demopage/audio")
K_VALUES = ("128", "512", "1024")
SEED = 1234


def get_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_wav_mono_resampled(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    wav = wav / (wav.abs().max() + 1e-8)
    return wav.squeeze(0)


def load_manifest(checkpoint_dir: Path) -> dict:
    with open(checkpoint_dir / "kmeans_manifest.json", "r") as f:
        return json.load(f)


def quantize_semantic_residual(model, z, padding_mask, codebook, chunk_size=16384):
    attrs = model.encode_attributes(z, padding_mask=padding_mask)
    z_sem = attrs.z_sem
    centroids = codebook["centroids"].to(device=z.device, dtype=z.dtype)

    if padding_mask is not None:
        valid_mask = ~padding_mask
        x = z_sem[valid_mask]
    else:
        valid_mask = None
        x = z_sem.view(-1, z_sem.shape[-1])

    distances = []
    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start : start + chunk_size]
        distances.append(torch.cdist(chunk, centroids))
    indices = torch.argmin(torch.cat(distances, dim=0), dim=-1)
    z_sem_q = centroids[indices]

    z_sem_residual = torch.zeros_like(z_sem)
    if valid_mask is not None:
        z_sem_residual[valid_mask] = x - z_sem_q
    else:
        z_sem_residual = (x - z_sem_q).view(*z_sem.shape)

    return z_sem_residual + attrs.z_pros + attrs.z_mean


def encode_decode_with_residual_kmeans(
    model,
    audios_srs,
    codebook,
    num_steps,
    temperature,
    guidance_scale,
    generator,
):
    enc_features, enc_padding_mask, dec_features, dec_padding_mask = model.extract_features(
        audios_srs,
        target_audios_srs=audios_srs,
    )
    encoder_output = model.encode(enc_features, enc_padding_mask)
    speaker_embedding = model.extract_speaker_embedding(audios_srs)
    z = quantize_semantic_residual(
        model,
        encoder_output.z,
        encoder_output.padding_mask,
        codebook,
    )
    reconstructed_mel, _ = model.sample(
        num_steps=num_steps,
        temperature=temperature,
        guidance_scale=guidance_scale,
        z=z,
        generator=generator,
        padding_mask=encoder_output.padding_mask,
        speaker_embedding=speaker_embedding,
    )
    audio = model.vocoder.decode(reconstructed_mel.permute(0, 2, 1))
    return audio / (audio.abs().max(dim=-1, keepdim=True)[0] + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate male.wav reconstructions using checkpoint KMeans codebooks."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--audio-path", type=Path, default=AUDIO_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--k", nargs="+", default=list(K_VALUES))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--guidance-scale", type=float, default=1.3)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument(
        "--mode",
        choices=["quantized", "residual"],
        default="quantized",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint_dir} on {device}...")
    model = load_pretrained_model(str(args.checkpoint_dir))
    print(f"Moving model to {device}...")
    model = model.to(device).eval()
    print("Model ready.")
    manifest = load_manifest(args.checkpoint_dir)
    wav = load_wav_mono_resampled(args.audio_path, model.config.sample_rate).to(device)
    audios_srs = [(wav, model.config.sample_rate)]

    for k in args.k:
        if k not in manifest["codebooks"]:
            raise KeyError(f"KMeans codebook {k} not found in manifest.")
        codebook_path = args.checkpoint_dir / manifest["codebooks"][k]["path"]
        print(f"Generating with k={k}: {codebook_path}")
        codebook = torch.load(codebook_path, map_location="cpu")
        model.kmeans_codebook = codebook if args.mode == "quantized" else None

        generator = torch.Generator(device=device)
        generator.manual_seed(SEED + int(k))
        if args.mode == "residual":
            audio = encode_decode_with_residual_kmeans(
                model=model,
                audios_srs=audios_srs,
                codebook=codebook,
                num_steps=args.num_steps,
                temperature=args.temperature,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).detach().cpu()
        else:
            out = model.encode_decode(
                audios_srs=audios_srs,
                num_steps=args.num_steps,
                temperature=args.temperature,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            audio = out["audio_waveform"].detach().cpu()
        output_path = args.output_dir / f"male_kmeans_{k}{args.output_suffix}.wav"
        torchaudio.save(str(output_path), audio, model.config.sample_rate)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
