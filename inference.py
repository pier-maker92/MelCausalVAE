import os
import torch
import argparse
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from dicodec.modules.builder import load_pretrained_model


def load_wav_mono_resampled(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    # normalize audio
    wav = wav / wav.abs().max()
    return wav.squeeze(0)


def normalize_quantized_step(value: str) -> tuple[str, int | None]:
    value = str(value).strip().lower()
    value = value.removesuffix("step")
    if value.endswith("k"):
        step_count = int(float(value[:-1]) * 1000)
        return f"{int(step_count / 1000)}k", step_count
    step_count = int(value)
    if step_count % 1000 == 0:
        return f"{step_count // 1000}k", step_count
    return str(step_count), step_count


def infer_quantizer_type_from_path(path: Path, fallback: str) -> str:
    config_path = path / "config.json" if path.is_dir() else None
    if config_path is not None and config_path.exists():
        import json

        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("quantizer_type", fallback)

    name = path.name
    for quantizer_type in ("vq_ema", "std_vq", "bsq", "fsq"):
        if quantizer_type in name:
            return quantizer_type
    return fallback


def normalize_semantic_quantizer_variant(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower().replace("-", "_")
    aliases = {
        "z": "z",
        "z_sem": "z_sem",
        "zsem": "z_sem",
        "z_semantic": "z_sem",
        "semantic": "z_sem",
    }
    if value not in aliases:
        raise ValueError("--semantic_quantizer_variant must be either 'z' or 'z_sem'.")
    return aliases[value]


def config_codebook_size(config: dict) -> int | None:
    value = config.get(
        "codebook_size",
        config.get("num_embeddings", config.get("num_codebooks")),
    )
    return int(value) if value is not None else None


def configured_quantizer_dirs(
    quantized_dir: Path,
    codebook_size: int,
    variant: str | None,
) -> list[Path]:
    dirs = []
    variant_aliases = {
        "z": {"z"},
        "z_sem": {"z_sem", "zsem", "z_semantic", "semantic"},
    }
    for path in sorted(p for p in quantized_dir.iterdir() if p.is_dir()):
        if variant is not None and path.name not in variant_aliases[variant]:
            continue
        config_path = path / "config.json"
        if not config_path.exists():
            continue
        import json

        with open(config_path, "r") as f:
            config = json.load(f)
        if config_codebook_size(config) == codebook_size:
            dirs.append(path)
    return dirs


def resolve_semantic_quantizer_checkpoint(args) -> Path | None:
    if args.semantic_quantizer_checkpoint is not None:
        return Path(args.semantic_quantizer_checkpoint)

    if args.semantic_quantizer_steps is None and args.semantic_codebook_size is None:
        return None
    if args.semantic_quantizer_steps is None or args.semantic_codebook_size is None:
        raise ValueError(
            "Pass both --semantic_quantizer_steps and --semantic_codebook_size, "
            "or pass --semantic_quantizer_checkpoint explicitly."
        )

    step_label, step_count = normalize_quantized_step(args.semantic_quantizer_steps)
    quantized_dir = Path(args.checkpoint_dir) / "quantized" / f"{step_label}step"
    if not quantized_dir.is_dir():
        raise FileNotFoundError(f"Quantized checkpoint directory not found: {quantized_dir}")

    variant = normalize_semantic_quantizer_variant(args.semantic_quantizer_variant)
    configured_dirs = configured_quantizer_dirs(
        quantized_dir,
        args.semantic_codebook_size,
        variant,
    )
    if len(configured_dirs) == 1:
        return configured_dirs[0]
    if len(configured_dirs) > 1:
        formatted = "\n".join(str(path) for path in configured_dirs)
        raise RuntimeError(
            "Multiple matching semantic quantizer variants found. "
            "Pass --semantic_quantizer_variant z or --semantic_quantizer_variant z_sem.\n"
            f"{formatted}"
        )
    if variant is not None:
        raise FileNotFoundError(
            "No semantic quantizer checkpoint found for "
            f"variant={variant}, steps={args.semantic_quantizer_steps}, "
            f"codebook_size={args.semantic_codebook_size} in {quantized_dir}."
        )

    candidates = sorted(
        {
            path
            for pattern in (
                f"*cb{args.semantic_codebook_size}*",
                str(args.semantic_codebook_size),
            )
            for path in quantized_dir.glob(pattern)
            if path.is_dir() or path.suffix == ".pt"
        }
    )
    configured_dirs = [
        path for path in candidates if path.is_dir() and (path / "config.json").exists()
    ]
    if configured_dirs:
        candidates = configured_dirs
    if step_count is not None:
        exact = [
            path
            for path in candidates
            if f"step_{step_count}_" in path.name
            or (step_count == 1000 and "model_epoch_1_" in path.name)
            or path.is_dir()
        ]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            candidates = exact

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "No semantic quantizer checkpoint found for "
            f"steps={args.semantic_quantizer_steps}, "
            f"codebook_size={args.semantic_codebook_size} in {quantized_dir}."
        )
    formatted = "\n".join(str(path) for path in candidates)
    raise RuntimeError(f"Multiple matching semantic quantizer checkpoints:\n{formatted}")


def main(args):
    checkpoint_dir = args.checkpoint_dir
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise RuntimeError(
            "No CUDA or MPS device is available. CPU inference is not supported."
        )

    print(f"Loading model from {checkpoint_dir}...")
    model = load_pretrained_model(checkpoint_dir)
    model.to(device)
    semantic_quantizer_checkpoint = resolve_semantic_quantizer_checkpoint(args)
    if semantic_quantizer_checkpoint is not None:
        args.semantic_quantizer_type = infer_quantizer_type_from_path(
            semantic_quantizer_checkpoint,
            args.semantic_quantizer_type,
        )
        print(f"Loading semantic quantizer from {semantic_quantizer_checkpoint}...")
        model.load_external_semantic_quantizer(
            checkpoint_path=str(semantic_quantizer_checkpoint),
            quantizer_type=args.semantic_quantizer_type,
            codebook_size=args.semantic_codebook_size,
            target_source=args.semantic_quantizer_target_override,
        )
    assert not model.training, "Model must be in eval mode"
    assert not model.encoder.training, (
        "Encoder must be in eval mode: reparameterization trick and "
        "dropout regularizer are only disabled when training=False"
    )

    audio_path = args.audio_path
    print(f"Processing audio: {audio_path}")

    with torch.inference_mode():
        wav = load_wav_mono_resampled(audio_path, model.config.sample_rate).to(device)

        # Prepare inputs as expected by Dicodec encode_decode: list of (audio_tensor, sr)
        audios_srs = [(wav, model.config.sample_rate)]

        params = {
            "audios_srs": audios_srs,
            "num_steps": args.num_steps,
            "temperature": args.temperature,
            "guidance_scale": args.guidance_scale,
        }
        if args.target_audio is not None:
            target_wav = load_wav_mono_resampled(
                args.target_audio, model.config.sample_rate
            ).to(device)
            speaker_embedding = model.extract_speaker_embedding(
                [(target_wav, model.config.sample_rate)]
            )
            if speaker_embedding is None:
                raise RuntimeError(
                    "Speaker embedding swapping requires a checkpoint with a speaker "
                    "encoder."
                )
            params["speaker_embedding"] = speaker_embedding

        if getattr(args, "zero_speaker", False):
            params["zero_speaker"] = True

        if getattr(args, "guide_only_speaker", False):
            params["guide_only_speaker"] = True

        out = model.encode_decode(**params)
        audio = out["audio_waveform"]
        output_path = args.output_path
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(audio_path),
                f"reconstructed_{os.path.basename(audio_path)}",
            )
        torchaudio.save(output_path, audio.cpu(), model.config.sample_rate)
        print("Saved reconstructed audio to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--checkpoint_dir", type=str, default="checkpoints/vq-refactored"
    )
    parser.add_argument("-i", "--audio_path", type=str, default="audio_assets/male.wav")
    parser.add_argument(
        "--target_audio",
        type=str,
        default=None,
        help="Audio file whose speaker embedding conditions the reconstruction",
    )
    parser.add_argument("-o", "--output_path", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--guidance_scale", type=float, default=1.3)
    parser.add_argument(
        "--semantic_quantizer_checkpoint",
        type=str,
        default=None,
        help="External quantizer folder or legacy .pt checkpoint.",
    )
    parser.add_argument(
        "--semantic_quantizer_steps",
        type=str,
        default=None,
        help="Reference checkpoint under checkpoint_dir/quantized, e.g. 1k, 5k, 11k, 1000, 5000, 11000.",
    )
    parser.add_argument(
        "--semantic_quantizer_type",
        type=str,
        choices=["vq_ema", "bsq", "std_vq", "fsq"],
        default="std_vq",
    )
    parser.add_argument(
        "--semantic_codebook_size",
        type=int,
        default=None,
        help="Optional override; inferred from checkpoint when possible.",
    )
    parser.add_argument(
        "--semantic_quantizer_variant",
        type=str,
        choices=["z", "z_sem", "zsem", "z_semantic", "semantic"],
        default=None,
        help="Choose which quantized/<step>step subfolder to load: z or z_sem.",
    )
    parser.add_argument(
        "--semantic_quantizer_target_override",
        "--semantic_quantizer_target",
        dest="semantic_quantizer_target_override",
        type=str,
        choices=["z", "z_sem"],
        default=None,
        help="Override target_source from the quantizer config.",
    )
    parser.add_argument(
        "-qq", "--zero_speaker", action="store_true", help="Zero out speaker embedding"
    )
    parser.add_argument(
        "--guide_only_speaker",
        action="store_true",
        help="Apply guidance scale only to speaker embedding",
    )

    args = parser.parse_args()
    main(args)
