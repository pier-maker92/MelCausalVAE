import argparse
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_wav_mono_resampled(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    return (wav / wav.abs().max().clamp(min=1e-6)).squeeze(0)


def main(args):
    from dicodec.modules.builder import (
        load_external_semantic_quantizer,
        load_pretrained_model,
    )
    from inference import (
        infer_quantizer_type_from_path,
        resolve_semantic_quantizer_checkpoint,
    )

    device = choose_device()
    model = load_pretrained_model(args.checkpoint_dir)
    model.to(device).eval()

    semantic_quantizer_checkpoint = resolve_semantic_quantizer_checkpoint(args)
    if semantic_quantizer_checkpoint is None:
        raise ValueError("Pass a semantic quantizer checkpoint or steps/codebook flags.")

    quantizer_type = infer_quantizer_type_from_path(
        semantic_quantizer_checkpoint,
        args.semantic_quantizer_type,
    )
    load_external_semantic_quantizer(
        model,
        checkpoint_path=str(semantic_quantizer_checkpoint),
        quantizer_type=quantizer_type,
        codebook_size=args.semantic_codebook_size,
        target_source=args.semantic_quantizer_variant,
    )

    audios_srs = [
        (
            load_wav_mono_resampled(str(audio_path), model.config.sample_rate).to(
                device
            ),
            model.config.sample_rate,
        )
        for audio_path in args.audio_paths
    ]

    with torch.inference_mode():
        features, padding_mask, _, _ = model.extract_features(audios_srs)
        encoder_output = model.encode(features, padding_mask)
        quantizer_output = encoder_output.quantizer_output

    if quantizer_output is None:
        raise RuntimeError("Quantizer did not run; check that it was loaded correctly.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantizer_output, output_path)
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--audio_paths",
        nargs="+",
        required=True,
        help="One or more audio files to quantize as a single batch.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="quantizer_output.pt",
        help="Path where the batched QuantizeOutput is saved.",
    )
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
        help="Reference checkpoint under checkpoint_dir/quantized, e.g. 11k.",
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
    )
    parser.add_argument(
        "--semantic_quantizer_variant",
        type=str,
        choices=["z", "z_sem"],
        default=None,
        help="Quantizer target/source variant to resolve and run.",
    )
    args = parser.parse_args()
    main(args)
