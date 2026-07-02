import os

# --- WORKAROUND FOR FAIRSEQ/HYDRA ON PYTHON 3.11 ---
import dataclasses

_orig_get_field = dataclasses._get_field


def _patched_get_field(cls, a_name, a_type, *args, **kwargs):
    try:
        return _orig_get_field(cls, a_name, a_type, *args, **kwargs)
    except ValueError as e:
        if "mutable default" in str(e):
            default = getattr(cls, a_name, dataclasses.MISSING)
            actual_default = (
                default.default if isinstance(default, dataclasses.Field) else default
            )
            if actual_default is not dataclasses.MISSING:
                default_cls = actual_default.__class__
                orig_hash = getattr(default_cls, "__hash__", None)
                try:
                    default_cls.__hash__ = lambda self: id(self)
                except TypeError:
                    pass

                try:
                    return _orig_get_field(cls, a_name, a_type, *args, **kwargs)
                finally:
                    try:
                        if orig_hash is None:
                            default_cls.__hash__ = None
                        else:
                            default_cls.__hash__ = orig_hash
                    except TypeError:
                        pass
        raise


dataclasses._get_field = _patched_get_field
# ---------------------------------------------------

import json
import torch
import argparse
import torchaudio
from vocos import Vocos
import torchaudio.transforms as T
from modules.builder import build_model


# TODO: move to utils
def load_wav_mono_resampled(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    # normalize audio
    wav = wav / wav.abs().max()
    return wav.squeeze(0)


def main(args):
    checkpoint_dir = args.checkpoint_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {checkpoint_dir}...")
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    model = build_model(cfg_dict)

    checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
    model.from_pretrained(checkpoint_path)
    model.eval()
    model.to(device)

    # Initialize Vocoder
    print("Initializing Vocoder...")
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)

    audio_path = args.audio_path
    target_path = args.target
    print(f"Processing input audio: {audio_path}")
    print(f"Processing target audio: {target_path}")

    with torch.inference_mode():
        wav_in = load_wav_mono_resampled(audio_path, model.config.sample_rate).to(
            device
        )
        wav_tg = load_wav_mono_resampled(target_path, model.config.sample_rate).to(
            device
        )

        # Encode both
        feat_in, mask_in, _, _, _ = model.extract_features(
            [(wav_in, model.config.sample_rate)]
        )
        feat_tg, mask_tg, _, _, _ = model.extract_features(
            [(wav_tg, model.config.sample_rate)]
        )

        out_in = model.encode(feat_in, mask_in)
        out_tg = model.encode(feat_tg, mask_tg)

        # Reconstruct context_vector component by component.
        qd = model.encoder._qd
        context_vector = torch.zeros_like(out_in.z)

        if getattr(args, "quantized", False):
            # Use only quantized content from input (no residual, no tail)
            if out_in.quantized is not None:
                context_vector[..., :qd] += out_in.quantized
        else:
            # Use quantized + residual from input
            if out_in.quantized is not None:
                context_vector[..., :qd] += out_in.quantized
            if out_in.residual is not None:
                context_vector[..., :qd] += out_in.residual
            # Build tail from chunks:
            # - First chunk_idx chunks from INPUT
            # - Last target_chunk_idx chunks from TARGET
            # - Zeros in between
            tail_dim = out_in.z.shape[-1] - qd
            T_in = context_vector.shape[1]

            # Get input tail
            tail_in = out_in.tail  # [B, T_in, tail_dim] or None

            # Get target tail (interpolated or averaged to match input length)
            tail_tg = None
            if out_tg.tail is not None:
                tail_tg = out_tg.tail  # [B, T_tg, tail_dim]

                if getattr(args, "temporal_smooth", False):
                    k = getattr(args, "smooth_size", 5)
                    pad_left = k // 2
                    pad_right = k - 1 - pad_left
                    tail_tg_perm = tail_tg.permute(0, 2, 1)  # [B, tail_dim, T_tg]
                    tail_tg_pad = torch.nn.functional.pad(
                        tail_tg_perm, (pad_left, pad_right), mode="replicate"
                    )
                    tail_tg = torch.nn.functional.avg_pool1d(
                        tail_tg_pad, kernel_size=k, stride=1
                    )
                    tail_tg = tail_tg.permute(0, 2, 1)

                if getattr(args, "target_avg", False):
                    # Average over time and expand to T_in
                    tail_tg = tail_tg.mean(dim=1, keepdim=True).expand(-1, T_in, -1)
                elif tail_tg.shape[1] != T_in:
                    tail_tg = tail_tg.permute(0, 2, 1)
                    tail_tg = torch.nn.functional.interpolate(
                        tail_tg, size=T_in, mode="linear", align_corners=False
                    )
                    tail_tg = tail_tg.permute(0, 2, 1)

            chunk_size = args.chunk_size
            n_chunks = tail_dim // chunk_size

            # Build tail as zeros, then fill in
            tail = torch.zeros(
                context_vector.shape[0],
                T_in,
                tail_dim,
                device=context_vector.device,
                dtype=context_vector.dtype,
            )

            if args.target_start_chunk is not None and args.target_end_chunk is not None:
                # Mode: Zeros everywhere except for the specified chunk range from TARGET
                start_idx = min(args.target_start_chunk * chunk_size, tail_dim)
                end_idx = min(args.target_end_chunk * chunk_size, tail_dim)
                if tail_tg is not None:
                    tail[..., start_idx:end_idx] = tail_tg[..., start_idx:end_idx]
                print(
                    f"Tail ({tail_dim}d, {n_chunks} chunks of {chunk_size}): "
                    f"zeros everywhere except target[{start_idx}:{end_idx}]"
                )
            else:
                in_chunks = args.chunk_idx if args.chunk_idx is not None else n_chunks
                tg_chunks = args.target_chunk_idx if args.target_chunk_idx is not None else 0

                # Fill first in_chunks from input
                in_keep = min(in_chunks * chunk_size, tail_dim)
                if in_keep > 0 and tail_in is not None:
                    tail[..., :in_keep] = tail_in[..., :in_keep]

                # Fill last tg_chunks from target
                tg_keep = min(tg_chunks * chunk_size, tail_dim)
                if tg_keep > 0 and tail_tg is not None:
                    tail[..., -tg_keep:] = tail_tg[..., -tg_keep:]

                if in_keep < tail_dim or tg_keep > 0:
                    print(
                        f"Tail ({tail_dim}d, {n_chunks} chunks of {chunk_size}): "
                        f"input[:{in_keep}] + zeros[{in_keep}:{tail_dim - tg_keep}] + target[{tail_dim - tg_keep}:]"
                    )

            context_vector[..., qd:] += tail

        padding_mask = out_in.padding_mask

        # Use the speaker embedding from the target audio (may be None if no instance norm)
        speaker_embedding_tg = getattr(out_tg, "speaker_embedding", None)

        # Decode the mixed latent representation
        reconstructed_mel, reconstructed_padding_mask = model.sample(
            num_steps=args.num_steps,
            temperature=args.temperature,
            guidance_scale=args.guidance_scale,
            z=context_vector,
            padding_mask=padding_mask,
            speaker_embedding=speaker_embedding_tg,
            guide_only_speaker=getattr(args, "guide_only_speaker", False),
        )

        audio = vocoder.decode(reconstructed_mel.permute(0, 2, 1))

        # normalize audio
        audio = audio / audio.abs().max()
        output_path = args.output_path
        if output_path is None:
            base_in = os.path.splitext(os.path.basename(audio_path))[0]
            base_tg = os.path.splitext(os.path.basename(target_path))[0]
            output_path = os.path.join(
                os.path.dirname(audio_path),
                f"swap_spk_{base_in}_to_{base_tg}.wav",
            )

        torchaudio.save(output_path, audio.cpu(), model.config.sample_rate)
        print("Saved swapped audio to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        type=str,
        default="checkpoints/exps/vq-setting-5-dropout-fixed",
    )
    parser.add_argument("-i", "--audio_path", type=str, default="ablations/male.wav")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to the target audio for swapping components",
    )
    parser.add_argument("-o", "--output_path", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument(
        "-q",
        "--quantized",
        action="store_true",
        help="Use only quantized head for content",
    )
    parser.add_argument(
        "--chunk_idx",
        type=int,
        default=None,
        help="Number of tail chunks to keep from INPUT (from start). Default: all",
    )
    parser.add_argument(
        "--target_chunk_idx",
        type=int,
        default=None,
        help="Number of tail chunks to keep from TARGET (from end). Default: 0",
    )
    parser.add_argument(
        "--target_start_chunk",
        type=int,
        default=None,
        help="Start chunk index to copy from TARGET tail (source tail will be zeros)",
    )
    parser.add_argument(
        "--target_end_chunk",
        type=int,
        default=None,
        help="End chunk index to copy from TARGET tail",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=2, help="Size of each tail chunk (default: 2)"
    )
    parser.add_argument(
        "--guide_only_speaker",
        action="store_true",
        help="Apply guidance scale only to speaker embedding",
    )
    parser.add_argument(
        "--target_avg",
        action="store_true",
        help="Average the target tail over time before applying it to the source",
    )
    parser.add_argument(
        "--temporal_smooth",
        action="store_true",
        help="Apply temporal smoothing to the target tail",
    )
    parser.add_argument(
        "--smooth_size",
        type=int,
        default=5,
        help="Window size (in frames) for temporal smoothing",
    )

    args = parser.parse_args()
    main(args)
