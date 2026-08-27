import os
import torch
import argparse
import torchaudio
import torchaudio.transforms as T
from modules.builder import load_pretrained_model


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
        "-qq", "--zero_speaker", action="store_true", help="Zero out speaker embedding"
    )
    parser.add_argument(
        "--guide_only_speaker",
        action="store_true",
        help="Apply guidance scale only to speaker embedding",
    )

    args = parser.parse_args()
    main(args)
