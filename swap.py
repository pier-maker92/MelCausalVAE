import os
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

        context_vector = out_in.z
        padding_mask = out_in.padding_mask

        # Use the speaker embedding from the target audio
        speaker_embedding_tg = getattr(out_tg, "speaker_embedding", None)
        if speaker_embedding_tg is None:
            raise RuntimeError(
                "target speaker embedding is None. Ensure the model was trained with use_instance_norm=True."
            )

        # Decode the mixed latent representation
        reconstructed_mel, reconstructed_padding_mask = model.sample(
            num_steps=args.num_steps,
            temperature=args.temperature,
            guidance_scale=args.guidance_scale,
            z=context_vector,
            padding_mask=padding_mask,
            speaker_embedding=speaker_embedding_tg,
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

    args = parser.parse_args()
    main(args)
