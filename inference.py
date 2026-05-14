import os
import json
import torch
import argparse
import torchaudio
from vocos import Vocos
import torchaudio.transforms as T
from modules.builder import build_model


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
        device = torch.device("cpu")

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
    print(f"Processing audio: {audio_path}")

    with torch.inference_mode():
        wav = load_wav_mono_resampled(audio_path, model.config.sample_rate).to(device)

        # Prepare inputs as expected by VAE encode_decode: list of (audio_tensor, sr)
        audios_srs = [(wav, model.config.sample_rate)]

        params = {
            "audios_srs": audios_srs,
            "num_steps": args.num_steps,
            "temperature": args.temperature,
            "guidance_scale": args.guidance_scale,
        }
        if args.quantized or args.residual or args.tail:
            params["quantized"] = False
            params["residual"] = False
            params["tail"] = False

        if args.quantized:
            params["quantized"] = True
        if args.residual:
            params["residual"] = True
        if args.tail:
            params["tail"] = True

        out = model.encode_decode(**params)

        reconstructed_mel = out["decoder_output"].audio_features
        padding_mask = out["decoder_output"].padding_mask
        audio = vocoder.decode(reconstructed_mel.permute(0, 2, 1))

        # normalize audio
        audio = audio / audio.abs().max()
        output_path = args.output_path
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(audio_path),
                f"reconstructed_{os.path.basename(audio_path)}",
            )
        torchaudio.save(output_path, audio, model.config.sample_rate)
        print("Saved reconstructed audio to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--checkpoint_dir", type=str, default="checkpoints/vq-refactored"
    )
    parser.add_argument("-i", "--audio_path", type=str, default="ablations/male.wav")
    parser.add_argument("-o", "--output_path", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--guidance_scale", type=float, default=1.3)
    parser.add_argument("-q", "--quantized", action="store_true")
    parser.add_argument("-r", "--residual", action="store_true")
    parser.add_argument("-t", "--tail", action="store_true")
    args = parser.parse_args()
    main(args)
