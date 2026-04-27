#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio

# Add current directory to path to import vq_inference
sys.path.append(str(Path(__file__).parent))
from vq_inference import MelVAEInference

def main():
    parser = argparse.ArgumentParser(description="Minimal VQ Mel-VAE inference script")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input wav file")
    parser.add_argument("-o", "--output", type=str, default="reconstructed.wav", help="Output wav file")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to model checkpoint (.safetensors)")
    parser.add_argument("--config", type=str, default="configs/inference/vq_inference_full.yaml", help="Path to config yaml")
    parser.add_argument("--steps", type=int, default=16, help="Number of CFM steps")
    parser.add_argument("--temp", type=float, default=0.2, help="Temperature for sampling")
    parser.add_argument("--guidance", type=float, default=1.3, help="Guidance scale")
    
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = MelVAEInference.load(args.config, args.checkpoint)
    
    # Load audio
    print(f"Loading audio from {args.input}...")
    wav = model.load_wav_mono_resampled(args.input)
    
    # Run inference
    print("Running inference...")
    with torch.inference_mode():
        latent = model.encode(wav, model.sample_rate)
        audio = model.sample(
            latent, 
            #vq_ids = None,
            residual=None,
            tail=None,
            num_steps=args.steps, 
            temperature=args.temp, 
            guidance_scale=args.guidance
        )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torchaudio.save(str(output_path), audio.unsqueeze(0), model.sample_rate)
    print(f"Saved reconstructed audio to {output_path}")

if __name__ == "__main__":
    main()
