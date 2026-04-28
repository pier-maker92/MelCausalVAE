import os

import torch
import torchaudio

from vq_inference import MelVAEInference

model = MelVAEInference.load(
    "configs/inference/vq_inference_full.yaml",
    "checkpoints/exps/1d-8x-NAR-AE-vq-chunks/model.safetensors",
)
model.eval()

with torch.inference_mode():
    wav = model.load_wav_mono_resampled("ablations/violin.wav")
    latent = model.encode(wav, model.sample_rate)
    audio = model.sample(
        latent,
        #vq_ids=None,
        #residual=None,
        #tail=None,
        num_steps=16,
        temperature=0.2,
        guidance_scale=1.3,
    )
   

os.makedirs("audio_outputs", exist_ok=True)
torchaudio.save(
    "audio_outputs/violin.wav",
    audio.unsqueeze(0),
    model.sample_rate,
)
print("Audio saved to audio_outputs/violin.wav")