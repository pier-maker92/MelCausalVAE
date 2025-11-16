# best to use both feature extractor and model with GPU!
import torch
import torchaudio
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)


class SeamlessM4Tv2Encoder(nn.Module):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "WueNLP/seamless-m4t-v2-large-speech-encoder", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "WueNLP/seamless-m4t-v2-large-speech-encoder",
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device=device)
        self.resampler = torchaudio.transforms.Resample(orig_freq=24_000, new_freq=16_000)
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, audio):
        audio = self.resampler(audio)
        audio_inputs = self.feature_extractor(audio, return_attention_mask=True, return_tensors="pt", device=device)
        audio_inputs = audio_inputs.to(device)
        with torch.autocast(dtype=self.dtype, device_type=self.device):
            audio_hidden_states = self.model(**audio_inputs)[0].detach().cpu().numpy().squeeze()
        return audio_hidden_states


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    seamless_m4t_encoder = SeamlessM4Tv2Encoder(device=device, dtype=torch.float32)
    audio, orig_freq = torchaudio.load("/Users/pierfrancescomelucci/Research/male.wav")
    audio = seamless_m4t_encoder(audio)
    breakpoint()
