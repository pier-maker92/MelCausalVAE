# best to use both feature extractor and model with GPU!
import torch
import torchaudio
import torch.nn as nn
from typing import List, Tuple
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModel, AutoFeatureExtractor


@dataclass
class SeamlessM4Tv2EncoderOutput:
    semantic_features: torch.Tensor
    padding_mask: torch.Tensor


class SeamlessM4Tv2Encoder(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "WueNLP/seamless-m4t-v2-large-speech-encoder", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "WueNLP/seamless-m4t-v2-large-speech-encoder",
            trust_remote_code=True,
            torch_dtype=dtype,
        ).eval()
        self.resampler = torchaudio.transforms.Resample(orig_freq=24_000, new_freq=16_000)

    def set_dtype(self, dtype: torch.dtype):
        self.model.to(dtype=dtype)
        self.feature_extractor
        self.resampler
        self.dtype = dtype

    def set_device(self, device: torch.device):
        self.model.to(device=device)
        self.feature_extractor
        self.resampler
        self.device = device

    def _resize_padding_mask(self, attention_mask: torch.BoolTensor, target_length: int) -> torch.BoolTensor:
        attention_mask = (
            F.interpolate(
                attention_mask.unsqueeze(1).float(),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            > 0.5  # Use threshold instead of .bool()
        )
        return ~attention_mask

    @torch.no_grad()
    def forward(self, audios_srs: List[Tuple[torch.FloatTensor, int]], **kwargs):
        audios, sampling_rates = zip(*audios_srs)
        device = audios[0].device
        dtype = torch.bfloat16
        audios = [self.resampler(audio).to(dtype=torch.float32).cpu() for audio in audios]
        processed_audios = self.feature_extractor(
            audios, return_attention_mask=True, return_tensors="pt", sampling_rate=16000
        )
        processed_audios["input_features"] = processed_audios["input_features"].to(device=device, dtype=dtype)
        processed_audios["attention_mask"] = processed_audios["attention_mask"].to(device=device, dtype=bool)
        audio_hidden_states = self.model(**processed_audios).last_hidden_state
        padding_mask = self._resize_padding_mask(
            processed_audios["attention_mask"], target_length=audio_hidden_states.shape[1]
        )
        return SeamlessM4Tv2EncoderOutput(
            semantic_features=audio_hidden_states,
            padding_mask=padding_mask,
        )
