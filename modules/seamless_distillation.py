import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoProcessor, SeamlessM4TModel

processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-large")
seamless_encoder = (
    SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-large")
).speech_encoder


@dataclass
class SeamlessM4Tv2EncoderOutput:
    semantic_features: torch.Tensor  # feature at 6.25Hz sampling rate
    padding_mask: torch.Tensor  # 0 = valid, 1 = padding


class SeamlessEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = processor
        self.seamless_encoder = seamless_encoder
        self.sr = processor.feature_extractor.sampling_rate

    def downsample_mask(self, mask: torch.BoolTensor, target_len: int):
        """
        mask: (B, T) boolean
        target_len: int (downsampled length)
        In SeamlessM4T, 1 = valid, 0 = padding
        """
        assert mask.dtype == torch.bool, "mask must be a boolean tensor"
        mask = mask.float()  # interpolate needs float
        mask = mask.unsqueeze(1)  # (B, 1, T)

        mask_ds = F.interpolate(mask, size=target_len, mode="nearest")

        return mask_ds.squeeze(1).bool()

    def preprocess_audio(self, audios_srs: List[Tuple[torch.FloatTensor, int]]):
        audios, sampling_rates = zip(*audios_srs)
        unique_sampling_rates = set(sampling_rates)
        if len(unique_sampling_rates) > 1:
            raise ValueError(
                "All audios must have the same sampling rate. "
                f"Found {len(unique_sampling_rates)} unique sampling rates: "
                f"{unique_sampling_rates}."
            )
        sr = unique_sampling_rates.pop()
        if sr != self.sr:
            print(f"Resampling audio from {sr} to {self.sr}")
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            audios = [resampler(audio) for audio in audios]

        audio_inputs = processor(
            audio=audios,
            return_tensors="pt",
            sampling_rate=self.sr,
        )
        return audio_inputs

    def forward(
        self,
        audios_srs,
    ):
        audio_inputs = self.preprocess_audio(audios_srs)
        seamless_audio_features = self.seamless_encoder(audio_inputs.input_features)
        attention_mask = self.downsample_mask(
            audio_inputs.attention_mask, seamless_audio_features.shape[1]
        )
        return SeamlessM4Tv2EncoderOutput(
            semantic_features=seamless_audio_features,
            padding_mask=~attention_mask,
        )
