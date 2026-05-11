import sys
import json
import torch
import einops
from torch import nn
from einops import rearrange
from typing import Tuple, List
from .configs import MelSpectrogramConfig
from torchaudio.transforms import MelSpectrogram
from .output_dataclasses import FeatureExtractorOutput


# TODO fix implementation for bigvgan
# from meldataset import get_mel_spectrogram as get_mel_spectrogram_bigvgan
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        config: MelSpectrogramConfig,
        **kwargs,
    ):
        super().__init__()
        self.sampling_rate = config.sampling_rate
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.n_mels = config.n_mels
        self.padding = config.padding
        self.normalize = config.normalize
        self.use_bigvgan_mel = config.use_bigvgan_mel

        if self.use_bigvgan_mel:
            raise NotImplementedError("BigVGAN mel not implemented yet")

        self.mel_transform = MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=self.padding == "center",
            power=1,
        )

        self.register_buffer("std", torch.tensor(2.080231189727783))
        self.register_buffer("mean", torch.tensor(-1.0173088312149048))

    @torch.no_grad()
    def _update_std_mean_with_momentum(
        self, mel_spec: torch.Tensor, padding_mask: torch.BoolTensor
    ):
        # mel_spec: (B, T, C), padding_mask: (B, T)
        valid_mel = mel_spec[~padding_mask]
        if valid_mel.numel() > 0:
            self.std.copy_(self.std * 0.99 + valid_mel.std() * 0.01)
            self.mean.copy_(self.mean * 0.99 + valid_mel.mean() * 0.01)

    def forward(self, audios_srs: List[Tuple[torch.FloatTensor, int]], **kwargs):
        audios, sampling_rates = zip(*audios_srs)
        # audios = [audio.unsqueeze(0) for audio in audios if audio.dim() == 1]
        unique_sampling_rates = set(sampling_rates)
        if len(unique_sampling_rates) > 1:
            raise ValueError(
                "All audios must have the same sampling rate. "
                f"Found {len(unique_sampling_rates)} unique sampling rates: "
                f"{unique_sampling_rates}."
            )
        sr = unique_sampling_rates.pop()
        if sr != self.sampling_rate:
            raise ValueError(
                f"Sampling rate {sr} is not supported by this model. "
                f"Expected {self.sampling_rate}."
            )
        dtype = audios[0].dtype
        device = audios[0].device
        # Get max length for padding
        # Get max length
        if len(audios) > 1:
            max_length = max(audio.size(-1) for audio in audios)
            batch_size = len(audios)

            # Create padded tensor using torch.nn.utils.rnn.pad_sequence
            padded_audios = torch.nn.utils.rnn.pad_sequence(
                audios, batch_first=True, padding_value=0.0
            )
            # Create padding mask
            padding_mask = torch.ones(
                (batch_size, max_length),
                dtype=torch.bool,
                device=audios[0].device,
            )
            for i, audio in enumerate(audios):
                padding_mask[i, : audio.size(-1)] = False
        else:
            padded_audios = audios[0].unsqueeze(0)
            padding_mask = torch.zeros(
                1,
                audios[0].size(-1),
                dtype=torch.bool,
                device=audios[0].device,
            )

        self.mel_transform.to(device=device, dtype=torch.float32)
        mel_spec = self.mel_transform(padded_audios.to(torch.float32))
        # Keep in fp32 for log operation to avoid fp16 underflow
        mel_spec = torch.log(mel_spec + 1e-6)

        mel_spec = einops.rearrange(mel_spec, "b c t -> b t c")
        # Convert to target dtype after log operation
        mel_spec = mel_spec.to(dtype)

        # Interpolate padding mask to match mel_spec temporal dimension
        # padding_mask is (B, L), we want (B, T)
        padding_mask = (
            torch.nn.functional.interpolate(
                padding_mask.unsqueeze(1).to(torch.float32),
                size=mel_spec.shape[1],
                mode="nearest",
            )
            .squeeze(1)
            .to(torch.bool)
        )

        assert padding_mask.shape[1] == mel_spec.shape[1], (
            f"Temporal dimensions mismatch: padding_mask {padding_mask.shape[1]} vs "
            f"mel_spec {mel_spec.shape[1]}"
        )

        if self.training:
            self._update_std_mean_with_momentum(mel_spec, padding_mask)

        if self.normalize:
            mel_spec = (mel_spec - self.mean) / self.std

        return FeatureExtractorOutput(
            audio_features=mel_spec,
            padding_mask=padding_mask,
        )
