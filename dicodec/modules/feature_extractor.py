import json
import torch
import einops
from torch import nn
from typing import Tuple, List
import torchaudio.functional as F
from .configs import MelSpectrogramConfig, WavLMConfig
from torchaudio.transforms import MelSpectrogram
from .output_dataclasses import FeatureExtractorOutput


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
            audios = [F.resample(audio, sr, self.sampling_rate) for audio in audios]
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
            mel_spec = (
                mel_spec - self.mean.to(device=mel_spec.device, dtype=mel_spec.dtype)
            ) / self.std.to(device=mel_spec.device, dtype=mel_spec.dtype)

        return FeatureExtractorOutput(
            audio_features=mel_spec,
            padding_mask=padding_mask,
        )


class WavLMFeatureExtractor(nn.Module):
    def __init__(
        self,
        config: WavLMConfig,
        wavlm: nn.Module,
        **kwargs,
    ):
        super().__init__()
        self.sampling_rate = config.sampling_rate
        self.layer = config.layer
        self.normalize = config.normalize

        object.__setattr__(self, "_wavlm", wavlm)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad_(False)

        self.register_buffer("std", torch.tensor(1.0))
        self.register_buffer("mean", torch.tensor(0.0))

    @property
    def wavlm(self) -> nn.Module:
        return self._wavlm

    def train(self, mode: bool = True):
        super().train(mode)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad_(False)
        return self

    @staticmethod
    def _normalize_features_per_channel(
        features: torch.Tensor,
        padding_mask: torch.BoolTensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(features.dtype)
        valid_count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (features * valid).sum(dim=1, keepdim=True) / valid_count
        variance = ((features - mean) ** 2 * valid).sum(dim=1, keepdim=True)
        variance = variance / (valid_count - 1).clamp_min(1.0)
        normalized = (features - mean) / (torch.sqrt(variance) + eps)
        return normalized.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        audios_srs: List[Tuple[torch.FloatTensor, int]] | None = None,
        audio_16khz: List[torch.FloatTensor] | None = None,
        **kwargs,
    ):
        if audio_16khz is not None:
            audios = tuple(audio_16khz)
        elif audios_srs is not None:
            audios, sampling_rates = zip(*audios_srs)
            unique_sampling_rates = set(sampling_rates)
            if len(unique_sampling_rates) > 1:
                raise ValueError(
                    "All audios must have the same sampling rate. "
                    f"Found {len(unique_sampling_rates)} unique sampling rates: "
                    f"{unique_sampling_rates}."
                )
            sr = unique_sampling_rates.pop()
            if sr != self.sampling_rate:
                import torchaudio.functional as F

                audios = [F.resample(audio, sr, self.sampling_rate) for audio in audios]
        else:
            raise ValueError(
                "WavLMFeatureExtractor requires audios_srs or audio_16khz."
            )

        dtype = audios[0].dtype
        device = audios[0].device

        if len(audios) > 1:
            max_length = max(audio.size(-1) for audio in audios)
            batch_size = len(audios)

            padded_audios = torch.nn.utils.rnn.pad_sequence(
                audios, batch_first=True, padding_value=0.0
            )
            padding_mask = torch.ones(
                (batch_size, max_length),
                dtype=torch.bool,
                device=device,
            )
            for i, audio in enumerate(audios):
                padding_mask[i, : audio.size(-1)] = False
        else:
            padded_audios = audios[0].unsqueeze(0)
            padding_mask = torch.zeros(
                1,
                audios[0].size(-1),
                dtype=torch.bool,
                device=device,
            )

        if self.wavlm.device != device:
            self.wavlm.to(device)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad_(False)

        with torch.no_grad():
            outputs = self.wavlm(
                padded_audios.float(),  # WavLM weights are always fp32
                attention_mask=(~padding_mask).long(),
                output_hidden_states=True,
            )
            features = outputs.hidden_states[self.layer]

        features = features.to(dtype)

        feat_padding_mask = (
            torch.nn.functional.interpolate(
                padding_mask.unsqueeze(1).to(torch.float32),
                size=features.shape[1],
                mode="nearest",
            )
            .squeeze(1)
            .to(torch.bool)
        )

        if self.normalize:
            features = self._normalize_features_per_channel(features, feat_padding_mask)

        return FeatureExtractorOutput(
            audio_features=features,
            padding_mask=feat_padding_mask,
        )
