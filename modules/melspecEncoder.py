import torch
import einops
from torch import nn
from typing import Optional
from einops import rearrange
from typing import List, Tuple
from dataclasses import dataclass
from transformers import PretrainedConfig
from torchaudio.transforms import MelSpectrogram


@dataclass
class MelSpectrogramConfig:
    mel_channels: int = 100
    sampling_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    padding: str = "center"
    normalize: bool = True
    # def __init__(
    #     self,
    #     sampling_rate: int = 24000,
    #     n_fft: int = 1024,
    #     hop_length: int = 256,
    #     n_mels: int = 100,
    #     padding: str = "center",
    #     normalize: bool = True,
    #     **kwargs,
    # ):
    #     super().__init__(**kwargs)
    #     self.sampling_rate = sampling_rate
    #     self.n_fft = n_fft
    #     self.hop_length = hop_length
    #     self.n_mels = n_mels
    #     self.padding = padding
    #     self.normalize = normalize


class MelSpectrogramEncoder(torch.nn.Module):
    _supports_flash_attn_2 = True
    _supports_sdpa = True

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
        self.mel_transform = MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=self.padding == "center",
            power=1,
        )
        self.std = 2.0798065662384033
        self.mean = -0.9009257555007935
        self.normalize = config.normalize

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
            raise ValueError(f"Sampling rate {sr} is not supported by this model. " f"Expected {self.sampling_rate}.")
        dtype = audios[0].dtype
        device = audios[0].device
        # Get max length for padding
        # Get max length
        if len(audios) > 1:
            max_length = max(audio.size(-1) for audio in audios)
            batch_size = len(audios)

            # Create padded tensor using torch.nn.utils.rnn.pad_sequence
            padded_audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0)
            # Create padding mask
            padding_mask = torch.ones(
                (batch_size, max_length),
                dtype=torch.bool,
                device=audios[0].device,
            )
            for i, audio in enumerate(audios):
                padding_mask[i, : audio.size(-1)] = False
        else:
            padded_audios = audios[0]
            padding_mask = torch.zeros(
                1,
                audios[0].size(-1),
                dtype=torch.bool,
                device=audios[0].device,
            )
        self.mel_transform.to(device=device, dtype=torch.float32)
        # Apply mel transform to padded audio (always in fp32 for stability)
        # self.mel_transform.mel_scale.fb.to(torch.float32).to(device=device)

        mel_spec = self.mel_transform(padded_audios.to(torch.float32))
        # Keep in fp32 for log operation to avoid fp16 underflow
        mel_spec = torch.log(mel_spec + 1e-6)
        mel_spec = einops.rearrange(mel_spec, "b c t -> b t c")
        # Convert to target dtype after log operation
        mel_spec = mel_spec.to(dtype)

        padding_mask = (
            torch.nn.functional.interpolate(
                padding_mask.unsqueeze(0).unsqueeze(0).to(mel_spec.dtype),
                size=(mel_spec.shape[:2]),
                mode="bicubic",
                align_corners=False,
            )
            .to(dtype=torch.bool)
            .squeeze(0)
            .squeeze(0)
        )  # 93.75Hz

        assert padding_mask.shape[1] == mel_spec.shape[1], (
            f"Temporal dimensions mismatch: padding_mask {padding_mask.shape[1]} vs " f"mel_spec {mel_spec.shape[1]}"
        )

        if self.normalize:
            mel_spec = (mel_spec - self.mean) / self.std

        return MelSpectrogramOutput(
            audio_features=mel_spec,
            padding_mask=padding_mask,
            codes=None,
            embeddings=mel_spec,
        )


@dataclass
class MelSpectrogramOutput:
    audio_features: torch.Tensor
    padding_mask: torch.Tensor
    codes: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None
