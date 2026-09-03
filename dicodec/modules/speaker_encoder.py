from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF

from .configs import SpeakerEncoderConfig
from transformers import WavLMModel


class MaskedAttentiveStatsPool(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        attention_channels: int = 128,
    ):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(input_channels, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, input_channels, kernel_size=1),
        )
        self.proj = nn.Linear(input_channels * 2, output_channels)
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, x: torch.Tensor, padding_mask: torch.BoolTensor) -> torch.Tensor:
        x_cf = x.transpose(1, 2)
        attn_logits = self.attn(x_cf)
        attn_logits = attn_logits.masked_fill(padding_mask[:, None, :], -torch.inf)
        alpha = torch.softmax(attn_logits, dim=2)
        alpha = alpha.masked_fill(padding_mask[:, None, :], 0.0)

        mean = torch.sum(alpha * x_cf, dim=2)
        residuals = torch.sum(alpha * (x_cf**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-4, max=1e4))
        return self.norm(self.proj(torch.cat([mean, std], dim=1)))


class WavLMSpeakerEncoder(nn.Module):
    def __init__(
        self,
        config: SpeakerEncoderConfig,
        wavlm: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.sampling_rate = config.sampling_rate
        self.layers = list(config.wavlm_layers)
        self.layer_combine = config.wavlm_layer_combine
        self.pooling = config.wavlm_pooling
        self.freeze_wavlm = True

        if self.layer_combine not in {"mean", "weighted_sum", "concat"}:
            raise ValueError(
                "wavlm_layer_combine must be one of: mean, weighted_sum, concat."
            )
        if self.pooling not in {"mean", "mean_std", "attentive_stats"}:
            raise ValueError(
                "wavlm_pooling must be one of: mean, mean_std, attentive_stats."
            )

        object.__setattr__(self, "_wavlm", wavlm)
        max_layer = self.wavlm.config.num_hidden_layers
        if min(self.layers) < 0 or max(self.layers) > max_layer:
            raise ValueError(
                f"wavlm_layers must be in [0, {max_layer}] for "
                "the configured WavLM model."
            )
        self.wavlm.eval()
        for parameter in self.wavlm.parameters():
            parameter.requires_grad_(False)

        hidden_size = self.wavlm.config.hidden_size
        if self.layer_combine == "weighted_sum":
            if config.wavlm_layer_weights is not None:
                if len(config.wavlm_layer_weights) != len(self.layers):
                    raise ValueError(
                        "wavlm_layer_weights must have the same length as "
                        "wavlm_layers."
                    )
                alpha = torch.tensor(config.wavlm_layer_weights, dtype=torch.float32)
                if (alpha < 0).any() or alpha.sum() <= 0:
                    raise ValueError(
                        "wavlm_layer_weights must be non-negative and have "
                        "positive sum."
                    )
                alpha = alpha / alpha.sum()
                self.layer_alpha = nn.Parameter(torch.log(alpha.clamp(min=1e-8)))
            else:
                self.layer_alpha = nn.Parameter(torch.zeros(len(self.layers)))
            pooled_input_dim = hidden_size
        elif self.layer_combine == "concat":
            pooled_input_dim = hidden_size * len(self.layers)
        else:
            self.layer_alpha = None
            pooled_input_dim = hidden_size

        if self.pooling == "attentive_stats":
            self.pool = MaskedAttentiveStatsPool(
                input_channels=pooled_input_dim,
                output_channels=config.embedding_dim,
                attention_channels=config.wavlm_attention_channels,
            )
            self.proj = nn.Identity()
        else:
            multiplier = 2 if self.pooling == "mean_std" else 1
            self.pool = None
            self.proj = nn.Sequential(
                nn.Linear(pooled_input_dim * multiplier, config.embedding_dim),
                nn.LayerNorm(config.embedding_dim),
            )

    @property
    def wavlm(self) -> nn.Module:
        return self._wavlm

    def train(self, mode: bool = True):
        super().train(mode)
        self.wavlm.eval()
        for parameter in self.wavlm.parameters():
            parameter.requires_grad_(False)
        return self

    def _prepare_audio(
        self,
        audios_srs: List[Tuple[torch.FloatTensor, int]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        waveforms = []
        lengths = []
        for audio, sample_rate in audios_srs:
            waveform = audio.to(device=device, dtype=torch.float32)
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=0)
            if sample_rate != self.sampling_rate:
                waveform = AF.resample(waveform, sample_rate, self.sampling_rate)
            waveforms.append(waveform)
            lengths.append(waveform.numel())

        padded = torch.nn.utils.rnn.pad_sequence(
            waveforms, batch_first=True, padding_value=0.0
        )
        padding_mask = torch.ones(padded.shape, device=padded.device, dtype=torch.bool)
        for index, length in enumerate(lengths):
            padding_mask[index, :length] = False
        return padded, padding_mask

    def _combine_layers(self, hidden_states) -> torch.Tensor:
        selected = [hidden_states[layer] for layer in self.layers]
        if self.layer_combine == "concat":
            return torch.cat(selected, dim=-1)
        stacked = torch.stack(selected, dim=0)
        if self.layer_combine == "weighted_sum":
            alpha = torch.softmax(self.layer_alpha, dim=0).view(-1, 1, 1, 1)
            return (alpha * stacked).sum(dim=0)
        return stacked.mean(dim=0)

    def _masked_pool(
        self,
        features: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(features.dtype)
        denom = valid.sum(dim=1).clamp(min=1.0)
        mean = (features * valid).sum(dim=1) / denom
        if self.pooling == "mean":
            return self.proj(mean)

        variance = ((features - mean[:, None, :]) ** 2 * valid).sum(dim=1) / denom
        std = torch.sqrt(variance.clamp(min=1e-4, max=1e4))
        return self.proj(torch.cat([mean, std], dim=-1))

    def forward(
        self,
        audios_srs: List[Tuple[torch.FloatTensor, int]],
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        waveforms, padding_mask = self._prepare_audio(audios_srs, device)

        self.wavlm.eval()
        for parameter in self.wavlm.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            outputs = self.wavlm(
                waveforms.float(),
                attention_mask=(~padding_mask).long(),
                output_hidden_states=True,
            )
        features = self._combine_layers(outputs.hidden_states)

        feat_padding_mask = (
            F.interpolate(
                padding_mask.unsqueeze(1).to(torch.float32),
                size=features.shape[1],
                mode="nearest",
            )
            .squeeze(1)
            .to(torch.bool)
        )

        features = features.to(dtype=next(self.parameters()).dtype)

        if self.pool is not None:
            return self.pool(features, feat_padding_mask)
        return self._masked_pool(features, feat_padding_mask)
