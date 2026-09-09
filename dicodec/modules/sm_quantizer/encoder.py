"""Configurable framewise MLP or strictly causal residual convolution encoder."""

import torch
from torch import nn
from torch.nn import functional as F

from .configs import EncoderConfig


class CausalBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.left_padding = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x).transpose(1, 2)
        hidden = self.conv(F.pad(hidden, (self.left_padding, 0))).transpose(1, 2)
        return x + self.dropout(self.output(F.silu(hidden)))


def build_encoder(input_dim: int, output_dim: int, fallback_hidden_dim: int, config: EncoderConfig) -> nn.Sequential:
    hidden_dim = config.hidden_dim or fallback_hidden_dim
    layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
    if config.type == "mlp":
        for _ in range(config.layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
    else:
        layers.extend(CausalBlock(hidden_dim, config.kernel_size, config.dropout) for _ in range(config.layers))
    if config.dropout:
        layers.append(nn.Dropout(config.dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    # Default MLP retains encoder.0 / encoder.2 keys and initialization order.
    return nn.Sequential(*layers)
