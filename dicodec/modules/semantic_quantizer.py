from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import SemanticQuantizerConfig
from .output_dataclasses import VQStats
from .quantizer.vq import VectorQuantizer


@dataclass
class SemanticQuantizerOutput:
    z: torch.Tensor
    z_sem_q: torch.Tensor
    z_pros: torch.Tensor
    quantizer_loss: torch.Tensor
    recon_loss: torch.Tensor
    stats: VQStats | None = None
    indices: torch.Tensor | None = None


class ResNetBlock1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.act = nn.SiLU()

    @staticmethod
    def _normalize(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.conv1(self.act(self._normalize(x, self.norm1)))
        if valid_mask is not None:
            x = x * valid_mask
        x = self.conv2(self.act(self._normalize(x, self.norm2)))
        x = residual + x
        if valid_mask is not None:
            x = x * valid_mask
        return x


class ResNetStack1D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
    ):
        super().__init__()
        dilations = dilations or [1] * num_blocks
        if len(dilations) != num_blocks:
            raise ValueError("dilations length must match num_blocks.")
        self.blocks = nn.ModuleList(
            ResNetBlock1D(dim, kernel_size=kernel_size, dilation=d) for d in dilations
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, valid_mask=valid_mask)
        return x


class ResNetEncoder1D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_blocks: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.resnet = ResNetStack1D(hidden_dim, num_blocks, kernel_size=kernel_size)
        self.out_proj = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.transpose(1, 2)
        channel_mask = None
        if valid_mask is not None:
            channel_mask = valid_mask.transpose(1, 2)
            x = x * channel_mask
        x = self.in_proj(x)
        if channel_mask is not None:
            x = x * channel_mask
        x = self.out_proj(self.resnet(x, valid_mask=channel_mask))
        if channel_mask is not None:
            x = x * channel_mask
        return x.transpose(1, 2)


class SemanticQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        config: SemanticQuantizerConfig,
    ):
        super().__init__()
        if config.vq_config is None:
            raise ValueError("semantic_quantizer_config.vq_config must be set.")

        quant_dim = config.quant_dim or dim
        if (
            config.vq_config.vq_type in {"vq", "vq_ema"}
            and config.vq_config.vq_dim is None
        ):
            config.vq_config.vq_dim = quant_dim

        self.config = config
        self.dim = dim
        self.quant_dim = quant_dim
        self.semantic_encoder = ResNetEncoder1D(
            in_dim=dim,
            hidden_dim=config.hidden_dim,
            out_dim=quant_dim,
            num_blocks=config.num_sem_blocks,
            kernel_size=config.kernel_size,
        )
        self.quantizer = VectorQuantizer(config.vq_config, dim=quant_dim)
        self.prosody_encoder = ResNetEncoder1D(
            in_dim=dim,
            hidden_dim=config.hidden_dim,
            out_dim=quant_dim,
            num_blocks=config.num_pros_blocks,
            kernel_size=config.kernel_size,
        )
        self.out_proj = nn.Linear(quant_dim * 2, dim)

    def forward(
        self,
        z_sem: torch.Tensor,
        z_pros_mean: torch.Tensor,
        padding_mask: torch.BoolTensor | None = None,
    ) -> SemanticQuantizerOutput:
        valid_mask = None
        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).to(dtype=z_sem.dtype)

        z_sem_enc = self.semantic_encoder(z_sem, valid_mask=valid_mask)
        vq_out = self.quantizer(z_sem_enc, padding_mask=padding_mask)
        z_sem_q = vq_out.quantized
        z_pros = self.prosody_encoder(z_pros_mean, valid_mask=valid_mask)
        z = self.out_proj(torch.cat([z_sem_q, z_pros], dim=-1))

        if padding_mask is not None:
            valid = ~padding_mask
            recon_loss = (
                F.mse_loss(z_sem_q[valid], z_sem_enc.detach()[valid])
                if valid.any()
                else z.new_zeros(())
            )
        else:
            recon_loss = F.mse_loss(z_sem_q, z_sem_enc.detach())

        total_loss = vq_out.loss
        if self.config.recon_weight > 0.0:
            total_loss = total_loss + self.config.recon_weight * recon_loss

        return SemanticQuantizerOutput(
            z=z,
            z_sem_q=z_sem_q,
            z_pros=z_pros,
            quantizer_loss=total_loss,
            recon_loss=recon_loss,
            stats=vq_out.stats,
            indices=vq_out.indices,
        )
