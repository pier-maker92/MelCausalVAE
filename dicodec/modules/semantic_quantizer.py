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


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.linear2(self.act(self.linear1(self.norm1(x))))
        x = x + residual
        if valid_mask is not None:
            x = x * valid_mask
        return x


class MLPResNetStack(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(MLPResNetBlock(dim) for _ in range(num_blocks))

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, valid_mask=valid_mask)
        return x


class MLPResNetEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.resnet = MLPResNetStack(hidden_dim, num_blocks)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_mask is not None:
            x = x * valid_mask
        x = self.in_proj(x)
        if valid_mask is not None:
            x = x * valid_mask
        x = self.out_proj(self.resnet(x, valid_mask=valid_mask))
        if valid_mask is not None:
            x = x * valid_mask
        return x


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
        self.semantic_encoder = MLPResNetEncoder(
            in_dim=dim,
            hidden_dim=config.hidden_dim,
            out_dim=quant_dim,
            num_blocks=config.num_sem_blocks,
        )
        self.quantizer = VectorQuantizer(config.vq_config, dim=quant_dim)
        self.prosody_encoder = MLPResNetEncoder(
            in_dim=dim,
            hidden_dim=config.hidden_dim,
            out_dim=quant_dim,
            num_blocks=config.num_pros_blocks,
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
        drop_acoustic_p = self.config.vq_config.drop_acoustic_p
        if self.training and drop_acoustic_p > 0.0:
            keep = torch.rand(
                z_pros.shape[0],
                1,
                1,
                device=z_pros.device,
                dtype=z_pros.dtype,
            ) >= drop_acoustic_p
            z_pros = z_pros * keep.to(dtype=z_pros.dtype)
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
