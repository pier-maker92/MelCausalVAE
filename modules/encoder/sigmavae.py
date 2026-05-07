import math
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from modules.configs import SigmaVAEencoderConfig


class SigmaVAEEncoder(nn.Module):
    def __init__(self, config: SigmaVAEencoderConfig):
        super().__init__()
        self.config = config
        self.std_activation = (
            nn.Softplus() if self.config.use_sofplus else nn.Identity()
        )
        self.kl_loss_weight = float(config.kl_loss_weight)

    def forward(self, **kwargs):
        pass

    def reparameterize(
        self,
        mu: torch.FloatTensor,
        logvar: Optional[torch.FloatTensor] = None,
        std: Optional[float] = None,
    ) -> torch.FloatTensor:
        eps = torch.randn_like(mu)
        if logvar is None:
            std = self.sample_scalar_std(mu, std)
            while std.dim() < mu.dim():
                std = std.unsqueeze(-1)
        else:
            std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def kl_divergence(
        self,
        mu: torch.FloatTensor,
        logvar: Optional[torch.FloatTensor],
        padding_mask: torch.BoolTensor,
        dtype: torch.dtype,
    ) -> torch.FloatTensor:
        if logvar is None:
            # Compute in fp32 for numerical stability
            mu_valid = mu[~padding_mask].to(dtype)
            return F.mse_loss(mu_valid, torch.zeros_like(mu_valid)).to(mu.dtype)
        # Compute KL divergence in fp32 for numerical stability with fp16
        mu_valid = mu[~padding_mask].to(dtype)
        logvar_valid = logvar[~padding_mask].to(dtype)
        kl = -0.5 * torch.sum(1 + logvar_valid - mu_valid.pow(2) - logvar_valid.exp())
        return kl.to(mu.dtype) * self.kl_loss_weight

    def kl_divergence_weighted(
        self,
        mu: torch.FloatTensor,
        logvar: Optional[torch.FloatTensor],
        padding_mask: torch.BoolTensor,
        dtype: torch.dtype,
        channel_weights: torch.Tensor,
    ) -> torch.FloatTensor:
        """
        Same KL as kl_divergence but each latent dimension is scaled by channel_weights [latent_dim].
        When weights are all 1.0, matches kl_divergence (up to dtype handling).
        """
        w = channel_weights.to(device=mu.device, dtype=dtype).view(1, 1, -1)
        valid = (~padding_mask).unsqueeze(-1).to(dtype=dtype)
        if logvar is None:
            mu_f = mu.to(dtype)
            denom = (valid.sum() * mu.shape[-1]).clamp(min=1.0)
            return (mu_f.pow(2) * w * valid).sum() / denom
        mu_f = mu.to(dtype)
        logvar_f = logvar.to(dtype)
        kl_elem = -0.5 * (1 + logvar_f - mu_f.pow(2) - logvar_f.exp())
        return (kl_elem * w * valid).sum().to(mu.dtype)

    def sample_scalar_std(
        self, mu: torch.FloatTensor, std: Optional[float] = None
    ) -> torch.FloatTensor:
        weight = self.std_activation(
            torch.randn(mu.shape[0], mu.shape[1], dtype=mu.dtype, device=mu.device)
            * (self.config.target_std if std is None else std)
        )
        return torch.clamp(weight, min=-2 * std, max=2 * std)

    def _resize_padding_mask(
        self, padding_mask: torch.BoolTensor, target_length: int, dtype: torch.dtype
    ) -> torch.BoolTensor:
        padding_mask = (
            F.interpolate(
                padding_mask.unsqueeze(1).to(dtype),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            > 0.5  # Use threshold instead of .bool()
        )
        return padding_mask

    def get_kl_cosine_schedule(self, step):
        """
        Returns the scaled KL loss weight following a cosine schedule
        ranging from 0 to 1 over total_steps.
        Once step surpasses total_steps, stays at 1.
        """
        if (
            self.config.kl_loss_warmup_steps is None
            or self.config.kl_loss_warmup_steps == 0
        ):
            return 1.0
        if step >= self.config.kl_loss_warmup_steps:
            return 1.0
        # Cosine schedule: start at 0, increase to kl_loss_weight in total_steps
        cosine = 0.5 * (1 - math.cos(math.pi * step / self.config.kl_loss_warmup_steps))
        return cosine
