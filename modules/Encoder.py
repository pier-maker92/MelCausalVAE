import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
import torch.nn as nn
from einops import rearrange
from dataclasses import dataclass
from typing import Optional, List, Tuple
from modules.Tadastride.resnet import ResNet
from modules.flash_attn_encoder import FlashTransformerEncoder
from modules.Tadastride.alignement import extract_durations


# FIXME these classes should be pruned from useless fields
@dataclass
class ConvformerOutput:
    z: torch.FloatTensor
    kl_loss: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None
    mu: Optional[torch.FloatTensor] = None
    align_list: Optional[List[torch.FloatTensor]] = None
    align_loss: Optional[torch.FloatTensor] = None
    durations: Optional[torch.LongTensor] = None
    z_lengths: Optional[torch.LongTensor] = None


@dataclass
class SigmaVAEencoderConfig:
    logvar_layer: bool = True
    kl_loss_weight: float = 1e-3
    target_std: Optional[float] = None
    use_sofplus: Optional[bool] = None
    kl_loss_warmup_steps: int = 1000
    semantic_regulation: bool = True


@dataclass
class ConvformerEncoderConfig(SigmaVAEencoderConfig):
    compress_factor_C: int = 8
    tf_heads: int = 8
    tf_layers: int = 4
    drop_p: float = 0.1
    latent_dim: int = 64
    n_residual_blocks: int = 3
    hidden_dim: int = 512


class SigmaVAEEncoder(nn.Module):
    def __init__(self, config: SigmaVAEencoderConfig):
        super().__init__()
        self.config = config
        self.std_activation = nn.Softplus() if self.config.use_sofplus else nn.Identity()
        self.kl_loss_weight = float(config.kl_loss_weight)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.semantic_regulator = InterpolateRegulator(
        #     depth=2, in_channels=1024, channels=256, out_channels=config.latent_dim
        # )

    def forward(self, **kwargs):
        pass

    def reparameterize(self, mu: torch.FloatTensor, logvar: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        eps = torch.randn_like(mu)
        if logvar is None:
            std = self.sample_scalar_std(mu)
            while std.dim() < mu.dim():
                std = std.unsqueeze(-1)
        else:
            std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def kl_divergence(
        self, mu: torch.FloatTensor, logvar: Optional[torch.FloatTensor], padding_mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        if logvar is None:
            # Compute in fp32 for numerical stability
            mu_valid = mu[~padding_mask].float()
            return torch.nn.functional.mse_loss(mu_valid, torch.zeros_like(mu_valid)).to(mu.dtype)
        # Compute KL divergence in fp32 for numerical stability with fp16
        mu_valid = mu[~padding_mask].float()
        logvar_valid = logvar[~padding_mask].float()
        kl = -0.5 * torch.sum(1 + logvar_valid - mu_valid.pow(2) - logvar_valid.exp())
        return kl.to(mu.dtype)

    def sample_scalar_std(self, mu: torch.FloatTensor) -> torch.FloatTensor:
        return self.std_activation(
            torch.randn(mu.shape[0], mu.shape[1], dtype=mu.dtype, device=mu.device) * self.config.target_std
        )

    def _resize_padding_mask(self, padding_mask: torch.BoolTensor, target_length: int) -> torch.BoolTensor:
        padding_mask = (
            torch.nn.functional.interpolate(
                padding_mask.unsqueeze(1).float(),
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
        ranging from 0 to self.kl_loss_weight over total_steps.
        Once step surpasses total_steps, stays at self.kl_loss_weight.
        """
        if self.config.kl_loss_warmup_steps == 0:
            return self.kl_loss_weight
        if step >= self.config.kl_loss_warmup_steps:
            return self.kl_loss_weight
        # Cosine schedule: start at 0, increase to kl_loss_weight in total_steps
        cosine = 0.5 * (1 - math.cos(math.pi * step / self.config.kl_loss_warmup_steps))
        return self.kl_loss_weight * cosine


# ---------- causal Transformer tail ----------
class CausalTransformerTail(nn.Module):
    def __init__(self, d_model=512, nheads=8, nlayers=4, drop_p=0.1):
        super().__init__()
        self.enc = FlashTransformerEncoder(d_model=d_model, nhead=nheads, nlayers=nlayers, drop_p=drop_p)

    def forward(self, tokens):  # [B, T_tok, d_model]
        return self.enc(tokens, causal=True)


# FIXME hardcoded parameters for Tadastride ResNet
class hparams:
    hidden_dim = 512
    kernel_size = 3
    sigma_square = 5.0
    ver_f = False
    n_mel_channels = 100
    causal = True


class ConvformerEncoder(SigmaVAEEncoder):
    def __init__(self, config: ConvformerEncoderConfig):
        super().__init__(config)

        compress_factor_C = config.compress_factor_C
        if compress_factor_C != 8:
            raise NotImplementedError("Only compress_factor_C = 8 is supported for now")
        tf_heads = config.tf_heads
        tf_layers = config.tf_layers
        drop_p = config.drop_p
        latent_dim = config.latent_dim
        n_residual_blocks = config.n_residual_blocks
        assert (compress_factor_C & (compress_factor_C - 1)) == 0, "C must be power of 2"
        # Tadastride ResNet
        hp = hparams()
        hp.hidden_dim = config.hidden_dim
        self.resnet = ResNet(hp)

        # Causal Transformer tail operating on tokens of size 512
        self.transformer = CausalTransformerTail(
            d_model=config.hidden_dim, nheads=tf_heads, nlayers=tf_layers, drop_p=drop_p
        )

        self.mu = nn.Linear(config.hidden_dim, latent_dim)
        if config.logvar_layer:
            self.logvar = nn.Linear(config.hidden_dim, latent_dim)

    def forward(self, x: torch.FloatTensor, padding_mask: torch.BoolTensor = None, **kwargs):
        x = rearrange(x, "b t c -> b c t")  # [B, T, 100] -> [B, 100, T] time in the last dimension
        x, x_mask, alignment_list, score_loss_total = self.resnet(
            x=x, x_lengths=(~padding_mask).sum(dim=-1)
        )  # padding mask is False where the audio is valid, this is why the ~ operation is used
        x = rearrange(x, "b c t -> b t c")  # [B, C, T] -> [B, T, C] hidden features in the last dimension
        x = self.transformer(x)  # [B, T, C] -> [B, T, C]
        # reparameterize the latent features
        mu = self.mu(x)
        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(z)
        z = self.reparameterize(mu, logvar)
        kl_loss = None
        if kwargs.get("step", None) is not None:
            kl_loss = self.kl_divergence(mu, logvar, padding_mask=~x_mask) * self.get_kl_cosine_schedule(
                kwargs["step"]
            )

        return ConvformerOutput(
            z=z,
            mu=mu,
            align_list=alignment_list,
            align_loss=score_loss_total,
            padding_mask=~x_mask,
            kl_loss=kl_loss,
        )

    @torch.no_grad()
    def encode(self, x: torch.FloatTensor, padding_mask: torch.BoolTensor = None):
        x = rearrange(x, "b t c -> b c t")
        x, z_mask, durations, _, _, _, z_lengths = extract_durations(self.resnet, x, (~padding_mask).sum(dim=-1))
        x = rearrange(x, "b c t -> b t c")
        z = self.transformer(x)
        mu = self.mu(z)
        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(z)
        z = self.reparameterize(mu, logvar)

        return ConvformerOutput(
            z=z,
            durations=durations,
            z_lengths=z_lengths,
            padding_mask=~z_mask,
        )
