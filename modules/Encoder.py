import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
import torch.nn as nn
from einops import rearrange
from dataclasses import dataclass
from typing import Optional, List


from modules.flash_attn_encoder import FlashTransformerEncoder
from modules.downsampler import DownSampler
from modules.aligner import Aligner


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
    num_embeddings: int = 100
    embedding_dim: int = 80
    phoneme_parsing_mode: str = "phoneme"
    vocab_path: str = "data/vocab.json"
    use_aligner: bool = False


class SigmaVAEEncoder(nn.Module):
    def __init__(self, config: SigmaVAEencoderConfig):
        super().__init__()
        self.config = config
        self.std_activation = (
            nn.Softplus() if self.config.use_sofplus else nn.Identity()
        )
        self.kl_loss_weight = float(config.kl_loss_weight)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.semantic_regulator = InterpolateRegulator(
        #     depth=2, in_channels=1024, channels=256, out_channels=config.latent_dim
        # )

    def forward(self, **kwargs):
        pass

    def reparameterize(
        self, mu: torch.FloatTensor, logvar: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        eps = torch.randn_like(mu)
        if logvar is None:
            std = self.sample_scalar_std(mu)
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
    ) -> torch.FloatTensor:
        if logvar is None:
            # Compute in fp32 for numerical stability
            mu_valid = mu[~padding_mask].float()
            return torch.nn.functional.mse_loss(
                mu_valid, torch.zeros_like(mu_valid)
            ).to(mu.dtype)
        # Compute KL divergence in fp32 for numerical stability with fp16
        mu_valid = mu[~padding_mask].float()
        logvar_valid = logvar[~padding_mask].float()
        kl = -0.5 * torch.sum(1 + logvar_valid - mu_valid.pow(2) - logvar_valid.exp())
        return kl.to(mu.dtype)

    def sample_scalar_std(self, mu: torch.FloatTensor) -> torch.FloatTensor:
        return self.std_activation(
            torch.randn(mu.shape[0], mu.shape[1], dtype=mu.dtype, device=mu.device)
            * self.config.target_std
        )

    def _resize_padding_mask(
        self, padding_mask: torch.BoolTensor, target_length: int
    ) -> torch.BoolTensor:
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


class CausalTransformerTail(nn.Module):
    def __init__(self, d_model=512, nheads=8, nlayers=4, drop_p=0.1):
        super().__init__()
        self.enc = FlashTransformerEncoder(
            d_model=d_model, nhead=nheads, nlayers=nlayers, drop_p=drop_p
        )

    def forward(self, tokens):  # [B, T_tok, d_model]
        return self.enc(tokens, causal=True)


# TransformerEncoder with causal masking via is_causal for left-only attention [web:84][web:92]


class ConvformerEncoder(SigmaVAEEncoder):
    def __init__(self, config: ConvformerEncoderConfig):
        super().__init__(config)

        compress_factor_C = config.compress_factor_C
        tf_heads = config.tf_heads
        tf_layers = config.tf_layers
        drop_p = config.drop_p
        latent_dim = config.latent_dim
        # n_residual_blocks = config.n_residual_blocks FIXME handle

        self.downsampler = DownSampler(
            d_in=100,
            d_hidden=512,
            d_out=512,
            compress_factor=compress_factor_C,
            causal=True,
        )

        # Causal Transformer tail operating on tokens of size 512
        self.transformer = CausalTransformerTail(
            d_model=512, nheads=tf_heads, nlayers=tf_layers, drop_p=drop_p
        )

        self.mu = nn.Linear(512, latent_dim)
        if config.logvar_layer:
            self.logvar = nn.Linear(512, latent_dim)

        if config.use_aligner:
            self.aligner = Aligner(
                attn_dim=2048,
                text_dim=config.embedding_dim,
                audio_dim=512,
                embedding_dim=config.embedding_dim,
                num_embeddings=config.num_embeddings,
                parsing_mode=config.phoneme_parsing_mode,
                vocab_path=config.vocab_path,
            )
        # print number of parameters

    def forward(
        self,
        x: torch.FloatTensor,
        phonemes: List[str],
        padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):  # x: [B, T, 100]

        x, padding_mask = self.downsampler(x, padding_mask)
        z = self.transformer(x)  # [B, T/C, 512]

        # get alignement
        align_loss, durations = None, None
        if self.config.use_aligner:
            z_aligned, durations, padding_mask_aligned, align_loss = self.aligner(
                mels=z,
                phonemes=phonemes,
                mels_mask=~padding_mask,  # NOTE: padding_mask is False for padding tokens, but aligner expects True for padding tokens
            )
            # z = z_aligned.to(x.dtype)
            #padding_mask = padding_mask_aligned
            assert not torch.isnan(z).any(), "z contains nan after aligner"

        mu = self.mu(z)
        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(z)
        z = self.reparameterize(mu, logvar)
        assert not torch.isnan(z).any(), "z contains nan after reparameterization"

        kl_loss = None
        if kwargs.get("step", None) is not None:
            kl_loss = self.kl_divergence(
                mu, logvar, padding_mask
            ) * self.get_kl_cosine_schedule(kwargs["step"])

        return ConvformerOutput(
            z=z,
            kl_loss=kl_loss,
            padding_mask=padding_mask,
            mu=mu,
            durations=durations,
            align_loss=align_loss,
        )


@torch.no_grad()
def forward_latent(model, x):
    model.eval()
    return model(x)  # returns mu: [B, T/C, latent_dim]


def test_time_causality_invariance(model):
    torch.manual_seed(0)
    B, T, F = 1, 64, 100

    x = torch.randn(B, T, F)
    y_ref = forward_latent(model, x)  # [B, T/C, latent_dim]

    for j in range(1, y_ref.shape[1]):
        t_max = j * model.config.compress_factor_C
        x_pert = x.clone()
        if t_max < T:
            x_pert[:, t_max:, :] = torch.randn_like(x_pert[:, t_max:, :]) * 5.0
        y_pert = forward_latent(model, x_pert)

        ok = torch.allclose(y_pert[:, j - 1, :], y_ref[:, j - 1, :], atol=1e-7, rtol=0)
        assert ok, f"Causality violated at output index {j}"


if __name__ == "__main__":
    config = ConvformerEncoderConfig(
        compress_factor_C=8,
        tf_heads=8,
        tf_layers=4,
        drop_p=0.1,
        latent_dim=64,
        n_residual_blocks=3,
    )
    model = ConvformerEncoder(config=config)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    test_time_causality_invariance(model)
    breakpoint()
    # print number of parameters
