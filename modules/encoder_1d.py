import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from modules.Encoder import (
    SigmaVAEEncoder,
    ConvformerEncoderConfig,
    ConvformerOutput,
    CausalTransformerTail,
)


# ---------- 1D building blocks ----------


class TimeCausalConv1d(nn.Conv1d):
    """Causal Conv1d: left-pad only so output at time t depends only on inputs <= t."""

    def __init__(self, c_in, c_out, k, d=1, s=1):
        super().__init__(c_in, c_out, k, dilation=d, stride=s, padding=0)
        self.k, self.d, self.s = k, d, s

    def forward(self, x):  # x: [B, C, T]
        pad_left = (self.k - 1) * self.d
        x = F.pad(x, (pad_left, 0))
        return super().forward(x)


class PreNormResCausalBlock1d(nn.Module):
    """GroupNorm(1) -> GELU -> CausalConv1d -> Dropout + skip."""

    def __init__(self, c_in, c_out, *, k=3, d=1, s=1, drop_p=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, c_in)
        self.act = nn.GELU()
        self.main = TimeCausalConv1d(c_in, c_out, k=k, d=d, s=s)
        if c_in != c_out or s != 1:
            self.skip = TimeCausalConv1d(c_in, c_out, k=1, d=1, s=s)
        else:
            self.skip = nn.Identity()
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):  # x: [B, C, T]
        h = self.act(self.norm(x))
        h = self.dropout(h)
        return self.main(h) + self.skip(x)


class CausalDownsamplingBlock1d(nn.Module):
    """Dilated residual blocks followed by a stride-2 downsampler."""

    def __init__(self, c_in, c_out, n_residual_blocks=3, drop_p=0.1):
        super().__init__()
        dilations = [1, 2, 4, 8]
        self.residual_blocks = nn.ModuleList(
            [
                PreNormResCausalBlock1d(c_in, c_in, k=5, d=dilation, drop_p=drop_p)
                for dilation in dilations[:n_residual_blocks]
            ]
        )
        self.downsampling = PreNormResCausalBlock1d(
            c_in, c_out, k=5, d=1, s=2, drop_p=drop_p
        )

    def forward(self, x):  # x: [B, C, T]
        for block in self.residual_blocks:
            x = block(x)
        return self.downsampling(x)


# ---------- full 1D encoder ----------


class ConvformerEncoder1d(SigmaVAEEncoder):
    """
    1D convolutional encoder: treats 100 mel bins as input channels and uses
    only temporal (causal) Conv1d operations. Drop-in replacement for the 2D
    ConvformerEncoder with the same ConvformerEncoderConfig.
    """

    def __init__(self, config: ConvformerEncoderConfig):
        super().__init__(config)

        compress_factor_C = config.compress_factor_C
        tf_heads = config.tf_heads
        tf_layers = config.tf_layers
        drop_p = config.drop_p
        latent_dim = config.latent_dim
        n_residual_blocks = config.n_residual_blocks
        d_model = config.d_model

        assert (
            compress_factor_C >= 1
            and (compress_factor_C & (compress_factor_C - 1)) == 0
        ), "C must be power of 2"
        self.C = compress_factor_C

        # Input projection: [B, 100, T] -> [B, 256, T]
        self.in_proj = TimeCausalConv1d(100, d_model // 2, k=7)

        # Mixer: dilated causal blocks with increasing channels
        self.mixer = nn.Sequential(
            PreNormResCausalBlock1d(d_model // 2, d_model, k=7, d=1, drop_p=drop_p),
            PreNormResCausalBlock1d(d_model, d_model, k=5, d=2, drop_p=drop_p),
            PreNormResCausalBlock1d(d_model, d_model, k=3, d=4, drop_p=drop_p),
        )

        # Temporal downsampling: log2(C) stages of stride-2
        num_stages = int(math.log2(compress_factor_C))
        self.downsampling = nn.ModuleDict()
        for i in range(num_stages):
            factor = 2 ** (i + 1)
            self.downsampling[f"downsample@{factor}"] = CausalDownsamplingBlock1d(
                d_model, d_model, n_residual_blocks=n_residual_blocks, drop_p=drop_p
            )

        # Causal Transformer tail
        self.transformer = CausalTransformerTail(
            d_model=d_model, nheads=tf_heads, nlayers=tf_layers, drop_p=drop_p
        )

        self.mu = nn.Linear(d_model, latent_dim)
        if config.logvar_layer:
            self.logvar = nn.Linear(d_model, latent_dim)

    def forward(
        self,
        x: torch.FloatTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        # x: [B, T, 100]
        x = x.transpose(1, 2)  # [B, 100, T]
        x = self.in_proj(x)  # [B, d_model//2, T]
        x = self.mixer(x)  # [B, d_model, T]

        for layer in self.downsampling.values():
            x = layer(x)  # [B, d_model, T/C]

        hiddens = x.transpose(1, 2)  # [B, T/C, 512]
        z = self.transformer(hiddens)  # [B, T/C, 512]

        mu = self.mu(z)
        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(z)
        z = self.reparameterize(mu, logvar)

        semantic_loss = None
        if kwargs.get("semantic_guidance", None) is not None:
            raise NotImplementedError("Semantic guidance is not implemented yet")

        kl_loss = None
        if kwargs.get("step", None) is not None:
            kl_loss = self.kl_divergence(
                mu,
                logvar,
                self._resize_padding_mask(padding_mask, mu.shape[1], dtype=z.dtype),
                dtype=z.dtype,
            ) * self.get_kl_cosine_schedule(kwargs["step"])

        return ConvformerOutput(
            z=z,
            kl_loss=kl_loss,
            padding_mask=self._resize_padding_mask(
                padding_mask, mu.shape[1], dtype=z.dtype
            ),
            mu=mu,
            semantic_loss=semantic_loss,
        )


# ---------- causality test ----------


@torch.no_grad()
def forward_latent_1d(model, x):
    model.eval()
    return model(x)


def test_time_causality_invariance_1d(model):
    torch.manual_seed(0)
    B, T, F = 1, 64, 100

    x = torch.randn(B, T, F)
    y_ref = forward_latent_1d(model, x)

    for j in range(1, y_ref.z.shape[1]):
        t_max = j * model.config.compress_factor_C
        x_pert = x.clone()
        if t_max < T:
            x_pert[:, t_max:, :] = torch.randn_like(x_pert[:, t_max:, :]) * 5.0
        y_pert = forward_latent_1d(model, x_pert)

        ok = torch.allclose(
            y_pert.z[:, j - 1, :], y_ref.z[:, j - 1, :], atol=1e-7, rtol=0
        )
        assert ok, f"Causality violated at output index {j}"
    print("Causality test passed!")


if __name__ == "__main__":
    config = ConvformerEncoderConfig(
        compress_factor_C=8,
        tf_heads=8,
        tf_layers=4,
        drop_p=0.1,
        latent_dim=64,
        n_residual_blocks=3,
    )
    model = ConvformerEncoder1d(config=config)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    test_time_causality_invariance_1d(model)
