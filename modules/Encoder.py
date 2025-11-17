import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
import torch.nn.functional as F
from dataclasses import dataclass
from modules.semantic_module import SeamlessM4Tv2Encoder
from modules.flash_attn_encoder import FlashTransformerEncoder
from modules.regulator import InterpolateRegulator


@dataclass
class ConvformerOutput:
    z: torch.FloatTensor
    kl_loss: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None
    µ: Optional[torch.FloatTensor] = None
    semantic_loss: Optional[torch.FloatTensor] = None


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


class SigmaVAEEncoder(nn.Module):
    def __init__(self, config: SigmaVAEencoderConfig):
        super().__init__()
        self.config = config
        self.std_activation = nn.Softplus() if self.config.use_sofplus else nn.Identity()
        self.kl_loss_weight = float(config.kl_loss_weight)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.semantic_regulator = InterpolateRegulator(
            depth=2, in_channels=1024, channels=256, out_channels=config.latent_dim
        )

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
            return F.mse_loss(mu_valid, torch.zeros_like(mu_valid)).to(mu.dtype)
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
            F.interpolate(
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


# ---------- utils ----------


class ChannelLastLayerNorm(nn.Module):
    # LayerNorm over channels for tensors in [B, C, T, F]
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):  # x: [B, C, T, F]
        x = rearrange(x, "b c t f -> b t f c")  # [B, T, F, C]
        x = self.ln(x)  # normalize over C
        return rearrange(x, "b t f c -> b c t f")  # [B, C, T, F]


class TimeCausalConv2d(nn.Conv2d):
    # Causal in time (left pad only); "same" in frequency (symmetric left/right)
    def __init__(self, c_in, c_out, kt, kf, dt=1, df=1, st=1, sf=1):
        super().__init__(c_in, c_out, (kt, kf), dilation=(dt, df), stride=(st, sf), padding=0)
        self.kt, self.kf, self.dt, self.df, self.st, self.sf = kt, kf, dt, df, st, sf

    def _same_freq_pad(self, F_in):
        # Solve for total pad so L_out = ceil(F_in / sf)
        # L_out = floor((F_in + p - df*(kf-1) - 1)/sf + 1)
        target = (F_in + self.sf - 1) // self.sf
        p_total = max(0, (target - 1) * self.sf + self.df * (self.kf - 1) + 1 - F_in)
        left = p_total // 2
        right = p_total - left
        return left, right

    def forward(self, x):  # x: [B, C, T, F]
        t_left = (self.kt - 1) * self.dt  # causal time pad (left only)
        f_left, f_right = self._same_freq_pad(x.shape[-1])
        x = F.pad(x, (f_left, f_right, t_left, 0))
        return super().forward(x)


class ChannelLastLinear(nn.Module):
    # Linear over frequency dimension F: [B, C, T, F_in] -> [B, C, T, F_out]
    def __init__(self, f_in, f_out):
        super().__init__()
        self.lin = nn.Linear(f_in, f_out)

    def forward(self, x):
        B, C, T, F_in = x.shape
        x = rearrange(x, "b c t f -> b t c f")
        x = self.lin(x)  # [B, T, C, F_out]
        return rearrange(x, "b t c f -> b c t f")  # [B, C, T, F_out]


class PreNormResCausalBlock(nn.Module):
    # LN -> act -> causal conv; skip mirrors stride if T or F changes, else 1x1
    def __init__(self, c_in, c_out, *, kt=3, kf=7, dt=1, df=1, st=1, sf=1, act=nn.GELU, drop_p=0.1):
        super().__init__()
        self.ln = ChannelLastLayerNorm(c_in)
        self.act = act()
        self.main = TimeCausalConv2d(c_in, c_out, kt=kt, kf=kf, dt=dt, df=df, st=st, sf=sf)
        shape_changes = (c_in != c_out) or (st != 1) or (sf != 1)
        if shape_changes:
            # Match spatial sizes when stride or non-same padding is used
            self.skip = TimeCausalConv2d(c_in, c_out, kt=1, kf=1, dt=1, df=1, st=st, sf=sf)
        else:
            if c_in != c_out:
                self.skip = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1)
            else:
                self.skip = nn.Identity()
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        h = self.act(self.ln(x))
        h = self.dropout(h)
        return self.main(h) + self.skip(x)


class CausalDownsamplingBlock(nn.Module):
    def __init__(self, c_in, c_out, n_residual_blocks=2, compress_freq=4, drop_p=0.1):
        super().__init__()
        # residual blocks
        dilations = [1, 2, 4, 8]
        self.residual_blocks = nn.ModuleList(
            [
                PreNormResCausalBlock(c_in, c_in, kt=3, kf=5, dt=dilation, df=1, st=1, sf=1, drop_p=drop_p)
                for dilation in dilations[:n_residual_blocks]
            ]
        )
        # downsampler
        self.downsampling = PreNormResCausalBlock(
            c_in, c_out, kt=5, kf=7, dt=1, df=1, st=2, sf=compress_freq, drop_p=drop_p
        )

    def forward(self, x):
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return self.downsampling(x)


# ---------- causal Transformer tail ----------


class CausalTransformerTail(nn.Module):
    def __init__(self, d_model=512, nheads=8, nlayers=4, drop_p=0.1):
        super().__init__()
        # layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=nheads,
        #     dim_feedforward=d_model * 4,
        #     dropout=drop_p,
        #     batch_first=True,
        #     norm_first=True,
        # )
        # self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.enc = FlashTransformerEncoder(d_model=d_model, nhead=nheads, nlayers=nlayers, drop_p=drop_p)

    def forward(self, tokens):  # [B, T_tok, d_model]
        # L = tokens.size(1)
        # causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1)
        return self.enc(tokens, causal=True)


# TransformerEncoder with causal masking via is_causal for left-only attention [web:84][web:92]

# ---------- full model as specified ----------


class ConvformerEncoder(SigmaVAEEncoder):
    """
    Spec:
      - Start with 3 causal conv2d blocks (dilations 1,2,4) that reach 512 channels and reduce frequency to 64.
      - Then two downsamplers: ds2 (time/2, freq/4) and ds4 (time/4, freq/4), with channels 256 then 512.
      - Then extra temporal downsampling to reach T/C with sf=1, keeping channels at 512.
      - Collapse frequency from 4 -> 1 with a causal conv (no pooling), channels remain 512.
      - Finally a causal Transformer over tokens of size 512.
    """

    def __init__(self, config: ConvformerEncoderConfig):
        super().__init__(config)

        compress_factor_C = config.compress_factor_C
        tf_heads = config.tf_heads
        tf_layers = config.tf_layers
        drop_p = config.drop_p
        latent_dim = config.latent_dim
        n_residual_blocks = config.n_residual_blocks

        assert compress_factor_C >= 1 and (compress_factor_C & (compress_factor_C - 1)) == 0, "C must be power of 2"
        self.C = compress_factor_C
        self.in_freq_proj = nn.Linear(100, 100)

        # Input projection: [B, T, 100] -> [B, 32, T, 100]
        self.in_proj = TimeCausalConv2d(1, 32, kt=3, kf=5, dt=1, df=1, st=1, sf=1)

        # Three causal blocks with dilations 1,2,4; last block also projects frequency -> 64
        self.freq_mixer = nn.Sequential(
            PreNormResCausalBlock(32, 64, kt=7, kf=7, dt=1, df=1, st=1, sf=1, drop_p=drop_p),
            PreNormResCausalBlock(64, 128, kt=5, kf=5, dt=1, df=4, st=1, sf=1, drop_p=drop_p),
            PreNormResCausalBlock(128, 256, kt=3, kf=3, dt=1, df=8, st=1, sf=1, drop_p=drop_p),
        )

        # Two downsamplers:
        # ds2: time /2, freq /4, channels 512 -> 256
        self.downsampling = nn.ModuleDict(
            {
                "downsample@2": CausalDownsamplingBlock(256, 512, n_residual_blocks=n_residual_blocks, drop_p=drop_p),
                "downsample@4": CausalDownsamplingBlock(512, 512, n_residual_blocks=n_residual_blocks, drop_p=drop_p),
            }
        )
        # Extra temporal downsampling to reach T/C (sf=1), keeping channels at 512
        extra_stages = int(math.log2(self.C)) - 2  # we already did /4 in time via ds2 and ds4
        for i in range(max(0, extra_stages)):
            self.downsampling[f"downsample@{2**(i+2+1)}"] = CausalDownsamplingBlock(
                512, 512, n_residual_blocks=n_residual_blocks, compress_freq=1, drop_p=drop_p
            )

        # Collapse frequency 4 -> 1 with a single causal conv, keep channels at 512
        # Use kf=4, sf=4, no frequency padding so F_out = 1 exactly when F_in = 4
        self.freq_collapse = TimeCausalConv2d(512, 512, kt=1, kf=8, dt=1, df=1, st=1, sf=8)

        # Causal Transformer tail operating on tokens of size 512
        self.transformer = CausalTransformerTail(d_model=512, nheads=tf_heads, nlayers=tf_layers, drop_p=drop_p)

        self.mu = nn.Linear(512, latent_dim)
        if config.logvar_layer:
            self.logvar = nn.Linear(512, latent_dim)

    def forward(self, x: torch.FloatTensor, padding_mask: torch.BoolTensor = None, **kwargs):  # x: [B, T, 100]
        B, T, F = x.shape
        x = self.in_freq_proj(x)
        x = x.unsqueeze(1)  # [B, 1, T, 100]
        x = self.in_proj(x)  # [B, 32, T, 100]

        # Three dilated causal blocks; end with [B, 512, T, 64]
        x = self.freq_mixer(x)  # [B, 512, T, 64]

        # Downsampling blocks
        for layer in self.downsampling.values():
            x = layer(x)

        # Collapse frequency 4 -> 1 without changing channels
        x = self.freq_collapse(x).squeeze(-1)  # [B, 512, T/C]

        # Flatten freq to tokens and run causal Transformer
        hiddens = x.transpose(1, 2)  # [B, T/C, 512]
        z = self.transformer(hiddens)  # [B, T/C, 512]
        mu = self.mu(z)
        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(z)

        semantic_loss = None
        if kwargs.get("semantic_guidance", None) is not None:
            semantic_loss = self.semantic_regulator(
                target=mu,
                guidance=kwargs["semantic_guidance"].semantic_features,
                guidance_mask=kwargs["semantic_guidance"].padding_mask,
                target_padding_mask=self._resize_padding_mask(padding_mask, mu.shape[1]),
            )

        z = self.reparameterize(mu, logvar)
        kl_loss = None
        if kwargs.get("step", None) is not None:
            kl_loss = self.kl_divergence(
                mu, logvar, self._resize_padding_mask(padding_mask, mu.shape[1])
            ) * self.get_kl_cosine_schedule(kwargs["step"])

        return ConvformerOutput(
            z=z,
            kl_loss=kl_loss,
            padding_mask=self._resize_padding_mask(padding_mask, mu.shape[1]),
            µ=mu,
            semantic_loss=semantic_loss,
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
