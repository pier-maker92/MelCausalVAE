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
from modules.similarity import SimilarityPoolingBatch


@dataclass
class ConvformerOutput:
    z: torch.FloatTensor
    kl_loss: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None
    mu: Optional[torch.FloatTensor] = None
    semantic_loss: Optional[torch.FloatTensor] = None
    semantic_features: Optional[torch.FloatTensor] = None
    durations: Optional[torch.LongTensor] = None
    z_pooled_fps: Optional[torch.FloatTensor] = None


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
    use_bigvgan_mel: bool = False
    use_1d_encoder: bool = False
    use_similarity: bool = False
    similarity_threshold: float = 0.9
    similarity_threshold_in_01: bool = False
    freeze_encoder_before_latent_heads: bool = False
    d_model: int = 512
    # Latent ablations (independent toggles): chunk size 2, linear schedule along chunk index.
    latent_chunk_ablate_dropout: bool = False
    latent_chunk_ablate_kl: bool = False
    latent_chunk_size: int = 2
    latent_chunk_dropout_start: float = 0.0
    latent_chunk_dropout_end: float = 0.8
    # True: one random cutoff, drop chunks s..end (chunk 0 never dropped).
    # False: each chunk dropped independently with linear p_i across chunk index.
    latent_chunk_dropout_hierarchical: bool = True
    latent_chunk_kl_weight_start: float = 1e-10
    latent_chunk_kl_weight_end: float = 1e-2


def _assert_latent_chunks_divisible(latent_dim: int, chunk_size: int) -> None:
    assert latent_dim % chunk_size == 0, (
        f"latent_dim ({latent_dim}) must be divisible by latent_chunk_size ({chunk_size})"
    )


def _latent_chunk_dropout_probs_per_chunk(
    *,
    num_chunks: int,
    start_p: float,
    end_p: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-chunk dropout probability (independent mode), linear from start to end."""
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if num_chunks == 1:
        return torch.tensor([start_p], device=device, dtype=dtype)
    return torch.linspace(start_p, end_p, num_chunks, device=device, dtype=dtype)


def apply_latent_chunk_dropout(
    z: torch.Tensor,
    *,
    chunk_size: int,
    dropout_start: float,
    dropout_end: float,
    training: bool,
    hierarchical: bool = True,
) -> torch.Tensor:
    """
    Latent chunk dropout on ``z`` [B, T, latent_dim].

    If ``hierarchical`` (default): chunk 0 never dropped; with probability
    ``dropout_end`` a single cutoff ``s`` is sampled (weights linear from
    ``dropout_start`` at ``s=1`` to ``dropout_end`` at the last start index);
    chunks ``s..n-1`` are zeroed. Same mask across ``T`` per batch row.

    If not ``hierarchical``: each chunk is dropped independently with
    probability ``p_i`` linear in chunk index (``dropout_start`` .. ``dropout_end``).
    """
    if not training:
        return z
    B, T, D = z.shape
    _assert_latent_chunks_divisible(D, chunk_size)
    n_chunks = D // chunk_size

    if not hierarchical:
        probs = _latent_chunk_dropout_probs_per_chunk(
            num_chunks=n_chunks,
            start_p=dropout_start,
            end_p=dropout_end,
            device=z.device,
            dtype=z.dtype,
        )
        zv = z.view(B, T, n_chunks, chunk_size)
        u = torch.rand(B, 1, n_chunks, 1, device=z.device, dtype=z.dtype)
        keep = u >= probs.view(1, 1, n_chunks, 1)
        return (zv * keep).view(B, T, D)

    if n_chunks <= 1:
        return z

    device = z.device
    # Possible start indices s = 1 .. n_chunks-1 (chunk 0 always kept)
    n_starts = n_chunks - 1
    if n_starts == 1:
        w = torch.tensor([1.0], device=device, dtype=torch.float32)
    else:
        # s = 1 + k for k = 0 .. n_starts-1; weight linear in k
        k = torch.arange(n_starts, device=device, dtype=torch.float32)
        t = k / max(n_starts - 1, 1)
        w = dropout_start + (dropout_end - dropout_start) * t
        w = torch.clamp(w, min=0.0)

    w_sum = float(w.sum().item())
    if w_sum <= 1e-12:
        return z

    probs = (w / w.sum()).expand(B, -1)
    starts_rel = torch.multinomial(probs, num_samples=1).squeeze(-1)
    start_chunk = starts_rel + 1

    u = torch.rand(B, device=device, dtype=torch.float32)
    active = u < float(dropout_end)
    start_chunk = torch.where(
        active,
        start_chunk,
        torch.full_like(start_chunk, n_chunks),
    )

    chunk_idx = torch.arange(n_chunks, device=device).view(1, n_chunks).expand(B, -1)
    keep = chunk_idx < start_chunk.unsqueeze(1)

    zv = z.view(B, T, n_chunks, chunk_size)
    keep_4 = keep.unsqueeze(1).unsqueeze(-1).to(dtype=zv.dtype)
    return (zv * keep_4).view(B, T, D)


def latent_chunk_kl_channel_weights(
    *,
    latent_dim: int,
    chunk_size: int,
    weight_start: float,
    weight_end: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """[latent_dim] weights (same for both channels in each chunk), linear across chunks."""
    _assert_latent_chunks_divisible(latent_dim, chunk_size)
    n_chunks = latent_dim // chunk_size
    if n_chunks == 1:
        chunk_w = torch.tensor([weight_start], device=device, dtype=dtype)
    else:
        chunk_w = torch.linspace(
            weight_start, weight_end, n_chunks, device=device, dtype=dtype
        )
    return chunk_w.repeat_interleave(chunk_size)


class SigmaVAEEncoder(nn.Module):
    def __init__(self, config: SigmaVAEencoderConfig):
        super().__init__()
        self.config = config
        self.std_activation = (
            nn.Softplus() if self.config.use_sofplus else nn.Identity()
        )
        self.kl_loss_weight = float(config.kl_loss_weight)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.semantic_regulator = InterpolateRegulator(
            depth=2, in_channels=1024, channels=256, out_channels=config.latent_dim
        )

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
        return kl.to(mu.dtype)

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

    def sample_scalar_std(self, mu: torch.FloatTensor) -> torch.FloatTensor:
        return self.std_activation(
            torch.randn(mu.shape[0], mu.shape[1], dtype=mu.dtype, device=mu.device)
            * self.config.target_std
        )

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
        super().__init__(
            c_in, c_out, (kt, kf), dilation=(dt, df), stride=(st, sf), padding=0
        )
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
    def __init__(
        self,
        c_in,
        c_out,
        *,
        kt=3,
        kf=7,
        dt=1,
        df=1,
        st=1,
        sf=1,
        act=nn.GELU,
        drop_p=0.1,
    ):
        super().__init__()
        self.ln = ChannelLastLayerNorm(c_in)
        self.act = act()
        self.main = TimeCausalConv2d(
            c_in, c_out, kt=kt, kf=kf, dt=dt, df=df, st=st, sf=sf
        )
        shape_changes = (c_in != c_out) or (st != 1) or (sf != 1)
        if shape_changes:
            # Match spatial sizes when stride or non-same padding is used
            self.skip = TimeCausalConv2d(
                c_in, c_out, kt=1, kf=1, dt=1, df=1, st=st, sf=sf
            )
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
                PreNormResCausalBlock(
                    c_in, c_in, kt=3, kf=5, dt=dilation, df=1, st=1, sf=1, drop_p=drop_p
                )
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
        self.enc = FlashTransformerEncoder(
            d_model=d_model, nhead=nheads, nlayers=nlayers, drop_p=drop_p
        )

    def forward(self, tokens):  # [B, T_tok, d_model]
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

        assert (
            compress_factor_C >= 1
            and (compress_factor_C & (compress_factor_C - 1)) == 0
        ), "C must be power of 2"
        self.C = compress_factor_C
        self.in_freq_proj = nn.Linear(100, 100)

        # Input projection: [B, T, 100] -> [B, 32, T, 100]
        self.in_proj = TimeCausalConv2d(1, 32, kt=3, kf=5, dt=1, df=1, st=1, sf=1)

        # Three causal blocks with dilations 1,2,4; last block also projects frequency -> 64
        self.freq_mixer = nn.Sequential(
            PreNormResCausalBlock(
                32, 64, kt=7, kf=7, dt=1, df=1, st=1, sf=1, drop_p=drop_p
            ),
            PreNormResCausalBlock(
                64, 128, kt=5, kf=5, dt=1, df=4, st=1, sf=1, drop_p=drop_p
            ),
            PreNormResCausalBlock(
                128, 256, kt=3, kf=3, dt=1, df=8, st=1, sf=1, drop_p=drop_p
            ),
        )

        # Two downsamplers:
        # ds2: time /2, freq /4, channels 512 -> 256
        self.downsampling = nn.ModuleDict(
            {
                "downsample@2": CausalDownsamplingBlock(
                    256, 512, n_residual_blocks=n_residual_blocks, drop_p=drop_p
                ),
                "downsample@4": CausalDownsamplingBlock(
                    512, 512, n_residual_blocks=n_residual_blocks, drop_p=drop_p
                ),
            }
        )
        # Extra temporal downsampling to reach T/C (sf=1), keeping channels at 512
        extra_stages = (
            int(math.log2(self.C)) - 2
        )  # we already did /4 in time via ds2 and ds4
        for i in range(max(0, extra_stages)):
            self.downsampling[f"downsample@{2**(i+2+1)}"] = CausalDownsamplingBlock(
                512,
                512,
                n_residual_blocks=n_residual_blocks,
                compress_freq=1,
                drop_p=drop_p,
            )

        # Collapse frequency 4 -> 1 with a single causal conv, keep channels at 512
        # Use kf=4, sf=4, no frequency padding so F_out = 1 exactly when F_in = 4
        self.freq_collapse = TimeCausalConv2d(
            512, 512, kt=1, kf=8, dt=1, df=1, st=1, sf=8
        )

        # Causal Transformer tail operating on tokens of size 512
        self.transformer = CausalTransformerTail(
            d_model=512, nheads=tf_heads, nlayers=tf_layers, drop_p=drop_p
        )

        self.mu = nn.Linear(512, latent_dim)
        if config.logvar_layer:
            self.logvar = nn.Linear(512, latent_dim)
        self.similarity_pooler = (
            SimilarityPoolingBatch(
                threshold=config.similarity_threshold,
                threshold_in_01=config.similarity_threshold_in_01,
            )
            if config.use_similarity
            else None
        )
        if config.latent_chunk_ablate_dropout or config.latent_chunk_ablate_kl:
            _assert_latent_chunks_divisible(config.latent_dim, config.latent_chunk_size)
        if config.freeze_encoder_before_latent_heads:
            self._freeze_encoder_before_latent_heads()

    def _freeze_encoder_before_latent_heads(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mu.parameters():
            param.requires_grad = True
        if hasattr(self, "logvar"):
            for param in self.logvar.parameters():
                param.requires_grad = True

    def forward(
        self, x: torch.FloatTensor, padding_mask: torch.BoolTensor = None, **kwargs
    ):  # x: [B, T, 100]
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
        latent_padding_mask = (
            self._resize_padding_mask(padding_mask, z.shape[1], dtype=z.dtype)
            if padding_mask is not None
            else torch.zeros(
                (z.shape[0], z.shape[1]), device=z.device, dtype=torch.bool
            )
        )
        durations = None
        z_pooled_fps = None
        if self.similarity_pooler is not None:
            z, durations, pooled_mask = self.similarity_pooler(z, latent_padding_mask.long())
            latent_padding_mask = pooled_mask.bool()
            valid_durs = durations[durations > 0].float()
            if valid_durs.numel() > 0:
                latent_fps = 93.75 / float(self.C)
                z_pooled_fps = torch.tensor(
                    latent_fps, device=z.device, dtype=z.dtype
                ) / valid_durs.mean()

        mu = self.mu(z)
        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(z)

        z = self.reparameterize(mu, logvar)

        if self.config.latent_chunk_ablate_dropout:
            z = apply_latent_chunk_dropout(
                z,
                chunk_size=self.config.latent_chunk_size,
                dropout_start=self.config.latent_chunk_dropout_start,
                dropout_end=self.config.latent_chunk_dropout_end,
                training=self.training,
                hierarchical=self.config.latent_chunk_dropout_hierarchical,
            )

        semantic_loss = None
        if kwargs.get("semantic_guidance", None) is not None:
            raise NotImplementedError("Semantic guidance is not implemented yet")
            # semantic_loss = self.semantic_regulator(
            #     target=z,
            #     guidance=kwargs["semantic_guidance"].semantic_features,
            #     guidance_mask=kwargs["semantic_guidance"].padding_mask,
            #     target_padding_mask=self._resize_padding_mask(padding_mask, mu.shape[1]),
            # )

        kl_loss = None
        if kwargs.get("step", None) is not None:
            if self.config.latent_chunk_ablate_kl:
                ch_w = latent_chunk_kl_channel_weights(
                    latent_dim=self.config.latent_dim,
                    chunk_size=self.config.latent_chunk_size,
                    weight_start=self.config.latent_chunk_kl_weight_start,
                    weight_end=self.config.latent_chunk_kl_weight_end,
                    device=z.device,
                    dtype=torch.float32,
                )
                kl_term = self.kl_divergence_weighted(
                    mu,
                    logvar,
                    latent_padding_mask,
                    dtype=torch.float32,
                    channel_weights=ch_w,
                )
            else:
                kl_term = self.kl_divergence(
                    mu,
                    logvar,
                    latent_padding_mask,
                    dtype=z.dtype,
                )
            kl_loss = kl_term * self.get_kl_cosine_schedule(kwargs["step"])

        return ConvformerOutput(
            z=z,
            kl_loss=kl_loss,
            padding_mask=latent_padding_mask,
            mu=mu,
            semantic_loss=semantic_loss,
            durations=durations,
            z_pooled_fps=z_pooled_fps,
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
