"""
AlignmentQFormer
================
Q-Former whose cross-attention is hard-gated by the binary alignment matrix
produced by ``AlignmentMatrixBuilder``, followed by a weighted ``PhonemeQueryPooler``
that collapses the Q per-phoneme queries into a single vector via mean+std projection.

Masking rule
------------
Each phoneme segment ``n`` owns ``num_queries_per_phoneme`` (default 4)
learnable queries.  In cross-attention query ``n·Q + k`` may **only** attend
to mel frame ``t`` where::

    alignment[b, t, n] == 1

All other (query, frame) pairs receive an additive ``-inf`` mask, so they
contribute zero weight after softmax.
Pooling
-------
Weighted mean+std pooling using cross-attention weights:

    weights = attn_weights.unsqueeze(-1)           # (B, N, T, 1)
    mu = (weights * z_t).sum(dim=2)
    var = (weights * (z_t - mu.unsqueeze(2))**2).sum(dim=2)
    z_n = torch.cat([mu, torch.sqrt(var + 1e-5)], dim=-1)
    pooled = proj(z_n)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Sinusoidal positional encoding
# -------------------------


def sinusoidal_pos_emb(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=x.device) / max(half - 1, 1)
    )
    args = x.unsqueeze(-1) * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


class RelativePositionEncoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        assert d_model % 2 == 0
        self.proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_pos_emb(rel_pos, self.proj.in_features)
        return self.proj(emb.to(dtype=self.proj.weight.dtype))


def compute_relative_positions(alignment: torch.Tensor) -> torch.Tensor:
    B, T, N = alignment.shape
    phon_idx = alignment.argmax(dim=-1)
    cum = alignment.cumsum(dim=1)
    cum_frame = cum.gather(dim=2, index=phon_idx.unsqueeze(-1)).squeeze(-1).float()
    durations = alignment.sum(dim=1)
    dur_frame = durations.gather(dim=1, index=phon_idx).float()
    rel_pos = (cum_frame - 1.0) / (dur_frame - 1.0).clamp(min=1.0)
    return rel_pos.clamp(0.0, 1.0)


# -------------------------
# Feed-forward block
# -------------------------


class _FFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Q-Former Layer
# -------------------------


class QFormerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_ca = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = _FFN(d_model, ffn_dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        mel_features: torch.Tensor,
        cross_mask: torch.Tensor,
        causal_mask: torch.Tensor,
        sa_key_padding_mask: torch.Tensor,
        last_layer: bool = False,
        temperature: float = 1.0,
    ):
        q = self.norm_ca(queries)
        if temperature != 1.0:
            q = q / temperature
        ca_out, attn_weights = self.cross_attn(
            q, mel_features, mel_features, attn_mask=cross_mask
        )
        if not last_layer:
            queries = queries + self.drop(ca_out)
            q = self.norm_sa(queries)
            sa_out, _ = self.self_attn(
                q, q, q, attn_mask=causal_mask, key_padding_mask=sa_key_padding_mask
            )
            queries = queries + self.drop(sa_out)
            queries = queries + self.ffn(self.norm_ff(queries))
            out = queries
        else:
            out = attn_weights
        return out


# -------------------------
# AlignmentQFormer
# -------------------------
@dataclass
class QFormerOutput:
    pooled: torch.Tensor
    rel_pos: torch.Tensor
    weights: torch.Tensor


class AlignmentQFormer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_queries_per_phoneme: int = 4,
        num_layers: int = 2,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        out_dim: int = 64,
        context_expansion: int = 5,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.Q_learnable = num_queries_per_phoneme
        self.Q_tot = num_queries_per_phoneme + 1
        self.context_expansion = context_expansion
        self.temperature = temperature
        ffn_dim = ffn_dim or 4 * d_model

        self.query_proto = nn.Parameter(torch.empty(num_queries_per_phoneme, d_model))
        nn.init.trunc_normal_(self.query_proto, std=0.02)

        self.pos_encoder = RelativePositionEncoder(d_model)
        self.layers = nn.ModuleList(
            [
                QFormerLayer(d_model, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(2 * d_model, out_dim)

    def _build_causal_mask(self, N: int, device: torch.device) -> torch.Tensor:
        """
        Build an additive causal mask for self-attention, allowing each query
        to attend **only to queries of the same phoneme**.

        Returns:
            mask: (N*Q_tot, N*Q_tot) additive float mask (-inf where blocked)
        """
        Q = self.Q_tot
        total = N * Q

        # Gruppi di queries per fonema
        group = torch.arange(total, device=device) // Q  # (total,)

        # Mask: True = allowed, False = blocked
        allowed = group.unsqueeze(0) == group.unsqueeze(1)  # (total, total)

        mask = torch.zeros(total, total, device=device)
        mask = mask.masked_fill(~allowed, -1e8)
        return mask

    def _build_cross_attn_mask(
        self, alignment: torch.Tensor, phoneme_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        B, T, N = alignment.shape
        Q_tot, H = self.Q_tot, self.num_heads
        aligned_t = alignment.permute(0, 2, 1)
        if self.context_expansion > 0:
            pad = self.context_expansion
            expanded = F.max_pool1d(
                aligned_t, kernel_size=2 * pad + 1, stride=1, padding=pad
            )
        else:
            expanded = aligned_t
        mask_bool = (
            expanded.unsqueeze(2).expand(B, N, Q_tot, T).reshape(B, N * Q_tot, T).bool()
        )
        additive = expanded.new_zeros(B, N * Q_tot, T).masked_fill(~mask_bool, -1e8)
        empty = (~mask_bool).all(dim=-1, keepdim=True)
        additive = additive.masked_fill(empty, 0.0)
        if phoneme_mask is not None:
            pad_rows = (
                phoneme_mask.unsqueeze(2)
                .expand(B, N, self.Q_tot)
                .reshape(B, N * self.Q_tot)
                .unsqueeze(-1)
            )
            additive = additive.masked_fill(pad_rows, 0.0)
        return (
            additive.unsqueeze(1)
            .expand(B, H, N * Q_tot, T)
            .reshape(B * H, N * Q_tot, T)
            .contiguous()
        )

    def forward(
        self,
        mel_features: torch.Tensor,
        alignment: torch.Tensor,
        phoneme_embeddings: torch.Tensor,
        phoneme_mask: Optional[torch.BoolTensor] = None,
        mel_values: Optional[torch.Tensor] = None,
    ) -> QFormerOutput:
        B, T, N = alignment.shape
        Q_tot = self.Q_tot

        rel_pos = compute_relative_positions(alignment)
        pos_emb = self.pos_encoder(rel_pos)
        mel_pos = mel_features + pos_emb

        proto = self.query_proto.view(1, 1, self.Q_learnable, self.d_model).expand(
            B, N, self.Q_learnable, self.d_model
        )
        queries = (
            torch.cat([phoneme_embeddings.unsqueeze(2), proto], dim=2)
            .reshape(B, N * Q_tot, self.d_model)
            .contiguous()
        )

        cross_mask = self._build_cross_attn_mask(alignment, phoneme_mask)
        causal_mask = self._build_causal_mask(N, device=mel_features.device)
        if phoneme_mask is not None:
            sa_key_padding_mask = (
                phoneme_mask.unsqueeze(2).expand(B, N, Q_tot).reshape(B, N * Q_tot)
            )
        else:
            sa_key_padding_mask = torch.zeros(
                B, N * Q_tot, dtype=torch.bool, device=mel_features.device
            )

        # Pass through layers
        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) - 1
            out = layer(
                queries,
                mel_pos,
                cross_mask,
                causal_mask,
                sa_key_padding_mask,
                last_layer=last,
                temperature=self.temperature,
            )
            if not last:
                queries = out

        # Weighted mean+std pooling
        weights = out.reshape(B, N, Q_tot, T)[:, :, 0, :]
        weights = weights.nan_to_num(0.0)

        # # plot weigths
        # import matplotlib.pyplot as plt
        # for i, w in enumerate(weights):
        #     plt.imshow(w.unsqueeze(0).T.detach().cpu().numpy(), aspect="auto")
        #     plt.colorbar()
        #     plt.savefig(f"weights_{i}.png")
        #     plt.close()

        values = mel_features if mel_values is None else mel_values
        pooled = torch.bmm(weights, values)
        # diff = mel_features.unsqueeze(1) - mu.unsqueeze(2)  # (B, N, T, D)
        # var = (weights.unsqueeze(-1) * diff**2).sum(dim=2)  # (B, N, D)
        # z_n = torch.cat([mu, torch.sqrt(var + 1e-5)], dim=-1)
        # pooled = self.proj(z_n)

        return QFormerOutput(
            pooled=pooled,
            rel_pos=rel_pos,
            weights=weights,
        )


# ---------------------------------------------------------------------------
# DiT conditioning projector
# ---------------------------------------------------------------------------


class CausalConv1dBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pad = kernel_size - 1

        self.norm = nn.LayerNorm(channels)

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=0,
        )

        self.act = nn.GELU()

        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.norm(x)
        x = self.act(x)

        x = x.transpose(1, 2)

        x = F.pad(x, (self.pad, 0))  # causal padding

        x = self.conv(x)

        x = x.transpose(1, 2)

        return residual + self.drop(x)


class DurationConditioningProjector(nn.Module):

    def __init__(
        self,
        d_in,
        d_out=None,
        channels=None,
        kernel_size=31,
        n_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        d_out = d_out or d_in
        channels = channels or d_in

        self.in_proj = nn.Linear(d_in, channels) if d_in != channels else nn.Identity()

        self.pos_encoder = RelativePositionEncoder(channels)

        self.conv_blocks = nn.ModuleList(
            [
                CausalConv1dBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.out_norm = nn.LayerNorm(channels)

        self.out_proj = (
            nn.Linear(channels, d_out) if channels != d_out else nn.Identity()
        )

    def forward(
        self,
        pooled,  # (B,N,D)
        durations,  # (B,N)
        rel_pos,  # (B,T)
    ):

        x = upsample_by_duration(pooled, durations)

        x = self.in_proj(x)

        x = x + self.pos_encoder(rel_pos)

        for block in self.conv_blocks:
            x = block(x)

        x = self.out_norm(x)

        return self.out_proj(x)


def upsample_by_duration(pooled, durations):

    B, N, D = pooled.shape

    T = durations.sum(dim=1).max()

    device = pooled.device

    frame_index = torch.arange(T, device=device)

    cum_dur = torch.cumsum(durations, dim=1)

    start = cum_dur - durations

    mask = (frame_index[None, None, :] >= start[..., None]) & (
        frame_index[None, None, :] < cum_dur[..., None]
    )

    idx = mask.float().argmax(dim=1)

    return torch.gather(
        pooled,
        1,
        idx.unsqueeze(-1).expand(-1, -1, D),
    )
