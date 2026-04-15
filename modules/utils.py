import math
from collections import namedtuple
from functools import wraps
from typing import Union, Optional

import torch
from beartype import beartype
from einops import rearrange
from packaging import version
from torch import einsum, nn
from torch.cuda.amp import autocast
from torch.nn import Module
from torch.nn import functional as F

# constants

FlashAttentionConfig = namedtuple(
    "FlashAttentionConfig",
    ["enable_flash", "enable_math", "enable_mem_efficient"],
)

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def _mps_is_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and bool(mps.is_available())


# FlashAttention v2: CUDA-only. When MPS is available, skip importing flash_attn
# (no suitable wheel) and use PyTorch SDPA on MPS/CPU instead.
_HAS_FLASH_ATTN = False
flash_attn_qkvpacked_func = None  # type: ignore[assignment]
flash_attn_func = None  # type: ignore[assignment]
flash_attn_varlen_qkvpacked_func = None  # type: ignore[assignment]

if not _mps_is_available():
    try:
        from flash_attn import (
            flash_attn_qkvpacked_func,
            flash_attn_func,
            flash_attn_varlen_qkvpacked_func,
        )

        _HAS_FLASH_ATTN = True
    except ImportError:
        pass


class Attend(nn.Module):
    """
    Attention core on [B, H, N, D]: FlashAttention on CUDA when available,
    otherwise PyTorch scaled_dot_product_attention (CPU / MPS / CUDA without flash).

    Flash path rules:
      - CUDA only; dtype float16/bfloat16; D % 8 == 0 and D <= 256.
      - Self-attention with no mask: packed kernel.
      - Self-attention with padding mask [B,S] or [B,1,1,S]: varlen packed kernel.
      - Cross-attention without mask: non-packed kernel.
      - Cross-attention with mask: not implemented here.

    When MPS is available at import time, flash_attn is not loaded (no macOS wheel);
    forward uses SDPA instead.
    """

    def __init__(self, dropout: float = 0.0, flash: bool = False, scale: Optional[float] = None, window_size: Optional[int] = None):
        super().__init__()
        self.dropout = float(dropout)
        self.scale = scale
        self.flash = flash and _HAS_FLASH_ATTN  # keep flag for compatibility
        self._window_size = window_size

    @staticmethod
    def _normalize_mask(mask: torch.Tensor, S: int) -> torch.Tensor:
        """
        Normalize to [B, S] bool with True=valid.
        Accepts [B,S] or [B,1,1,S].
        """
        if mask.ndim == 2:
            m = mask
        elif mask.ndim == 4 and mask.shape[-1] == S:
            m = mask.squeeze(1).squeeze(1)
        else:
            raise ValueError("mask must be [B,S] or [B,1,1,S] with last dim == seqlen")
        if m.dtype != torch.bool:
            m = m != 0
        return m

    @staticmethod
    def _build_cu_seqlens(mask_bs: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        From [B,S] bool True=valid -> cu_seqlens [B+1], max_seqlen int.
        """
        B, S = mask_bs.shape
        lengths = mask_bs.sum(dim=1, dtype=torch.int32)
        cu = torch.empty(B + 1, dtype=torch.int32, device=mask_bs.device)
        cu[0] = 0
        torch.cumsum(lengths, dim=0, out=cu[1:])
        max_len = int(lengths.max().item()) if B > 0 else 0
        return cu, max_len

    def _build_group_causal_mask(
        self,
        Nq: int,
        Nk: int,
        group_size: int,
        device: torch.device,
        dtype: torch.dtype,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build a block-causal attention mask: bidirectional within each group
        of `group_size` frames, causal across groups. Optionally combined with
        sliding-window and padding constraints.

        Returns float mask [B?, 1, Nq, Nk] with 0 = attend, -inf = masked.
        """
        idx_q = torch.arange(Nq, device=device)
        idx_k = torch.arange(Nk, device=device)

        # Block-causal: key's group <= query's group
        attn_mask = (idx_k[None, :] // group_size) <= (idx_q[:, None] // group_size)

        # Sliding window constraint (left side only)
        window_left = 0
        if self._window_size is not None:
            ws = self._window_size
            window_left = int(ws[0]) if isinstance(ws, (tuple, list)) else int(ws)
        if window_left > 0:
            attn_mask = attn_mask & (idx_k[None, :] >= (idx_q[:, None] - window_left))

        # Expand and combine with padding mask
        if padding_mask is not None:
            mask_bs = self._normalize_mask(padding_mask, S=Nk)  # [B, Nk] True=valid
            attn_mask = attn_mask.unsqueeze(0) & mask_bs.unsqueeze(1)  # [B, Nq, Nk]
            float_mask = torch.zeros(attn_mask.shape[0], 1, Nq, Nk, device=device, dtype=dtype)
            float_mask.masked_fill_(~attn_mask.unsqueeze(1), float("-inf"))
        else:
            float_mask = torch.zeros(1, 1, Nq, Nk, device=device, dtype=dtype)
            float_mask.masked_fill_(~attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        return float_mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        group_size: Optional[int] = None,
    ):
        """
        q, k, v: [B, H, N, D]
        mask: None or [B,S] / [B,1,1,S] with True=valid
        group_size: if set, use block-causal attention (bidirectional within
                    groups of this size, causal across groups) via SDPA.
        returns: [B, H, Nq, D]
        """
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "q,k,v must be [B,H,N,D]"
        B, H, Nq, D = q.shape
        _, Hk, Nk, Dk = k.shape
        assert H == Hk and D == Dk, "Mismatched heads or head_dim between q and k/v"

        use_flash = _HAS_FLASH_ATTN and q.is_cuda
        if use_flash:
            assert q.dtype in (torch.float16, torch.bfloat16), (
                "Use fp16 or bf16 for FlashAttention"
            )
            assert D % 8 == 0 and D <= 256, "head_dim must be divisible by 8 and <= 256"
        else:
            if q.device.type not in ("cuda", "cpu", "mps"):
                raise ValueError(f"Unsupported attention device: {q.device}")

        p = self.dropout if self.training else 0.0
        softmax_scale = self.scale if self.scale is not None else (D**-0.5)

        # Block-causal path: SDPA with explicit mask (Flash Attn can't express this)
        if group_size is not None:
            float_mask = self._build_group_causal_mask(
                Nq, Nk, group_size, q.device, q.dtype, padding_mask=mask,
            )
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=float_mask, dropout_p=p, scale=softmax_scale,
            )

        if not use_flash:
            if Nq == Nk:
                return self._sdpa_self_attention(
                    q, k, v, mask=mask, causal=causal, dropout_p=p, scale=softmax_scale
                )
            if mask is not None:
                raise NotImplementedError(
                    "Cross-attention with mask requires separate q and kv masks; use varlen APIs for both."
                )
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=p, scale=softmax_scale, is_causal=causal
            )

        # --- FlashAttention (CUDA fp16/bf16) ---
        assert k.is_cuda and v.is_cuda
        # to [B, S, H, D]
        q_ = q.permute(0, 2, 1, 3).contiguous()
        k_ = k.permute(0, 2, 1, 3).contiguous()
        v_ = v.permute(0, 2, 1, 3).contiguous()

        # Window size: (left, right). Causal → right=0; bidirectional → symmetric.
        if self._window_size is not None:
            w = (
                (self._window_size, 0)
                if causal
                else (self._window_size, self._window_size)
            )
        else:
            w = (-1, -1)
        if Nq == Nk:
            if mask is None:
                qkv = torch.stack([q_, k_, v_], dim=2)  # [B, S, 3, H, D]
                out = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=w,
                )  # [B, S, H, D]
                return out.permute(0, 2, 1, 3).contiguous()
            else:
                mask_bs = self._normalize_mask(mask, S=Nk)  # [B,S]
                keep = mask_bs.reshape(B * Nk)
                qkv = torch.stack([q_, k_, v_], dim=2)  # [B, S, 3, H, D]
                qkv_flat = qkv.view(B * Nk, 3, H, D)
                qkv_packed = qkv_flat[keep]  # [T, 3, H, D], T=sum(valid)
                cu_seqlens, max_seqlen = self._build_cu_seqlens(mask_bs)
                out_packed = flash_attn_varlen_qkvpacked_func(
                    qkv_packed,
                    cu_seqlens,
                    max_seqlen,
                    dropout_p=p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=w,
                )  # [T, H, D]
                out_flat = torch.zeros(B * Nk, H, D, device=q.device, dtype=q.dtype)
                out_flat[keep] = out_packed
                out = out_flat.view(B, Nk, H, D).permute(0, 2, 1, 3).contiguous()
                return out

        # cross-attention
        if mask is not None:
            raise NotImplementedError(
                "Cross-attention with mask requires separate q and kv masks; use varlen APIs for both."
            )
        out = flash_attn_func(
            q_, k_, v_, dropout_p=p, softmax_scale=softmax_scale, causal=causal, window_size=w
        )  # [B, Nq, H, D]
        return out.permute(0, 2, 1, 3).contiguous()

    def _sdpa_self_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        causal: bool,
        dropout_p: float,
        scale: float,
    ) -> torch.Tensor:
        """PyTorch SDPA fallback for non-CUDA (e.g. MPS) or when flash_attn is off."""
        B, H, N, _D = q.shape
        if mask is None:
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, scale=scale, is_causal=causal
            )
        mask_bs = self._normalize_mask(mask, S=N)
        key_m = mask_bs[:, None, None, :]  # [B,1,1,N]
        q_m = mask_bs[:, None, :, None]  # [B,1,N,1]
        valid = key_m & q_m
        attn_mask = torch.zeros(B, 1, N, N, device=q.device, dtype=q.dtype)
        attn_mask.masked_fill_(~valid, float("-inf"))
        if causal:
            causal_block = torch.triu(
                torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1
            )
            attn_mask = attn_mask.masked_fill(causal_block, float("-inf"))
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale
        )


class LearnedSinusoidalPosEmb(Module):
    """used by @crowsonkb"""

    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fouriered


class RotaryEmbedding(Module):
    def __init__(self, dim, theta=50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @autocast(enabled=False)
    @beartype
    def forward(self, t: Union[int, torch.Tensor]):
        if not torch.is_tensor(t):
            t = torch.arange(t, device=self.device)
        t = t.type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@autocast(enabled=False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


# convolutional positional generating module


class ConvPositionEmbed(Module):
    def __init__(self, dim, *, kernel_size, groups=None):
        super().__init__()
        assert is_odd(kernel_size)
        groups = default(groups, dim)  # full depthwise conv by default
        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.GELU(),
        )

    def forward(self, x, mask=None):
        if exists(mask):
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = rearrange(x, "b n c -> b c n")
        x = self.dw_conv1d(x)
        out = rearrange(x, "b c n -> b n c")
        if exists(mask):
            out = out.masked_fill(~mask, 0.0)
        return out


# --------- norms ---------


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class AdaptiveRMSNorm(Module):
    def __init__(self, dim, cond_dim=None):
        super().__init__()
        cond_dim = default(cond_dim, dim)
        self.scale = dim**0.5
        self.to_gamma = nn.Linear(cond_dim, dim)
        self.to_beta = nn.Linear(cond_dim, dim)
        # init to identity
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, cond):
        normed = F.normalize(x, dim=-1) * self.scale
        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = map(lambda t: rearrange(t, "b d -> b 1 d"), (gamma, beta))
        return normed * gamma + beta


# --------- attention ---------

# NOTE MultiheadRMSNorm is not used by default; keep Attention API unchanged.


class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        dropout=0,
        flash=False,
        qk_norm=False,
        qk_norm_scale=10,
        window_size=None,
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads
        scale = qk_norm_scale if qk_norm else None
        self.attend = Attend(dropout, flash=flash, scale=scale, window_size=window_size)
        self.qk_norm = qk_norm
        # If qk_norm is ever enabled, define simple per-head RMSNorms or raise.
        if qk_norm:
            raise NotImplementedError("qk_norm path not enabled in FlashAttention-only build")
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, mask=None, rotary_emb=None, causal: bool = False, group_size: Optional[int] = None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))
        out = self.attend(q, k, v, mask=mask, causal=causal, group_size=group_size)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# feedforward


class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, dropout=0.0):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim),
    )
