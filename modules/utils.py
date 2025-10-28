import math
from collections import namedtuple
from functools import wraps
from typing import Union

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


from typing import Optional

print_once = once(print)
from flash_attn import flash_attn_qkvpacked_func  # packed QKV is the most convenient

_HAS_FLASH_ATTN = True


class Attend(nn.Module):
    """
    Attend module that accepts q, k, v shaped [B, H, N, D].

    Behavior:
      - If `flash=True` and the flash-attn package is installed, use flash_attn_qkvpacked_func
        with packed qkv (fastest).
      - Else, if PyTorch's scaled_dot_product_attention is available, call it inside
        torch.backends.cuda.sdp_kernel(...) to select optimized kernels.
      - Else, fall back to explicit matmul/softmax.
    Args:
      dropout: attention dropout probability
      flash: prefer flash-attn (if available)
      scale: optional softmax scale (float). If None, defaults to 1/sqrt(dim_head)
    """

    def __init__(self, dropout: float = 0.0, flash: bool = False, scale: Optional[float] = None):
        super().__init__()
        self.dropout = float(dropout)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.scale = scale
        self.flash = flash and _HAS_FLASH_ATTN  # only enable if package present

    def _bool_mask_to_additive(
        self, mask: torch.Tensor, q_len: int, k_len: int, heads: int, dtype: torch.dtype, device: torch.device
    ):
        """
        Convert boolean mask to additive mask shaped [B, H, q_len, k_len] with -inf for masked positions.
        Expect mask in either shape [B, S] (True = valid) or [B, 1, 1, S] (True = valid).
        """
        if mask is None:
            return None
        # normalize to [B, 1, 1, S]
        if mask.ndim == 2:
            pad = mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,S]
        else:
            pad = mask  # assume broadcastable
        # We assume `pad` is boolean with True -> valid token (matches your earlier code).
        # Create additive where invalid (False) -> -inf
        additive = (~pad).to(dtype=dtype) * float("-inf")  # invalid -> -inf
        # expand to heads and queries
        # [B,1,1,S] -> [B, H, q_len, k_len] by broadcasting (first expand to [B,1,q_len,S])
        additive = additive.expand(-1, 1, q_len, -1).expand(-1, heads, -1, -1)
        return additive.to(device)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ):
        """
        q, k, v: [B, H, N, D]  (N = sequence length)
        mask: either None, or boolean tensor [B, S] with True==valid, or [B,1,1,S]
        causal: whether to apply causal mask (no attend to future)
        returns: [B, H, Nq, D]
        """
        # sanity checks
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "q,k,v must be [B, H, N, D]"
        B, H, Nq, D = q.shape
        _, _, Nk, _ = k.shape
        device = q.device
        dtype = q.dtype

        # compute default softmax scale
        softmax_scale = self.scale if self.scale is not None else (D**-0.5)

        # If flash-attn package is available and requested, use packed kernel.
        # flash_attn expects packed QKV as shape [B, S, 3, H, D] (S=sequence length)
        if self.flash and _HAS_FLASH_ATTN and q.is_cuda:
            # We need Q/K/V in shape [B, S, H, D] (S=sequence length).
            # Our q/k/v are [B, H, N, D] -> transpose to [B, N, H, D]
            q_ = q.permute(0, 2, 1, 3).contiguous()  # [B, Nq, H, D]
            k_ = k.permute(0, 2, 1, 3).contiguous()  # [B, Nk, H, D]
            v_ = v.permute(0, 2, 1, 3).contiguous()  # [B, Nk, H, D]

            # If query and key lengths differ (cross-attention), flash_attn packed kernel
            # requires Q,K,V to have same sequence length; the most common use-case here is self-attention.
            # For cross-attention you would call flash_attn_func(q,k,v) instead.
            if Nq != Nk:
                # fallback to torch scaled_dot_product_attention path below
                pass
            else:
                # pack qkv: shape [B, S, 3, H, D]
                qkv = torch.stack([q_, k_, v_], dim=2)  # [B, S, 3, H, D]

                # handle mask: flash_attn_qkvpacked_func does not have a standard 'attn_mask' param in older versions.
                # We choose to zero-out k and v at padding positions (simple approach) and zero outputs at pad tokens.
                if mask is not None:
                    # normalize mask to [B, S] boolean (True valid). If mask is [B,1,1,S], reduce to [B,S].
                    if mask.ndim == 4:
                        mask_bw = mask.squeeze(1).squeeze(1)  # [B, S]
                    elif mask.ndim == 2:
                        mask_bw = mask
                    else:
                        mask_bw = mask  # assume broadcastable
                    # keep_mask: 1.0 for valid, 0.0 for pad
                    keep_mask = mask_bw.to(dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]
                    # multiply k and v by keep_mask to zero-out padded kvs
                    qkv[:, :, 1, :, :] = qkv[:, :, 1, :, :] * keep_mask  # k
                    qkv[:, :, 2, :, :] = qkv[:, :, 2, :, :] * keep_mask  # v

                # call flash-attn packed QKV kernel
                # note: flash_attn_qkvpacked_func signature accepts (qkv, dropout_p=..., softmax_scale=None, causal=False)
                out = flash_attn_qkvpacked_func(
                    qkv, dropout_p=self.dropout if self.training else 0.0, softmax_scale=softmax_scale, causal=causal
                )
                # out -> [B, S, H, D]  -> convert to [B, H, S, D]
                out = out.permute(0, 2, 1, 3).contiguous()  # [B, H, S, D]
                return out

        # Next preferred path: use PyTorch scaled_dot_product_attention (which can select flash/math kernels via sdp_kernel)
        # This supports differing query/key lengths and accepts attn_mask (float additive mask) and is_causal param.
        try:
            # Build additive attn_mask shaped [B, H, Nq, Nk] with -inf where masked
            attn_mask = self._bool_mask_to_additive(mask, q_len=Nq, k_len=Nk, heads=H, dtype=dtype, device=device)

            # Note: scaled_dot_product_attention signature (PyTorch 2.1+):
            # F.scaled_dot_product_attention(q,k,v, attn_mask=None, dropout_p=0.0, softmax_scale=None, is_causal=False)
            with torch.backends.cuda.sdp_kernel():
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    softmax_scale=softmax_scale,
                    is_causal=causal,
                )
            return out  # [B, H, Nq, D]
        except Exception:
            # final fallback: manual attention (works everywhere, slower)
            scale = softmax_scale
            # sim: [B, H, Nq, Nk]
            sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale

            if mask is not None:
                # mask: [B, 1, 1, Nk] or [B, Nk]
                if mask.ndim == 2:
                    mask_exp = mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,Nk]
                else:
                    mask_exp = mask
                # mask boolean True=valid -> keep where True
                sim = sim.masked_fill(~mask_exp.expand(-1, H, Nq, -1), float("-inf"))

            attn = sim.softmax(dim=-1)
            attn = self.attn_dropout(attn).to(v.dtype)
            out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
            return out


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


# NOTE MultiheadRMSNorm is not used in the unet_voicebox.py file
# class MultiheadRMSNorm(Module):
#     def __init__(self, dim, heads):
#         super().__init__()
#         self.scale = dim**0.5
#         self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

#     def forward(self, x):
#         return F.normalize(x, dim=-1) * self.gamma * self.scale


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
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads

        scale = qk_norm_scale if qk_norm else None

        self.attend = Attend(dropout, flash=flash, scale=scale)

        self.qk_norm = qk_norm

        if qk_norm:
            self.q_norm = MultiheadRMSNorm(dim_head, heads=heads)
            self.k_norm = MultiheadRMSNorm(dim_head, heads=heads)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, mask=None, rotary_emb=None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        out = self.attend(q, k, v, mask=mask)

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
