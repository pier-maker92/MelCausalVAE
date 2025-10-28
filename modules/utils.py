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


print_once = once(print)


class Attend(nn.Module):
    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = scale
        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"
        # determine efficient attention configs for cuda and cpu
        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None
        if not torch.cuda.is_available() or not flash:
            return
        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once("A100 GPU detected, using flash attention if input tensor is on cuda")
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once("Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda")
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, dim_head, k_len, is_cuda, device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )
        # if scale is given, divide by the default scale that sdpa uses
        if exists(self.scale):
            q = q * (self.scale / (dim_head**-0.5))
        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)
        # Check if there is a compatible device for flash attention
        config = self.cuda_config if is_cuda else self.cpu_config
        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        return out

    def forward(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        scale = default(self.scale, q.shape[-1] ** -0.5)
        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, "b j -> b 1 1 j")
        if self.flash:
            return self.flash_attn(q, k, v, mask=mask)
        # similarity
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale
        # key padding mask
        if exists(mask):
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn).to(v.dtype)
        # aggregate values
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)
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
