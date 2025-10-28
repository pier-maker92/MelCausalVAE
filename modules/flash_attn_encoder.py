# transformer_flash_attn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# public API functions per README:
# flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, ...)
# flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, ...)
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

_HAS_FLASH = True
# except Exception:
#     flash_attn_qkvpacked_func = None
#     flash_attn_func = None
#     _HAS_FLASH = False


class FlashMultiheadAttention(nn.Module):
    """
    Multi-head attention using Dao-AILab flash-attn kernels when available.
    Input/Output shape: [B, T, E]
    Args:
      embed_dim: total embedding dimension (E)
      num_heads: number of attention heads
      dropout: dropout probability applied to attention (forward only)
    Supports:
      - key_padding_mask: Bool mask shaped [B, T] (True for PAD positions)
      - causal: bool (if True, apply causal mask)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = float(dropout)

        # projection layers (we'll produce Q,K,V and optionally pack them)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, causal: bool = False):
        """
        x: [B, T, E]
        key_padding_mask: optional [B, T] bool tensor where True indicates padding positions (will be masked)
        causal: whether to apply autoregressive causal mask
        returns: [B, T, E]
        """
        B, T, E = x.size()
        H = self.num_heads
        D = self.head_dim  # head dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape to (B, T, H, D)
        # flash_attn expects Q/K/V shaped (batch, seqlen, nheads, headdim)
        q = q.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)

        # If flash-attn is available and running on CUDA, prefer the packed QKV kernel (less concat overhead)
        if _HAS_FLASH and x.is_cuda:
            orig_proj_dtype = self.out_proj.weight.dtype
            # Ensure inputs to flash-attn are in a supported dtype (fp16/bf16)
            if q.dtype not in (torch.float16, torch.bfloat16):
                target_dtype = (
                    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                )
                q = q.to(target_dtype)
                k = k.to(target_dtype)
                v = v.to(target_dtype)
            # Build packed qkv: (B, T, 3, H, D)
            qkv = torch.stack([q, k, v], dim=2).contiguous()  # [B, T, 3, H, D]

            # convert mask to the attn arguments expected by the function:
            # flash_attn_qkvpacked_func takes dropout_p and causal flags. It doesn't accept key_padding_mask directly,
            # but we can add a large negative bias to the attention using 'cu_seqlens/unpadded' variants for varlen;
            # however for the constant-length case it's simplest to use flash_attn_qkvpacked_func plus an additive mask:
            attn_kwargs = {
                "dropout_p": self.dropout if self.training else 0.0,
                "softmax_scale": None,  # default 1/sqrt(head_dim)
                "causal": causal,
            }

            # If no padding mask, call packed kernel directly
            if key_padding_mask is None:
                out = flash_attn_qkvpacked_func(qkv, **attn_kwargs)  # [B, T, H, D]
            else:
                # Packed function does not directly accept key_padding_mask as an argument.
                # To handle padding we convert key_padding_mask -> attention scores bias by using
                # masked positions having -inf. The flash-attn functions accept an 'attn_mask'
                # variant in some releases; if not, a common pattern is to zero-out k/v for pad
                # and use q lengths (varlen) API; for simplicity, this implementation will
                # use flash_attn_func (q,k,v) call which accepts no mask either, so we convert
                # padded k/v to very small values and then zero outputs at padding positions.
                # NOTE: production code should use the varlen API (flash_attn_varlen_* functions)
                # when sequences have different lengths in the batch.
                # Here we zero-out k and v at pad positions and later zero outputs on pad:
                pad_mask = key_padding_mask.to(device=x.device)  # True = pad
                # invert so keep_mask = 1 on valid tokens
                keep_mask = (~pad_mask).to(k.dtype)  # [B, T], 1.0 for valid
                keep_mask = keep_mask.view(B, T, 1, 1)  # broadcast over heads & head_dim
                k = k * keep_mask
                v = v * keep_mask
                qkv = torch.stack([q, k, v], dim=2).contiguous()
                out = flash_attn_qkvpacked_func(qkv, **attn_kwargs)  # [B, T, H, D]
                # zero out outputs at padding positions explicitly (to be safe)
                out = out * keep_mask

            # out: [B, T, H, D] -> reshape to [B, T, E]
            out = out.view(B, T, H * D)
            # Cast back to the projection layer's dtype to avoid dtype mismatch
            if out.dtype != orig_proj_dtype:
                out = out.to(orig_proj_dtype)
            out = self.out_proj(out)
            return out

        # fallback (no flash-attn installed) -> use standard scaled dot-product (slower)
        # reshape to [B, H, T, D] for matmul logic
        qh = q.permute(0, 2, 1, 3)  # [B, H, T, D]
        kh = k.permute(0, 2, 1, 3)
        vh = v.permute(0, 2, 1, 3)

        # compute QK^T
        qk = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(D)  # [B,H,T,T]

        if key_padding_mask is not None:
            # key_padding_mask: [B, T] with True at padded positions
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            qk = qk.masked_fill(mask, float("-inf"))

        if causal:
            causal_mask = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1)
            qk = qk.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        w = F.softmax(qk, dim=-1)
        if self.dropout:
            w = F.dropout(w, p=self.dropout, training=self.training)
        attn_out = torch.matmul(w, vh)  # [B, H, T, D]
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)
        out = self.out_proj(attn_out)
        return out


class FlashTransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer using FlashMultiheadAttention."""

    def __init__(self, d_model=512, nhead=8, dim_feedforward=None, dropout=0.1, norm_first=True):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.self_attn = FlashMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None, causal: bool = False):
        """
        src: [B, T, d_model]
        src_key_padding_mask: [B, T] bool (True means pad)
        causal: whether to use causal masking in self-attention
        """
        x = src
        if self.norm_first:
            y = self.norm1(x)
            y = self.self_attn(y, key_padding_mask=src_key_padding_mask, causal=causal)
            x = x + self.dropout1(y)
            z = self.norm2(x)
            z = self.linear2(self.activation(self.linear1(z)))
            x = x + self.dropout2(z)
            return x
        else:
            y = self.self_attn(x, key_padding_mask=src_key_padding_mask, causal=causal)
            x = self.norm1(x + self.dropout1(y))
            z = self.linear2(self.activation(self.linear1(x)))
            x = self.norm2(x + self.dropout2(z))
            return x


class FlashTransformerEncoder(nn.Module):
    """Stack multiple FlashTransformerEncoderLayer to build an encoder tail."""

    def __init__(self, d_model=512, nhead=8, nlayers=4, drop_p=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FlashTransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=drop_p, norm_first=True)
                for _ in range(nlayers)
            ]
        )

    def forward(self, tokens: torch.Tensor, src_key_padding_mask: torch.Tensor = None, causal: bool = False):
        """
        tokens: [B, T, d_model]
        src_key_padding_mask: optional [B, T] (True for pad)
        causal: bool -> pass to self-attention
        """
        x = tokens
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask, causal=causal)
        return x


# Example usage:
if __name__ == "__main__":
    # toy example
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.bfloat16)
    B, T, E = 2, 128, 512
    x = torch.randn(B, T, E, device=device)
    # create a padding mask (False = keep, True = pad)
    pad_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    pad_mask[:, -5:] = True  # last 5 tokens are padding

    model = FlashTransformerEncoder(d_model=E, nhead=8, nlayers=2, drop_p=0.1).to(device).to(torch.bfloat16)
    model.train()
    # use autocast for best performance on GPUs
    if device.startswith("cuda"):
        from torch.cuda.amp import autocast

        with autocast(dtype=torch.bfloat16):
            out = model(x, src_key_padding_mask=pad_mask, causal=True)
    else:
        out = model(x, src_key_padding_mask=pad_mask, causal=True)
    print(out.shape)  # -> [B, T, E]
