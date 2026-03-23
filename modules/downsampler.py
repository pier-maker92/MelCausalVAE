import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_padding_mask(
    padding_mask: torch.BoolTensor, target_length: int
) -> torch.BoolTensor:
    padding_mask = (
        torch.nn.functional.interpolate(
            padding_mask.unsqueeze(1).float(),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        > 0.5
    )
    return padding_mask


class LayerNorm1d(nn.Module):
    def __init__(
        self, channels: int, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, T, C) -> LN(C) -> (B, C, T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class Conv1d(nn.Conv1d):
    def __init__(self, *args, causal=False, **kwargs):
        self.is_causal = causal
        if causal and kwargs.get("padding", 0) > 0:
            self._causal_padding = kwargs["padding"] * 2
            kwargs["padding"] = 0
        else:
            self._causal_padding = 0
        super(Conv1d, self).__init__(*args, **kwargs)

    def forward(self, x, x_mask=None):
        if self.is_causal and self._causal_padding > 0:
            x = F.pad(x, (self._causal_padding, 0))

        output = super(Conv1d, self).forward(x)

        if x_mask is not None:
            output = output * x_mask

        return output


# ── Convolution-based downsampling (stride=2, halves time) ──────────────

class ConvDownsampleBlock(nn.Module):
    """Causal strided Conv1d that halves the time dimension."""

    def __init__(self, channels, causal=True):
        super().__init__()
        self.conv = Conv1d(
            channels, channels, kernel_size=3, stride=2, padding=1, causal=causal,
        )
        self.norm = LayerNorm1d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: [B, C, T] → [B, C, T//2]
        return self.act(self.norm(self.conv(x)))


# ── Linear residual block (feature-dimension only, no temporal mixing) ──

class LinearResBlock(nn.Module):
    """Residual block with Linear layers on the feature dimension (pointwise)."""

    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, C]
            mask: [B, T] bool – True = padded
        """
        residual = x
        out = self.act(self.norm1(self.linear1(x)))
        out = self.norm2(self.linear2(out))

        if mask is not None:
            valid = (~mask).unsqueeze(-1).to(dtype=out.dtype)
            out = out * valid
            residual = residual * valid

        return self.act(out + residual)


# ── DownSampler: conv downsampling + linear refinement ──────────────────

class DownSampler(nn.Module):
    def __init__(
        self, d_in=100, d_hidden=256, d_out=64, compress_factor=4,
        causal=True, n_residual_blocks=3,
    ):
        super().__init__()

        if not (compress_factor > 0 and (compress_factor & (compress_factor - 1)) == 0):
            raise ValueError("compress_factor must be a power of 2")

        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden), nn.SiLU()
        )

        num_downsampling_steps = int(math.log2(compress_factor))

        self.downsample_convs = nn.ModuleList()
        self.downsample_refine = nn.ModuleList()
        for _ in range(num_downsampling_steps):
            self.downsample_convs.append(ConvDownsampleBlock(d_hidden, causal=causal))
            self.downsample_refine.append(LinearResBlock(d_hidden))

        self.refine_blocks = nn.ModuleList()
        for _ in range(n_residual_blocks):
            self.refine_blocks.append(LinearResBlock(d_hidden))

        self.output_proj = nn.Sequential(
            nn.Linear(d_hidden, d_out), nn.LayerNorm(d_out)
        )

    def forward(self, x, x_mask=None):
        """
        Args:
            x: [B, T, F]
            x_mask: [B, T] bool – True = padded
        Returns:
            x: [B, T', d_out]
            mask: [B, T'] bool (True = padded) or None
        """
        mask = x_mask

        x = self.input_proj(x)
        if mask is not None:
            x = x * (~mask).unsqueeze(-1).to(dtype=x.dtype)

        for conv_ds, lin_ref in zip(self.downsample_convs, self.downsample_refine):
            # Conv downsample: [B, T, C] → [B, C, T] → conv → [B, C, T'] → [B, T', C]
            x = conv_ds(x.transpose(1, 2)).transpose(1, 2)
            if mask is not None:
                mask = _resize_padding_mask(mask, x.shape[1])
                x = x * (~mask).unsqueeze(-1).to(dtype=x.dtype)
            # Linear refinement in [B, T', C]
            x = lin_ref(x, mask)

        for block in self.refine_blocks:
            x = block(x, mask)

        x = self.output_proj(x)
        if mask is not None:
            x = x * (~mask).unsqueeze(-1).to(dtype=x.dtype)

        return x, mask
