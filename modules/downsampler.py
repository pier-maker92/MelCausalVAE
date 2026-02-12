import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, causal=True):
        super().__init__()
        self.conv1 = Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            causal=causal,
        )
        self.norm1 = LayerNorm1d(out_channels)
        self.act = nn.SiLU()

        self.conv2 = Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=causal,
        )
        self.norm2 = LayerNorm1d(out_channels)

        self.shortcut = nn.Identity()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                LayerNorm1d(out_channels),
            )

    def forward(self, x, mask=None):
        residual = self.shortcut(x)
        if mask is not None:
            residual = residual * mask

        out = self.act(self.norm1(self.conv1(x, mask)))
        out = self.norm2(self.conv2(out, mask))

        return self.act(out + residual)


class DownSampler(nn.Module):
    def __init__(
        self, d_in=100, d_hidden=256, d_out=64, compress_factor=4, causal=True
    ):
        super().__init__()

        if not (compress_factor > 0 and (compress_factor & (compress_factor - 1)) == 0):
            raise ValueError("compress_factor must be a power of 2")

        # 1. Input Projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(d_in, d_hidden, kernel_size=1), LayerNorm1d(d_hidden), nn.SiLU()
        )

        # 2. ResNet Stack con raffinamento
        self.layers = nn.ModuleList()
        num_downsampling_steps = int(math.log2(compress_factor))

        for _ in range(num_downsampling_steps):
            # Downsampling (stride 2)
            self.layers.append(ResNetBlock(d_hidden, d_hidden, stride=2, causal=causal))
            # Raffinamento (stride 1)
            self.layers.append(ResNetBlock(d_hidden, d_hidden, stride=1, causal=causal))

        # 3. Output Projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(d_hidden, d_out, kernel_size=1), LayerNorm1d(d_out)
        )

    def forward(self, x, x_mask=None):
        x = x.transpose(1, 2)

        mask = None
        if x_mask is not None:
            mask = 1.0 - x_mask.to(x.dtype)
            mask = mask.unsqueeze(1)

        # Step 1: Expansion
        x = self.input_proj(x)
        if mask is not None:
            x = x * mask

        # Step 2: Loop sui blocchi
        for layer in self.layers:
            # Applichiamo il blocco
            x = layer(x, mask)

            # Se il layer ha fatto downsampling, aggiorniamo la maschera
            # (Controlliamo se lo stride del primo conv del blocco è > 1)
            if hasattr(layer.conv1, "stride") and layer.conv1.stride[0] > 1:
                if mask is not None:
                    mask = mask[:, :, :: layer.conv1.stride[0]]

        # Step 3: Final Compression
        x = self.output_proj(x)
        if mask is not None:
            x = x * mask
        return x.transpose(1, 2), mask.squeeze(1).bool() if mask is not None else None
