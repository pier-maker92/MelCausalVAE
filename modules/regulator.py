import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


class InterpolateRegulator(nn.Module):
    def __init__(
        self,
        depth: int,  # <-- number of conv-GN-Mish stacks
        in_channels: int,
        channels: int,
        out_channels: int,
        groups: int = 1,
        is_causal: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.is_causal = is_causal

        layers = nn.ModuleList([])

        # ----- convolution blocks -----
        for i in range(depth):
            conv = nn.Conv1d(
                in_channels if i == 0 else channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=0 if is_causal else 1,  # symmetric vs causal
            )
            layers.extend(
                [
                    conv,
                    nn.GroupNorm(groups, channels),
                    nn.Mish(),
                ]
            )

        # ----- final projection -----
        layers.append(nn.Conv1d(channels if depth > 0 else in_channels, out_channels, 1))

        self.model = nn.Sequential(*layers)

    def _causal_interpolate(self, x, out_length):
        """
        Nearest-neighbor causal upsampling:
        Only uses *past* frames (zero-order hold).
        """
        B, C, T = x.shape
        if T == out_length:
            return x

        scale = out_length / T
        idx = torch.floor(torch.arange(out_length, device=x.device) / scale).long()
        idx = torch.clamp(idx, 0, T - 1)
        return x[:, :, idx]

    def forward(self, x, xlens=None, ylens=None):
        # x: (B, T, D)
        target_len = ylens.max()

        # mask: (B, target_len, 1)
        mask = ~make_pad_mask(lengths=xlens, max_len=target_len).unsqueeze(-1)

        # (B, T, D) → (B, D, T)
        x = x.transpose(1, 2).contiguous()

        # ---- Interpolation ----
        if self.is_causal:
            x = self._causal_interpolate(x, target_len)
        else:
            x = F.interpolate(x, size=target_len, mode="linear")

        # ---- Convolutional stacks ----
        if self.is_causal:
            for layer in self.model:
                if isinstance(layer, nn.Conv1d) and layer.kernel_size[0] > 1:
                    pad = layer.kernel_size[0] - 1
                    x = F.pad(x, (pad, 0))  # left padding only
                    x = layer(x)
                else:
                    x = layer(x)
        else:
            x = self.model(x)

        out = x.transpose(1, 2).contiguous()
        return out * mask, ylens
