import torch.nn as nn


class LinearResNetBlock(nn.Module):
    """
    A ResNet block using only Linear layers (position-wise feed-forward).
    It is causal by definition because it operates on each timestep independently,
    without mixing information across the time dimension.
    Expects input shape of (B, T, C).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

        self.linear2 = nn.Linear(out_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
            self.shortcut_norm = nn.LayerNorm(out_channels)
        else:
            self.shortcut_norm = nn.Identity()

    def forward(self, x, mask=None):
        # x shape: (B, T, C)

        # Calculate shortcut
        if isinstance(self.shortcut, nn.Linear):
            residual = self.shortcut(x)
        else:
            residual = x
        residual = self.shortcut_norm(residual)

        if mask is not None:
            residual = residual * mask.unsqueeze(-1)

        # Layer 1
        out = self.linear1(x)
        out = self.norm1(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        out = self.act(out)

        # Layer 2
        out = self.linear2(out)
        out = self.norm2(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return self.act(out + residual)


class LinearResNet(nn.Module):
    """
    A stack of Linear ResNet blocks.
    """

    def __init__(self, in_dim: int, hidden_dim: int, output_dim: int, num_blocks: int):
        super().__init__()

        # Initial projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList(
            [LinearResNetBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x, reverse_mask=None):
        # x shape: (B, T, in_dim)
        mask = ~reverse_mask

        # 1. Input Projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # 2. ResNet Layers
        for layer in self.layers:
            x = layer(x, mask)

        # 3. Output Projection
        x = self.output_proj(x)
        x = self.output_norm(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x, reverse_mask
