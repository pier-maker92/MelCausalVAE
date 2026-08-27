from typing import Optional

import torch
from torch import Tensor, nn

from ..configs import FocalQuantizerConfig


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape: int, tanhscale_init: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1,), tanhscale_init))
        self.normalized_shape = normalized_shape
        self.tanhscale_init = tanhscale_init

    def forward(self, x: Tensor) -> Tensor:
        return (self.alpha * x).tanh()


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(ffn_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x


class FocalModulation(nn.Module):
    def __init__(
        self,
        dim: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int,
        dropout: float = 0.0,
        use_post_norm: bool = False,
        tanhscale_init: float = 0.5,
        normalize_modulator: bool = False,
        causal: bool = False,
        window_size: int = 128,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.focal_level = focal_level
        self.normalize_modulator = normalize_modulator
        self.causal = causal
        self.use_post_norm = use_post_norm

        self.in_proj = nn.Linear(dim, 2 * dim + focal_level + 1)
        self.layers = nn.ModuleList()
        self.activation = nn.GELU()
        self.context_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.causal_pads = []

        for level in range(focal_level):
            kernel_size = focal_factor * level + focal_window
            self.causal_pads.append(kernel_size - 1)
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        padding=0 if causal else "same",
                        groups=dim,
                    ),
                    nn.GELU(),
                )
            )

        if causal:
            self.causal_pads.append(window_size - 1)
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(dim, dim, window_size, groups=dim),
                    nn.GELU(),
                )
            )

        if use_post_norm:
            self.norm = DynamicTanh(dim, tanhscale_init) if causal else nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x).permute(0, 2, 1)
        query, context, gates = x.split([self.dim, self.dim, self.focal_level + 1], dim=1)

        context_all = 0.0
        for level, layer in enumerate(self.layers):
            if self.causal:
                causal_pad = self.causal_pads[level]
                context = nn.functional.pad(context, [causal_pad, 0], mode="replicate")
            context = layer(context)
            context_all = context_all + context * gates[:, level : level + 1]

        if not self.causal:
            context_global = self.activation(context.mean(dim=-1))
            context_all = context_all + context_global[..., None] * gates[:, self.focal_level :]

        if self.normalize_modulator:
            context_all = context_all / (self.focal_level + 1)

        modulator = self.context_proj(context_all)
        x = (query * modulator).permute(0, 2, 1)
        x = self.norm(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x


class FocalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int,
        dropout: float = 0.0,
        use_post_norm: bool = False,
        use_layerscale: bool = False,
        layerscale_init: float = 1e-4,
        tanhscale_init: float = 0.5,
        normalize_modulator: bool = False,
        causal: bool = False,
        window_size: int = 128,
    ) -> None:
        super().__init__()
        norm_cls = DynamicTanh if causal else nn.LayerNorm
        self.modulation_norm = norm_cls(dim, tanhscale_init) if causal else norm_cls(dim)
        self.modulation = FocalModulation(
            dim=dim,
            focal_window=focal_window,
            focal_level=focal_level,
            focal_factor=focal_factor,
            dropout=dropout,
            use_post_norm=use_post_norm,
            tanhscale_init=tanhscale_init,
            normalize_modulator=normalize_modulator,
            causal=causal,
            window_size=window_size,
        )
        self.feed_forward_norm = norm_cls(dim, tanhscale_init) if causal else norm_cls(dim)
        self.feed_forward = FeedForward(dim=dim, ffn_dim=ffn_dim, dropout=dropout)

        if use_layerscale:
            self.modulation_gamma = nn.Parameter(torch.full((dim,), layerscale_init))
            self.feed_forward_gamma = nn.Parameter(torch.full((dim,), layerscale_init))
        else:
            self.modulation_gamma = 1.0
            self.feed_forward_gamma = 1.0

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.modulation_gamma * self.modulation(self.modulation_norm(x))
        x = x + self.feed_forward_gamma * self.feed_forward(self.feed_forward_norm(x))
        return x


class FocalEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: FocalQuantizerConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim or input_dim
        ffn_dim = config.ffn_dim or hidden_dim * 4

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                FocalBlock(
                    dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    focal_window=config.focal_window,
                    focal_level=config.focal_level,
                    focal_factor=config.focal_factor,
                    dropout=config.dropout,
                    use_post_norm=config.use_post_norm,
                    use_layerscale=config.use_layerscale,
                    layerscale_init=config.layerscale_init,
                    tanhscale_init=config.tanhscale_init,
                    normalize_modulator=config.normalize_modulator,
                    causal=config.causal,
                    window_size=config.window_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class FocalDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: FocalQuantizerConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim or output_dim
        ffn_dim = config.ffn_dim or hidden_dim * 4

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                FocalBlock(
                    dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    focal_window=config.focal_window,
                    focal_level=config.focal_level,
                    focal_factor=config.focal_factor,
                    dropout=config.dropout,
                    use_post_norm=config.use_post_norm,
                    use_layerscale=config.use_layerscale,
                    layerscale_init=config.layerscale_init,
                    tanhscale_init=config.tanhscale_init,
                    normalize_modulator=config.normalize_modulator,
                    causal=config.causal,
                    window_size=config.window_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
