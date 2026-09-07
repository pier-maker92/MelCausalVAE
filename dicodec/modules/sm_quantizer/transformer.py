import torch
from torch import nn

from .configs import TransformerConfig


class CausalDecoder(nn.Module):
    """Decoder-only self attention; no cross attention or future context."""

    def __init__(self, input_dim: int, config: TransformerConfig):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, config.dim)
        self.positions = nn.Embedding(config.max_length, config.dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                config.dim, config.heads, config.ff_dim, config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(config.layers)
        ])
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, codes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        length = codes.shape[1]
        if length > self.positions.num_embeddings:
            raise ValueError("Sequence exceeds transformer.max_length.")
        x = self.input_projection(codes) + self.positions(torch.arange(length, device=codes.device))
        causal_mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=~valid)
        return self.norm(x).masked_fill(~valid.unsqueeze(-1), 0)
