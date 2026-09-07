"""Mask-aware EMA clustering with a straight-through training path."""

import torch
from torch import nn
from torch.nn import functional as F

from ..quantizer.vq_ema import EMAVectorQuantizer
from .configs import QuantizerConfig
from .output_dataclasses import QuantizerOutput


class OnlineQuantizer(nn.Module):
    def __init__(self, config: QuantizerConfig):
        super().__init__()
        self.codebook = EMAVectorQuantizer(**vars(config))
        # The shared quantizer normally omits this counter from checkpoints.
        # Persist it here so resuming preserves the dead-code reset schedule.
        self.codebook.register_buffer("_forward_count", self.codebook._forward_count, persistent=True)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> QuantizerOutput:
        # Gather before clustering: padding must never update the EMA statistics.
        frames = x[valid]
        indices, centers = self.codebook(frames.detach().float())
        centers = centers.to(frames.dtype)
        commitment = F.mse_loss(frames, centers.detach())
        straight_through = frames + (centers - frames).detach()
        codes = torch.zeros_like(x)
        codes[valid] = straight_through
        tokens = torch.full(valid.shape, -1, dtype=torch.long, device=x.device)
        tokens[valid] = indices
        return QuantizerOutput(codes, tokens, commitment)
