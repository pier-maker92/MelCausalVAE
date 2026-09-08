"""Mask-aware online quantization with backend-specific gradient estimators."""

import torch
from torch import nn
from torch.nn import functional as F

from ..quantizer.vq_ema import EMAVectorQuantizer
from ..quantizer.bsq import BinarySphericalQuantizer
from ..quantizer.fsq import FiniteScalarQuantizer
from .configs import QuantizerConfig
from .fsq_levels import FSQ_LEVELS
from .metrics import batch_codebook_metrics
from .output_dataclasses import QuantizerOutput


class OnlineQuantizer(nn.Module):
    def __init__(self, config: QuantizerConfig):
        super().__init__()
        self.config = config
        self.codebook = self.build_codebook(config)
        # The shared quantizer normally omits this counter from checkpoints.
        # Persist it here so resuming preserves the dead-code reset schedule.
        if config.type == "vq_ema":
            self.codebook.register_buffer("_forward_count", self.codebook._forward_count, persistent=True)

    @staticmethod
    def build_codebook(config: QuantizerConfig) -> nn.Module:
        if config.type == "bsq":
            return BinarySphericalQuantizer(config.codebook_size)
        if config.type == "fsq":
            quantizer = FiniteScalarQuantizer(config.codebook_size)
            if tuple(quantizer.levels_list) != FSQ_LEVELS[config.codebook_size]:
                raise ValueError("Shared FSQ levels do not match the SM quantizer preset.")
            return quantizer
        return EMAVectorQuantizer(
            dim=config.dim, codebook_size=config.codebook_size, decay=config.decay,
            eps=config.eps, reset_dead_codes=config.reset_dead_codes,
            reset_every_forward=config.reset_every_forward,
        )

    @property
    def dim(self) -> int:
        return self.codebook.dim

    def bsq_regularization(self, frames: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(frames.float() / self.config.entropy_temperature)
        average = probs.mean(dim=0)
        negative_entropy = (average * torch.log(average + 1e-5)
                            + (1 - average) * torch.log(1 - average + 1e-5)).mean()
        centered = probs - average
        covariance = centered.T @ centered / probs.shape[0]
        off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
        return negative_entropy + off_diagonal.square().mean()

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> QuantizerOutput:
        # Gather before clustering: padding must never update the EMA statistics.
        frames = x[valid]
        if frames.shape[0] == 0:
            raise ValueError("Quantization requires at least one valid frame.")
        # FSQ already implements STE rounding and must keep its tanh derivative.
        source = frames if self.config.type == "fsq" else frames.detach().float()
        indices, centers = self.codebook(source)
        centers = centers.to(frames.dtype)
        commitment = frames.new_zeros(())
        regularization = frames.new_zeros(())
        if self.config.type == "vq_ema":
            commitment = F.mse_loss(frames, centers.detach())
        elif self.config.type == "bsq":
            regularization = self.bsq_regularization(frames)
        straight_through = centers if self.config.type == "fsq" else frames + (centers - frames).detach()
        codes = torch.zeros_like(x)
        codes[valid] = straight_through
        tokens = torch.full(valid.shape, -1, dtype=torch.long, device=x.device)
        tokens[valid] = indices
        perplexity, utilization = batch_codebook_metrics(indices, self.codebook.codebook_size)
        return QuantizerOutput(codes, tokens, commitment, regularization, perplexity, utilization)
