import torch
from torch import nn
from torch.nn import functional as F

from .configs import ModelConfig
from .diffusion import DiffusionHead
from .output_dataclasses import QuantizerOutput, SMQuantizerOutput
from .quantizer import OnlineQuantizer
from .transformer import CausalDecoder


def projection(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


class SMQuantizer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Resolve fixed BSQ/FSQ dimensions before constructing any projections.
        # Keep EMA's parameter initialization order compatible with old runs.
        quant_dim = config.quantizer.resolved_dim
        self.encoder = projection(config.latent_dim, config.projection_hidden_dim, quant_dim)
        self.quantizer = OnlineQuantizer(config.quantizer)
        self.reconstruction_head = projection(quant_dim, config.projection_hidden_dim, config.latent_dim)
        self.decoder = CausalDecoder(quant_dim, config.transformer)
        self.diffusion_head = DiffusionHead(config.latent_dim, config.transformer.dim, config.diffusion)

    def validate_input(self, z: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        if z.ndim != 3 or z.shape[-1] != self.config.latent_dim or min(z.shape[:2]) < 1:
            raise ValueError("Expected nonempty latents [B, T, latent_dim].")
        if not z.is_floating_point():
            raise TypeError("Latents must be floating point.")
        if valid_mask is None:
            valid_mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        if valid_mask.shape != z.shape[:2] or valid_mask.dtype != torch.bool or valid_mask.device != z.device:
            raise ValueError("valid_mask must be bool [B, T] on the latent device (True = valid).")
        if not valid_mask[:, 0].all() or (valid_mask[:, 1:] & ~valid_mask[:, :-1]).any():
            raise ValueError("Every sequence must have a nonempty prefix followed by right padding.")
        if not torch.isfinite(z[valid_mask]).all():
            raise ValueError("Valid latent frames must be finite.")
        return valid_mask

    def encode(self, z: torch.Tensor, valid_mask: torch.Tensor | None = None) -> QuantizerOutput:
        valid = self.validate_input(z, valid_mask)
        z = z.detach().masked_fill(~valid.unsqueeze(-1), 0)
        return self.quantizer(self.encoder(z), valid)

    def forward(self, z: torch.Tensor, valid_mask: torch.Tensor | None = None,
                *, target: torch.Tensor | None = None) -> SMQuantizerOutput:
        valid = self.validate_input(z, valid_mask)
        target = z if target is None else target
        if target.shape != z.shape or target.device != z.device or target.dtype != z.dtype:
            raise ValueError("Target must match input shape, device and dtype.")
        self.validate_input(target, valid)
        pairs = valid[:, :-1] & valid[:, 1:]
        if not pairs.any():
            raise ValueError("Training requires at least one consecutive pair of latent frames.")
        if z.shape[1] > self.config.transformer.max_length:
            raise ValueError("Sequence exceeds transformer.max_length.")
        quantized = self.encode(z, valid)
        reconstruction = self.reconstruction_head(quantized.codes)
        l1 = F.l1_loss(reconstruction[valid], target[valid].detach())
        l2 = F.mse_loss(reconstruction[valid], target[valid].detach())
        # Context at t sees only q[0:t+1]; its target is the next selected frame.
        context = self.decoder(quantized.codes[:, :-1], valid[:, :-1])
        flow = self.diffusion_head(target[:, 1:], context, pairs)
        weights = self.config.loss
        loss = (weights.flow * flow.loss + weights.reconstruction_l1 * l1
                + weights.reconstruction_l2 * l2 + weights.commitment * quantized.commitment_loss
                + weights.bsq_regularization * quantized.bsq_regularization_loss)
        return SMQuantizerOutput(
            loss, flow.loss, l1, l2, quantized.commitment_loss, quantized.indices,
            quantized.codes, reconstruction.masked_fill(~valid.unsqueeze(-1), 0), context, pairs,
            quantized.bsq_regularization_loss,
            quantized.perplexity, quantized.codebook_utilization_pct,
        )

    @torch.no_grad()
    def predict_next(self, z: torch.Tensor, valid_mask: torch.Tensor | None = None, **sampling_kwargs) -> torch.Tensor:
        """Sample one frame [B, D] in target space from an input prefix; call eval() first."""
        if self.training:
            raise RuntimeError("Call eval() before prediction to freeze the EMA codebook and dropout.")
        valid = self.validate_input(z, valid_mask)
        codes = self.encode(z, valid).codes
        context = self.decoder(codes, valid)
        last = context[torch.arange(z.shape[0], device=z.device), valid.sum(1) - 1]
        return self.diffusion_head.sample(last, **sampling_kwargs)
