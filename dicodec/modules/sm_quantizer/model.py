import torch
from torch import nn
from torch.nn import functional as F

from .configs import ModelConfig
from .asr import ASRHead
from .encoder import build_encoder
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
        self.encoder = build_encoder(config.latent_dim, quant_dim, config.projection_hidden_dim, config.encoder)
        self.quantizer = OnlineQuantizer(config.quantizer)
        self.reconstruction_head = (
            projection(quant_dim, config.projection_hidden_dim, config.latent_dim)
            if config.loss.reconstruction_l1 > 0 or config.loss.reconstruction_l2 > 0 else None
        )
        self.decoder = CausalDecoder(quant_dim, config.transformer) if config.language_modeling else None
        self.diffusion_head = (DiffusionHead(config.latent_dim, config.transformer.dim, config.diffusion)
                               if config.language_modeling else None)
        self.asr_head = ASRHead(quant_dim, config.asr) if config.asr.enabled else None

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

    def _asr_curriculum_codes(
        self,
        encoded: torch.Tensor,
        quantized: torch.Tensor,
        ratio: float,
    ) -> tuple[torch.Tensor, float]:
        if self.asr_head is None or not self.config.asr.curriculum or ratio <= 0:
            return quantized, 0.0
        ratio = min(max(float(ratio), 0.0), 1.0)
        batch_size = encoded.shape[0]
        num_bypass = min(batch_size, max(0, int(batch_size * ratio + 0.5)))
        if num_bypass == 0:
            return quantized, 0.0
        selected = torch.randperm(batch_size, device=encoded.device)[:num_bypass]
        sample_mask = torch.zeros(batch_size, dtype=torch.bool, device=encoded.device)
        sample_mask[selected] = True
        codes = torch.where(sample_mask[:, None, None], encoded, quantized)
        return codes, 100.0 * num_bypass / batch_size

    def forward(self, z: torch.Tensor, valid_mask: torch.Tensor | None = None,
                *, target: torch.Tensor | None = None, text_targets: torch.Tensor | None = None,
                text_lengths: torch.Tensor | None = None,
                asr_curriculum_ratio: float = 0.0) -> SMQuantizerOutput:
        valid = self.validate_input(z, valid_mask)
        target = z if target is None else target
        if target.shape != z.shape or target.device != z.device or target.dtype != z.dtype:
            raise ValueError("Target must match input shape, device and dtype.")
        self.validate_input(target, valid)
        pairs = valid[:, :-1] & valid[:, 1:]
        if self.config.language_modeling and not pairs.any():
            raise ValueError("Training requires at least one consecutive pair of latent frames.")
        if self.config.language_modeling and z.shape[1] > self.config.transformer.max_length:
            raise ValueError("Sequence exceeds transformer.max_length.")
        if self.asr_head is not None:
            # Fail before quantization to avoid updating EMA on an invalid batch.
            self.asr_head.validate_targets(text_targets, text_lengths, valid.sum(1) * self.config.asr.upsample_factor)
        encoder_input = z.detach().masked_fill(~valid.unsqueeze(-1), 0)
        encoded = self.encoder(encoder_input)
        quantized = self.quantizer(encoded, valid)
        reconstruction = None
        l1, l2 = z.new_zeros(()), z.new_zeros(())
        if self.reconstruction_head is not None:
            reconstruction = self.reconstruction_head(quantized.codes)
            if self.config.loss.reconstruction_l1 > 0:
                l1 = F.l1_loss(reconstruction[valid], target[valid].detach())
            if self.config.loss.reconstruction_l2 > 0:
                l2 = F.mse_loss(reconstruction[valid], target[valid].detach())
            reconstruction = reconstruction.masked_fill(~valid.unsqueeze(-1), 0)
        # Context at t sees only q[0:t+1]; its target is the next selected frame.
        context = None
        flow_loss = z.new_zeros(())
        if self.config.language_modeling:
            context = self.decoder(quantized.codes[:, :-1], valid[:, :-1])
            flow_loss = self.diffusion_head(target[:, 1:], context, pairs).loss
        asr_loss = z.new_zeros(())
        asr_logits = None
        wer_errors = wer_words = None
        asr_curriculum_pct = 0.0
        if self.asr_head is not None:
            asr_codes, asr_curriculum_pct = self._asr_curriculum_codes(
                encoded, quantized.codes, asr_curriculum_ratio,
            )
            asr = self.asr_head(asr_codes, valid, text_targets, text_lengths)
            asr_loss, asr_logits = asr.loss, asr.logits
            if self.config.loss.asr > 0:
                wer_errors, wer_words = self.asr_head.word_error_counts(
                    asr.logits, asr.input_lengths, text_targets, text_lengths)
        weights = self.config.loss
        loss = (weights.flow * flow_loss + weights.reconstruction_l1 * l1
                + weights.reconstruction_l2 * l2 + weights.commitment * quantized.commitment_loss
                + weights.bsq_regularization * quantized.bsq_regularization_loss + weights.asr * asr_loss)
        return SMQuantizerOutput(
            loss, flow_loss, l1, l2, quantized.commitment_loss, quantized.indices,
            quantized.codes, reconstruction, context, pairs,
            quantized.bsq_regularization_loss,
            quantized.perplexity, quantized.codebook_utilization_pct,
            asr_loss, asr_logits, wer_errors, wer_words, self.config.quantizer.type == "bsq",
            asr_curriculum_pct,
        )

    @torch.no_grad()
    def predict_next(self, z: torch.Tensor, valid_mask: torch.Tensor | None = None, **sampling_kwargs) -> torch.Tensor:
        """Sample one frame [B, D] in target space from an input prefix; call eval() first."""
        if self.training:
            raise RuntimeError("Call eval() before prediction to freeze the EMA codebook and dropout.")
        if not self.config.language_modeling:
            raise RuntimeError("predict_next requires language_modeling=true.")
        valid = self.validate_input(z, valid_mask)
        codes = self.encode(z, valid).codes
        context = self.decoder(codes, valid)
        last = context[torch.arange(z.shape[0], device=z.device), valid.sum(1) - 1]
        return self.diffusion_head.sample(last, **sampling_kwargs)
