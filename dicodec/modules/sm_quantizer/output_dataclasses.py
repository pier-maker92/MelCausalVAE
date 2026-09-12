from dataclasses import dataclass

import torch


@dataclass
class LatentBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor
    text_targets: torch.Tensor | None = None
    text_lengths: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "LatentBatch":
        return LatentBatch(self.inputs.to(device), self.targets.to(device), self.valid_mask.to(device),
                           None if self.text_targets is None else self.text_targets.to(device),
                           None if self.text_lengths is None else self.text_lengths.to(device))


@dataclass
class ASROutput:
    loss: torch.Tensor
    logits: torch.Tensor
    input_lengths: torch.Tensor


@dataclass
class QuantizerOutput:
    codes: torch.Tensor
    indices: torch.Tensor
    commitment_loss: torch.Tensor
    bsq_regularization_loss: torch.Tensor
    perplexity: torch.Tensor
    codebook_utilization_pct: torch.Tensor


@dataclass
class DiffusionOutput:
    loss: torch.Tensor
    velocity: torch.Tensor
    target_velocity: torch.Tensor


@dataclass
class SMQuantizerOutput:
    loss: torch.Tensor
    flow_loss: torch.Tensor
    reconstruction_l1: torch.Tensor
    reconstruction_l2: torch.Tensor
    commitment_loss: torch.Tensor
    indices: torch.Tensor
    quantized: torch.Tensor
    reconstruction: torch.Tensor | None
    context: torch.Tensor | None
    next_frame_mask: torch.Tensor
    bsq_regularization_loss: torch.Tensor
    perplexity: torch.Tensor
    codebook_utilization_pct: torch.Tensor
    asr_loss: torch.Tensor
    asr_logits: torch.Tensor | None = None
    wer_errors: int | None = None
    wer_words: int | None = None
    bsq_active: bool = False
    asr_curriculum_pct: float = 0.0
    asr_curriculum_reconstruction_weight: float = 0.0

    def metrics(self) -> dict[str, float]:
        names = ("loss", "flow_loss", "reconstruction_l1", "reconstruction_l2",
                 "commitment_loss", "bsq_regularization_loss", "perplexity", "codebook_utilization_pct", "asr_loss")
        metrics = {name: getattr(self, name).detach().item() for name in names}
        if self.asr_curriculum_pct:
            metrics["asr_curriculum_pct"] = self.asr_curriculum_pct
        metrics["asr_curriculum_reconstruction_weight"] = self.asr_curriculum_reconstruction_weight
        if not self.bsq_active:
            metrics.pop("bsq_regularization_loss")
        if self.wer_errors is not None:
            from .wer import word_error_rate

            metrics["wer"] = word_error_rate(self.wer_errors, self.wer_words)
        return metrics
