from dataclasses import dataclass

import torch


@dataclass
class LatentBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "LatentBatch":
        return LatentBatch(self.inputs.to(device), self.targets.to(device), self.valid_mask.to(device))


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
    reconstruction: torch.Tensor
    context: torch.Tensor
    next_frame_mask: torch.Tensor
    bsq_regularization_loss: torch.Tensor
    perplexity: torch.Tensor
    codebook_utilization_pct: torch.Tensor

    def metrics(self) -> dict[str, float]:
        names = ("loss", "flow_loss", "reconstruction_l1", "reconstruction_l2",
                 "commitment_loss", "bsq_regularization_loss", "perplexity", "codebook_utilization_pct")
        return {name: getattr(self, name).detach().item() for name in names}
