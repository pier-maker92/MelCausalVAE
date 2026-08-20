import torch
import torch.nn as nn
from typing import Optional
from ..configs import VQConfig
import torch.nn.functional as F
from .fsq import FiniteScalarQuantizer
from .bsq import BinarySphericalQuantizer
from .pitch_loss import PitchLoss
from ..output_dataclasses import VQVAEOutput, VQStats


def _batch_vq_stats(
    indices_bt: torch.Tensor,
    valid: torch.BoolTensor,
    num_embeddings: int,
    ref: torch.Tensor,
) -> VQStats:
    """
    Empirical entropy perplexity exp(H) over code indices in the batch,
    and how many distinct codes appear (absolute and relative to codebook size).
    """
    if not valid.any():
        z = ref.new_zeros(())
        return VQStats(perplexity=z, codes_used=z, codes_used_frac=z)
    idx = indices_bt[valid].long()
    counts = torch.bincount(idx, minlength=num_embeddings).float()
    total = counts.sum().clamp(min=1.0)
    p = counts / total
    log_p = torch.log(p + 1e-10)
    entropy = -(p * log_p).sum()
    perplexity = entropy.exp()
    codes_used = (counts > 0).sum().to(dtype=torch.float32)
    codes_used_frac = codes_used / float(num_embeddings)
    return VQStats(
        perplexity=perplexity.detach(),
        codes_used=codes_used.detach(),
        codes_used_frac=codes_used_frac.detach(),
    )


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        config: VQConfig,
        dim: int,
        pitch_loss_config=None,
        acoustic_dim: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.pitch_loss = None
        if pitch_loss_config is not None:
            if acoustic_dim is None:
                raise ValueError(
                    "acoustic_dim is required when pitch_loss_config is set."
                )
            self.pitch_loss = PitchLoss(
                pitch_loss_config,
                semantic_dim=dim,
                acoustic_dim=acoustic_dim,
            )
        self.num_embeddings = config.num_embeddings
        self.vq_type = config.vq_type
        if self.vq_type == "fsq":
            self.quantizer = FiniteScalarQuantizer(codebook_size=self.num_embeddings)
        elif self.vq_type == "bsq":
            self.quantizer = BinarySphericalQuantizer(codebook_size=self.num_embeddings)
        elif self.vq_type == "vq":
            from .std_vq import StandardVectorQuantizer

            vq_dim = getattr(config, "vq_dim")
            self.quantizer = StandardVectorQuantizer(
                dim=vq_dim, codebook_size=self.num_embeddings
            )
        elif self.vq_type == "vq_ema":
            from .vq_ema import EMAVectorQuantizer

            vq_dim = getattr(config, "vq_dim")
            self.quantizer = EMAVectorQuantizer(
                dim=vq_dim,
                codebook_size=self.num_embeddings,
                decay=config.ema_decay,
                eps=config.ema_eps,
                reset_dead_codes=config.reset_dead_codes,
                reset_every_forward=config.reset_every_forward,
            )
        else:
            raise ValueError(f"Unknown vq_type: {self.vq_type}")

        self.proj_in = nn.Sequential(
            nn.Linear(self.dim, self.quantizer.dim, bias=False),
            nn.LayerNorm(self.quantizer.dim),
        )
        self.proj_out = nn.Sequential(
            nn.Linear(self.quantizer.dim, self.dim, bias=True),
        )

    def forward(
        self,
        z: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        acoustic: Optional[torch.Tensor] = None,
        pitch_targets=None,
        step: Optional[int] = None,
    ) -> VQVAEOutput:
        """
        Args:
            z: ``[B, T, dim]`` continuous features.
            padding_mask: ``[B, T]`` with ``True`` = padded (ignored in VQ loss).

        Returns:
            VQVAEOutput
        """
        B, T, D = z.shape

        if D != self.dim:
            raise ValueError(
                f"VectorQuantizer expected {self.dim} dimensions, received {D}."
            )

        z_proj = self.proj_in(z)

        if padding_mask is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=z.device)
        else:
            valid = ~padding_mask

        flat_valid = valid.reshape(-1)
        z_proj_flat = z_proj.reshape(-1, z_proj.shape[-1])
        z_q_proj_flat = z_proj_flat.clone()
        indices_flat = torch.zeros(
            z_proj_flat.shape[0], device=z.device, dtype=torch.long
        )

        z_proj_valid = None
        z_q_proj_valid = None
        if flat_valid.any():
            z_proj_valid = z_proj_flat[flat_valid]
            indices_valid, z_q_proj_valid = self.quantizer(z_proj_valid)
            indices_valid = indices_valid.long()
            z_q_proj_flat[flat_valid] = z_q_proj_valid.to(dtype=z_proj_flat.dtype)
            indices_flat[flat_valid] = indices_valid

        z_q_proj = z_q_proj_flat.view_as(z_proj)
        indices_bt = indices_flat.view(B, T)

        # Straight-Through Estimator (STE)
        z_q_proj_st = z_proj + (z_q_proj - z_proj).detach()

        z_q = self.proj_out(z_q_proj_st)

        z_residual = z - z_q.detach()

        # Compute stats using all valid positions
        stats = _batch_vq_stats(indices_bt, valid, self.num_embeddings, z)

        total_loss = z.new_zeros(())
        if self.vq_type in {"vq", "vq_ema"} and z_proj_valid is not None:
            commitment_loss = F.mse_loss(z_proj_valid, z_q_proj_valid.detach())
            if self.vq_type == "vq":
                codebook_loss = F.mse_loss(
                    z_q_proj_valid,
                    z_proj_valid.detach(),
                )
                total_loss = (
                    codebook_loss + self.config.commitment_weight * commitment_loss
                )
            else:
                total_loss = self.config.commitment_weight * commitment_loss

        pitch_losses = {}
        if self.pitch_loss is not None and pitch_targets is not None:
            if acoustic is None:
                raise ValueError(
                    "acoustic is required when pitch targets are provided."
                )
            pitch_losses = self.pitch_loss(
                semantic_quantized=z_q,
                acoustic=acoustic,
                pitch_targets=pitch_targets,
                padding_mask=padding_mask,
                step=step,
            )
            total_loss = total_loss + pitch_losses["loss"]

        return VQVAEOutput(
            indices=indices_bt,
            quantized=z_q,
            residual=z_residual,
            stats=stats,
            loss=total_loss,
            acoustic_f0_loss=pitch_losses.get("acoustic_f0_loss"),
            semantic_f0_adv_loss=pitch_losses.get("semantic_f0_adv_loss"),
            pitch_voiced_loss=pitch_losses.get("pitch_voiced_loss"),
            pitch_contour_loss=pitch_losses.get("pitch_contour_loss"),
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
