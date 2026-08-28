import torch
import torch.nn as nn
from typing import Optional
from ..configs import VQConfig
import torch.nn.functional as F
from .fsq import FiniteScalarQuantizer
from .bsq import BinarySphericalQuantizer
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
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.num_embeddings = config.num_embeddings
        self.vq_type = config.vq_type
        if self.vq_type == "fsq":
            self.quantizer = FiniteScalarQuantizer(codebook_size=self.num_embeddings)
        elif self.vq_type == "bsq":
            self.quantizer = BinarySphericalQuantizer(codebook_size=self.num_embeddings)
        elif self.vq_type == "vq":
            from .std_vq import StandardVectorQuantizer
            vq_dim = getattr(config, "vq_dim")
            self.quantizer = StandardVectorQuantizer(dim=vq_dim, codebook_size=self.num_embeddings)
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
        
        self.recon_weight = getattr(config, "recon_weight", None)
        self.proj_in = nn.Sequential(
            nn.Linear(self.dim, self.quantizer.dim, bias=False),
            nn.LayerNorm(self.quantizer.dim),
        )
        self.proj_out = nn.Linear(self.quantizer.dim, self.dim, bias=True)

    def forward(
        self,
        z: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
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

        z_proj = self._encode(z)
        
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
        
        z_q = self._decode(z_q_proj_st)

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

        if self.vq_type == "bsq" and z_proj_valid is not None:
            entropy_weight = getattr(self.config, "entropy_loss_weight", 0.0)
            if entropy_weight > 0.0:
                temp = getattr(self.config, "entropy_temperature", 1.0)
                # Probabilità marginale di bit = 1
                p = torch.sigmoid(z_proj_valid / temp)
                p_avg = p.mean(dim=0)
                # Vogliamo massimizzare l'entropia, quindi minimizziamo l'entropia negativa
                entropy = -p_avg * torch.log(p_avg + 1e-7) - (1 - p_avg) * torch.log(1 - p_avg + 1e-7)
                entropy_loss = -entropy.mean()
                total_loss = total_loss + entropy_weight * entropy_loss
        
        if self.recon_weight is not None:
            recon_loss = F.mse_loss(z_q[valid], z[valid])
            total_loss = total_loss + self.recon_weight * recon_loss

        return VQVAEOutput(
            indices=indices_bt,
            quantized=z_q,
            residual=z_residual,
            stats=stats,
            loss=total_loss,
        )

    def _encode(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj_in(z)

    def _decode(self, z_q_proj: torch.Tensor) -> torch.Tensor:
        return self.proj_out(z_q_proj)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
