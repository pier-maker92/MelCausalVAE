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
            vq_dim = getattr(config, "vq_dim", None)
            if vq_dim is None:
                vq_dim = 128
            self.quantizer = StandardVectorQuantizer(dim=vq_dim, codebook_size=self.num_embeddings)
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
            z_q_proj_flat[flat_valid] = z_q_proj_valid.to(dtype=z_proj_flat.dtype)
            indices_flat[flat_valid] = indices_valid.long()

        z_q_proj = z_q_proj_flat.view_as(z_proj)
        indices_bt = indices_flat.view(B, T)

        # Straight-Through Estimator (STE)
        z_q_proj_st = z_proj + (z_q_proj - z_proj).detach()
        
        z_q_rec = self.proj_out(z_q_proj_st)
        z_q = z_q_rec

        z_residual = z - z_q.detach()
        
        # Compute stats using all valid positions
        stats = _batch_vq_stats(indices_bt, valid, self.num_embeddings, z)

        total_loss = z.new_zeros(())
        if valid.any():
            total_loss = F.mse_loss(z_q_rec[valid], z[valid])

        return VQVAEOutput(
            indices=indices_bt,
            quantized=z_q,
            residual=z_residual,
            stats=stats,
            loss=total_loss,
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
