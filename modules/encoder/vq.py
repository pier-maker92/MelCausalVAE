import torch
import torch.nn as nn
from typing import Optional
from ..configs import VQConfig
import torch.nn.functional as F
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
    ):
        super().__init__()
        self.config = config
        self.dim = config.dim_to_quantize
        self.num_embeddings = config.num_embeddings
        
        self.quantizer = BinarySphericalQuantizer(codebook_size=self.num_embeddings)
        
        self.proj_in = nn.Sequential(
            nn.Linear(self.dim, self.dim*4),
            nn.GELU(),
            nn.Linear(self.dim*4, self.dim*4),
            nn.GELU(),
            nn.Linear(self.dim*4, self.quantizer.dim),
            nn.LayerNorm(self.quantizer.dim)
        )
        self.proj_out = nn.Sequential(
            nn.Linear(self.quantizer.dim, self.dim*4),
            nn.GELU(),
            nn.Linear(self.dim*4, self.dim*4),
            nn.GELU(),
            nn.Linear(self.dim*4, self.dim)
        )

    def forward(
        self,
        z: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        drop_acoustic: bool = False,
    ) -> VQVAEOutput:
        """
        Args:
            z: ``[B, T, dim]`` continuous features.
            padding_mask: ``[B, T]`` with ``True`` = padded (ignored in VQ loss).

        Returns:
            VQVAEOutput
        """
        B, T, D = z.shape
        
        # Split z if D > self.dim
        z_qtz = z[..., :self.dim]
        z_pass = z[..., self.dim:] if not drop_acoustic else torch.zeros_like(z[..., self.dim:])

        # Project down to BSQ dim
        z_proj = self.proj_in(z_qtz)
        
        # Quantize using BSQ
        indices_bt, z_q_proj = self.quantizer(z_proj)
        
        # Straight-Through Estimator (STE)
        z_q_proj_st = z_proj + (z_q_proj - z_proj).detach()
        
        # Project back to original quantized dim
        z_q_rec = self.proj_out(z_q_proj_st)
        
        # Recombine if there were extra dimensions
        z_q = torch.cat([z_q_rec, z_pass], dim=-1)

        z_residual = z - z_q.detach()
        
        if padding_mask is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=z.device)
        else:
            valid = ~padding_mask

        # Compute stats using all valid positions
        stats = _batch_vq_stats(indices_bt, valid, self.num_embeddings, z)
        
        recon_loss = F.mse_loss(z_q_rec[valid], z_qtz[valid])

        return VQVAEOutput(
            indices=indices_bt,
            quantized=z_q,
            residual=z_residual,
            stats=stats,
            loss=recon_loss,
        )
