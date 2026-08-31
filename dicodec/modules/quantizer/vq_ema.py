import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

__all__ = ["EMAVectorQuantizer"]


class EMAVectorQuantizer(nn.Module):
    """
    Standard nearest-neighbor VQ with EMA codebook updates.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 256,
        decay: float = 0.99,
        eps: float = 1e-5,
        reset_dead_codes: bool = False,
        reset_every_forward: int = 10,
    ):
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        if eps <= 0.0:
            raise ValueError("EMA eps must be > 0.")
        if reset_every_forward <= 0:
            raise ValueError("reset_every_forward must be > 0.")

        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps
        self.reset_dead_codes = reset_dead_codes
        self.reset_every_forward = reset_every_forward

        embedding = torch.empty(codebook_size, dim)
        nn.init.uniform_(embedding, -1.0 / codebook_size, 1.0 / codebook_size)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        self.register_buffer("embed_avg", embedding.clone())
        self.register_buffer("_forward_count", torch.zeros((), dtype=torch.long), persistent=False)

    def forward(self, lats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lats: [..., D]

        Returns:
            toks: [...]
            codes: [..., D]
        """
        flatten = lats.reshape(-1, self.dim)
        embedding = self.embedding.to(dtype=flatten.dtype)

        distances = (
            torch.sum(flatten**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.matmul(flatten, embedding.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        codes_flat = F.embedding(encoding_indices, embedding)

        if self.training:
            self._ema_update(flatten.detach(), encoding_indices)
            self._forward_count.add_(1)
            if (
                self.reset_dead_codes
                and self._forward_count.item() % self.reset_every_forward == 0
            ):
                self._reset_dead_codes(flatten.detach())

        toks = encoding_indices.view(*lats.shape[:-1])
        codes = codes_flat.view_as(lats).to(dtype=lats.dtype)
        return toks, codes

    @torch.no_grad()
    def _ema_update(
        self,
        flatten: torch.Tensor,
        encoding_indices: torch.Tensor,
    ) -> torch.Tensor:
        one_hot = F.one_hot(encoding_indices, self.codebook_size).to(
            dtype=self.cluster_size.dtype
        )
        counts = one_hot.sum(dim=0)
        embed_sum = torch.matmul(one_hot.t(), flatten.to(dtype=self.embed_avg.dtype))

        self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + self.eps)
            / (n + self.codebook_size * self.eps)
            * n.clamp(min=self.eps)
        )
        self.embedding.copy_(self.embed_avg / smoothed_cluster_size.unsqueeze(1))
        return counts

    @torch.no_grad()
    def _reset_dead_codes(
        self,
        flatten: torch.Tensor,
    ) -> bool:
        # A code is dead when its smoothed historical usage has decayed, not
        # merely because it was absent from the current mini-batch.
        dead_mask = self.cluster_size < 0.5
        num_dead = int(dead_mask.sum().item())
        if num_dead == 0 or flatten.numel() == 0:
            return False

        dead_indices = torch.nonzero(dead_mask, as_tuple=False).squeeze(1)
        random_indices = torch.randint(
            0,
            flatten.shape[0],
            (num_dead,),
            device=flatten.device,
        )
        replacement = flatten[random_indices].to(dtype=self.embedding.dtype)

        self.embedding[dead_indices] = replacement
        self.embed_avg[dead_indices] = replacement
        self.cluster_size[dead_indices] = 1.0
        return True
