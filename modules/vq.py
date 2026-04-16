"""
Minimal vector quantisation for encoder residuals.

Single codebook of ``num_embeddings`` vectors in ``dim`` space (encoder latent dim when VQ is after ``mu``).
Hard nearest-neighbour assignment + VQ-VAE style commitment loss.

Optional EMA codebook updates (no grad on embeddings); commitment still trains the encoder.

Forward returns ``z - z_q`` (quantised vector subtracted) for use as input
to Gaussian heads, while ``vq_loss`` trains the encoder (and codebook if not EMA).
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQBatchStats(NamedTuple):
    """Detached scalars for logging (valid positions only)."""

    perplexity: torch.Tensor
    codes_used: torch.Tensor
    codes_used_frac: torch.Tensor


def _batch_vq_stats(
    indices_bt: torch.Tensor,
    valid: torch.BoolTensor,
    num_embeddings: int,
    ref: torch.Tensor,
) -> VQBatchStats:
    """
    Empirical entropy perplexity exp(H) over code indices in the batch,
    and how many distinct codes appear (absolute and relative to codebook size).
    """
    if not valid.any():
        z = ref.new_zeros(())
        return VQBatchStats(perplexity=z, codes_used=z, codes_used_frac=z)
    idx = indices_bt[valid].long()
    counts = torch.bincount(idx, minlength=num_embeddings).float()
    total = counts.sum().clamp(min=1.0)
    p = counts / total
    log_p = torch.log(p + 1e-10)
    entropy = -(p * log_p).sum()
    perplexity = entropy.exp()
    codes_used = (counts > 0).sum().to(dtype=torch.float32)
    codes_used_frac = codes_used / float(num_embeddings)
    return VQBatchStats(
        perplexity=perplexity.detach(),
        codes_used=codes_used.detach(),
        codes_used_frac=codes_used_frac.detach(),
    )


def _deterministic_sample_indices(n: int, m: int, step: int, device: torch.device) -> torch.Tensor:
    """``m`` indices in ``[0, n)`` identical across ranks given the same ``step`` (DDP-safe)."""
    if n <= 0 or m <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    t = torch.arange(m, device=device, dtype=torch.int64)
    return (t * 2654435761 + int(step) * 1597334677) % int(n)


class HardVectorQuantizer(nn.Module):
    """
    Args:
        dim: feature dimension (e.g. encoder ``d_model``).
        num_embeddings: codebook size (number of discrete codes).
        commit_weight: weight on commitment term (encoder pulls toward codes).
        reset_dead_codes: if True (training), reassign unused codes each step to random
            encoder vectors (valid positions), mitigating collapse.
        reset_max_per_step: cap how many dead codes are reset per forward (None = all).
        reset_every_forward: run dead-code reset only every N forwards (use >1 if using
            gradient accumulation to avoid thrashing the codebook mid-accumulation).
        use_ema_codebook: if True, update ``codebook`` with EMA of batch cluster stats
            (no optimizer grad on embeddings); ``vq_loss`` is commitment-only.
        ema_decay: decay for EMA buffers (per-code updates only when the code appears
            in the current batch).
        ema_epsilon: floor on cluster size when dividing to form code vectors.
    """

    def __init__(
        self,
        dim: int,
        num_embeddings: int,
        commit_weight: float = 0.25,
        reset_dead_codes: bool = True,
        reset_max_per_step: Optional[int] = None,
        reset_every_forward: int = 1,
        use_ema_codebook: bool = False,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.num_embeddings = num_embeddings
        self.commit_weight = float(commit_weight)
        self.reset_dead_codes = bool(reset_dead_codes)
        self.reset_max_per_step = reset_max_per_step
        self.reset_every_forward = max(1, int(reset_every_forward))
        self.use_ema_codebook = bool(use_ema_codebook)
        self.ema_decay = float(ema_decay)
        self.ema_epsilon = float(ema_epsilon)
        self.register_buffer("_vq_forward_index", torch.tensor(0, dtype=torch.long))
        self.codebook = nn.Embedding(num_embeddings, dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=dim**-0.5)

        if self.use_ema_codebook:
            self.codebook.weight.requires_grad_(False)
            self.register_buffer(
                "_ema_cluster_size",
                torch.ones(num_embeddings, dtype=torch.float32),
            )
            self.register_buffer(
                "_ema_embedding_sum",
                self.codebook.weight.data.detach().float().clone(),
            )

    def forward(
        self,
        z: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        global_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, VQBatchStats]:
        """
        Args:
            z: ``[B, T, dim]`` continuous features.
            padding_mask: ``[B, T]`` with ``True`` = padded (ignored in VQ loss).
            global_step: training step for deterministic dead-code reset (DDP-safe).

        Returns:
            z_residual: ``[B, T, dim]`` = ``z - z_q`` (``z_q`` detached for subtraction).
            vq_loss: scalar commitment + optional codebook loss.
            z_q: ``[B, T, dim]`` quantised vectors (grad flows to codebook if not EMA).
            stats: perplexity and code usage on non-padded positions (detached).
        """
        B, T, D = z.shape
        if D != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {D}")

        flat = z.reshape(B * T, D)
        emb = self.codebook.weight
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ emb.t()
            + emb.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = dist.argmin(dim=1)
        z_q = self.codebook(indices).view(B, T, D)

        z_residual = z - z_q.detach()

        if padding_mask is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=z.device)
        else:
            valid = ~padding_mask

        indices_bt = indices.view(B, T)
        stats = _batch_vq_stats(indices_bt, valid, self.num_embeddings, z)

        if valid.any():
            z_e = z[valid]
            z_qv = z_q[valid]
            loss_commit = F.mse_loss(z_e, z_qv.detach())
            if self.use_ema_codebook:
                vq_loss = self.commit_weight * loss_commit
            else:
                loss_codebook = F.mse_loss(z_qv, z_e.detach())
                vq_loss = loss_codebook + self.commit_weight * loss_commit
        else:
            vq_loss = z.new_zeros(())

        if self.training and self.use_ema_codebook and valid.any():
            self._ema_update_codebook(indices_bt, valid, z)

        self._maybe_reset_dead_codes(
            indices_bt=indices_bt,
            valid=valid,
            z=z,
            global_step=global_step,
        )

        return z_residual, vq_loss, z_q, stats

    @torch.no_grad()
    def _ema_update_codebook(
        self,
        indices_bt: torch.Tensor,
        valid: torch.BoolTensor,
        z: torch.Tensor,
    ) -> None:
        idx = indices_bt[valid].long()
        z_v = z[valid].float()
        K = self.num_embeddings
        one_hot = F.one_hot(idx, num_classes=K).float()
        cluster_count = one_hot.sum(dim=0)
        cluster_sum = one_hot.t() @ z_v

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                torch.distributed.all_reduce(cluster_count)
                torch.distributed.all_reduce(cluster_sum)

        active = cluster_count > 0
        if not active.any():
            return

        decay = self.ema_decay
        dt_s = self._ema_embedding_sum.dtype
        dt_c = self._ema_cluster_size.dtype
        new_sum = (
            decay * self._ema_embedding_sum[active].float()
            + (1.0 - decay) * cluster_sum[active].float()
        ).to(dt_s)
        new_cnt = (
            decay * self._ema_cluster_size[active].float()
            + (1.0 - decay) * cluster_count[active].float()
        ).to(dt_c)
        self._ema_embedding_sum[active] = new_sum
        self._ema_cluster_size[active] = new_cnt
        denom = self._ema_cluster_size.unsqueeze(1).float().clamp(min=self.ema_epsilon)
        self.codebook.weight.data.copy_(
            (self._ema_embedding_sum.float() / denom).to(self.codebook.weight.dtype)
        )

    @torch.no_grad()
    def _maybe_reset_dead_codes(
        self,
        indices_bt: torch.Tensor,
        valid: torch.BoolTensor,
        z: torch.Tensor,
        global_step: Optional[int],
    ) -> None:
        if not self.training or not self.reset_dead_codes:
            return
        fwd_i = int(self._vq_forward_index.item())
        self._vq_forward_index.add_(1)
        if self.reset_every_forward > 1 and (fwd_i % self.reset_every_forward) != 0:
            return
        if not valid.any():
            return
        idx = indices_bt[valid].long()
        counts = torch.bincount(idx, minlength=self.num_embeddings)
        dead = (counts == 0).nonzero(as_tuple=False).reshape(-1)
        if dead.numel() == 0:
            return
        if self.reset_max_per_step is not None:
            cap = int(self.reset_max_per_step)
            if dead.numel() > cap:
                dead = dead[:cap]
        z_e = z[valid]
        n = int(z_e.shape[0])
        m = int(dead.numel())
        step = int(global_step) if global_step is not None else 0
        r = _deterministic_sample_indices(n, m, step, z.device)
        dead_f = dead.to(dtype=torch.float32).reshape(-1, 1)
        d_ix = torch.arange(self.dim, device=z.device, dtype=torch.float32).reshape(1, -1)
        jitter = (1e-3 * torch.sin(dead_f * 0.01745 + d_ix * 0.03142 + float(step))).to(
            self.codebook.weight.dtype
        )
        new_rows = z_e[r].to(self.codebook.weight.dtype) + jitter
        self.codebook.weight.data[dead] = new_rows
        if self.use_ema_codebook:
            self._ema_embedding_sum[dead] = new_rows.to(self._ema_embedding_sum.dtype)
            self._ema_cluster_size[dead] = torch.ones(
                dead.numel(),
                dtype=self._ema_cluster_size.dtype,
                device=z.device,
            )
