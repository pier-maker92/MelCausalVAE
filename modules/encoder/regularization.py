import torch
import torch.nn as nn
from typing import Optional
from modules.configs import DropoutConfig, KLChunkRegularizer


def _assert_latent_chunks_divisible(latent_dim: int, chunk_size: int) -> None:
    assert (
        latent_dim % chunk_size == 0
    ), f"latent_dim ({latent_dim}) must be divisible by latent_chunk_size ({chunk_size})"


class DropoutRegularizer(nn.Module):
    def __init__(self, config: DropoutConfig):
        super().__init__()
        self.dropout_start = config.dropout_start
        self.dropout_end = config.dropout_end
        self.chunk_size = config.chunk_size
        self.dropout_hierarchical = config.dropout_hierarchical
        self.training = True

    def set_training(self, training: bool) -> None:
        self.training = training

    def _latent_chunk_dropout_probs_per_chunk(
        self, num_chunks: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Per-chunk dropout probability (independent mode), linear from start to end."""
        if num_chunks <= 0:
            raise ValueError("num_chunks must be positive")
        if num_chunks == 1:
            return torch.tensor([self.dropout_start], device=device, dtype=dtype)
        return torch.linspace(
            self.dropout_start,
            self.dropout_end,
            num_chunks,
            device=device,
            dtype=dtype,
        )

    def forward(self, z: torch.FloatTensor) -> torch.Tensor:
        """
        Latent chunk dropout on ``z`` [B, T, latent_dim].

        If ``hierarchical`` (default): chunk 0 never dropped; with probability
        ``dropout_end`` a single cutoff ``s`` is sampled (weights linear from
        ``dropout_start`` at ``s=1`` to ``dropout_end`` at the last start index);
        chunks ``s..n-1`` are zeroed. Same mask across ``T`` per batch row.

        If not ``hierarchical``: each chunk is dropped independently with
        probability ``p_i`` linear in chunk index (``dropout_start`` .. ``dropout_end``).
        """
        if not self.training:
            return z
        B, T, D = z.shape
        _assert_latent_chunks_divisible(D, self.chunk_size)
        n_chunks = D // self.chunk_size

        if not self.dropout_hierarchical:
            probs = self._latent_chunk_dropout_probs_per_chunk(
                num_chunks=n_chunks, device=z.device, dtype=z.dtype
            )
            zv = z.view(B, T, n_chunks, self.chunk_size)
            u = torch.rand(B, 1, n_chunks, 1, device=z.device, dtype=z.dtype)
            keep = u >= probs.view(1, 1, n_chunks, 1)
            return (zv * keep).view(B, T, D)

        if n_chunks <= 1:
            return z

        device = z.device
        # Possible start indices s = 1 .. n_chunks-1 (chunk 0 always kept)
        n_starts = n_chunks - 1
        if n_starts == 1:
            w = torch.tensor([1.0], device=device, dtype=torch.float32)
        else:
            # s = 1 + k for k = 0 .. n_starts-1; weight linear in k
            k = torch.arange(n_starts, device=device, dtype=torch.float32)
            t = k / max(n_starts - 1, 1)
            w = self.dropout_start + (self.dropout_end - self.dropout_start) * t
            w = torch.clamp(w, min=0.0)

        w_sum = float(w.sum().item())
        if w_sum <= 1e-12:
            return z

        probs = (w / w.sum()).expand(B, -1)
        starts_rel = torch.multinomial(probs, num_samples=1).squeeze(-1)
        start_chunk = starts_rel + 1

        u = torch.rand(B, device=device, dtype=torch.float32)
        active = u < float(self.dropout_end)
        start_chunk = torch.where(
            active,
            start_chunk,
            torch.full_like(start_chunk, n_chunks),
        )

        chunk_idx = (
            torch.arange(n_chunks, device=device).view(1, n_chunks).expand(B, -1)
        )
        keep = chunk_idx < start_chunk.unsqueeze(1)

        zv = z.view(B, T, n_chunks, self.chunk_size)
        keep_4 = keep.unsqueeze(1).unsqueeze(-1).to(dtype=zv.dtype)
        return (zv * keep_4).view(B, T, D)


class KLChunkRegularizer(nn.Module):
    def __init__(self, config: KLChunkRegularizer, vq_quant_dim: Optional[int] = None):
        super().__init__()
        self.kl_start = config.kl_weight_start
        self.kl_end = config.kl_weight_end
        self.chunk_size = config.chunk_size
        self.vq_quant_dim = vq_quant_dim
        self.training = True

    def set_training(self, training: bool) -> None:
        self.training = training

    def latent_chunk_kl_weights(
        self,
        latent_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """[latent_dim] KL weights: first `zero_chunks` are 0, then linear from `weight_start` to `weight_end`."""
        _assert_latent_chunks_divisible(latent_dim, self.chunk_size)
        n_chunks = latent_dim // self.chunk_size
        zero_chunks = (
            0 if self.vq_quant_dim is None else self.vq_quant_dim // self.chunk_size
        )

        weights = torch.zeros(n_chunks, device=device, dtype=dtype)
        n_active = n_chunks - zero_chunks
        if n_active <= 0:
            raise ValueError(
                f"zero_chunks ({zero_chunks}) must be less than n_chunks ({n_chunks})"
            )
        weights[zero_chunks:] = torch.linspace(
            self.kl_start, self.kl_end, n_active, device=device, dtype=dtype
        )

        return weights.repeat_interleave(self.chunk_size)

    def forward(
        self,
        mu: torch.FloatTensor,
        logvar: Optional[torch.FloatTensor],
        padding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """
        Same KL as kl_divergence but each latent dimension is scaled by channel_weights [latent_dim].
        When weights are all 1.0, matches kl_divergence (up to dtype handling).
        """
        dtype = mu.dtype
        channel_weights = self.latent_chunk_kl_weights(
            latent_dim=mu.shape[-1],
            device=mu.device,
            dtype=dtype,
        )
        w = channel_weights.to(device=mu.device, dtype=dtype).view(1, 1, -1)
        valid = (~padding_mask).unsqueeze(-1).to(dtype=dtype)
        if logvar is None:
            mu_f = mu.to(dtype)
            denom = (valid.sum() * mu.shape[-1]).clamp(min=1.0)
            return (mu_f.pow(2) * w * valid).sum() / denom
        mu_f = mu.to(dtype)
        logvar_f = logvar.to(dtype)
        kl_elem = -0.5 * (1 + logvar_f - mu_f.pow(2) - logvar_f.exp())
        return (kl_elem * w * valid).sum().to(mu.dtype)
