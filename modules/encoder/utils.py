import torch
import torch.nn as nn
import torch.nn.functional as F


def _assert_latent_chunks_divisible(latent_dim: int, chunk_size: int) -> None:
    assert (
        latent_dim % chunk_size == 0
    ), f"latent_dim ({latent_dim}) must be divisible by latent_chunk_size ({chunk_size})"


def _latent_chunk_dropout_probs_per_chunk(
    *,
    num_chunks: int,
    start_p: float,
    end_p: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-chunk dropout probability (independent mode), linear from start to end."""
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if num_chunks == 1:
        return torch.tensor([start_p], device=device, dtype=dtype)
    return torch.linspace(start_p, end_p, num_chunks, device=device, dtype=dtype)


def apply_latent_chunk_dropout(
    z: torch.Tensor,
    *,
    chunk_size: int,
    dropout_start: float,
    dropout_end: float,
    training: bool,
    hierarchical: bool = True,
) -> torch.Tensor:
    """
    Latent chunk dropout on ``z`` [B, T, latent_dim].

    If ``hierarchical`` (default): chunk 0 never dropped; with probability
    ``dropout_end`` a single cutoff ``s`` is sampled (weights linear from
    ``dropout_start`` at ``s=1`` to ``dropout_end`` at the last start index);
    chunks ``s..n-1`` are zeroed. Same mask across ``T`` per batch row.

    If not ``hierarchical``: each chunk is dropped independently with
    probability ``p_i`` linear in chunk index (``dropout_start`` .. ``dropout_end``).
    """
    if not training:
        return z
    B, T, D = z.shape
    _assert_latent_chunks_divisible(D, chunk_size)
    n_chunks = D // chunk_size

    if not hierarchical:
        probs = _latent_chunk_dropout_probs_per_chunk(
            num_chunks=n_chunks,
            start_p=dropout_start,
            end_p=dropout_end,
            device=z.device,
            dtype=z.dtype,
        )
        zv = z.view(B, T, n_chunks, chunk_size)
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
        w = dropout_start + (dropout_end - dropout_start) * t
        w = torch.clamp(w, min=0.0)

    w_sum = float(w.sum().item())
    if w_sum <= 1e-12:
        return z

    probs = (w / w.sum()).expand(B, -1)
    starts_rel = torch.multinomial(probs, num_samples=1).squeeze(-1)
    start_chunk = starts_rel + 1

    u = torch.rand(B, device=device, dtype=torch.float32)
    active = u < float(dropout_end)
    start_chunk = torch.where(
        active,
        start_chunk,
        torch.full_like(start_chunk, n_chunks),
    )

    chunk_idx = torch.arange(n_chunks, device=device).view(1, n_chunks).expand(B, -1)
    keep = chunk_idx < start_chunk.unsqueeze(1)

    zv = z.view(B, T, n_chunks, chunk_size)
    keep_4 = keep.unsqueeze(1).unsqueeze(-1).to(dtype=zv.dtype)
    return (zv * keep_4).view(B, T, D)


def latent_chunk_kl_channel_weights(
    *,
    latent_dim: int,
    chunk_size: int,
    weight_start: float,
    weight_end: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """[latent_dim] weights (same for both channels in each chunk), linear across chunks."""
    _assert_latent_chunks_divisible(latent_dim, chunk_size)
    n_chunks = latent_dim // chunk_size
    if n_chunks == 1:
        chunk_w = torch.tensor([weight_start], device=device, dtype=dtype)
    else:
        chunk_w = torch.linspace(
            weight_start, weight_end, n_chunks, device=device, dtype=dtype
        )
    return chunk_w.repeat_interleave(chunk_size)


def latent_chunk_kl_channel_weights_vq_tail(
    *,
    latent_dim: int,
    chunk_size: int,
    zero_chunks: int,
    tail_weight_start: float,
    tail_weight_end: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Per-channel KL weights: first ``zero_chunks`` chunks are 0; remaining chunks
    linearly from ``tail_weight_start`` (at chunk index ``zero_chunks``) to
    ``tail_weight_end`` (last chunk).
    """
    _assert_latent_chunks_divisible(latent_dim, chunk_size)
    n_chunks = latent_dim // chunk_size
    if zero_chunks < 0 or zero_chunks > n_chunks:
        raise ValueError(
            f"vq_kl_zero_chunks ({zero_chunks}) must be in [0, {n_chunks}] "
            f"for latent_dim={latent_dim}, chunk_size={chunk_size}."
        )
    w = torch.zeros(latent_dim, device=device, dtype=dtype)
    n_tail = n_chunks - zero_chunks
    if n_tail <= 0:
        return w
    for c in range(zero_chunks, n_chunks):
        if n_tail == 1:
            wc = tail_weight_start
        else:
            t = (c - zero_chunks) / float(n_tail - 1)
            wc = tail_weight_start + (tail_weight_end - tail_weight_start) * t
        w[c * chunk_size : (c + 1) * chunk_size] = wc
    return w


class TimeCausalConv1d(nn.Conv1d):
    """Causal Conv1d: left-pad only so output at time t depends only on inputs <= t."""

    def __init__(self, c_in, c_out, k, d=1, s=1):
        super().__init__(c_in, c_out, k, dilation=d, stride=s, padding=0)
        self.k, self.d, self.s = k, d, s

    def forward(self, x):  # x: [B, C, T]
        pad_left = (self.k - 1) * self.d
        x = F.pad(x, (pad_left, 0))
        return super().forward(x)


class PreNormResCausalBlock1d(nn.Module):
    """GroupNorm(1) -> GELU -> CausalConv1d -> Dropout + skip."""

    def __init__(self, c_in, c_out, *, k=3, d=1, s=1, drop_p=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, c_in)
        self.act = nn.GELU()
        self.main = TimeCausalConv1d(c_in, c_out, k=k, d=d, s=s)
        if c_in != c_out or s != 1:
            self.skip = TimeCausalConv1d(c_in, c_out, k=1, d=1, s=s)
        else:
            self.skip = nn.Identity()
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):  # x: [B, C, T]
        h = self.act(self.norm(x))
        h = self.dropout(h)
        return self.main(h) + self.skip(x)


class CausalDownsamplingBlock1d(nn.Module):
    """Dilated residual blocks followed by a stride-2 downsampler."""

    def __init__(self, c_in, c_out, n_residual_blocks=3, drop_p=0.1):
        super().__init__()
        dilations = [1, 2, 4, 8]
        self.residual_blocks = nn.ModuleList(
            [
                PreNormResCausalBlock1d(c_in, c_in, k=5, d=dilation, drop_p=drop_p)
                for dilation in dilations[:n_residual_blocks]
            ]
        )
        self.downsampling = PreNormResCausalBlock1d(
            c_in, c_out, k=5, d=1, s=2, drop_p=drop_p
        )

    def forward(self, x):  # x: [B, C, T]
        for block in self.residual_blocks:
            x = block(x)
        return self.downsampling(x)


class Transformer(nn.Module):
    def __init__(self, d_model=512, nheads=8, nlayers=4, drop_p=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nheads,
            batch_first=True,
            norm_first=True,
            dropout=drop_p,
            dim_feedforward=4 * d_model,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)

    def forward(self, x, pad_mask=None):
        seq_len = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        return self.enc(x, mask=mask, src_key_padding_mask=pad_mask, is_causal=True)
