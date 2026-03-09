"""
AlignmentQFormer
================
Q-Former whose cross-attention is hard-gated by the binary alignment matrix
produced by ``AlignmentMatrixBuilder``, followed by a ``PhonemeQueryPooler``
that collapses the Q per-phoneme queries into a single vector via
concat + linear projection.

Masking rule
------------
Each phoneme segment ``n`` owns ``num_queries_per_phoneme`` (default 4)
learnable queries.  In cross-attention query ``n·Q + k`` may **only** attend
to mel frame ``t`` where::

    alignment[b, t, n] == 1

All other (query, frame) pairs receive an additive ``-inf`` mask, so they
contribute zero weight after softmax.

Architecture (per layer)
------------------------
1. **Cross-attention** — queries attend to pos-encoded mel frames, gated by alignment.
2. **Self-attention**  — queries attend to each other (no mask).
3. **FFN**             — position-wise feed-forward.

Cross-attention comes *first* so that the shared prototypes are immediately
differentiated by acoustic content before they interact via self-attention.
Each sub-layer uses pre-norm + residual.

Relative positional encoding
-----------------------------
For each mel frame ``t`` we compute its **relative position within its
phoneme** as a scalar ``r ∈ [0, 1]``::

    r[b, t] = (cumulative_frame_within_phoneme - 1)
              / (phoneme_duration - 1).clamp(min=1)

This scalar is mapped to ``d_model`` via a **sinusoidal embedding** (fixed,
no parameters) followed by a learned ``Linear(d_model, d_model)`` projection.
The result is added to ``mel_features`` *before* they are used as keys/values
in cross-attention — so the queries see positionally-tagged frames from the
very first layer.

The raw ``rel_pos`` tensor ``(B, T)`` is returned in ``QFormerOutput`` so the
DiT caller can build frame-level conditioning as::

    pos_emb = qformer.pos_encoder(rel_pos)          # (B, T, d_model)
    phon_idx = alignment.argmax(-1)                 # (B, T)
    cond = pooled.gather(1, phon_idx.unsqueeze(-1)
                         .expand(B, T, d_model))    # (B, T, d_model)
    cond = cond + pos_emb                           # (B, T, d_model)

Pooling
-------
``PhonemeQueryPooler`` takes ``(B, N, Q, d_model)`` and produces
``(B, N, d_model)`` via::

    concat(q_0, …, q_{Q-1})  →  Linear(Q*d_model, d_model)  →  LayerNorm

Outputs
-------
``QFormerOutput.hidden_states``  ``(B, N, Q, d_model)`` — pre-pool queries.
``QFormerOutput.pooled``         ``(B, N, d_model)``    — one vector per phoneme.
``QFormerOutput.rel_pos``        ``(B, T)``             — relative pos in [0,1].
``QFormerOutput.attn_mask``      ``(B*H, N*Q, T)``      — additive cross-attn mask.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal relative positional encoding
# ---------------------------------------------------------------------------


def sinusoidal_pos_emb(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Map a scalar tensor to a sinusoidal embedding.

    Args:
        x:   ``(*)`` float tensor with values in ``[0, 1]``.
        dim: output embedding dimension (must be even).

    Returns:
        ``(*, dim)`` float tensor.
    """
    assert dim % 2 == 0, f"sinusoidal_pos_emb requires even dim, got {dim}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=x.device)
        / max(half - 1, 1)
    )                                                    # (half,)
    args = x.unsqueeze(-1).float() * freqs               # (*, half)
    return torch.cat([args.sin(), args.cos()], dim=-1)   # (*, dim)


class RelativePositionEncoder(nn.Module):
    """
    Sinusoidal embedding of the intra-phoneme relative position scalar,
    followed by a learned linear projection into ``d_model``.

    Args:
        d_model: output dimension (must be even).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel_pos: ``(B, T)`` float in ``[0, 1]``.

        Returns:
            ``(B, T, d_model)``
        """
        emb = sinusoidal_pos_emb(rel_pos, self.d_model)  # (B, T, d_model)
        return self.proj(emb.to(dtype=self.proj.weight.dtype))


# ---------------------------------------------------------------------------
# Relative position computation
# ---------------------------------------------------------------------------


def compute_relative_positions(alignment: torch.Tensor) -> torch.Tensor:
    """
    Compute the intra-phoneme relative position for every mel frame.

    Given a binary alignment matrix ``(B, T, N)``:

    1. ``phon_idx[b, t]``  — which phoneme owns frame ``t`` (argmax over N).
    2. ``cum_frame[b, t]`` — 1-based frame count within that phoneme.
    3. ``dur[b, t]``       — duration of the owning phoneme.
    4. ``rel_pos``         — ``(cum_frame - 1) / (dur - 1).clamp(min=1)``
       → ``0`` at the first frame, ``1`` at the last.

    Args:
        alignment: ``(B, T, N)`` binary float tensor.

    Returns:
        ``(B, T)`` float tensor in ``[0, 1]``.
    """
    B, T, N = alignment.shape

    # Which phoneme owns each frame — (B, T)
    phon_idx = alignment.argmax(dim=-1)                          # (B, T)

    # Cumulative frame count per phoneme column — (B, T, N)
    cum = alignment.cumsum(dim=1)                                # (B, T, N)

    # Gather at the owning phoneme — (B, T)
    cum_frame = cum.gather(
        dim=2, index=phon_idx.unsqueeze(-1)
    ).squeeze(-1).float()                                        # (B, T)

    # Duration of each phoneme — (B, N); gather per frame — (B, T)
    durations = alignment.sum(dim=1)                             # (B, N)
    dur_frame = durations.gather(dim=1, index=phon_idx).float()  # (B, T)

    # Normalise to [0, 1] — clamp guards padding frames where alignment is all
    # zero (argmax returns 0, cum_frame=0 → raw value = -1.0 without clamp).
    rel_pos = (cum_frame - 1.0) / (dur_frame - 1.0).clamp(min=1.0)
    return rel_pos.clamp(0.0, 1.0)                               # (B, T)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class _FFN(nn.Module):
    """Position-wise feed-forward block."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QFormerLayer(nn.Module):
    """
    Single Q-Former layer: **cross-attention → self-attention → FFN**.

    Cross-attention is placed first so that the shared query prototypes are
    immediately differentiated by acoustic content before they interact via
    self-attention.  All sub-layers use pre-norm + residual.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Cross-attention (queries → pos-encoded mel frames, alignment-masked)
        self.norm_ca = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Self-attention (queries ↔ queries, no mask)
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # FFN
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = _FFN(d_model, ffn_dim, dropout)

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,              # (B, N*Q, d_model)
        mel_features: torch.Tensor,         # (B, T,   d_model) — already pos-encoded
        cross_mask: torch.Tensor,           # (B*H, N*Q, T)  additive float mask
        causal_mask: torch.Tensor,          # (N*Q, N*Q)     additive float mask
        sa_key_padding_mask: torch.Tensor,  # (B, N*Q)       True = padding key
    ) -> torch.Tensor:                      # (B, N*Q, d_model)

        # 1 — Cross-attention (pre-norm), alignment mask applied
        q = self.norm_ca(queries)
        ca_out, _ = self.cross_attn(
            q, mel_features, mel_features,
            attn_mask=cross_mask,
        )
        queries = queries + self.drop(ca_out)

        # 2 — Causal self-attention (pre-norm):
        #     group n attends only to groups 0..n (past + self, not future).
        #     key_padding_mask explicitly zeroes out padding phoneme keys so
        #     that real queries never draw weight from padding content,
        #     regardless of causal ordering.
        q = self.norm_sa(queries)
        sa_out, _ = self.self_attn(
            q, q, q,
            attn_mask=causal_mask,
            key_padding_mask=sa_key_padding_mask,
        )
        queries = queries + self.drop(sa_out)

        # 3 — FFN (pre-norm)
        queries = queries + self.ffn(self.norm_ff(queries))

        return queries


# ---------------------------------------------------------------------------
# Pooler
# ---------------------------------------------------------------------------


class PhonemeQueryPooler(nn.Module):
    """
    Collapse Q per-phoneme queries into a single ``d_model`` vector.

    Operation::

        (B, N, Q, d_model)
          → reshape  → (B, N, Q*d_model)
          → Linear(Q*d_model, d_model)
          → LayerNorm
          → (B, N, d_model)
    """

    def __init__(self, num_queries: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(num_queries * d_model, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, N, Q, d_model)``
        Returns:
            ``(B, N, d_model)``
        """
        B, N, Q, D = x.shape
        return self.norm(self.proj(x.reshape(B, N, Q * D)))


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class QFormerOutput:
    """Structured output of :py:meth:`AlignmentQFormer.forward`."""

    hidden_states: torch.Tensor
    """``(B, N, Q, d_model)`` — per-query representations per phoneme."""

    pooled: torch.Tensor
    """``(B, N, d_model)`` — one vector per phoneme after concat-pool."""

    rel_pos: torch.Tensor
    """``(B, T)`` — intra-phoneme relative position in ``[0, 1]``.

    Pass directly to :class:`DiTConditioningProjector` together with
    ``pooled`` and ``alignment`` to build frame-level DiT conditioning.
    """

    attn_mask: torch.Tensor
    """``(B*H, N*Q, T)`` additive cross-attention mask (0 / -inf)."""


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class AlignmentQFormer(nn.Module):
    """
    Q-Former with alignment-gated cross-attention, sinusoidal intra-phoneme
    relative positional encoding, and concat-pool.

    Args:
        d_model:                 Model / embedding dimension (must be even).
        num_heads:               Number of attention heads.
        num_queries_per_phoneme: Learnable queries per phoneme (default **4**).
        num_layers:              Number of Q-Former layers (default 2).
        ffn_dim:                 Inner FFN dimension (default ``4 * d_model``).
        dropout:                 Dropout probability.

    Example::

        qformer = AlignmentQFormer(d_model=256, num_heads=8)

        out = qformer(
            mel_features = mel_enc,                  # (B, T, 256)
            alignment    = align_out.alignments,     # (B, T, N)
            phoneme_mask = align_out.phoneme_mask,   # (B, N)
        )
        # out.hidden_states → (B, N, 4, 256)
        # out.pooled        → (B, N, 256)
        # out.rel_pos       → (B, T)     ← [0,1] per frame, for DiT cond
        #
        # DiT frame-level conditioning:
        #   pos_emb  = qformer.pos_encoder(out.rel_pos)
        #   phon_idx = align_out.alignments.argmax(-1)       # (B, T)
        #   expanded = phon_idx.unsqueeze(-1).expand(B, T, 256)
        #   cond     = out.pooled.gather(1, expanded) + pos_emb
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_queries_per_phoneme: int = 4,
        num_layers: int = 2,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"

        self.d_model = d_model
        self.num_heads = num_heads
        self.Q = num_queries_per_phoneme
        ffn_dim = ffn_dim or 4 * d_model

        # Shared learnable query prototypes (Q, d_model).
        # All groups start identical; symmetry is broken immediately by the
        # alignment-masked cross-attention in the first layer.
        self.query_proto = nn.Parameter(torch.empty(num_queries_per_phoneme, d_model))
        nn.init.trunc_normal_(self.query_proto, std=0.02)

        # Relative positional encoder: scalar [0,1] → (d_model,)
        self.pos_encoder = RelativePositionEncoder(d_model)

        # Q-Former layers (cross-attn → self-attn → FFN)
        self.layers = nn.ModuleList([
            QFormerLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm before pooling
        self.out_norm = nn.LayerNorm(d_model)

        # Concat-pool: (B, N, Q, d_model) → (B, N, d_model)
        self.pooler = PhonemeQueryPooler(num_queries_per_phoneme, d_model)

    # ---------------------------------------------------------------------- #
    # Mask construction                                                        #
    # ---------------------------------------------------------------------- #

    def _build_causal_mask(self, N: int, device: torch.device) -> torch.Tensor:
        """
        Build an additive causal mask ``(N*Q, N*Q)`` for self-attention.

        Query at flat index ``n*Q + k`` may attend to position ``j`` iff
        ``j // Q <= n`` (same or earlier phoneme group).

        * Allowed  → **0.0**
        * Blocked  → **-inf**
        """
        Q = self.Q
        total = N * Q
        # group index for each query position
        group = torch.arange(total, device=device) // Q          # (N*Q,)
        # allowed: group[j] <= group[i]  →  causal over phoneme groups
        allowed = group.unsqueeze(0) <= group.unsqueeze(1)        # (N*Q, N*Q)
        mask = torch.zeros(total, total, device=device)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask                                               # (N*Q, N*Q)

    def _build_cross_attn_mask(
        self,
        alignment: torch.Tensor,                   # (B, T, N)
        phoneme_mask: Optional[torch.BoolTensor],  # (B, N) or None
    ) -> torch.Tensor:                             # (B*H, N*Q, T)
        """
        Build an additive float mask ``(B*H, N*Q, T)``.

        * Allowed  — query n, frame t where alignment[b,t,n]==1  →  **0.0**
        * Blocked  — everything else                             →  **-inf**

        Two safety guards prevent NaN in softmax:

        1. Fully-empty phoneme rows (no assigned frames) → reset to 0.
        2. Padding phoneme rows (``phoneme_mask == True``) → reset to 0.
        """
        B, T, N = alignment.shape
        Q, H = self.Q, self.num_heads

        mask_bool = (
            alignment.permute(0, 2, 1)               # (B, N, T)
            .unsqueeze(2).expand(B, N, Q, T)         # (B, N, Q, T)
            .reshape(B, N * Q, T)
            .bool()
        )

        additive = alignment.new_zeros(B, N * Q, T)
        additive = additive.masked_fill(~mask_bool, float("-inf"))

        # Guard 1: empty phonemes
        empty = (~mask_bool).all(dim=-1, keepdim=True)
        additive = additive.masked_fill(empty, 0.0)

        # Guard 2: padding phonemes
        if phoneme_mask is not None:
            pad_rows = (
                phoneme_mask                                     # (B, N)
                .unsqueeze(2).expand(B, N, Q)
                .reshape(B, N * Q)
                .unsqueeze(-1)
            )
            additive = additive.masked_fill(pad_rows, 0.0)

        # Expand for heads → (B*H, N*Q, T)
        return (
            additive
            .unsqueeze(1).expand(B, H, N * Q, T)
            .reshape(B * H, N * Q, T)
            .contiguous()
        )

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        mel_features: torch.Tensor,                         # (B, T, d_model)
        alignment: torch.Tensor,                            # (B, T, N) float 0/1
        phoneme_mask: Optional[torch.BoolTensor] = None,   # (B, N) True=padding
    ) -> QFormerOutput:
        """
        Parameters
        ----------
        mel_features:
            Encoded mel-spectrogram frames ``(B, T, d_model)``.
        alignment:
            Binary alignment matrix ``(B, T, N)``; ``alignment[b,t,n]==1``
            iff frame ``t`` belongs to phoneme segment ``n``.
        phoneme_mask:
            Padding mask ``(B, N)``.  ``True`` = padded segment.

        Returns
        -------
        :class:`QFormerOutput`
        """
        B, T, N = alignment.shape
        Q = self.Q

        # ------------------------------------------------------------------ #
        # 1. Relative positional encoding                                      #
        #    rel_pos : (B, T) in [0, 1]  — 0 = first frame of phoneme,        #
        #                                  1 = last  frame of phoneme          #
        #    pos_emb : (B, T, d_model)   — injected into mel keys/values       #
        # ------------------------------------------------------------------ #
        rel_pos  = compute_relative_positions(alignment)         # (B, T)
        pos_emb  = self.pos_encoder(rel_pos)                     # (B, T, d_model)
        mel_pos  = mel_features + pos_emb                        # (B, T, d_model)

        # ------------------------------------------------------------------ #
        # 2. Initialise queries from shared prototypes                         #
        # ------------------------------------------------------------------ #
        queries = (
            self.query_proto                          # (Q, d_model)
            .view(1, 1, Q, self.d_model)
            .expand(B, N, Q, self.d_model)            # (B, N, Q, d_model)
            .contiguous()                             # make contiguous before reshape
            .reshape(B, N * Q, self.d_model)          # (B, N*Q, d_model)
        )

        # ------------------------------------------------------------------ #
        # 3. Cross-attention mask                                              #
        # ------------------------------------------------------------------ #
        cross_mask = self._build_cross_attn_mask(alignment, phoneme_mask)

        # ------------------------------------------------------------------ #
        # 4. Causal self-attention mask  (N*Q, N*Q)                           #
        #    Group n attends only to groups 0..n — monotonic, no future leak. #
        # ------------------------------------------------------------------ #
        causal_mask = self._build_causal_mask(N, device=mel_features.device)

        # ------------------------------------------------------------------ #
        # 5. Self-attention key padding mask  (B, N*Q)                        #
        #    Expands phoneme_mask from (B, N) to (B, N*Q) so that real        #
        #    queries never draw attention weight from padding phoneme keys,    #
        #    regardless of their position in the causal order.                #
        # ------------------------------------------------------------------ #
        if phoneme_mask is not None:
            sa_key_padding_mask = (
                phoneme_mask                              # (B, N)
                .unsqueeze(2).expand(B, N, Q)            # (B, N, Q)
                .contiguous().reshape(B, N * Q)          # (B, N*Q)
            )
        else:
            sa_key_padding_mask = torch.zeros(
                B, N * Q, dtype=torch.bool, device=mel_features.device
            )

        # ------------------------------------------------------------------ #
        # 6. Q-Former layers                                                   #
        # ------------------------------------------------------------------ #
        for layer in self.layers:
            queries = layer(
                queries, mel_pos, cross_mask, causal_mask, sa_key_padding_mask
            )

        # ------------------------------------------------------------------ #
        # 7. Norm + reshape → (B, N, Q, d_model)                              #
        # ------------------------------------------------------------------ #
        hidden_states = self.out_norm(queries).reshape(B, N, Q, self.d_model)

        # ------------------------------------------------------------------ #
        # 8. Concat-pool → (B, N, d_model)                                    #
        # ------------------------------------------------------------------ #
        pooled = self.pooler(hidden_states)

        return QFormerOutput(
            hidden_states=hidden_states,
            pooled=pooled,
            rel_pos=rel_pos,
            attn_mask=cross_mask,
        )


# ---------------------------------------------------------------------------
# DiT conditioning projector
# ---------------------------------------------------------------------------


class DiTConditioningProjector(nn.Module):
    """
    Prepare frame-level conditioning for a DiT from :class:`QFormerOutput`.

    Operation
    ---------
    Given:

    * ``pooled``    ``(B, N, d_model)`` — one compressed vector per phoneme.
    * ``alignment`` ``(B, T, N)``       — binary (or soft) alignment matrix.
    * ``rel_pos``   ``(B, T)``          — intra-phoneme relative position ∈ [0,1].

    Produces::

        phoneme_at_t  = bmm(alignment, pooled)          # (B, T, d_model)
        pos_feat      = pos_encoder(rel_pos)             # (B, T, d_model)
        cond          = proj(cat([phoneme_at_t,
                                  pos_feat], dim=-1))   # (B, T, d_model)

    No LayerNorm is applied here — the DiT's own LayerNorm on the conditioning
    input makes it redundant.

    ``bmm(alignment, pooled)`` broadcasts each phoneme vector onto the frames
    that belong to it.  With hard binary alignment this is identical to a
    gather, but the bmm formulation also supports **soft alignment** and is
    fully differentiable.

    The two signals are kept separate until the final ``Linear(2*d,  d)``
    so the projection can learn an arbitrary mixing — rather than committing
    to a sum that fuses them a priori.

    The RoPE inside the DiT handles absolute frame positions; ``rel_pos``
    carries complementary *intra-phoneme* position (onset / body / offset of
    the phoneme), so the two are orthogonal and non-redundant.

    Args:
        d_model:     Dimension shared by QFormer and DiT.
        pos_encoder: The :class:`RelativePositionEncoder` instance from the
                     paired :class:`AlignmentQFormer` (reused, not copied).
        dropout:     Dropout applied after the projection.

    Example::

        projector = DiTConditioningProjector(
            d_model     = 256,
            pos_encoder = qformer.pos_encoder,   # share weights
        )

        cond = projector(
            pooled    = out.pooled,              # (B, N, 256)
            alignment = align_out.alignments,    # (B, T, N)
            rel_pos   = out.rel_pos,             # (B, T)
        )
        # cond → (B, T, 256)  ready for DiT cross-attention / adaLN
    """

    def __init__(
        self,
        d_model: int,
        pos_encoder: RelativePositionEncoder,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_encoder = pos_encoder                          # shared
        self.proj = nn.Linear(2 * d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        pooled: torch.Tensor,                           # (B, N, d_model)
        alignment: torch.Tensor,                        # (B, T, N) hard or soft
        rel_pos: torch.Tensor,                          # (B, T) in [0, 1]
        soft_align: Optional[torch.Tensor] = None,      # (B, T, N) from corrector
    ) -> torch.Tensor:                                  # (B, T, d_model)
        """
        Parameters
        ----------
        pooled:
            Per-phoneme compressed vectors from :class:`AlignmentQFormer`.
        alignment:
            Hard binary alignment ``(B, T, N)`` — used for ``rel_pos`` only
            when ``soft_align`` is provided, otherwise used for upsampling.
        rel_pos:
            Intra-phoneme relative positions ``(B, T)`` from
            :attr:`QFormerOutput.rel_pos`.
        soft_align:
            Optional soft alignment ``(B, T, N)`` from
            :class:`DifferentiableBoundaryCorrector`.  When provided,
            replaces the hard ``alignment`` for upsampling — giving smooth,
            differentiable boundaries instead of hard steps.

        Returns
        -------
        ``(B, T, d_model)`` — frame-level conditioning ready for the DiT.
        """
        # use soft alignment for upsampling if available, else hard
        upsample_align = soft_align if soft_align is not None else alignment

        # (B, T, N) x (B, N, d_model) → (B, T, d_model)
        phoneme_at_t = torch.bmm(upsample_align, pooled)

        # scalar [0,1] → (B, T, d_model)
        pos_feat = self.pos_encoder(rel_pos)

        # concat → project
        cond = torch.cat([phoneme_at_t, pos_feat], dim=-1)     # (B, T, 2*d_model)
        return self.drop(self.proj(cond))                       # (B, T, d_model)


# ---------------------------------------------------------------------------
# Norm penalty
# ---------------------------------------------------------------------------


def pooled_norm_penalty(
    pooled: torch.Tensor,
    phoneme_mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    """
    L2 norm regularisation on per-phoneme pooled vectors.

    Penalises large-norm vectors to keep the latent space bounded and stable
    for the downstream AR+diffusion head.  Padding phonemes are excluded from
    the mean so that batch items with fewer phonemes are not inadvertently
    down-weighted.

    Loss per phoneme::

        ℓ_n = ‖pooled_n‖²

    Aggregated as the mean over valid (non-padding) phonemes across the batch::

        L = mean_{b,n : not padding} ‖pooled[b,n]‖²

    Usage::

        reg = pooled_norm_penalty(out.pooled, phoneme_mask) * λ
        loss = reconstruction_loss + reg

    Args:
        pooled:       ``(B, N, d_model)`` — per-phoneme vectors from
                      :class:`AlignmentQFormer`.
        phoneme_mask: ``(B, N)`` bool — ``True`` = padding phoneme.
                      If ``None``, all phonemes are treated as valid.

    Returns:
        Scalar tensor — mean squared norm over valid phonemes.
    """
    # squared L2 norm per phoneme — (B, N)
    sq_norm = pooled.pow(2).sum(dim=-1)

    if phoneme_mask is not None:
        # zero out padding positions before averaging
        sq_norm = sq_norm.masked_fill(phoneme_mask, 0.0)
        n_valid = (~phoneme_mask).sum().clamp(min=1)
        return sq_norm.sum() / n_valid
    else:
        return sq_norm.mean()


# ---------------------------------------------------------------------------
# Differentiable boundary corrector
# ---------------------------------------------------------------------------


class DifferentiableBoundaryCorrector(nn.Module):
    """
    Learns to shift phoneme boundaries to minimise reconstruction error,
    producing a **soft, differentiable alignment matrix** from hard durations.

    Motivation
    ----------
    Hard upsampling via ``bmm(alignment_hard, pooled)`` produces step
    discontinuities at phoneme boundaries that the DiT has never seen during
    training with fixed-rate compression.  This module replaces the hard
    alignment with a **Gaussian soft alignment** whose centres can be shifted
    by a small learned delta — making boundaries differentiable and smooth.

    Operation
    ---------
    1. Compute expected phoneme centres from durations::

        centers_n = cumsum(durations)_n - durations_n / 2   # (B, N)

    2. Predict a small shift ``δ_n`` from ``pooled_mean`` via a 2-layer MLP::

        δ = tanh(MLP(pooled_mean)) * max_shift              # (B, N)

    3. Build a Gaussian soft alignment::

        soft_align[b, t, n] = softmax_n( -(t - (center_n + δ_n))² / 2σ² )

       ``softmax`` over N ensures each frame's weights sum to 1 across phonemes.

    4. Return ``soft_align (B, T, N)`` — a drop-in replacement for the hard
       alignment in ``bmm(soft_align, pooled)``.

    Args:
        d_model:    Input dimension of ``pooled_mean``.
        max_shift:  Maximum boundary shift in frames (default 4).
        sigma:      Gaussian bandwidth in frames (default 2.0).
                    Larger → smoother boundaries, smaller → closer to hard.

    The gradient of the reconstruction loss flows through ``soft_align``
    → ``centers + δ`` → ``δ`` → MLP weights — the model learns to correct
    aligner errors end-to-end.

    Example::

        corrector = DifferentiableBoundaryCorrector(d_model=256)

        soft_align = corrector(
            pooled_mean = pooled_mean,   # (B, N, d_model)
            durations   = durations,     # (B, N)  integer frame counts
            T           = mel_T,         # target time dimension
            phoneme_mask = mask,         # (B, N)  True = padding
        )
        # soft_align → (B, T, N)
        phoneme_at_t = torch.bmm(soft_align, pooled)   # (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        max_shift: float = 4.0,
        sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_shift = max_shift
        self.sigma = sigma

        # lightweight MLP: d_model → d_model//2 → 1
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),   # scalar shift per phoneme
        )

    def forward(
        self,
        pooled_mean: torch.Tensor,                      # (B, N, d_model)
        durations: torch.Tensor,                        # (B, N) float/long
        T: int,                                         # target frames
        phoneme_mask: Optional[torch.BoolTensor] = None,  # (B, N) True=pad
    ) -> torch.Tensor:                                  # (B, T, N)
        """
        Parameters
        ----------
        pooled_mean:
            Mean-pooled phoneme vectors computed on the hard alignment.
            Used as input to the boundary shift MLP.
        durations:
            Hard frame counts per phoneme ``(B, N)`` from
            ``AlignmentOutput.durations``.
        T:
            Target time dimension — must match ``mel_features.shape[1]``.
        phoneme_mask:
            ``(B, N)`` padding mask.  Padding phonemes receive zero weight.

        Returns
        -------
        ``(B, T, N)`` soft alignment matrix where each row (frame) sums to 1
        across valid phonemes.
        """
        B, N, _ = pooled_mean.shape
        durations = durations.float()                    # (B, N)

        # ------------------------------------------------------------------ #
        # 1. Hard phoneme centres from cumulative durations                   #
        #    center_n = sum(dur_0..n-1) + dur_n/2  (1-based frame index)      #
        # ------------------------------------------------------------------ #
        cum = durations.cumsum(dim=1)                    # (B, N)
        centers = cum - durations / 2.0                  # (B, N)

        # ------------------------------------------------------------------ #
        # 2. Learned boundary shift  δ ∈ (-max_shift, +max_shift)             #
        # ------------------------------------------------------------------ #
        delta = torch.tanh(self.mlp(pooled_mean).squeeze(-1))  # (B, N)
        delta = delta * self.max_shift                          # (B, N)

        # zero shift for padding phonemes — don't move phantom boundaries
        if phoneme_mask is not None:
            delta = delta.masked_fill(phoneme_mask, 0.0)

        centers_shifted = centers + delta                # (B, N)

        # ------------------------------------------------------------------ #
        # 3. Gaussian soft alignment                                           #
        #    dist[b, t, n] = (t - center_shifted[b, n])²                      #
        # ------------------------------------------------------------------ #
        t_grid = torch.arange(T, dtype=torch.float32, device=pooled_mean.device)
        # (B, N, T) — squared distance from each frame to each phoneme centre
        dist_sq = (
            t_grid.view(1, 1, T) - centers_shifted.unsqueeze(-1)
        ).pow(2)                                         # (B, N, T)

        # logits for softmax over N — (B, N, T)
        logits = -dist_sq / (2.0 * self.sigma ** 2)

        # mask padding phonemes with -inf before softmax
        if phoneme_mask is not None:
            logits = logits.masked_fill(
                phoneme_mask.unsqueeze(-1), float("-inf")
            )

        # softmax over phoneme dim → each frame sums to 1 across phonemes
        soft_align = torch.softmax(logits, dim=1)        # (B, N, T)

        return soft_align.permute(0, 2, 1)               # (B, T, N)