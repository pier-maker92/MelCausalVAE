import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def upsample_soft_train(z, alignment_xz, eps: float = 1e-8):
    # Convert frame->z alignment into z->frame weights by normalizing over z (columns).
    col_sum = alignment_xz.sum(dim=1, keepdim=True)  # (B, 1, T)
    w = alignment_xz / (col_sum + eps)
    # (B, D, L) @ (B, L, T) -> (B, D, T)
    return torch.bmm(z, w)


def upsample_hard_infer(z, durations, T=None):
    if durations.dtype not in (torch.int32, torch.int64):
        durations = durations.long()

    B, D, L = z.size()
    lengths = durations.sum(dim=1)
    max_T = int(lengths.max().item()) if T is None else int(T)
    out_lengths = torch.clamp(lengths, max=max_T)

    h = z.new_zeros(B, D, max_T)
    for b in range(B):
        Tb = int(out_lengths[b].item())
        if Tb <= 0:
            continue
        # Build indices [0,0,...,1,1,...,L-1,...] according to durations
        idx = torch.repeat_interleave(torch.arange(L, device=z.device), durations[b].clamp(min=0))
        if T is None:
            h[b, :, :Tb] = z[b].index_select(1, idx)
        else:
            # crop/pad to fixed T
            idx = idx[:max_T]
            h[b, :, : idx.numel()] = z[b].index_select(1, idx)

    h_mask = torch.arange(max_T, device=z.device).unsqueeze(0) < out_lengths.unsqueeze(1)
    return h, h_mask


def compose_alignments(alignment_list):
    """
    Compose alignment matrices from chained stride>1 layers to map
    final z positions back to original input frames.

    Alignments are expected in order from input-side to output-side:
        alignment_list[0]: (B, T/s, T)       — first downsampling layer
        alignment_list[1]: (B, T/s^2, T/s)   — second downsampling layer
        ...
    The composed result maps the deepest z directly to original frames.

    Args:
        alignment_list: list of (B, L_out, L_in) alignment tensors.

    Returns:
        composed: (B, L_final, T_original)
    """
    if not alignment_list:
        return None

    composed = alignment_list[0]
    for a in alignment_list[1:]:
        # a: (B, L_next, L_prev)  @  composed: (B, L_prev, T_original)
        composed = torch.bmm(a, composed)

    return composed


def get_durations(composed_alignment, z_lengths=None, x_lengths=None):
    """
    Extract discrete durations from a composed alignment matrix.

    Each original frame t is assigned to the z position with the highest
    attention weight.  Monotonicity is enforced via cumulative-max so that
    z regions are always contiguous.

    Args:
        composed_alignment: (B, L, T) soft alignment from z to original frames.
        z_lengths:  (B,) valid z lengths (optional).
        x_lengths:  (B,) valid input lengths (optional).

    Returns:
        durations:  (B, L) integer durations (sum to x_lengths per sample).
        boundaries: (B, L) start-frame index for each z position.
    """
    B, L, T = composed_alignment.size()

    # For each frame t, find the z with highest attention weight
    assignment = composed_alignment.argmax(dim=1)  # (B, T)

    durations = torch.zeros(B, L, device=composed_alignment.device, dtype=torch.long)
    boundaries = torch.zeros(B, L, device=composed_alignment.device, dtype=torch.long)

    for b in range(B):
        t_max = int(x_lengths[b].item()) if x_lengths is not None else T
        l_max = int(z_lengths[b].item()) if z_lengths is not None else L

        assign_b = assignment[b, :t_max].clamp(0, l_max - 1)

        # Enforce monotonicity (running maximum)
        assign_b, _ = assign_b.cummax(dim=0)

        # Count how many frames fall into each z position
        durations[b].scatter_add_(0, assign_b, torch.ones_like(assign_b))

        # Boundaries are the cumulative sum of preceding durations
        if l_max > 1:
            boundaries[b, 1:l_max] = durations[b, : l_max - 1].cumsum(0)

    return durations, boundaries


@torch.no_grad()
def extract_durations(model, x, x_lengths):
    """
    Run a forward pass through the model, capture the full-batch alignment
    matrix from every Aligner (stride>1) layer via forward hooks, compose
    them, and return discrete durations.

    Args:
        model:     ResNet model (can be in eval or train mode).
        x:         (B, n_mel, T) input mel spectrogram.
        x_lengths: (B,) valid input lengths.

    Returns:
        durations:           (B, L_final) integer durations per z position.
        boundaries:          (B, L_final) start-frame index per z position.
        composed_alignment:  (B, L_final, T) composed soft alignment.
        layer_alignments:    list of per-layer (B, L_i, T_i) alignments.
        z_lengths:           (B,) valid z lengths at the final level.
    """
    layer_alignments = []

    def _hook(module, input, output):
        # Aligner with ver_f=False returns:
        #   (z, z_mask, z_lengths, alignment, score_loss)
        if not isinstance(output, tuple):
            return
        if len(output) == 5:
            alignment = output[3]
        elif len(output) == 8:
            # (z, z_mask, z_lengths, z_zeros, indices, x_weights, alignment, score_loss)
            alignment = output[6]
        else:
            return

        if isinstance(alignment, torch.Tensor) and alignment.dim() == 3:
            layer_alignments.append(alignment.detach())

    # Register hooks on every Aligner
    hooks = []
    for block in model.ResBlocks:
        if hasattr(block, "aligner"):
            hooks.append(block.aligner.register_forward_hook(_hook))

    # Forward pass
    x, x_mask, alignments_from_model, _ = model(x, x_lengths)

    # Prefer alignments returned by the model (already filtered per block),
    # otherwise fall back to hook-captured alignments.
    if isinstance(alignments_from_model, list):
        layer_alignments = [a.detach() for a in alignments_from_model if isinstance(a, torch.Tensor)]

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Compute z_lengths by tracing the stride chain
    z_lengths = x_lengths.float()
    for block in model.ResBlocks:
        if block.stride != 1:
            z_lengths = torch.ceil(z_lengths / block.stride)
    z_lengths = z_lengths.long()

    # Compose all layer alignments into a single matrix
    composed = compose_alignments(layer_alignments)

    # Derive discrete durations
    durations, boundaries = get_durations(composed, z_lengths, x_lengths)

    return x, x_mask, durations, boundaries, composed, layer_alignments, z_lengths


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_duration_regions(mel_spectrogram, durations, z_length=None, title=None):
    """
    Plot a mel spectrogram with coloured regions showing which original
    frames each z position covers, plus a duration bar-chart below.

    Args:
        mel_spectrogram: (n_mel, T) tensor or ndarray.
        durations:       (L,) integer durations for **one** sample.
        z_length:        valid number of z positions (defaults to all non-zero).
        title:           optional figure title.

    Returns:
        fig: matplotlib Figure.
    """
    if isinstance(mel_spectrogram, torch.Tensor):
        mel_spectrogram = mel_spectrogram.detach().cpu().numpy()

    if isinstance(durations, torch.Tensor):
        durations_np = durations.detach().cpu().numpy()
    else:
        durations_np = np.asarray(durations)

    # Make durations a strict 1D numeric array (matplotlib can recurse on object/2D inputs)
    durations_np = np.asarray(durations_np, dtype=np.int64).reshape(-1)

    if mel_spectrogram.ndim != 2:
        raise ValueError(f"mel_spectrogram must be 2D (n_mel, T), got shape {mel_spectrogram.shape}")

    n_mel, T = mel_spectrogram.shape

    if z_length is not None:
        if isinstance(z_length, torch.Tensor):
            L = int(z_length.detach().cpu().item())
        else:
            L = int(z_length)
    else:
        L = int(np.count_nonzero(durations_np))

    L = max(0, min(L, durations_np.shape[0]))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(12, T / 20), 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ---- top: mel spectrogram with z regions ----
    ax = axes[0]
    ax.imshow(mel_spectrogram, aspect="auto", origin="lower", interpolation="none")

    cmap = plt.cm.tab20
    start = 0
    for l in range(L):
        dur = int(durations_np[l])
        if dur == 0:
            continue
        color = cmap(l % 20)
        rect = plt.Rectangle(
            (start - 0.5, -0.5),
            dur,
            n_mel,
            linewidth=0,
            facecolor=(*color[:3], 0.18),
        )
        ax.add_patch(rect)
        if l > 0 and start > 0:
            ax.axvline(
                x=start - 0.5,
                color="white",
                linewidth=0.8,
                linestyle="--",
                alpha=0.7,
            )
        ax.text(
            start + dur / 2,
            n_mel * 0.95,
            f"z{l}",
            ha="center",
            va="top",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
        start += dur

    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.5, n_mel - 0.5)
    ax.set_ylabel("Mel bin")
    ax.set_title(title or "Mel spectrogram — z coverage regions")

    # ---- bottom: duration bar chart ----
    ax2 = axes[1]
    colors = [cmap(l % 20) for l in range(L)]
    x_pos = np.arange(L, dtype=np.int64)
    ax2.bar(x_pos, durations_np[:L], color=colors, edgecolor="black", linewidth=0.3)
    ax2.set_xlabel("z position")
    ax2.set_ylabel("Duration (frames)")
    ax2.set_xlim(-0.5, L - 0.5)

    plt.tight_layout()
    return fig


def plot_composed_alignment(composed_alignment, x_length=None, z_length=None, title=None):
    """
    Visualise the composed alignment matrix for a single sample.

    Args:
        composed_alignment: (L, T) or (1, L, T) tensor.
        x_length: valid T (optional, for cropping).
        z_length: valid L (optional, for cropping).
        title:    optional figure title.

    Returns:
        fig: matplotlib Figure.
    """
    if composed_alignment.dim() == 3:
        composed_alignment = composed_alignment[0]

    if isinstance(composed_alignment, torch.Tensor):
        data = composed_alignment.cpu().numpy()
    else:
        data = np.asarray(composed_alignment)

    L, T = data.shape
    if x_length is not None:
        T = int(x_length)
    if z_length is not None:
        L = int(z_length)
    data = data[:L, :T]

    fig, ax = plt.subplots(figsize=(max(10, T / 25), max(4, L / 5)))
    ax.imshow(data, aspect="auto", origin="lower", interpolation="none")
    ax.set_xlabel("Original frame")
    ax.set_ylabel("z position")
    ax.set_title(title or "Composed alignment (z → original frames)")
    plt.tight_layout()
    return fig
