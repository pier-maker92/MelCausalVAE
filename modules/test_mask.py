import torch
import numpy as np
import matplotlib.pyplot as plt
from modules.alignement import GAP_TOKEN
from modules.qformer import AlignmentQFormer
from modules.alignement import AlignmentMatrixBuilder

phonemes = [
    [
        {"end": 0.6, "phoneme": "AE0", "start": 0.49},
        {"end": 0.67, "phoneme": "N", "start": 0.6},
        {"end": 0.76, "phoneme": "T", "start": 0.67},
        {"end": 0.79, "phoneme": "OW1", "start": 0.76},
        {"end": 0.85, "phoneme": "N", "start": 0.79},
        {"end": 0.96, "phoneme": "IY0", "start": 0.85},
        {"end": 1.04, "phoneme": "AH0", "start": 0.96},
        {"end": 1.14, "phoneme": "P", "start": 1.04},
        {"end": 1.24, "phoneme": "OY1", "start": 1.14},
        {"end": 1.27, "phoneme": "N", "start": 1.24},
        {"end": 1.31, "phoneme": "T", "start": 1.27},
        {"end": 1.35, "phoneme": "AH0", "start": 1.31},
        {"end": 1.42, "phoneme": "D", "start": 1.35},
        {"end": 1.5, "phoneme": "AH1", "start": 1.42},
        {"end": 1.59, "phoneme": "P", "start": 1.5},
        {"end": 1.64, "phoneme": "T", "start": 1.59},
        {"end": 1.69, "phoneme": "AH0", "start": 1.64},
        {"end": 1.74, "phoneme": "DH", "start": 1.69},
        {"end": 1.81, "phoneme": "AH0", "start": 1.74},
        {"end": 1.89, "phoneme": "S", "start": 1.81},
        {"end": 1.98, "phoneme": "K", "start": 1.89},
        {"end": 2.27, "phoneme": "AY1", "start": 1.98},
        {"end": 2.44, "phoneme": "AE1", "start": 2.36},
        {"end": 2.49, "phoneme": "N", "start": 2.44},
        {"end": 2.53, "phoneme": "D", "start": 2.49},
        {"end": 2.59, "phoneme": "K", "start": 2.53},
        {"end": 2.63, "phoneme": "W", "start": 2.59},
        {"end": 2.7, "phoneme": "EH1", "start": 2.63},
        {"end": 2.79, "phoneme": "S", "start": 2.7},
        {"end": 2.85, "phoneme": "CH", "start": 2.79},
        {"end": 2.89, "phoneme": "AH0", "start": 2.85},
        {"end": 2.93, "phoneme": "N", "start": 2.89},
        {"end": 3, "phoneme": "D", "start": 2.93},
        {"end": 3.05, "phoneme": "M", "start": 3},
        {"end": 3.11, "phoneme": "IY1", "start": 3.05},
        {"end": 3.16, "phoneme": "W", "start": 3.11},
        {"end": 3.2, "phoneme": "IH0", "start": 3.16},
        {"end": 3.27, "phoneme": "TH", "start": 3.2},
        {"end": 3.31, "phoneme": "HH", "start": 3.27},
        {"end": 3.36, "phoneme": "ER0", "start": 3.31},
        {"end": 3.4, "phoneme": "G", "start": 3.36},
        {"end": 3.48, "phoneme": "L", "start": 3.4},
        {"end": 3.61, "phoneme": "AE1", "start": 3.48},
        {"end": 3.72, "phoneme": "N", "start": 3.61},
        {"end": 3.9, "phoneme": "S", "start": 3.72},
    ],
    [
        {"end": 0.29, "phoneme": "M", "start": 0.21},
        {"end": 0.33, "phoneme": "IH1", "start": 0.29},
        {"end": 0.38, "phoneme": "S", "start": 0.33},
        {"end": 0.46, "phoneme": "T", "start": 0.38},
        {"end": 0.49, "phoneme": "ER0", "start": 0.46},
        {"end": 0.82, "phoneme": "spn", "start": 0.49},
        {"end": 0.9, "phoneme": "W", "start": 0.85},
        {"end": 0.94, "phoneme": "AH0", "start": 0.9},
        {"end": 1.04, "phoneme": "Z", "start": 0.94},
        {"end": 1.14, "phoneme": "F", "start": 1.04},
        {"end": 1.17, "phoneme": "UH1", "start": 1.14},
        {"end": 1.24, "phoneme": "L", "start": 1.17},
        {"end": 1.27, "phoneme": "IY0", "start": 1.24},
        {"end": 1.35, "phoneme": "R", "start": 1.27},
        {"end": 1.4, "phoneme": "IH0", "start": 1.35},
        {"end": 1.48, "phoneme": "S", "start": 1.4},
        {"end": 1.55, "phoneme": "P", "start": 1.48},
        {"end": 1.65, "phoneme": "AA1", "start": 1.55},
        {"end": 1.7, "phoneme": "N", "start": 1.65},
        {"end": 1.76, "phoneme": "S", "start": 1.7},
        {"end": 1.79, "phoneme": "IH0", "start": 1.76},
        {"end": 1.86, "phoneme": "V", "start": 1.79},
        {"end": 1.92, "phoneme": "N", "start": 1.86},
        {"end": 2.16, "phoneme": "AW1", "start": 1.92},
    ],
]


def plot_cross_attn_mask(mask, batch_idx=0, num_heads=8):
    fig = plt.imshow(
        mask[batch_idx * num_heads].T.cpu().numpy(),
        aspect="auto",
    )
    plt.savefig(f"cross_attn_mask.png")
    plt.close()
    print("Cross-attention mask saved to cross_attn_mask.png")


def plot_self_attn_mask(mask):
    fig = plt.imshow(mask.T.cpu().numpy(), aspect="auto")
    plt.savefig("self_attn_mask.png")
    plt.close()
    print("Self-attention mask saved to self_attn_mask.png")


def plot_alignment(alignments, batch_idx=0):
    # Create an RGB image to highlight GAPs in green
    matrix = alignments.alignments[batch_idx].cpu().numpy()  # (T, N)
    labels = alignments.segment_labels[batch_idx]

    # Initialize with black (zeros) - (T, N, 3)
    rgb = np.zeros((*matrix.shape, 3))

    for j, label in enumerate(labels):
        mask = matrix[:, j] > 0
        if label == GAP_TOKEN:
            rgb[mask, j] = [0.0, 0.8, 0.0]  # Green for gaps
        else:
            rgb[mask, j] = [0.9, 0.9, 0.9]  # Light gray for phonemes

    plt.imshow(rgb, aspect="auto")
    # set x ticks to be phonemes
    plt.xticks(
        range(len(labels)),
        [p for p in labels],
        rotation=90,
    )
    plt.xlabel("Phonemes")
    plt.ylabel("Frames")
    plt.title("Phoneme Alignment (GAPs in Green)")
    plt.tight_layout()
    plt.savefig("alignments.png")
    plt.close()

    print("Alignment plot saved to alignments.png")


if __name__ == "__main__":
    builder = AlignmentMatrixBuilder(
        compress_factor=1,
        embedding_dim=128,
    )
    num_heads = 4
    qformer = AlignmentQFormer(
        num_queries_per_phoneme=1,
        context_expansion=10,
        d_model=128,
        num_heads=num_heads,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # alignements
    nframes = 410, 203
    alignments = builder.build(phonemes, nframes, device=device, dtype=dtype)

    plot_alignment(alignments, 1)

    # qformer
    cross_mask = qformer._build_cross_attn_mask(
        alignments.alignments, alignments.phoneme_mask
    )
    self_mask = qformer._build_causal_mask(alignments.alignments.shape[-1], device)
    plot_cross_attn_mask(cross_mask, 0, num_heads=num_heads)
    plot_self_attn_mask(self_mask)

    frames = torch.randn(2, 410, 128, device=device, dtype=dtype)
    out = qformer(
        frames, alignments.alignments, alignments.embeddings, alignments.phoneme_mask
    )

    # Test DurationConditioningProjector
    from modules.qformer import DurationConditioningProjector

    print(f"Pooled shape: {out.pooled.shape}")

    projector = DurationConditioningProjector(
        d_in=out.pooled.shape[-1],
        d_out=256,
        channels=256,
        kernel_size=31,
        n_layers=3
    ).to(device).to(dtype)

    durations = alignments.durations
    cond = projector(out.pooled, durations, out.rel_pos)

    print(f"Conditioning shape: {cond.shape}")
    print("Success: QFormer and DurationConditioningProjector test passed.")
