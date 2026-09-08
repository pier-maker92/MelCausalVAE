"""Codebook statistics pooled over all valid tokens in one batch."""

import torch


@torch.no_grad()
def batch_codebook_metrics(indices: torch.Tensor, codebook_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exp(entropy) and percentage of used codes; -1 denotes padding."""
    tokens = indices.reshape(-1)
    tokens = tokens[tokens >= 0]
    if tokens.numel() == 0:
        raise ValueError("Codebook metrics require at least one valid token.")
    counts = torch.bincount(tokens, minlength=codebook_size).float()
    used_counts = counts[counts > 0]
    probabilities = used_counts / used_counts.sum()
    perplexity = (-(probabilities * probabilities.log()).sum()).exp()
    utilization = counts.new_tensor(100.0 * used_counts.numel() / codebook_size)
    return perplexity, utilization
