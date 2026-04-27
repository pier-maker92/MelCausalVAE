import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class EncoderOutput:
    z: torch.FloatTensor
    kl_loss: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None
    mu: Optional[torch.FloatTensor] = None
    semantic_loss: Optional[torch.FloatTensor] = None
    semantic_features: Optional[torch.FloatTensor] = None
    durations: Optional[torch.LongTensor] = None
    z_pooled_fps: Optional[torch.FloatTensor] = None
    vq_loss: Optional[torch.FloatTensor] = None
    vq_perplexity: Optional[torch.FloatTensor] = None
    vq_codes_used: Optional[torch.FloatTensor] = None
    vq_codes_used_frac: Optional[torch.FloatTensor] = None
    # [B, T, vq_quant_dim] pre-additive VQ residual (mu_head - mu_q), detached.
    vq_latent_residual: Optional[torch.FloatTensor] = None
    # [B, T] codebook indices (padded positions still hold an index; mask with padding_mask).
    vq_indices: Optional[torch.LongTensor] = None
