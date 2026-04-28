import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class EncoderOutput:
    z: torch.FloatTensor
    kl_loss: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None
    mu: Optional[torch.FloatTensor] = None
    vq_loss: Optional[torch.FloatTensor] = None
    vq_perplexity: Optional[torch.FloatTensor] = None
    vq_codes_used: Optional[torch.FloatTensor] = None
    vq_codes_used_frac: Optional[torch.FloatTensor] = None
    vq_latent_residual: Optional[torch.FloatTensor] = None
    vq_indices: Optional[torch.LongTensor] = None


@dataclass
class DecoderOutput:
    loss: Optional[torch.FloatTensor] = None
    mel_generated: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None


@dataclass
class FeatureExtractorOutput:
    audio_features: torch.FloatTensor
    padding_mask: torch.BoolTensor
