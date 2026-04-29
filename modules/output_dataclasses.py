from pandas.core.internals.construction import dataclasses_to_dicts
import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class VQStats:
    perplexity: torch.Tensor
    codes_used: torch.Tensor
    codes_used_frac: torch.Tensor


@dataclass
class VQVAEOutput:
    indices: torch.LongTensor
    quantized: torch.FloatTensor
    residual: torch.FloatTensor
    stats: VQStats
    loss: Optional[torch.FloatTensor] = None


@dataclass
class EncoderOutput:
    z: torch.FloatTensor
    kl_loss: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None
    mu: Optional[torch.FloatTensor] = None
    vq_output: Optional[VQVAEOutput] = None
    mu_pre_vq: Optional[torch.FloatTensor] = None


@dataclass
class VAEOutput:
    audio_loss: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    mu_mean: Optional[torch.FloatTensor] = None
    mu_var: Optional[torch.FloatTensor] = None
    vq_stats: Optional[VQStats] = None
    vq_loss: Optional[torch.FloatTensor] = None


@dataclass
class DecoderOutput:
    loss: Optional[torch.FloatTensor] = None
    audio_features: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None


@dataclass
class FeatureExtractorOutput:
    audio_features: torch.FloatTensor
    padding_mask: torch.BoolTensor
