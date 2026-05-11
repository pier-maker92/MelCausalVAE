from transformers.utils import ModelOutput
import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class VQStats(ModelOutput):
    perplexity: Optional[torch.Tensor] = None
    codes_used: Optional[torch.Tensor] = None
    codes_used_frac: Optional[torch.Tensor] = None


@dataclass
class VQVAEOutput(ModelOutput):
    indices: Optional[torch.LongTensor] = None
    quantized: Optional[torch.FloatTensor] = None
    residual: Optional[torch.FloatTensor] = None
    stats: Optional[VQStats] = None
    loss: Optional[torch.FloatTensor] = None


@dataclass
class EncoderOutput(ModelOutput):
    z: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None
    mu: Optional[torch.FloatTensor] = None
    vq_stats: Optional[VQStats] = None
    vq_loss: Optional[torch.FloatTensor] = None
    mu_pre_vq: Optional[torch.FloatTensor] = None
    quantized: Optional[torch.FloatTensor] = None
    residual: Optional[torch.FloatTensor] = None
    tail: Optional[torch.FloatTensor] = None
    indices: Optional[torch.LongTensor] = None


@dataclass
class VAEOutput(ModelOutput):
    audio_loss: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    mu_mean: Optional[torch.FloatTensor] = None
    mu_var: Optional[torch.FloatTensor] = None
    vq_loss: Optional[torch.FloatTensor] = None
    vq_stats: Optional[VQStats] = None
    quantized: Optional[torch.FloatTensor] = None
    residual: Optional[torch.FloatTensor] = None
    tail: Optional[torch.FloatTensor] = None


@dataclass
class DecoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    audio_features: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None


@dataclass
class FeatureExtractorOutput(ModelOutput):
    audio_features: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None
