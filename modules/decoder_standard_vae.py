import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
import torch.nn as nn
from einops import rearrange
from dataclasses import dataclass
from typing import Optional, List


from modules.flash_attn_encoder import FlashTransformerEncoder
from modules.upsampler import Upsampler


@dataclass
class DecoderOutput:
    reconstructed_mel: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


@dataclass
class DecoderConfig:
    compress_factor_C: int
    tf_heads: int = 8
    tf_layers: int = 4
    drop_p: float = 0.1
    latent_dim: int = 64
    n_residual_blocks: int = 3
    phoneme_parsing_mode: str = "phoneme"
    vocab_path: str = "data/vocab.json"


class CausalTransformerTail(nn.Module):
    def __init__(self, d_model=512, nheads=8, nlayers=4, drop_p=0.1):
        super().__init__()
        self.enc = FlashTransformerEncoder(
            d_model=d_model, nhead=nheads, nlayers=nlayers, drop_p=drop_p
        )

    def forward(self, tokens):  # [B, T_tok, d_model]
        return self.enc(tokens, causal=True)


# TransformerEncoder with causal masking via is_causal for left-only attention [web:84][web:92]


class DecoderVAE(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()

        compress_factor_C = config.compress_factor_C
        tf_heads = config.tf_heads
        tf_layers = config.tf_layers
        drop_p = config.drop_p
        latent_dim = config.latent_dim
        # n_residual_blocks = config.n_residual_blocks FIXME handle

        self.upsampler = Upsampler(
            d_in=100,
            d_hidden=512,
            d_out=512,
            compress_factor=compress_factor_C,
            causal=True,
        )

        # Causal Transformer tail operating on tokens of size 512
        self.transformer = CausalTransformerTail(
            d_model=512, nheads=tf_heads, nlayers=tf_layers, drop_p=drop_p
        )

    def forward(
        self,
        z: torch.FloatTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        target_mel: Optional[torch.FloatTensor] = None,
        target_mel_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):  # x: [B, T, 100]

        x = self.transformer(z)  # [B, T/C, 512]
        x, mask, loss = self.upsampler(
            x=x,
            x_mask=padding_mask,
            target=target_mel,
            target_mask=target_mel_mask,
        )

        return DecoderOutput(
            reconstructed_mel=x,
            loss=loss,
        )
