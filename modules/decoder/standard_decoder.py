import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..configs import StandardDecoderConfig
from ..output_dataclasses import DecoderOutput


class PreNormResBlock1d(nn.Module):
    """Non-causal mirror of PreNormResCausalBlock1d — symmetric Conv1d padding."""

    def __init__(self, c_in: int, c_out: int, *, k: int = 3, d: int = 1, drop_p: float = 0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, c_in)
        self.act = nn.GELU()
        pad = (k - 1) * d // 2
        self.main = nn.Conv1d(c_in, c_out, k, dilation=d, padding=pad)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        h = self.act(self.norm(x))
        h = self.dropout(h)
        return self.main(h) + self.skip(x)


class UpsamplingBlock1d(nn.Module):
    """Mirror of CausalDownsamplingBlock1d: transpose-conv 2× upsample then residual stack."""

    def __init__(self, c_in: int, c_out: int, n_residual_blocks: int = 3, drop_p: float = 0.1):
        super().__init__()
        # ConvTranspose1d(k=4, s=2, p=1): output = (input-1)*2 - 2 + 4 = 2*input ✓
        self.upsample = nn.ConvTranspose1d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        dilations = [1, 2, 4, 8][:n_residual_blocks]
        self.residual_blocks = nn.ModuleList([
            PreNormResBlock1d(c_out, c_out, k=5, d=d, drop_p=drop_p)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T] → [B, C, 2T]
        x = self.upsample(x)
        for block in self.residual_blocks:
            x = block(x)
        return x


class BidirectionalTransformer(nn.Module):
    """TransformerEncoder without causal mask (bidirectional attention)."""

    def __init__(self, d_model: int = 512, nheads: int = 8, nlayers: int = 4, drop_p: float = 0.1):
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

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.enc(x, src_key_padding_mask=pad_mask)


class ConvDecoder(nn.Module):
    """
    Deterministic CNN decoder — non-causal mirror of the encoder.

    Encoder path:  mel [B,T,mel] → in_proj → mixer → downsample×N → transformer → latent
    Decoder path:  latent → latent_proj → transformer → upsample×N → demixer → out_proj → mel
    """

    def __init__(self, config: StandardDecoderConfig):
        super().__init__()
        d = config.d_model
        num_stages = int(math.log2(config.compress_factor))
        self.compress_factor = config.compress_factor

        self.latent_proj = nn.Linear(config.audio_latent_dim, d)

        self.transformer = BidirectionalTransformer(
            d_model=d,
            nheads=config.tf_heads,
            nlayers=config.tf_layers,
            drop_p=config.drop_p,
        )

        # Upsample log2(C) times, each 2×
        self.upsampling = nn.ModuleDict()
        for i in range(num_stages):
            factor = 2 ** (num_stages - i)
            self.upsampling[f"upsample@{factor}"] = UpsamplingBlock1d(
                d, d, n_residual_blocks=config.n_residual_blocks, drop_p=config.drop_p
            )

        # Demixer: reverse of encoder mixer (d → d → d//2), dilations reversed
        self.demixer = nn.Sequential(
            PreNormResBlock1d(d,     d,     k=3, d=4, drop_p=config.drop_p),
            PreNormResBlock1d(d,     d,     k=5, d=2, drop_p=config.drop_p),
            PreNormResBlock1d(d, d // 2,   k=7, d=1, drop_p=config.drop_p),
        )

        self.out_proj = nn.Conv1d(d // 2, config.mel_dim, kernel_size=7, padding=3)

    def _decode(
        self,
        context_vector: torch.Tensor,
        target_len: Optional[int] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """context_vector: [B, T/C, latent_dim] → mel: [B, T, mel_dim]"""
        x = self.latent_proj(context_vector)           # [B, T/C, d]
        x = self.transformer(x, pad_mask=padding_mask) # [B, T/C, d]
        x = x.transpose(1, 2)                          # [B, d, T/C]
        for layer in self.upsampling.values():
            x = layer(x)                               # [B, d, T]
        x = self.demixer(x)                            # [B, d//2, T]
        x = self.out_proj(x)                           # [B, mel_dim, T]
        x = x.transpose(1, 2)                          # [B, T, mel_dim]

        # Trim/pad to match target_len if provided
        if target_len is not None and x.shape[1] != target_len:
            if x.shape[1] > target_len:
                x = x[:, :target_len, :]
            else:
                pad = target_len - x.shape[1]
                x = F.pad(x, (0, 0, 0, pad))
        return x

    def forward(
        self,
        context_vector: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.BoolTensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> DecoderOutput:
        target_len = target.shape[1] if target is not None else None
        mel_pred = self._decode(context_vector, target_len=target_len, padding_mask=None)

        loss = None
        if target is not None:
            T = mel_pred.shape[1]
            if target_padding_mask is not None:
                valid = ~target_padding_mask[:, :T]        # [B, T]
                if valid.any():
                    loss = F.l1_loss(mel_pred[valid], target[:, :T, :][valid])
                else:
                    loss = F.l1_loss(mel_pred, target[:, :T, :])
            else:
                loss = F.l1_loss(mel_pred, target[:, :T, :])

        return DecoderOutput(
            loss=loss,
            audio_features=mel_pred,
            padding_mask=target_padding_mask,
        )

    def generate(
        self,
        context_vector: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> DecoderOutput:
        mel = self._decode(context_vector, padding_mask=None)
        # Upsample mask from latent framerate [B, T/C] to mel framerate [B, T]
        if padding_mask is not None:
            mel_mask = torch.repeat_interleave(padding_mask, self.compress_factor, dim=1)
            mel_mask = mel_mask[:, : mel.shape[1]]
        else:
            mel_mask = None
        return DecoderOutput(
            loss=None,
            audio_features=mel,
            padding_mask=mel_mask,
        )
