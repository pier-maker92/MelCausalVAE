"""
ConvformerDecoder — causal mirror of ConvformerEncoder.

Reverses the encoder path with explicit target-size matching at each stage.
All convolutions are causal in time (left-padded only).

Encoder trace (C=8, T=128, F=100):
  in_proj:       [B, 32, 128, 100]
  freq_mixer:    [B, 256, 128, 100]
  downsample@2:  [B, 512, 64, 25]      (time/2, freq/4)
  downsample@4:  [B, 512, 32, 7]       (time/2, freq ~= 25/4 ≈ 7)
  downsample@8:  [B, 512, 16, 7]       (time/2, freq/1)
  freq_collapse: [B, 512, 16, 1]       (freq 7→1 via kf=8,sf=8)
  → mu:          [B, 16, 64]

Decoder reverses:
  z [B,16,64] → linear+transformer [B,16,512]
  → reshape [B,512,16,1]
  → freq_expand [B,512,16,F_expand]    (ConvTranspose2d 1→7)
  → upsample@8to4 [B,512,32,F]        (time×2, freq×1)
  → upsample@4to2 [B,512,64,F']       (time×2, freq×4)
  → upsample@2to1 [B,256,128,F'']     (time×2, freq×4)
  → freq_demixer [B,32,128,F''] → out_proj → [B,128,100]
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
import torch.nn.functional as F
from dataclasses import dataclass
from modules.cfm import DiTOutput

from modules.Encoder import (
    ChannelLastLayerNorm,
    PreNormResCausalBlock,
    TimeCausalConv2d,
    CausalTransformerTail,
    ConvformerEncoderConfig,
)


# ---------- Causal Transposed Conv2d ----------


class TimeCausalConvTranspose2d(nn.Module):
    """
    Transposed convolution that is causal in time (produce no future dependency)
    and targets a specific output frequency size.

    Strategy: use ConvTranspose2d with no padding, then trim to exact target size.
    Since this is a *decoder*, causality means each output time step t depends only
    on latent steps ≤ t (guaranteed by trimming the right side of time output).
    """

    def __init__(self, c_in, c_out, kt, kf, st=1, sf=1):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            c_in, c_out,
            kernel_size=(kt, kf),
            stride=(st, sf),
            padding=0,
            output_padding=0,
        )
        self.kt, self.kf = kt, kf
        self.st, self.sf = st, sf

    def forward(self, x, target_t=None, target_f=None):
        """
        x: [B, C, T, F]
        target_t: desired output time dimension (default: T * st)
        target_f: desired output freq dimension (default: F * sf)
        """
        T_in, F_in = x.shape[2], x.shape[3]
        y = self.conv_t(x)  # raw transposed conv output

        # Target sizes
        t_want = target_t if target_t is not None else T_in * self.st
        f_want = target_f if target_f is not None else F_in * self.sf

        # Trim/pad time (keep left = causal)
        if y.shape[2] > t_want:
            y = y[:, :, :t_want, :]
        elif y.shape[2] < t_want:
            y = F.pad(y, (0, 0, 0, t_want - y.shape[2]))

        # Trim/pad freq (center crop for symmetry)
        if y.shape[3] > f_want:
            excess = y.shape[3] - f_want
            left = excess // 2
            y = y[:, :, :, left: left + f_want]
        elif y.shape[3] < f_want:
            y = F.pad(y, (0, f_want - y.shape[3]))

        return y


# ---------- Causal Upsampling Block ----------


class CausalUpsamplingBlock(nn.Module):
    """Mirrors CausalDownsamplingBlock: residual blocks + transposed-conv upsample."""

    def __init__(self, c_in, c_out, n_residual_blocks=2, expand_freq=4, drop_p=0.1):
        super().__init__()
        dilations = [1, 2, 4, 8]
        self.residual_blocks = nn.ModuleList(
            [
                PreNormResCausalBlock(
                    c_in, c_in, kt=3, kf=5, dt=dilation, df=1, st=1, sf=1, drop_p=drop_p
                )
                for dilation in dilations[:n_residual_blocks]
            ]
        )
        self.ln = ChannelLastLayerNorm(c_in)
        self.act = nn.GELU()
        self.upsampling = TimeCausalConvTranspose2d(
            c_in, c_out, kt=5, kf=7, st=2, sf=expand_freq
        )

    def forward(self, x, target_t=None, target_f=None):
        for block in self.residual_blocks:
            x = block(x)
        x = self.act(self.ln(x))
        return self.upsampling(x, target_t=target_t, target_f=target_f)


# ---------- Decoder Config ----------


@dataclass
class ConvformerDecoderConfig:
    """Config that mirrors ConvformerEncoderConfig for the decoder."""

    compress_factor_C: int = 8
    tf_heads: int = 8
    tf_layers: int = 4
    drop_p: float = 0.1
    latent_dim: int = 64
    n_residual_blocks: int = 3
    mel_channels: int = 100

    @staticmethod
    def from_encoder_config(enc_cfg: ConvformerEncoderConfig) -> "ConvformerDecoderConfig":
        return ConvformerDecoderConfig(
            compress_factor_C=enc_cfg.compress_factor_C,
            tf_heads=enc_cfg.tf_heads,
            tf_layers=enc_cfg.tf_layers,
            drop_p=enc_cfg.drop_p,
            latent_dim=enc_cfg.latent_dim,
            n_residual_blocks=enc_cfg.n_residual_blocks,
        )


# ---------- Full Decoder ----------


class ConvformerDecoder(nn.Module):
    """
    Classic VAE decoder mirroring ConvformerEncoder, fully causal.
    """

    def __init__(self, config: ConvformerDecoderConfig):
        super().__init__()
        self.config = config
        C = config.compress_factor_C
        assert C >= 1 and (C & (C - 1)) == 0, "C must be power of 2"
        self.C = C
        mel_channels = config.mel_channels
        latent_dim = config.latent_dim
        tf_heads = config.tf_heads
        tf_layers = config.tf_layers
        drop_p = config.drop_p
        n_residual_blocks = config.n_residual_blocks

        # Compute the frequency dimensions at each encoder stage by simulating
        # the encoder's "same" padding behavior for the freq axis.
        # Encoder: F_out = ceil(F_in / sf) for each downsampling stage.
        # Starting from mel_channels=100:
        #   ds@2: 100 → ceil(100/4) = 25
        #   ds@4: 25  → ceil(25/4)  = 7
        #   extra ds (sf=1): 7 → 7 (no freq change)
        #   freq_collapse (sf=8): 7 → ceil(7/8) = 1
        # Decoder reverses: 1 → 7 → 7 → 25 → 100

        self._compute_freq_schedule(mel_channels)

        # --- Latent → hidden ---
        self.latent_proj = nn.Linear(latent_dim, 512)

        # --- Transformer tail ---
        self.transformer = CausalTransformerTail(
            d_model=512, nheads=tf_heads, nlayers=tf_layers, drop_p=drop_p
        )

        # --- Frequency expansion: 1 → freq_after_collapse ---
        self.freq_expand = nn.Sequential(
            ChannelLastLayerNorm(512),
            nn.GELU(),
            TimeCausalConvTranspose2d(512, 512, kt=1, kf=8, st=1, sf=8),
        )

        # --- Upsampling blocks (reverse order of downsampling) ---
        self.upsampling = nn.ModuleDict()

        # Extra temporal upsampling stages (mirrors extra temporal downsamples, sf=1)
        extra_stages = int(math.log2(C)) - 2
        for i in range(max(0, extra_stages) - 1, -1, -1):
            stage_name = f"upsample@{2**(i+2+1)}to{2**(i+2)}"
            self.upsampling[stage_name] = CausalUpsamplingBlock(
                512, 512, n_residual_blocks=n_residual_blocks, expand_freq=1, drop_p=drop_p
            )

        # us@4→2: [B,512,T/4,7] → [B,512,T/2,25]
        self.upsampling["upsample@4to2"] = CausalUpsamplingBlock(
            512, 512, n_residual_blocks=n_residual_blocks, expand_freq=4, drop_p=drop_p
        )

        # us@2→1: [B,512,T/2,25] → [B,256,T,100]
        self.upsampling["upsample@2to1"] = CausalUpsamplingBlock(
            512, 256, n_residual_blocks=n_residual_blocks, expand_freq=4, drop_p=drop_p
        )

        # --- Frequency demixer (mirrors freq_mixer: 256→128→64→32) ---
        self.freq_demixer = nn.Sequential(
            PreNormResCausalBlock(256, 128, kt=3, kf=3, dt=1, df=8, st=1, sf=1, drop_p=drop_p),
            PreNormResCausalBlock(128, 64, kt=5, kf=5, dt=1, df=4, st=1, sf=1, drop_p=drop_p),
            PreNormResCausalBlock(64, 32, kt=7, kf=7, dt=1, df=1, st=1, sf=1, drop_p=drop_p),
        )

        # --- Output projection: [B,32,T,F] → [B,1,T,F] ---
        self.out_proj = TimeCausalConv2d(32, 1, kt=3, kf=5, dt=1, df=1, st=1, sf=1)

        # --- Linear to ensure exactly mel_channels ---
        self.out_freq_proj = nn.Linear(mel_channels, mel_channels)

    def _compute_freq_schedule(self, mel_channels: int):
        """
        Simulate the encoder's frequency reduction to compute target sizes
        at each decoder upsampling stage.
        """
        C = self.C

        # Encoder freq schedule (forward):
        # After freq_mixer: F = mel_channels (100) — sf=1 throughout
        # downsample@2: sf=4 → ceil(F / 4)
        # downsample@4: sf=4 → ceil(F / 4)
        # extra stages: sf=1 → F stays same
        # freq_collapse: sf=8 → ceil(F / 8)

        freqs = [mel_channels]  # stage 0: after freq_mixer

        # ds@2: sf=4
        f = (freqs[-1] + 4 - 1) // 4  # ceil division
        freqs.append(f)  # stage 1: after ds@2

        # ds@4: sf=4
        f = (freqs[-1] + 4 - 1) // 4
        freqs.append(f)  # stage 2: after ds@4

        # extra temporal stages: sf=1 (freq unchanged)
        extra_stages = int(math.log2(C)) - 2
        for _ in range(max(0, extra_stages)):
            freqs.append(freqs[-1])

        # freq_collapse: sf=8
        f_collapsed = (freqs[-1] + 8 - 1) // 8
        freqs.append(f_collapsed)

        # Now freqs[-1] = 1, and we reverse
        # Decoder freq targets (reverse of freqs, skipping the last which is 1):
        self.freq_schedule = list(reversed(freqs))
        # freq_schedule[0] = 1 (input)
        # freq_schedule[1] = 7 (after freq_expand)
        # freq_schedule[2] = 7 (after extra temporal upsample, if any)
        # freq_schedule[-2] = 25 (after us@4→2)
        # freq_schedule[-1] = 100 (after us@2→1)

    def _decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """
        z: [B, T/C, latent_dim] → mel: [B, T, mel_channels]
        """
        B = z.shape[0]
        T_latent = z.shape[1]
        T_out = T_latent * self.C

        # Latent projection
        h = self.latent_proj(z)  # [B, T/C, 512]

        # Transformer
        h = self.transformer(h)  # [B, T/C, 512]

        # Reshape to 4D: [B, 512, T/C, 1]
        h = h.transpose(1, 2).unsqueeze(-1)  # [B, 512, T/C, 1]

        # Frequency expansion: [B, 512, T/C, 1] → [B, 512, T/C, freq_schedule[1]]
        target_f = self.freq_schedule[1]
        for layer in self.freq_expand:
            if isinstance(layer, TimeCausalConvTranspose2d):
                h = layer(h, target_t=T_latent, target_f=target_f)
            else:
                h = layer(h)

        # Upsampling blocks with target sizes
        # Build time targets: T/C, then T/(C/2), T/(C/4), ..., T
        time_mult = 1
        freq_idx = 2  # next freq schedule index
        for name, layer in self.upsampling.items():
            time_mult *= 2
            target_t = T_latent * time_mult
            target_f = self.freq_schedule[freq_idx] if freq_idx < len(self.freq_schedule) else None
            h = layer(h, target_t=target_t, target_f=target_f)
            freq_idx += 1

        # Frequency demixer: [B, 256, T, 100] → [B, 32, T, 100]
        h = self.freq_demixer(h)

        # Output projection: [B, 32, T, F] → [B, 1, T, F]
        h = self.out_proj(h)
        h = h.squeeze(1)  # [B, T, F]

        # Ensure exact mel_channels via linear projection
        if h.shape[-1] != self.config.mel_channels:
            # Trim or pad freq to mel_channels before linear
            if h.shape[-1] > self.config.mel_channels:
                h = h[:, :, :self.config.mel_channels]
            else:
                h = F.pad(h, (0, self.config.mel_channels - h.shape[-1]))

        h = self.out_freq_proj(h)  # [B, T, mel_channels]

        return h

    def forward(
        self,
        target: torch.FloatTensor,
        target_padding_mask: torch.BoolTensor,
        context_vector: torch.FloatTensor,
    ):
        """
        Training forward pass. Mirrors DiT.forward() API.

        Args:
            target: [B, T, mel_channels] — target mel spectrogram
            target_padding_mask: [B, T] — True for padded positions
            context_vector: [B, T/C, latent_dim] — encoder latent z
        """
        reconstructed = self._decode(context_vector)

        # Trim to match target length
        min_len = min(reconstructed.shape[1], target.shape[1])
        reconstructed = reconstructed[:, :min_len]
        target = target[:, :min_len]
        mask = target_padding_mask[:, :min_len]

        # MSE loss on non-padded positions
        valid_recon = reconstructed[~mask]
        valid_target = target[~mask]
        loss = F.mse_loss(valid_recon.float(), valid_target.float()).to(reconstructed.dtype)

        # Reuse DiTOutput for compatibility
        

        return DiTOutput(loss=loss)

    def generate(
        self,
        context_vector: torch.FloatTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        # These args are accepted for API compatibility with DiT but ignored
        num_steps: int = 1,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        std: float = 0.0,
    ) -> torch.FloatTensor:
        """
        Generate mel spectrogram from latent. Single forward pass (no ODE).
        """
        return self._decode(context_vector)


# ---------- Test ----------

if __name__ == "__main__":
    print("=" * 60)
    print("ConvformerDecoder shape test")
    print("=" * 60)

    config = ConvformerDecoderConfig(
        compress_factor_C=8,
        tf_heads=8,
        tf_layers=4,
        drop_p=0.1,
        latent_dim=64,
        n_residual_blocks=3,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    decoder = ConvformerDecoder(config).to(device, dtype)
    print(f"Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"Freq schedule (decoder): {decoder.freq_schedule}")

    B, T, mel = 2, 128, 100
    T_latent = T // config.compress_factor_C  # 16

    z = torch.randn(B, T_latent, config.latent_dim, device=device, dtype=dtype)
    target = torch.randn(B, T, mel, device=device, dtype=dtype)
    padding_mask = torch.zeros(B, T, device=device, dtype=torch.bool)

    # Test forward (training)
    print("\n--- Forward (training) ---")
    output = decoder(target=target, target_padding_mask=padding_mask, context_vector=z)
    print(f"Loss: {output.loss.item():.4f}")

    # Test generate
    print("\n--- Generate ---")
    recon = decoder.generate(context_vector=z)
    print(f"Generated mel shape: {recon.shape}")
    assert recon.shape == (B, T, mel), f"Expected {(B, T, mel)}, got {recon.shape}"

    # Test with different T
    print("\n--- Generate (T=256) ---")
    T2 = 256
    z2 = torch.randn(B, T2 // config.compress_factor_C, config.latent_dim, device=device, dtype=dtype)
    recon2 = decoder.generate(context_vector=z2)
    print(f"Generated mel shape: {recon2.shape}")
    assert recon2.shape == (B, T2, mel), f"Expected {(B, T2, mel)}, got {recon2.shape}"

    print("\n✓ All shape tests passed!")
