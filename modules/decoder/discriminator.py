"""
Multi-scale mel-spectrogram discriminator for adversarial training of ConvDecoder.

Architecture: 3 discriminators at temporal scales 1×, 2×, 4× (AvgPool along time).
Each discriminator uses spectral-norm Conv1d layers (SOTA for audio GANs).
Losses: hinge GAN + feature matching (HiFi-GAN / EnCodec style).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def _sn(module: nn.Module) -> nn.Module:
    return nn.utils.spectral_norm(module)


class MelDiscriminator(nn.Module):
    """Single-scale 1-D discriminator operating on [B, T, mel_dim]."""

    CHANNELS = (128, 256, 512, 512, 512)

    def __init__(self, mel_dim: int = 100):
        super().__init__()
        C = self.CHANNELS
        self.pre_conv = _sn(nn.Conv1d(mel_dim, C[0], kernel_size=7, padding=3))
        self.convs = nn.ModuleList([
            _sn(nn.Conv1d(C[i], C[i + 1], kernel_size=5, stride=2, padding=2))
            for i in range(len(C) - 1)
        ])
        self.post_conv = _sn(nn.Conv1d(C[-1], 1, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: [B, T, mel_dim]
        x = x.transpose(1, 2)  # [B, mel_dim, T]
        features: List[torch.Tensor] = []

        x = F.leaky_relu(self.pre_conv(x), 0.1)
        features.append(x)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)

        logits = self.post_conv(x).squeeze(1)  # [B, T']
        features.append(logits.unsqueeze(1))
        return logits, features


class MultiScaleMelDiscriminator(nn.Module):
    """
    Three MelDiscriminators at temporal scales 1×, 2×, 4×.
    Average-pooling along the time axis creates the coarser views.
    """

    def __init__(self, mel_dim: int = 100):
        super().__init__()
        self.discriminators = nn.ModuleList([MelDiscriminator(mel_dim) for _ in range(3)])
        self.pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.AvgPool1d(kernel_size=4, stride=4),
        ])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        x: [B, T, mel_dim]
        Returns:
            all_logits:   list of 3 tensors [B, T_s]
            all_features: list of 3 lists of feature tensors
        """
        all_logits, all_features = [], []
        for pool, disc in zip(self.pools, self.discriminators):
            # pool operates on [B, mel_dim, T] → transpose in/out
            x_pooled = pool(x.transpose(1, 2)).transpose(1, 2)
            logits, feats = disc(x_pooled)
            all_logits.append(logits)
            all_features.append(feats)
        return all_logits, all_features


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def discriminator_hinge_loss(
    real_logits: List[torch.Tensor],
    fake_logits: List[torch.Tensor],
) -> torch.Tensor:
    """Hinge loss for the discriminator, averaged across scales."""
    loss = torch.tensor(0.0, device=real_logits[0].device)
    for r, f in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
    return loss / len(real_logits)


def generator_hinge_loss(fake_logits: List[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for the generator, averaged across scales."""
    loss = torch.tensor(0.0, device=fake_logits[0].device)
    for f in fake_logits:
        loss = loss + (-f).mean()
    return loss / len(fake_logits)


def feature_matching_loss(
    real_features: List[List[torch.Tensor]],
    fake_features: List[List[torch.Tensor]],
) -> torch.Tensor:
    """L1 feature-matching loss across all discriminator layers and scales."""
    loss = torch.tensor(0.0, device=real_features[0][0].device)
    n = 0
    for r_feats, f_feats in zip(real_features, fake_features):
        for r, f in zip(r_feats, f_feats):
            T = min(r.shape[-1], f.shape[-1])
            loss = loss + F.l1_loss(f[..., :T], r[..., :T].detach())
            n += 1
    return loss / max(n, 1)
