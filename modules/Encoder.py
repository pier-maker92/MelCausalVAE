import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Optional, List

from modules.flash_attn_encoder import FlashTransformerEncoder
from modules.resnet import LinearResNet as ResNet
from modules.downsampler import DownSampler
from modules.alignment import AlignmentMatrixBuilder


@dataclass
class ConvformerOutput:
    z: torch.FloatTensor
    kl_loss: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None
    mu: Optional[torch.FloatTensor] = None
    align_loss: Optional[torch.FloatTensor] = None
    durations: Optional[torch.LongTensor] = None
    z_upsampled: Optional[torch.FloatTensor] = None
    upsampled_padding_mask: Optional[torch.BoolTensor] = None
    segment_labels: Optional[List[List[str]]] = None


@dataclass
class SigmaVAEencoderConfig:
    logvar_layer: bool = True
    kl_loss_weight: float = 1e-3
    target_std: Optional[float] = None
    use_sofplus: Optional[bool] = None
    kl_loss_warmup_steps: int = 1000
    semantic_regulation: bool = True
    freeze_encoder: bool = False
    semantic_kl_loss_weight: Optional[float] = None


@dataclass
class ConvformerEncoderConfig(SigmaVAEencoderConfig):
    d_model: int = 512
    compress_factor_C: int = 8
    tf_heads: int = 8
    tf_layers: int = 4
    drop_p: float = 0.1
    latent_dim: int = 64
    n_residual_blocks: int = 3
    num_embeddings: int = 100
    embedding_dim: int = 80
    phoneme_parsing_mode: str = "phoneme"
    vocab_path: str = "data/vocab.json"
    use_aligner: bool = False
    threshold: float = 0.95
    force_downsample: bool = False
    split_route: bool = False
    semantic_dim: Optional[int] = None
    use_segment_embedding_residual: bool = False
    segment_embedding_scale: float = 1.0


class SigmaVAEEncoder(nn.Module):
    def __init__(self, config: SigmaVAEencoderConfig):
        super().__init__()
        self.config = config
        self.std_activation = (
            nn.Softplus() if self.config.use_sofplus else nn.Identity()
        )
        self.kl_loss_weight = float(config.kl_loss_weight)
        if self.config.semantic_kl_loss_weight is not None:
            self.semantic_kl_loss_weight = float(config.semantic_kl_loss_weight)

    def forward(self, **kwargs):
        raise NotImplementedError

    def reparameterize(
        self,
        mu: torch.FloatTensor,
        logvar: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        eps = torch.randn_like(mu)
        if logvar is None:
            std = self.sample_scalar_std(mu)
            while std.dim() < mu.dim():
                std = std.unsqueeze(-1)
        else:
            std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def kl_divergence(
        self,
        mu: torch.FloatTensor,
        logvar: Optional[torch.FloatTensor],
        padding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        if logvar is None:
            mu_valid = mu[~padding_mask].float()
            return F.mse_loss(mu_valid, torch.zeros_like(mu_valid)).to(mu.dtype)

        mu_valid = mu[~padding_mask].float()
        logvar_valid = logvar[~padding_mask].float()
        kl = -0.5 * torch.sum(
            1 + logvar_valid - mu_valid.pow(2) - logvar_valid.exp()
        )
        return kl.to(mu.dtype)

    def sample_scalar_std(self, mu: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.target_std is None:
            return torch.ones(
                mu.shape[0], mu.shape[1], dtype=mu.dtype, device=mu.device
            )
        return self.std_activation(
            torch.randn(mu.shape[0], mu.shape[1], dtype=mu.dtype, device=mu.device)
            * self.config.target_std
        )

    def _resize_padding_mask(
        self,
        padding_mask: torch.BoolTensor,
        target_length: int,
    ) -> torch.BoolTensor:
        padding_mask = (
            F.interpolate(
                padding_mask.unsqueeze(1).float(),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            > 0.5
        )
        return padding_mask

    def get_kl_cosine_schedule(self, step: int, kl_loss_weight: float) -> float:
        if self.config.kl_loss_warmup_steps == 0:
            return kl_loss_weight
        if step >= self.config.kl_loss_warmup_steps:
            return kl_loss_weight
        cosine = 0.5 * (
            1 - math.cos(math.pi * step / self.config.kl_loss_warmup_steps)
        )
        return kl_loss_weight * cosine


class CausalTransformerTail(nn.Module):
    def __init__(self, d_model=512, nheads=8, nlayers=4, drop_p=0.1):
        super().__init__()
        self.enc = FlashTransformerEncoder(
            d_model=d_model,
            nhead=nheads,
            nlayers=nlayers,
            drop_p=drop_p,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        return self.enc(tokens, causal=True)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ConvformerEncoder(SigmaVAEEncoder):
    _ALIGN_EMBED_DIM = 128

    def __init__(self, config: ConvformerEncoderConfig):
        super().__init__(config)

        compress_factor_C = config.compress_factor_C
        tf_heads = config.tf_heads
        tf_layers = config.tf_layers
        drop_p = config.drop_p
        latent_dim = config.latent_dim
        n_residual_blocks = config.n_residual_blocks
        d_model = config.d_model
        embed_dim = self._ALIGN_EMBED_DIM

        if config.force_downsample:
            self.downsampler = DownSampler(
                d_in=100,
                d_hidden=d_model,
                d_out=d_model,
                compress_factor=compress_factor_C,
                causal=True,
                n_residual_blocks=n_residual_blocks,
            )
        else:
            self.downsampler = ResNet(
                in_dim=100,
                hidden_dim=d_model,
                output_dim=d_model,
                num_blocks=n_residual_blocks,
            )

        self.alignment_matrix_builder = AlignmentMatrixBuilder(
            embedding_dim=embed_dim,
        )

        self.use_segment_embedding_residual = config.use_segment_embedding_residual
        if self.use_segment_embedding_residual:
            self.seg_emb_proj = nn.Linear(embed_dim, latent_dim)
            self.segment_embedding_scale = float(config.segment_embedding_scale)

        self.transformer = CausalTransformerTail(
            d_model=d_model,
            nheads=tf_heads,
            nlayers=tf_layers,
            drop_p=drop_p,
        )

        self.mu = nn.Linear(d_model, latent_dim)
        if config.logvar_layer:
            self.logvar = nn.Linear(d_model, latent_dim)

        self.upsample_refine = nn.Sequential(
            CausalConv1d(latent_dim + 1, latent_dim * 2, kernel_size=5),
            nn.GELU(),
            CausalConv1d(latent_dim * 2, latent_dim, kernel_size=5),
        )

        self.frame_target_proj = nn.Linear(d_model, latent_dim)

        if config.freeze_encoder:
            for name, param in self.named_parameters():
                if name.split(".")[0] in ["mu", "logvar"]:
                    continue
                param.requires_grad = False

    @staticmethod
    def _compute_phase(
        alignments: torch.Tensor,
        durations: torch.Tensor,
    ) -> torch.Tensor:
        """
        alignments: [B, T, N]
        durations:  [B, N]
        returns:    [B, T, 1]
        """
        cumpos = torch.cumsum(alignments, dim=1).to(alignments.dtype)
        durations = durations.to(alignments.dtype)
        phase_per_seg = (cumpos - 1.0) / (durations.unsqueeze(1) + 1e-8)
        phase = (phase_per_seg * alignments).sum(dim=2, keepdim=True)
        return phase

    def forward(
        self,
        x: torch.FloatTensor,
        phonemes: List[str],
        padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> ConvformerOutput:
        if padding_mask is None:
            padding_mask = torch.zeros(
                x.size(0), x.size(1), dtype=torch.bool, device=x.device
            )
        original_padding_mask = padding_mask.clone()
        x, frame_padding_mask = self.downsampler(x, padding_mask.bool())
        x = self.transformer(x)
        x_pre_pool = x

        target_T = (~frame_padding_mask).sum(dim=1)

        align_out = self.alignment_matrix_builder.build(
            phonemes,
            target_T,
            device=x.device,
            dtype=x.dtype,
        )
        if self.config.use_aligner:

            alignments = align_out.alignments              # [B, T, N]
            segment_padding_mask = align_out.phoneme_mask # [B, N]
            durations = align_out.durations               # [B, N]
            segment_labels = align_out.segment_labels
            embeddings = align_out.embeddings

            if x.shape[1] != alignments.shape[1]:
                raise RuntimeError(
                    f"Temporal mismatch: x has T={x.shape[1]}, "
                    f"alignments have T={alignments.shape[1]}"
                )

            align_float = alignments.to(x.dtype)                               # [B, T, N]
            dur = align_float.sum(dim=1).unsqueeze(-1)                         # [B, N, 1]
            x = torch.bmm(align_float.transpose(1, 2), x) / (dur + 1e-8)  # [B, N, D]
        else:
            durations = None
            segment_padding_mask = padding_mask
            segment_labels = None
            embeddings = None

        if self.config.split_route:
            raise NotImplementedError("split_route not implemented yet")

        mu = self.mu(x)
        logvar = self.logvar(x) if hasattr(self, "logvar") else None
        z = self.reparameterize(mu, logvar)
        if self.config.use_segment_embedding_residual and self.config.use_aligner:
            z = z + self.segment_embedding_scale * self.seg_emb_proj(embeddings)

        if torch.isnan(z).any():
            raise RuntimeError("z contains NaNs after reparameterization")

        if self.config.use_aligner:
            z_upsampled = torch.bmm(align_float, z)                            # [B, T, latent_dim]
            phase = self._compute_phase(align_float, durations)                # [B, T, 1]
            z_cat = torch.cat([z_upsampled, phase], dim=-1)                   # [B, T, latent_dim+1]

            z_upsampled = self.upsample_refine(
                z_cat.transpose(1, 2)
            ).transpose(1, 2)

            upsampled_padding_mask = frame_padding_mask
        else:
            z_upsampled = z
            upsampled_padding_mask = padding_mask

        if torch.isnan(z_upsampled).any():
            raise RuntimeError("z_upsampled contains NaNs after upsample refinement")

        kl_loss = None
        align_loss = None

        if kwargs.get("step", None) is not None:
            kl_weight = self.get_kl_cosine_schedule(
                kwargs["step"],
                kl_loss_weight=self.kl_loss_weight,
            )
            kl_loss = self.kl_divergence(
                mu,
                logvar,
                segment_padding_mask,
            ) * kl_weight

            # Auxiliary frame consistency loss disabled for now.
            # Uncomment only after base reconstruction is stable.
            #
            # x_pre_latent = self.frame_target_proj(x_pre_pool)
            # x_valid = x_pre_latent[~frame_padding_mask].float()
            # z_up_valid = z_upsampled[~frame_padding_mask].float()
            # if x_valid.numel() > 0:
            #     align_loss = F.mse_loss(z_up_valid, x_valid).to(x.dtype)
            # else:
            #     align_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return ConvformerOutput(
            z=z,
            kl_loss=kl_loss,
            padding_mask=segment_padding_mask,
            mu=mu,
            align_loss=align_loss,
            durations=durations,
            z_upsampled=z_upsampled,
            upsampled_padding_mask=upsampled_padding_mask,
            segment_labels=segment_labels,
        )


@torch.no_grad()
def forward_latent(
    model: ConvformerEncoder,
    x: torch.Tensor,
    phonemes: List[str],
    padding_mask: Optional[torch.BoolTensor] = None,
):
    model.eval()
    out = model(x=x, phonemes=phonemes, padding_mask=padding_mask)
    return out.mu


def smoke_test():
    torch.manual_seed(0)

    config = ConvformerEncoderConfig(
        compress_factor_C=8,
        tf_heads=8,
        tf_layers=4,
        drop_p=0.1,
        latent_dim=64,
        n_residual_blocks=3,
        use_segment_embedding_residual=False,
    )
    model = ConvformerEncoder(config=config)

    B, T, F = 2, 64, 100
    x = torch.randn(B, T, F)
    padding_mask = torch.zeros(B, T, dtype=torch.bool)
    phonemes = ["ciao", "come stai"]

    out = model(
        x=x,
        phonemes=phonemes,
        padding_mask=padding_mask,
        step=100,
    )

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print("z:", out.z.shape)
    print("mu:", out.mu.shape if out.mu is not None else None)
    print("z_upsampled:", out.z_upsampled.shape if out.z_upsampled is not None else None)
    print("kl_loss:", out.kl_loss.item() if out.kl_loss is not None else None)


if __name__ == "__main__":
    smoke_test()