import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
import torch.nn as nn
from einops import rearrange
from dataclasses import dataclass
from typing import Optional, List


from modules.flash_attn_encoder import FlashTransformerEncoder

from modules.downsampler import DownSampler
#from modules.downsampler_2d import DownSamplerConfig, DownSampler
from modules.resnet import LinearResNet as ResNet
from modules.similarity import Aligner, SimilarityUpsamplerBatch
from modules.alignement import AlignmentMatrixBuilder
from modules.qformer import (
    AlignmentQFormer,   
    DurationConditioningProjector,
    compute_relative_positions,
)


# Inizializzazione
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
    l1_l2_loss: Optional[torch.FloatTensor] = None


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
    d_model: int = 128
    n_residual_blocks: int = 3
    compress_factor_C: int = 1
    latent_dim: int = 64
    drop_p: float = 0.1
    tf_heads: int = 8
    tf_layers: int = 2
    num_embeddings: int = 100
    phoneme_parsing_mode: str = "phoneme"
    vocab_path: str = "data/vocab.json"
    use_aligner: bool = False
    threshold: float = 0.95
    force_downsample: bool = False
    split_route: bool = False
    semantic_dim: Optional[int] = None
    kl_loss_pooler_weight: float = 1e-3
    source: str = "z"


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.semantic_regulator = InterpolateRegulator(
        #     depth=2, in_channels=1024, channels=256, out_channels=config.latent_dim
        # )

    def forward(self, **kwargs):
        pass

    def reparameterize(
        self, mu: torch.FloatTensor, logvar: Optional[torch.FloatTensor] = None
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
            # Compute in fp32 for numerical stability
            mu_valid = mu[~padding_mask].float()
            if mu_valid.numel() == 0:
                return torch.zeros((), dtype=mu.dtype, device=mu.device)
            return torch.nn.functional.mse_loss(
                mu_valid, torch.zeros_like(mu_valid)
            ).to(mu.dtype)
        # Average KL over valid timesteps so its scale does not grow with sequence length.
        valid_mask = ~padding_mask
        num_valid = valid_mask.sum()
        if num_valid.item() == 0:
            return torch.zeros((), dtype=mu.dtype, device=mu.device)
        mu_fp32 = mu.float()
        logvar_fp32 = logvar.float()
        kl_per_dim = -0.5 * (
            1 + logvar_fp32 - mu_fp32.pow(2) - logvar_fp32.exp()
        )
        kl_per_token = kl_per_dim.sum(dim=-1)
        kl = kl_per_token.masked_fill(~valid_mask, 0.0).sum() / num_valid
        return kl.to(mu.dtype)

    def get_latent_source(
        self,
        mu: torch.FloatTensor,
        z: torch.FloatTensor,
    ) -> torch.FloatTensor:
        source_name = getattr(self.config, "source", "z")
        if source_name == "mu":
            return mu
        if source_name == "z":
            return z
        raise ValueError(f"Unsupported latent source: {source_name}")

    def sample_scalar_std(self, mu: torch.FloatTensor) -> torch.FloatTensor:
        return self.std_activation(
            torch.randn(mu.shape[0], mu.shape[1], dtype=mu.dtype, device=mu.device)
            * self.config.target_std
        )

    def _resize_padding_mask(
        self, padding_mask: torch.BoolTensor, target_length: int
    ) -> torch.BoolTensor:
        padding_mask = (
            torch.nn.functional.interpolate(
                padding_mask.unsqueeze(1).float(),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            > 0.5  # Use threshold instead of .bool()
        )
        return padding_mask

    def get_kl_cosine_schedule(self, step, kl_loss_weight):
        """
        Returns the scaled KL loss weight following a cosine schedule
        ranging from 0 to self.kl_loss_weight over total_steps.
        Once step surpasses total_steps, stays at self.kl_loss_weight.
        """
        if self.config.kl_loss_warmup_steps == 0:
            return kl_loss_weight
        if step >= self.config.kl_loss_warmup_steps:
            return kl_loss_weight
        # Cosine schedule: start at 0, increase to kl_loss_weight in total_steps
        cosine = 0.5 * (1 - math.cos(math.pi * step / self.config.kl_loss_warmup_steps))
        return kl_loss_weight * cosine


class CausalTransformerTail(nn.Module):
    def __init__(self, d_model=512, nheads=8, nlayers=4, drop_p=0.1):
        super().__init__()
        self.enc = FlashTransformerEncoder(
            d_model=d_model, nhead=nheads, nlayers=nlayers, drop_p=drop_p
        )

    def forward(self, tokens):  # [B, T_tok, d_model]
        return self.enc(tokens, causal=True)


class ConvformerEncoder(SigmaVAEEncoder):
    def __init__(self, config: ConvformerEncoderConfig):
        super().__init__(config)
        if config.source not in {"mu", "z"}:
            raise ValueError(f"Unsupported encoder source: {config.source}")

        compress_factor_C = config.compress_factor_C
        d_model = config.d_model
        tf_heads = config.tf_heads
        tf_layers = config.tf_layers
        drop_p = config.drop_p
        latent_dim = config.latent_dim
        n_residual_blocks = config.n_residual_blocks

        # downsampler_config = DownSamplerConfig(
        #     d_model=d_model,
        #     separable=False,
        #     n_residual_blocks=n_residual_blocks,
        #     compress_factor=compress_factor_C,
        #     drop_p=drop_p,
        # )

        if config.force_downsample:
            #self.downsampler = DownSampler(downsampler_config)
            self.downsampler = DownSampler(
                d_in = 100,
                d_hidden = int(d_model*4),
                d_out = d_model,
                compress_factor = compress_factor_C,
            )
        else:
            raise ValueError("Deprecated")
            # self.downsampler = ResNet(
            #     in_dim=100,
            #     hidden_dim=512,
            #     output_dim=512,
            #     num_blocks=n_residual_blocks,
            # )

        self.alignment_matrix_builder = AlignmentMatrixBuilder(
            compress_factor=config.compress_factor_C,
            embedding_dim=d_model,
        )
        self.proj_mean = nn.Linear(d_model, latent_dim)

        # Causal Transformer tail operating on tokens of size 512
        self.transformer = CausalTransformerTail(
            d_model=d_model, nheads=tf_heads, nlayers=tf_layers, drop_p=drop_p
        )

        self.mu = nn.Linear(d_model, latent_dim)
        if config.logvar_layer:
            self.logvar = nn.Linear(d_model, latent_dim)

        # self.pooler = AlignmentQFormer(
        #     d_model=d_model,
        #     num_queries_per_phoneme=3,
        #     num_layers=4,
        #     dropout=drop_p,
        #     out_dim=d_model,
        # )

        # TODO create an upsample convolution causal for conditioning_proj
        self.conditioning_proj = DurationConditioningProjector(
            d_in=latent_dim,
        )

        if config.freeze_encoder:
            for name, param in self.named_parameters():
                if name.split(".")[0] in ["pooler"]:
                    continue
                param.requires_grad = False

    def forward(
        self,
        x: torch.FloatTensor,
        phonemes: List[str],
        padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):  # x: [B, T, 100]

        original_padding_mask = padding_mask.clone()
        x, padding_mask = self.downsampler(x, padding_mask.bool())
        target_T = (~padding_mask).sum(dim=1)
        x = self.transformer(x)  # [B, T/C, 512]

        # get alignement
        durations, align_loss, l1_l2_loss = None, None, None
        if self.config.use_aligner:
            align_out = self.alignment_matrix_builder.build(
                phonemes,
                target_T,
                device=x.device,
                dtype=x.dtype,
            )
            alignments = align_out.alignments  # [B, T, N]
            padding_mask = align_out.phoneme_mask  # [B, N]
            durations = align_out.durations  # [B, N]
            segment_labels = align_out.segment_labels
            embeddings = align_out.embeddings
            rel_pos = compute_relative_positions(alignments)

            if x.shape[1] != alignments.shape[1]:
                raise RuntimeError(
                    f"Temporal mismatch: x has T={x.shape[1]}, "
                    f"alignments have T={alignments.shape[1]}"
                )

            align_float = alignments.to(x.dtype)  # [B, T, N]

            # Forward pass
            out = self.pooler(
                mel_features=x,
                alignment=alignments,
                phoneme_embeddings=embeddings,
                phoneme_mask=padding_mask,
            )
            x = out.pooled #torch.bmm(align_float.permute(0,2,1), x) / (durations.unsqueeze(-1) + 1e-8)
            pos = out.rel_pos

        mu = self.mu(x)
        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        assert not torch.isnan(z).any(), "z contains nan after reparameterization"

        kl_loss = None
        if kwargs.get("step", None) is not None:
            kl_loss = self.kl_divergence(
                mu, logvar, padding_mask
            ) * self.get_kl_cosine_schedule(
                kwargs["step"], kl_loss_weight=self.kl_loss_weight
            )

        source = self.get_latent_source(mu, z)

        if self.config.use_aligner:
            z_upsampled = self.conditioning_proj(
                pooled=source, durations=durations, rel_pos=pos
            )
            z_upsampled = torch.repeat_interleave(
                z_upsampled, self.config.compress_factor_C, dim=1
            )
            upsampled_padding_mask = original_padding_mask
        else:
            z_upsampled = torch.repeat_interleave(
                source, self.config.compress_factor_C, dim=1
            )
            upsampled_padding_mask = original_padding_mask

        return ConvformerOutput(
            z=z,
            kl_loss=kl_loss,
            padding_mask=padding_mask,
            mu=mu,
            durations=durations,
            z_upsampled=z_upsampled,
            align_loss=align_loss,
            upsampled_padding_mask=upsampled_padding_mask,
            l1_l2_loss=l1_l2_loss,
        )


@torch.no_grad()
def forward_latent(model, x):
    model.eval()
    return model(x)  # returns mu: [B, T/C, latent_dim]


def test_time_causality_invariance(model):
    torch.manual_seed(0)
    B, T, F = 1, 64, 100

    x = torch.randn(B, T, F)
    y_ref = forward_latent(model, x)  # [B, T/C, latent_dim]

    for j in range(1, y_ref.shape[1]):
        t_max = j * model.config.compress_factor_C
        x_pert = x.clone()
        if t_max < T:
            x_pert[:, t_max:, :] = torch.randn_like(x_pert[:, t_max:, :]) * 5.0
        y_pert = forward_latent(model, x_pert)

        ok = torch.allclose(y_pert[:, j - 1, :], y_ref[:, j - 1, :], atol=1e-7, rtol=0)
        assert ok, f"Causality violated at output index {j}"


if __name__ == "__main__":
    config = ConvformerEncoderConfig(
        compress_factor_C=8,
        tf_heads=8,
        tf_layers=4,
        drop_p=0.1,
        latent_dim=64,
        n_residual_blocks=3,
    )
    model = ConvformerEncoder(config=config)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    test_time_causality_invariance(model)
    breakpoint()
    # print number of parameters
