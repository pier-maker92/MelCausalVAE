import math
import torch
import random
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from .vq import HardVectorQuantizer, FiniteScalarQuantizer
from ..configs import EncoderConfig, VQConfig
from ..output_dataclasses import EncoderOutput
from .sigmavae import SigmaVAEEncoder
from .regularization import (
    DropoutRegularizer,
    KLChunkRegularizer,
)
from .utils import (
    TimeCausalConv1d,
    PreNormResCausalBlock1d,
    CausalDownsamplingBlock1d,
    Transformer,
)


class Encoder(SigmaVAEEncoder):
    """
    1D convolutional encoder: treats 100 mel bins as input channels and uses
    only temporal (causal) Conv1d operations. Drop-in replacement for the 2D
    Encoder with the same EncoderConfig.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__(config)

        compress_factor_C = config.compress_factor_C
        tf_heads = config.tf_heads
        tf_layers = config.tf_layers
        drop_p = config.drop_p
        latent_dim = config.latent_dim
        n_residual_blocks = config.n_residual_blocks
        d_model = config.d_model
        mel_dim = config.mel_dim

        assert (
            compress_factor_C >= 1
            and (compress_factor_C & (compress_factor_C - 1)) == 0
        ), "C must be power of 2"
        self.C = compress_factor_C

        # Input projection: [B, mel_dim, T] -> [B, d_model // 2, T]
        self.in_proj = TimeCausalConv1d(mel_dim, d_model // 2, k=7)

        # Mixer: dilated causal blocks with increasing channels
        self.mixer = nn.Sequential(
            PreNormResCausalBlock1d(d_model // 2, d_model, k=7, d=1, drop_p=drop_p),
            PreNormResCausalBlock1d(d_model, d_model, k=5, d=2, drop_p=drop_p),
            PreNormResCausalBlock1d(d_model, d_model, k=3, d=4, drop_p=drop_p),
        )

        # Temporal downsampling: log2(C) stages of stride-2
        num_stages = int(math.log2(compress_factor_C))
        self.downsampling = nn.ModuleDict()
        for i in range(num_stages):
            factor = 2 ** (i + 1)
            self.downsampling[f"downsample@{factor}"] = CausalDownsamplingBlock1d(
                d_model, d_model, n_residual_blocks=n_residual_blocks, drop_p=drop_p
            )

        # Causal Transformer
        self.transformer = Transformer(
            d_model=d_model, nheads=tf_heads, nlayers=tf_layers, drop_p=drop_p
        )

        semantic_dim = 0
        if config.vq_config:
            semantic_dim = config.vq_config.dim_to_quantize
            self.mu_semantic = nn.Linear(d_model, semantic_dim)
            if getattr(config.vq_config, "fsq_levels", None) is not None:
                self.vq = FiniteScalarQuantizer(config.vq_config)
            else:
                self.vq = HardVectorQuantizer(config.vq_config)

        acoustic_dim = getattr(config, "acoustic_dim", 0) or 0
        if acoustic_dim > 0:
            self.mu_acoustic = nn.Linear(d_model, acoustic_dim)
            self.logvar_acoustic = nn.Linear(d_model, acoustic_dim)

        if semantic_dim + acoustic_dim > 0:
            self.out_proj = nn.Linear(semantic_dim + acoustic_dim, latent_dim)

        if config.dropout_regularizer_config:
            self.dropout_regularizer = DropoutRegularizer(
                config=config.dropout_regularizer_config
            )
            self.use_pre_quant_dropout = (
                config.dropout_regularizer_config.pre_quantization
            )
        else:
            self.use_pre_quant_dropout = False

        if config.kl_chunk_regularizer_config:
            self.kl_chunk_regularizer = KLChunkRegularizer(
                config=config.kl_chunk_regularizer_config,
                vq_quant_dim=None,
            )

        if getattr(config, "speaker_cond_dim", None) is not None:
            self.speaker_proj = nn.Linear(2 * d_model, config.speaker_cond_dim)

        if config.freeze_encoder_before_latent_heads:
            self._freeze_encoder_before_latent_heads()

        self.config = config

    def slt(self, x: torch.FloatTensor):
        """The sign-log transform
        f(x) = sign(x) ln(|x| + 1)
        """
        return x.sign() * (x.abs() + 1).log()

    def _freeze_encoder_before_latent_heads(self):
        for param in self.parameters():
            param.requires_grad = False
        if hasattr(self, "mu_semantic"):
            for param in self.mu_semantic.parameters():
                param.requires_grad = True
        if hasattr(self, "mu_acoustic"):
            for param in self.mu_acoustic.parameters():
                param.requires_grad = True
        if hasattr(self, "logvar_acoustic"):
            for param in self.logvar_acoustic.parameters():
                param.requires_grad = True
        if hasattr(self, "out_proj"):
            for param in self.out_proj.parameters():
                param.requires_grad = True
        if hasattr(self, "vq"):
            for param in self.vq.parameters():
                param.requires_grad = True
        if hasattr(self, "speaker_proj"):
            for param in self.speaker_proj.parameters():
                param.requires_grad = True

    def forward(
        self,
        x: torch.FloatTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        # x: [B, T, 100]
        x = x.transpose(1, 2)  # [B, 100, T]
        x = self.in_proj(x)  # [B, d_model//2, T]
        x = self.mixer(x)  # [B, d_model, T]

        for layer in self.downsampling.values():
            x = layer(x)  # [B, d_model, T/C]
        padding_mask = (
            self._resize_padding_mask(padding_mask, x.shape[2], dtype=x.dtype)
            if padding_mask is not None
            else torch.zeros(
                (x.shape[0], x.shape[2]), device=x.device, dtype=torch.bool
            )
        )

        hiddens = x.transpose(1, 2)  # [B, T/C, 512]
        h = self.transformer(hiddens)  # [B, T/C, 512]

        speaker_embedding = None
        if getattr(self.config, "use_instance_norm", False):
            valid_mask = (~padding_mask).unsqueeze(-1).to(h.dtype)  # [B, T, 1]
            valid_count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

            spk_mu = (h * valid_mask).sum(dim=1, keepdim=True) / valid_count
            spk_sigma = torch.sqrt(
                ((h - spk_mu) ** 2 * valid_mask).sum(dim=1, keepdim=True) / valid_count
                + 1e-6
            )

            h = (h - spk_mu) / (spk_sigma + 1e-6)
            h = h * valid_mask  # Re-apply mask
            speaker_embedding = torch.cat(
                [spk_mu.squeeze(1), spk_sigma.squeeze(1)], dim=-1
            )

            if hasattr(self, "speaker_proj"):
                speaker_embedding = self.speaker_proj(speaker_embedding)

        semantic_features = None
        mu_semantic = None
        vq_out = None
        vq_quantized_ste = None
        if hasattr(self, "mu_semantic"):
            mu_semantic = self.mu_semantic(h)
            if getattr(self.config, "use_slt", False):
                mu_semantic = self.slt(mu_semantic)

            vq_out = self.vq(
                mu_semantic,
                padding_mask,
                global_step=kwargs.get("step", None),
            )
            # No reparameterization for semantic, just STE
            vq_quantized_ste = mu_semantic + (vq_out.quantized - mu_semantic).detach()
            semantic_features = vq_quantized_ste

        acoustic_features = None
        mu_acoustic = None
        logvar_acoustic = None
        if hasattr(self, "mu_acoustic"):
            mu_acoustic = self.mu_acoustic(h)
            logvar_acoustic = self.logvar_acoustic(h)
            if getattr(self.config, "use_slt", False):
                mu_acoustic = self.slt(mu_acoustic)

            z_acoustic = self.reparameterize(mu_acoustic, logvar_acoustic)

            if hasattr(self, "dropout_regularizer"):
                z_acoustic = self.dropout_regularizer(z_acoustic)

            acoustic_features = z_acoustic

        features_to_concat = []
        if semantic_features is not None:
            features_to_concat.append(semantic_features)
        if acoustic_features is not None:
            features_to_concat.append(acoustic_features)

        if len(features_to_concat) > 1:
            combined_features = torch.cat(features_to_concat, dim=-1)
        elif len(features_to_concat) == 1:
            combined_features = features_to_concat[0]
        else:
            raise ValueError("No semantic or acoustic features computed.")

        z = self.out_proj(combined_features)

        ortho_loss = None
        if mu_semantic is not None and mu_acoustic is not None:
            ortho_weight = 1.0
            if getattr(self.config, "semantic_distillation_config", None) is not None:
                ortho_weight = (
                    self.config.semantic_distillation_config.ortho_loss_weight
                )

            if ortho_weight > 0.0:
                mask = ~padding_mask
                h1 = mu_semantic[mask]
                h2 = mu_acoustic[mask]

                h1_centered = h1 - h1.mean(dim=0, keepdim=True)
                h2_centered = h2 - h2.mean(dim=0, keepdim=True)

                h1_norm = h1_centered / (h1_centered.std(dim=0, keepdim=True) + 1e-8)
                h2_norm = h2_centered / (h2_centered.std(dim=0, keepdim=True) + 1e-8)

                N = h1_centered.size(0)
                cross_corr = torch.matmul(h1_norm.T, h2_norm) / (N - 1 + 1e-8)
                ortho_loss = (cross_corr**2).mean()

        kl_loss = None
        if kwargs.get("step", None) is not None and hasattr(self, "mu_acoustic"):
            if hasattr(self, "kl_chunk_regularizer"):
                kl_term = self.kl_chunk_regularizer(
                    mu_acoustic, logvar_acoustic, padding_mask
                )
            else:
                kl_term = self.kl_divergence(
                    mu_acoustic,
                    logvar_acoustic,
                    padding_mask,
                    dtype=z.dtype,
                )
            kl_loss = kl_term * self.get_kl_cosine_schedule(kwargs["step"])

        out = {
            "z": z,
            "kl_loss": kl_loss,
            "padding_mask": padding_mask,
            "mu": mu_acoustic,
            "mu_pre_vq": mu_semantic,
            "ortho_loss": ortho_loss,
            "speaker_embedding": speaker_embedding,
        }

        if vq_out is not None:
            out["vq_stats"] = vq_out.stats
            out["vq_loss"] = vq_out.loss
            out["quantized"] = vq_quantized_ste
            out["residual"] = None
            out["tail"] = acoustic_features
            out["indices"] = vq_out.indices

        return EncoderOutput(**out)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
