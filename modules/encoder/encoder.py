import math
import torch
import random
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from .vq import VectorQuantizer
from ..configs import EncoderConfig, VQConfig
from ..output_dataclasses import EncoderOutput
from .sigmavae import SigmaVAEEncoder
from .regularization import (
    DropoutRegularizer,
    KLChunkRegularizer,
    NoiseRegularizer,
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

        self.mu = nn.Linear(d_model, latent_dim)
        if config.logvar_layer:
            self.logvar = nn.Linear(d_model, latent_dim)

        if config.vq_config:
            if config.vq_config.dim_to_quantize > config.latent_dim:
                raise ValueError(
                    f"dim_to_quantize ({config.vq_config.dim_to_quantize}) must be <= latent_dim ({config.latent_dim})."
                )
            self.vq = VectorQuantizer(config.vq_config)

        if config.dropout_regularizer_config:
            self.dropout_regularizer = DropoutRegularizer(
                config=config.dropout_regularizer_config
            )

        if config.kl_chunk_regularizer_config:
            self.kl_chunk_regularizer = KLChunkRegularizer(
                config=config.kl_chunk_regularizer_config,
                vq_quant_dim=None,
            )

        if config.noise_regularizer_config:
            if getattr(config, "use_reparameterization_trick", False):
                raise ValueError(
                    "Cannot use both noise_regularizer and use_reparameterization_trick=True"
                )
            self.noise_regularizer = NoiseRegularizer(
                config=config.noise_regularizer_config,
            )

        self.semantic_downsample_factor = getattr(
            config, "semantic_downsample_factor", 1
        )
        if self.semantic_downsample_factor > 1:
            if not hasattr(self, "vq"):
                raise ValueError(
                    "semantic_downsample_factor > 1 is only supported when VQ is enabled"
                )
            self.semantic_downsampler = TimeCausalConv1d(
                self._qd,
                self._qd,
                k=self.semantic_downsample_factor * 2,
                d=1,
                s=self.semantic_downsample_factor,
            )
            if self.add_vq_residual_to_stoch and hasattr(self, "logvar"):
                self.logvar_downsampler = TimeCausalConv1d(
                    self._qd,
                    self._qd,
                    k=self.semantic_downsample_factor * 2,
                    d=1,
                    s=self.semantic_downsample_factor,
                )

        if config.freeze_encoder_before_latent_heads:
            self._freeze_encoder_before_latent_heads()

        self.config = config
        self.use_reparameterization_trick = getattr(
            config, "use_reparameterization_trick"
        )

    def _freeze_encoder_before_latent_heads(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mu.parameters():
            param.requires_grad = False
        if hasattr(self, "logvar"):
            for param in self.logvar.parameters():
                param.requires_grad = False
        if hasattr(self, "vq"):
            for param in self.vq.parameters():
                param.requires_grad = True

    def _apply_instance_norm(self, mu, padding_mask):
        valid_mask = ~padding_mask

        valid_lens = valid_mask.sum(dim=1, keepdim=True).float()
        valid_lens = valid_lens.clamp(min=1.0)
        valid_lens = valid_lens.unsqueeze(-1)
        valid_mask_expanded = valid_mask.unsqueeze(-1).to(mu.dtype)

        spk_mu = (mu * valid_mask_expanded).sum(dim=1, keepdim=True) / valid_lens
        spk_variance = (((mu - spk_mu) ** 2) * valid_mask_expanded).sum(
            dim=1, keepdim=True
        ) / valid_lens
        spk_sigma = torch.sqrt(spk_variance + 1e-6)

        mu = (mu - spk_mu) / (spk_sigma + 1e-6)
        mu = mu * valid_mask_expanded

        speaker_embedding = torch.cat([spk_mu.squeeze(1), spk_sigma.squeeze(1)], dim=-1)
        return mu, speaker_embedding

    def _calculate_ortho_loss(self, mu_head, mu_tail, padding_mask):
        if mu_tail.shape[-1] == 0:
            return None

        ortho_weight = 0.0
        if getattr(self.config, "semantic_distillation_config", None) is not None:
            ortho_weight = self.config.semantic_distillation_config.ortho_loss_weight

        if ortho_weight <= 0.0:
            return None

        mask = ~padding_mask
        h1 = mu_head[mask]
        h2 = mu_tail[mask]

        beta = (
            getattr(self.config.semantic_distillation_config, "ortho_beta", 0.01)
            if getattr(self.config, "semantic_distillation_config", None) is not None
            else 0.01
        )

        cos_sim = F.cosine_similarity(h1, h2, dim=-1)
        abs_cos_sim = torch.abs(cos_sim)
        mean_abs_cos_sim = abs_cos_sim.mean()

        return (mean_abs_cos_sim - beta) ** 2

    def _quantize_and_sample(self, mu, logvar, padding_mask, step=None):
       pass

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

        mu = self.mu(h)
        speaker_embedding = None
        if getattr(self.config, "use_instance_norm", False):
            mu, speaker_embedding = self._apply_instance_norm(mu, padding_mask)
        
        if hasattr(self, "dropout_regularizer"):
            mu = self.dropout_regularizer(mu)
        if hasattr(self, "noise_regularizer"):
            mu = self.noise_regularizer(mu)

        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(h)

        vq_output = None
        if hasattr(self, "vq"):
            drop_acoustic = random.random() < self.config.vq_config.drop_acoustic_p
            vq_output = self.vq(mu, padding_mask, drop_acoustic=drop_acoustic)
            mu = vq_output.quantized
            if self.config.vq_config.add_residual and not drop_acoustic: # NOTE : Should be added the residual when drop_acoustic is True?
                mu = mu + vq_output.residual

        kl_weight = (
            self.get_kl_cosine_schedule(kwargs["step"])
            if kwargs.get("step", None) is not None
            else 0.0
        )
        if self.use_reparameterization_trick and self.training:
            z = self.reparameterize(mu, logvar, std=1.0)
        else:
            z = mu
        # KL loss computation
        kl_loss = None
        if self.training:
            kl_term = self.kl_divergence(
                mu,
                logvar,
                padding_mask,
                dtype=mu.dtype,
            )
            kl_loss = kl_term * kl_weight

        out = {
            "z": z,
            "kl_loss": kl_loss,
            "mu": mu,
            "padding_mask": padding_mask,
            "speaker_embedding": speaker_embedding,
        }

        if hasattr(self, "vq"):
            out["vq_stats"] = vq_output.stats
            out["vq_loss"] = vq_output.loss
            out["quantized"] = vq_output.quantized
            out["residual"] = vq_output.residual
            out["indices"] = vq_output.indices

        return EncoderOutput(**out)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
