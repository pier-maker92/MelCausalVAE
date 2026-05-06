import math
import torch
import random
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from modules.encoder.vq import HardVectorQuantizer
from modules.configs import EncoderConfig, VQConfig
from modules.output_dataclasses import EncoderOutput
from modules.encoder.sigmavae import SigmaVAEEncoder
from modules.encoder.regularization import (
    DropoutRegularizer,
    KLChunkRegularizer,
)
from modules.encoder.utils import (
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
            self.vq = HardVectorQuantizer(config.vq_config)
            self.residual_and_tail_dropout_p = (
                config.vq_config.residual_and_tail_dropout_p
            )

        if config.dropout_regularizer_config:
            self.dropout_regularizer = DropoutRegularizer(
                config=config.dropout_regularizer_config
            )

        if config.kl_chunk_regularizer_config:
            self.kl_chunk_regularizer = KLChunkRegularizer(
                config=config.kl_chunk_regularizer_config,
                vq_quant_dim=(
                    config.vq_config.dim_to_quantize if config.vq_config else None
                ),
            )

        if config.freeze_encoder_before_latent_heads:
            self._freeze_encoder_before_latent_heads()

        self.config = config

    def _freeze_encoder_before_latent_heads(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mu.parameters():
            param.requires_grad = True
        if hasattr(self, "logvar"):
            for param in self.logvar.parameters():
                param.requires_grad = True
        if hasattr(self, "vq"):
            for param in self.vq.parameters():
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

        mu_pre_vq = self.mu(h)

        if hasattr(self, "dropout_regularizer"):
            mu_pre_vq = self.dropout_regularizer(mu_pre_vq)

        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(h)

        mu_stoch = mu_pre_vq
        z_quantized = torch.zeros_like(mu_pre_vq)
        if hasattr(self, "vq"):
            qd = self.config.vq_config.dim_to_quantize
            mu_head = mu_pre_vq[..., :qd]
            mu_tail = mu_pre_vq[..., qd:]

            vq_out = self.vq(
                mu_head,
                padding_mask,
                global_step=kwargs.get("step", None),
            )

            mu_stoch = torch.cat([torch.zeros_like(mu_head), mu_tail], dim=-1)

            vq_quantized_ste = mu_head + (vq_out.quantized - mu_head).detach()
            z_quantized = torch.cat(
                [vq_quantized_ste, torch.zeros_like(mu_tail)], dim=-1
            )

            logvar_tail = logvar[..., qd:] if logvar is not None else None
            z_stoch_tail = self.reparameterize(mu_tail, logvar_tail)
            z_stoch = torch.cat([torch.zeros_like(mu_head), z_stoch_tail], dim=-1)
        else:
            z_stoch = self.reparameterize(mu_stoch, logvar)

        # dropout per sample
        if self.training and getattr(self, "residual_and_tail_dropout_p", 0.0) > 0.0:
            B = z_stoch.shape[0]
            keep_mask = (torch.rand(B, 1, 1, device=z_stoch.device) >= self.residual_and_tail_dropout_p).to(
                z_stoch.dtype
            )
            z = z_quantized + z_stoch * keep_mask
        else:
            z = z_quantized + z_stoch

        mu = mu_stoch
        kl_loss = None
        if kwargs.get("step", None) is not None:
            if hasattr(self, "kl_chunk_regularizer"):
                kl_term = self.kl_chunk_regularizer(mu, logvar, padding_mask)
            else:
                if hasattr(self, "vq"):
                    qd = self.config.vq_config.dim_to_quantize
                    mu_tail = mu[..., qd:]
                    logvar_tail = logvar[..., qd:] if logvar is not None else None
                    kl_term = self.kl_divergence(
                        mu_tail,
                        logvar_tail,
                        padding_mask,
                        dtype=z.dtype,
                    )
                else:
                    kl_term = self.kl_divergence(
                        mu,
                        logvar,
                        padding_mask,
                        dtype=z.dtype,
                    )
            kl_loss = kl_term * self.get_kl_cosine_schedule(kwargs["step"])

        out = {
            "z": z,
            "kl_loss": kl_loss,
            "padding_mask": padding_mask,
            "mu": mu,
            "mu_pre_vq": mu_pre_vq,
        }

        if hasattr(self, "vq"):
            out["vq_stats"] = vq_out.stats
            out["vq_loss"] = vq_out.loss
            out["quantized"] = z_quantized
            out["residual"] = torch.zeros_like(z_quantized)
            out["tail"] = torch.cat([torch.zeros_like(mu_head), mu_tail], dim=-1)

        return EncoderOutput(**out)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
