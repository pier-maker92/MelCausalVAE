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
            self.add_vq_residual_to_stoch = config.vq_config.add_vq_residual_to_stoch

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

        mu_original = self.mu(h)
        if getattr(self.config, "use_slt", False):
            mu_original = self.slt(mu_original)

        if self.use_pre_quant_dropout:
            mu = self.dropout_regularizer(mu_original)
        else:
            mu = mu_original

        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(h)

        z_quantized = torch.zeros_like(mu)
        # vq
        if hasattr(self, "vq"):
            qd = self.config.vq_config.dim_to_quantize
            mu_head = mu[..., :qd]
            mu_tail = mu[..., qd:]

            vq_out = self.vq(
                mu_head,
                padding_mask,
                global_step=kwargs.get("step", None),
            )
            # get z_quantized (straight-through estimator)
            vq_quantized_ste = mu_head + (vq_out.quantized - mu_head).detach()
            z_quantized = torch.cat(
                [vq_quantized_ste, torch.zeros_like(mu_tail)], dim=-1
            )

            logvar_head = logvar[..., :qd] if logvar is not None else None
            logvar_tail = logvar[..., qd:] if logvar is not None else None

            # 1. Define stochastic parts and their distributions
            if self.add_vq_residual_to_stoch:
                mu_stoch = torch.cat([vq_out.residual, mu_tail], dim=-1)
                logvar_stoch = logvar  # [B, T, D]
            else:
                mu_stoch = mu_tail
                logvar_stoch = logvar_tail  # [B, T, D-qd]

            # 2. Sample z_stoch (active parts only)
            if self.training:
                z_stoch = self.reparameterize(mu_tail, logvar_tail, std=1.0)
            else:
                z_stoch = mu_tail

            if self.add_vq_residual_to_stoch:
                if self.training:
                    z_stoch_head = self.reparameterize(
                        vq_out.residual, logvar_head, std=0.1
                    )
                else:
                    z_stoch_head = vq_out.residual
                z_stoch = torch.cat([z_stoch_head, z_stoch], dim=-1)
        else:
            mu_stoch = mu
            logvar_stoch = logvar
            z_stoch = self.reparameterize(mu, logvar)

        # 3. Dropout Regularizer (on active parts only)
        if hasattr(self, "dropout_regularizer") and not self.use_pre_quant_dropout:
            z_stoch_dropped = self.dropout_regularizer(z_stoch)
        else:
            z_stoch_dropped = z_stoch

        # 4. Pad with zeros if residual was skipped
        if hasattr(self, "vq") and not self.add_vq_residual_to_stoch:
            z_stoch_dropped = torch.cat(
                [torch.zeros_like(mu_head), z_stoch_dropped], dim=-1
            )

        # 5. Dropout per sample (on full dimension)
        if self.training and getattr(self, "residual_and_tail_dropout_p", 0.0) > 0.0:
            B = z_stoch_dropped.shape[0]
            keep_mask = (
                torch.rand(B, 1, 1, device=z_stoch_dropped.device)
                >= self.residual_and_tail_dropout_p
            ).to(z_stoch_dropped.dtype)
            z_stoch_dropped = z_stoch_dropped * keep_mask

        z = z_quantized + z_stoch_dropped

        kl_loss = None
        if kwargs.get("step", None) is not None:
            if hasattr(self, "kl_chunk_regularizer"):
                kl_term = self.kl_chunk_regularizer(
                    mu_stoch, logvar_stoch, padding_mask
                )
            else:
                kl_term = self.kl_divergence(
                    (
                        mu_stoch
                        if not self.use_pre_quant_dropout
                        else mu_original[..., qd:]  # only tail
                    ),
                    logvar_stoch,
                    padding_mask,
                    dtype=z.dtype,
                )
            kl_loss = kl_term * self.get_kl_cosine_schedule(kwargs["step"])

        out = {
            "z": z,
            "kl_loss": kl_loss,
            "padding_mask": padding_mask,
            "mu": mu_stoch,
            "mu_pre_vq": mu,
        }

        if hasattr(self, "vq"):
            out["vq_stats"] = vq_out.stats
            out["vq_loss"] = vq_out.loss
            out["quantized"] = z_quantized
            out["residual"] = torch.cat(
                [vq_out.residual, torch.zeros_like(mu_tail)], dim=-1
            )
            out["tail"] = torch.cat([torch.zeros_like(mu_head), mu_tail], dim=-1)
            out["indices"] = vq_out.indices

        return EncoderOutput(**out)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
