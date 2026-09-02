import math
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from ..configs import EncoderConfig
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

        self.config = config
        self.use_reparameterization_trick = getattr(
            config, "use_reparameterization_trick"
        )

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
        logvar = None
        if hasattr(self, "logvar"):
            raise NotImplementedError(
                "logvar is not supported. Sigma-VAE does not use logvar, sigma is drawn from a Normal distribution with fixed std=1.0."
            )

        # regularization
        if hasattr(self, "dropout_regularizer"):
            mu = self.dropout_regularizer(mu)
        if hasattr(self, "noise_regularizer"):
            mu = self.noise_regularizer(mu)

        if self.use_reparameterization_trick and self.training:
            z = self.reparameterize(mu, logvar, std=1.0)
        else:
            z = mu

        # L2 penalty computation
        kl_weight = (
            self.get_kl_cosine_schedule(kwargs["step"])
            if kwargs.get("step", None) is not None
            else 0.0
        )
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
        }

        return EncoderOutput(**out)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
