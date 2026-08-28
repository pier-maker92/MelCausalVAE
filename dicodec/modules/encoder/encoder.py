import math
import torch
import torch.nn as nn
from typing import Optional
from ..configs import EncoderConfig
from ..output_dataclasses import EncoderOutput
from ..lp_filter import LowPassFilter
from ..quantizer.vq import VectorQuantizer
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

        if config.vq_config is not None:
            if config.lowpass_filter_config is None:
                raise ValueError(
                    "encoder.lowpass_filter_config is required when encoder.vq_config is set."
                )
            lowpass_config = config.lowpass_filter_config
            self.lowpass_filter = LowPassFilter(
                cutoff_hz=lowpass_config.cutoff_hz,
                sample_rate=lowpass_config.sample_rate,
                order=lowpass_config.order,
            )
            self.quantizer = VectorQuantizer(config=config.vq_config, dim=latent_dim)
            self.conv_pros = nn.Conv1d(
                latent_dim,
                latent_dim,
                kernel_size=3,
                padding=1,
            )
            self.conv_sem = nn.Conv1d(
                latent_dim,
                latent_dim,
                kernel_size=3,
                padding=1,
            )
            self.attribute_projection = nn.Linear(latent_dim * 2, latent_dim)

        self.config = config
        self.use_reparameterization_trick = getattr(
            config, "use_reparameterization_trick"
        )

    def _encode_quantized_attributes(
        self,
        z: torch.FloatTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ):
        valid_mask = None
        if padding_mask is not None:
            if padding_mask.shape != z.shape[:2]:
                raise ValueError(
                    "padding_mask must have shape [batch, time], got "
                    f"{tuple(padding_mask.shape)} for z shape {tuple(z.shape)}."
                )
            valid_mask = (
                (~padding_mask).to(device=z.device, dtype=z.dtype).unsqueeze(-1)
            )
            valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            valid_count = z.new_full((z.shape[0], 1, 1), z.shape[1])

        if valid_mask is None:
            z_mean = z.mean(dim=1, keepdim=True)
        else:
            z_mean = (z * valid_mask).sum(dim=1, keepdim=True) / valid_count

        z_centered = z - z_mean
        if valid_mask is not None:
            z_centered = z_centered * valid_mask

        z_lp = self.lowpass_filter(z_centered, valid_mask=valid_mask)
        dot_product = torch.sum(z_centered * z_lp, dim=1, keepdim=True)
        norm_sq = torch.sum(z_lp.square(), dim=1, keepdim=True)
        z_pros = (dot_product / (norm_sq + 1e-8)) * z_lp
        z_sem = z_centered - z_pros
        if valid_mask is not None:
            z_pros = z_pros * valid_mask
            z_sem = z_sem * valid_mask

        vq_output = self.quantizer(z_sem, padding_mask=padding_mask)
        z_sem = vq_output.quantized

        z_pros = self.conv_pros((z_pros + z_mean).transpose(1, 2)).transpose(1, 2)
        z_sem = self.conv_sem(z_sem.transpose(1, 2)).transpose(1, 2)
        if valid_mask is not None:
            z_pros = z_pros * valid_mask
            z_sem = z_sem * valid_mask

        z = self.attribute_projection(torch.cat([z_sem, z_pros], dim=-1))
        if valid_mask is not None:
            z = z * valid_mask

        return z, vq_output

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

        vq_output = None
        if hasattr(self, "quantizer"):
            z, vq_output = self._encode_quantized_attributes(
                z,
                padding_mask=padding_mask,
            )

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
            "vq_loss": vq_output.loss if vq_output is not None else None,
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
