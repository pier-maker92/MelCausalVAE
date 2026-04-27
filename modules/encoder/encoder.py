import math
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from modules.configs import EncoderConfig
from modules.encoder.vq import HardVectorQuantizer
from modules.output_dataclasses import EncoderOutput
from modules.encoder.sigma_vae_encoder import SigmaVAEEncoder
from modules.encoder.utils import (
    _assert_latent_chunks_divisible,
    apply_latent_chunk_dropout,
    latent_chunk_kl_channel_weights_vq_tail,
    latent_chunk_kl_channel_weights,
)
from modules.encoder.regularization import (
    DropoutRegularizer,
    KLChunkRegularizer,
)


class TimeCausalConv1d(nn.Conv1d):
    """Causal Conv1d: left-pad only so output at time t depends only on inputs <= t."""

    def __init__(self, c_in, c_out, k, d=1, s=1):
        super().__init__(c_in, c_out, k, dilation=d, stride=s, padding=0)
        self.k, self.d, self.s = k, d, s

    def forward(self, x):  # x: [B, C, T]
        pad_left = (self.k - 1) * self.d
        x = F.pad(x, (pad_left, 0))
        return super().forward(x)


class PreNormResCausalBlock1d(nn.Module):
    """GroupNorm(1) -> GELU -> CausalConv1d -> Dropout + skip."""

    def __init__(self, c_in, c_out, *, k=3, d=1, s=1, drop_p=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, c_in)
        self.act = nn.GELU()
        self.main = TimeCausalConv1d(c_in, c_out, k=k, d=d, s=s)
        if c_in != c_out or s != 1:
            self.skip = TimeCausalConv1d(c_in, c_out, k=1, d=1, s=s)
        else:
            self.skip = nn.Identity()
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):  # x: [B, C, T]
        h = self.act(self.norm(x))
        h = self.dropout(h)
        return self.main(h) + self.skip(x)


class CausalDownsamplingBlock1d(nn.Module):
    """Dilated residual blocks followed by a stride-2 downsampler."""

    def __init__(self, c_in, c_out, n_residual_blocks=3, drop_p=0.1):
        super().__init__()
        dilations = [1, 2, 4, 8]
        self.residual_blocks = nn.ModuleList(
            [
                PreNormResCausalBlock1d(c_in, c_in, k=5, d=dilation, drop_p=drop_p)
                for dilation in dilations[:n_residual_blocks]
            ]
        )
        self.downsampling = PreNormResCausalBlock1d(
            c_in, c_out, k=5, d=1, s=2, drop_p=drop_p
        )

    def forward(self, x):  # x: [B, C, T]
        for block in self.residual_blocks:
            x = block(x)
        return self.downsampling(x)


class Transformer(nn.Module):
    def __init__(self, d_model=512, nheads=8, nlayers=4, drop_p=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nheads,
            batch_first=True,
            norm_first=True,
            dropout=drop_p,
            dim_feedforward=4 * d_model,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)

    def forward(self, x, pad_mask=None):
        return self.enc(x, src_key_padding_mask=pad_mask, is_causal=True)


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
                    f"vq_quant_dim ({config.vq_quant_dim}) must be <= latent_dim ({config.latent_dim})."
                )
            self.vq = HardVectorQuantizer(
                dim=config.vq_quant_dim,
                num_embeddings=config.vq_num_embeddings,
                commit_weight=config.vq_commit_weight,
                reset_dead_codes=config.vq_reset_dead_codes,
                reset_max_per_step=config.vq_reset_max_per_step,
                reset_every_forward=config.vq_reset_every_forward,
                use_ema_codebook=config.vq_use_ema_codebook,
                ema_decay=config.vq_ema_decay,
                ema_epsilon=config.vq_ema_epsilon,
            )
            self.vq_residual_drop = nn.Dropout(p=config.vq_residual_dropout)

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
        z = self.transformer(hiddens)  # [B, T/C, 512]

        vq_loss = None
        vq_perplexity = None
        vq_codes_used = None
        vq_codes_used_frac = None

        mu_pre_vq = self.mu(z)
        if hasattr(self, "dropout_regularizer"):
            mu_pre_vq = self.dropout_regularizer(mu_pre_vq)

        vq_latent_residual = None
        vq_indices = None
        if hasattr(self, "vq"):
            qd = self.config.vq_config.dim_to_quantize
            mu_head = mu_pre_vq[..., :qd]
            mu_tail = mu_pre_vq[..., qd:]
            mu_res_vq, vq_loss, mu_q, vq_stats, vq_indices = self.vq(
                mu_head,
                padding_mask,
                global_step=kwargs.get("step", None),
            )
            vq_perplexity = vq_stats.perplexity
            vq_codes_used = vq_stats.codes_used
            vq_codes_used_frac = vq_stats.codes_used_frac
            vq_latent_residual = (mu_head - mu_q.detach()).detach()
            r_add = self.vq_residual_drop(mu_res_vq)
            z_head = mu_q + r_add
            mu = torch.cat([mu_res_vq, mu_tail], dim=-1)
        else:
            mu = mu_pre_vq

        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(z)
        z_stoch = self.reparameterize(mu, logvar)

        if self.config.latent_chunk_ablate_dropout:
            z_stoch = apply_latent_chunk_dropout(
                z_stoch,
                chunk_size=self.config.latent_chunk_size,
                dropout_start=self.config.latent_chunk_dropout_start,
                dropout_end=self.config.latent_chunk_dropout_end,
                training=self.training,
                hierarchical=self.config.latent_chunk_dropout_hierarchical,
            )

        if hasattr(self, "vq"):
            qd = self.config.vq_config.dim_to_quantize
            z = torch.cat(
                [
                    z_head.to(dtype=z_stoch.dtype),
                    z_stoch[..., qd:].to(dtype=z_stoch.dtype),
                ],
                dim=-1,
            )
        else:
            z = z_stoch

        if kwargs.get("semantic_guidance", None) is not None:
            raise NotImplementedError("Semantic guidance is not implemented yet")

        kl_loss = None
        if kwargs.get("step", None) is not None:
            if hasattr(self, "kl_chunk_regularizer"):
                kl_term = self.kl_chunk_regularizer(mu_pre_vq, padding_mask)
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
            "mu": mu_pre_vq,
            "logvar": logvar,
        }

        if hasattr(self, "vq"):
            out["vq_loss"] = vq_loss
            out["vq_perplexity"] = vq_perplexity
            out["vq_codes_used"] = vq_codes_used
            out["vq_codes_used_frac"] = vq_codes_used_frac
            out["vq_latent_residual"] = vq_latent_residual
            out["vq_indices"] = vq_indices

        return EncoderOutput(**out)
