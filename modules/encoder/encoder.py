import math
import torch
import random
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from .vq import HardVectorQuantizer, FiniteScalarQuantizer, BinarySphericalQuantizer
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
            if getattr(config.vq_config, "fsq_levels", None) is not None:
                self.vq = FiniteScalarQuantizer(config.vq_config)
            else:
                self.vq = HardVectorQuantizer(config.vq_config)
            self.residual_and_tail_dropout_p = (
                config.vq_config.residual_and_tail_dropout_p
            )
            self.add_vq_residual_to_stoch = config.vq_config.add_vq_residual_to_stoch

            qd = config.vq_config.dim_to_quantize
            self._qd = qd
            tail_dim = config.latent_dim - qd

        elif config.bsq_config:
            if config.bsq_config.dim_to_quantize > config.latent_dim:
                raise ValueError(
                    f"BSQ dim_to_quantize ({config.bsq_config.dim_to_quantize}) must be <= latent_dim ({config.latent_dim})."
                )
            self.vq = BinarySphericalQuantizer(config.bsq_config)
            self.residual_and_tail_dropout_p = (
                config.bsq_config.residual_and_tail_dropout_p
            )
            self.add_vq_residual_to_stoch = config.bsq_config.add_vq_residual_to_stoch

            qd = config.bsq_config.dim_to_quantize
            self._qd = qd
            tail_dim = config.latent_dim - qd

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
            if getattr(config, "use_reparameterization_trick"):
                raise ValueError(
                    "Cannot use both noise_regularizer and use_reparameterization_trick=True"
                )
            self.noise_regularizer = NoiseRegularizer(
                config=config.noise_regularizer_config,
            )

        if config.freeze_encoder_before_latent_heads:
            self._freeze_encoder_before_latent_heads()

        self.config = config

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
        if not hasattr(self, "vq"):
            if (
                self.training
                and getattr(self.config, "use_reparameterization_trick", True)
                and not hasattr(self, "noise_regularizer")
            ):
                z_stoch = self.reparameterize(mu, logvar)
            else:
                z_stoch = mu
            return {
                "z_quantized": torch.zeros_like(mu),
                "z_stoch": z_stoch,
                "mu_stoch": mu,
                "logvar_stoch": logvar,
                "ortho_loss": None,
                "vq_dict": {}
            }
        
        qd = self._qd
        mu_head = mu[..., :qd]
        mu_tail = mu[..., qd:]

        ortho_loss = self._calculate_ortho_loss(mu_head, mu_tail, padding_mask)

        vq_out = self.vq(
            mu_head,
            padding_mask,
            global_step=step,
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
        if (
            self.training
            and getattr(self.config, "use_reparameterization_trick", True)
            and not hasattr(self, "noise_regularizer")
        ):
            z_stoch = self.reparameterize(mu_tail, logvar_tail, std=1.0)
        else:
            z_stoch = mu_tail

        if self.add_vq_residual_to_stoch:
            if (
                self.training
                and getattr(self.config, "use_reparameterization_trick", True)
                and not hasattr(self, "noise_regularizer")
            ):
                z_stoch_head = self.reparameterize(
                    vq_out.residual, logvar_head, std=0.1
                )
            else:
                z_stoch_head = vq_out.residual
            z_stoch = torch.cat([z_stoch_head, z_stoch], dim=-1)

        vq_dict = {
            "vq_stats": vq_out.stats,
            "vq_loss": vq_out.loss,
            "quantized": vq_quantized_ste,
            "residual": vq_out.residual,
            "tail": mu_tail,
            "indices": vq_out.indices,
            "mu_head": mu_head,
        }

        return {
            "z_quantized": z_quantized,
            "z_stoch": z_stoch,
            "mu_stoch": mu_stoch,
            "logvar_stoch": logvar_stoch,
            "ortho_loss": ortho_loss,
            "vq_dict": vq_dict
        }

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
        speaker_embedding = None
        if getattr(self.config, "use_instance_norm", False):
            mu_original, speaker_embedding = self._apply_instance_norm(
                mu_original, padding_mask
            )

        mu = mu_original
        if hasattr(self, "dropout_regularizer"):
            mu = self.dropout_regularizer(mu)
        if hasattr(self, "noise_regularizer"):
            mu = self.noise_regularizer(mu)

        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(h)

        q_out = self._quantize_and_sample(mu, logvar, padding_mask, step=kwargs.get("step", None))
        z_quantized = q_out["z_quantized"]
        z_stoch = q_out["z_stoch"]
        mu_stoch = q_out["mu_stoch"]
        logvar_stoch = q_out["logvar_stoch"]
        ortho_loss = q_out["ortho_loss"]
        vq_dict = q_out["vq_dict"]

        # 3. Regularizers are applied before quantization
        z_stoch_dropped = z_stoch

        # 4. Pad with zeros if residual was skipped
        if hasattr(self, "vq") and not self.add_vq_residual_to_stoch:
            z_stoch_dropped = torch.cat(
                [torch.zeros_like(vq_dict["mu_head"]), z_stoch_dropped], dim=-1
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
                _qd = self._qd if hasattr(self, "vq") else 0
                kl_term = self.kl_divergence(
                    mu_original[..., _qd:],  # only tail
                    logvar if logvar is None else logvar[..., _qd:],
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
            "ortho_loss": ortho_loss,
            "speaker_embedding": speaker_embedding,
        }

        if hasattr(self, "vq"):
            out["vq_stats"] = vq_dict["vq_stats"]
            out["vq_loss"] = vq_dict["vq_loss"]
            out["quantized"] = vq_dict["quantized"]
            out["residual"] = vq_dict["residual"]
            out["tail"] = vq_dict["tail"]
            out["indices"] = vq_dict["indices"]

        return EncoderOutput(**out)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
