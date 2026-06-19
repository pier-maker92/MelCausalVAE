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

            if tail_dim > 0:
                self.ortho_proj_head = nn.Linear(qd, qd)
                self.ortho_proj_tail = nn.Linear(tail_dim, qd)

        elif config.bsq_config:
            if config.bsq_config.dim_to_quantize > config.latent_dim:
                raise ValueError(
                    f"BSQ dim_to_quantize ({config.bsq_config.dim_to_quantize}) must be <= latent_dim ({config.latent_dim})."
                )
            self.vq = BinarySphericalQuantizer(config.bsq_config)
            self.residual_and_tail_dropout_p = config.bsq_config.residual_and_tail_dropout_p
            self.add_vq_residual_to_stoch = config.bsq_config.add_vq_residual_to_stoch

            qd = config.bsq_config.dim_to_quantize
            self._qd = qd
            tail_dim = config.latent_dim - qd

            if tail_dim > 0:
                self.ortho_proj_head = nn.Linear(qd, qd)
                self.ortho_proj_tail = nn.Linear(tail_dim, qd)

        if hasattr(self, "vq"):
            self.vq_pre_proj = nn.Linear(self._qd, self._qd)

        if getattr(config, "use_instance_norm", False) and getattr(config, "speaker_embedding_dim", 0) > 0:
            if config.vq_config or config.bsq_config:
                qd_val = config.vq_config.dim_to_quantize if config.vq_config else config.bsq_config.dim_to_quantize
                spk_raw_dim = (config.latent_dim - qd_val) * 2
            else:
                spk_raw_dim = config.latent_dim * 2
            self.spk_proj = nn.Linear(spk_raw_dim, config.speaker_embedding_dim)

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

        if config.noise_regularizer_config:
            if getattr(config, "use_reparameterization_trick", True):
                raise ValueError(
                    "Cannot use both noise_regularizer and use_reparameterization_trick=True"
                )
            self.noise_regularizer = NoiseRegularizer(
                config=config.noise_regularizer_config,
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
        if hasattr(self, "vq_pre_proj"):
            for param in self.vq_pre_proj.parameters():
                param.requires_grad = True
        if hasattr(self, "spk_proj"):
            for param in self.spk_proj.parameters():
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

        speaker_embedding = None
        if getattr(self.config, "use_instance_norm", False):
            if hasattr(self, "vq"):
                # Apply IN only to the tail (continuous dims), not the quantized head.
                qd = self._qd
                mu_head_raw = mu_original[..., :qd]
                mu_tail_raw = mu_original[..., qd:]
                spk_mu = mu_tail_raw.mean(dim=1, keepdim=True)
                spk_sigma = mu_tail_raw.std(dim=1, keepdim=True)
                mu_original = torch.cat(
                    [mu_head_raw, (mu_tail_raw - spk_mu) / (spk_sigma + 1e-6)], dim=-1
                )
            else:
                spk_mu = mu_original.mean(dim=1, keepdim=True)
                spk_sigma = mu_original.std(dim=1, keepdim=True)
                mu_original = (mu_original - spk_mu) / (spk_sigma + 1e-6)
            speaker_embedding = torch.cat(
                [spk_mu.squeeze(1), spk_sigma.squeeze(1)], dim=-1
            )
            if hasattr(self, "spk_proj"):
                speaker_embedding = self.spk_proj(speaker_embedding)

        if self.use_pre_quant_dropout:
            mu = self.dropout_regularizer(mu_original)
        else:
            mu = mu_original

        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(h)

        z_quantized = torch.zeros_like(mu)
        ortho_loss = None
        # vq
        if hasattr(self, "vq"):
            qd = self._qd
            mu_head = mu[..., :qd]
            mu_tail = mu[..., qd:]

            ortho_weight = 1.0
            if getattr(self.config, "semantic_distillation_config", None) is not None:
                ortho_weight = (
                    self.config.semantic_distillation_config.ortho_loss_weight
                )

            if mu_tail.shape[-1] > 0 and ortho_weight > 0.0:
                mask = ~padding_mask
                # h1 shape: [N, qd], h2 shape: [N, tail_dim]
                h1 = mu_head[mask]
                h2 = mu_tail[mask]

                # 1. Proietta alla stessa dimensione se necessario
                if hasattr(self, "ortho_proj_head") and hasattr(
                    self, "ortho_proj_tail"
                ):
                    h1_proj = self.ortho_proj_head(h1)
                    h2_proj = self.ortho_proj_tail(h2)
                else:
                    h1_proj = h1
                    h2_proj = h2

                # 2. Definisci il parametro beta
                beta = (
                    getattr(
                        self.config.semantic_distillation_config, "ortho_beta", 0.01
                    )
                    if getattr(self.config, "semantic_distillation_config", None)
                    is not None
                    else 0.01
                )

                # 3. Calcola la Cosine Similarity
                cos_sim = F.cosine_similarity(h1_proj, h2_proj, dim=-1)
                abs_cos_sim = torch.abs(cos_sim)
                mean_abs_cos_sim = abs_cos_sim.mean()

                # 4. Sottrai beta e eleva al quadrato
                ortho_loss = (mean_abs_cos_sim - beta) ** 2

            if hasattr(self, "vq_pre_proj"):
                mu_head = self.vq_pre_proj(mu_head)

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
            if self.training and getattr(
                self.config, "use_reparameterization_trick", True
            ) and not hasattr(self, "noise_regularizer"):
                z_stoch = self.reparameterize(mu_tail, logvar_tail, std=1.0)
            else:
                z_stoch = mu_tail

            if self.add_vq_residual_to_stoch:
                if self.training and getattr(
                    self.config, "use_reparameterization_trick", True
                ) and not hasattr(self, "noise_regularizer"):
                    z_stoch_head = self.reparameterize(
                        vq_out.residual, logvar_head, std=0.1
                    )
                else:
                    z_stoch_head = vq_out.residual
                z_stoch = torch.cat([z_stoch_head, z_stoch], dim=-1)
        else:
            mu_stoch = mu
            logvar_stoch = logvar
            if self.training and getattr(
                self.config, "use_reparameterization_trick", True
            ) and not hasattr(self, "noise_regularizer"):
                z_stoch = self.reparameterize(mu, logvar)
            else:
                z_stoch = mu

        # 3. Dropout Regularizer (on active parts only)
        if hasattr(self, "dropout_regularizer") and not self.use_pre_quant_dropout:
            z_stoch_dropped = self.dropout_regularizer(z_stoch)
        else:
            z_stoch_dropped = z_stoch

        if hasattr(self, "noise_regularizer"):
            z_stoch_dropped = self.noise_regularizer(z_stoch_dropped)

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
                _qd = self._qd if hasattr(self, "vq") else 0
                kl_term = self.kl_divergence(
                    (
                        mu_stoch
                        if not self.use_pre_quant_dropout
                        else mu_original[..., _qd:]  # only tail
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
            "ortho_loss": ortho_loss,
            "speaker_embedding": speaker_embedding,
        }

        if hasattr(self, "vq"):
            out["vq_stats"] = vq_out.stats
            out["vq_loss"] = vq_out.loss
            out["quantized"] = vq_quantized_ste
            out["residual"] = vq_out.residual
            out["tail"] = mu_tail
            out["indices"] = vq_out.indices

        return EncoderOutput(**out)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
