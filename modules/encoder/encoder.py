import math
import torch
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

        # dual encoder heads: semantic and acoustic
        self.semantic_proj = nn.Linear(d_model, config.semantic_dim)
        self.acoustic_proj = nn.Linear(d_model, config.acoustic_dim)
        # vq is mandatory for the semantic head, but optional for the acoustic head
        self.vq = VectorQuantizer(
            config.vq_config,
            dim=config.semantic_dim,
            pitch_loss_config=config.pitch_loss_config,
            acoustic_dim=config.acoustic_dim,
        )
        if self.config.vq_acoustic_config is not None:
            self.vq_acoustic = VectorQuantizer(
                config.vq_acoustic_config, dim=config.acoustic_dim
            )

        if config.dropout_regularizer_config:
            raise NotImplementedError("Dropout regularizer is not supported yet")
            # self.dropout_regularizer = DropoutRegularizer(
            #     config=config.dropout_regularizer_config
            # )

        self.use_reparameterization_trick = getattr(
            config, "use_reparameterization_trick"
        )

        if self.config.acoustic_logvar:
            assert (
                self.use_reparameterization_trick
            ), "acoustic_logvar requires reparameterization trick"
            self.acoustic_logvar = nn.Linear(d_model, config.acoustic_dim)

        self.proj_out = nn.Linear(config.semantic_dim + config.acoustic_dim, latent_dim)

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
        raise DeprecationWarning("Instance norm is deprecated.")

    def handle_drop_acoustic(self, acoustic):
        if self.config.vq_config.drop_acoustic_p > 0.0 and self.training:
            drop_mask = (
                (
                    torch.rand(acoustic.shape[0], device=acoustic.device)
                    < self.config.vq_config.drop_acoustic_p
                )
                .to(acoustic.dtype)
                .view(-1, 1, 1)
            )
        else:
            drop_mask = torch.ones(
                (acoustic.shape[0], 1, 1), device=acoustic.device, dtype=acoustic.dtype
            )
        acoustic = acoustic * drop_mask
        return acoustic

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

        semantic = self.semantic_proj(h)  # [B, T/C, semantic_dim]
        acoustic = self.acoustic_proj(h)  # [B, T/C, acoustic
        logvar = None
        if hasattr(self, "acoustic_logvar"):
            logvar = self.acoustic_logvar(h)  # [B, T/C, acoustic_dim]

        # acoustic branch
        if self.use_reparameterization_trick and self.training:
            acoustic = self.reparameterize(acoustic, logvar)

        acoustic = self.handle_drop_acoustic(acoustic)

        # quantized semantic branch
        vq_output = self.vq(
            semantic,
            padding_mask,
            acoustic=acoustic,
            pitch_targets=kwargs.get("pitch_targets", None),
            step=kwargs.get("step", None),
        )
        quantized = vq_output.quantized

        # concat the semantic and acoustic branches
        z = torch.cat([quantized, acoustic], dim=-1)
        z = self.proj_out(z)  # [B, T/C, latent_dim]

        # KL penalty computation
        kl_weight = (
            self.get_kl_cosine_schedule(kwargs["step"])
            if kwargs.get("step", None) is not None
            else 0.0
        )
        kl_loss = None
        if self.training:
            kl_term = self.kl_divergence(
                acoustic,
                logvar,
                padding_mask,
                dtype=acoustic.dtype,
            )
            kl_loss = kl_term * kl_weight

        out = {
            "z": z,
            "kl_loss": kl_loss,
            "mu": acoustic,
            "padding_mask": padding_mask,
            "speaker_embedding": None,
        }

        if hasattr(self, "vq"):
            out["vq_stats"] = vq_output.stats
            out["vq_loss"] = vq_output.loss
            out["quantized"] = vq_output.quantized
            out["residual"] = vq_output.residual
            out["indices"] = vq_output.indices
            out["acoustic_f0_loss"] = vq_output.acoustic_f0_loss
            out["semantic_f0_adv_loss"] = vq_output.semantic_f0_adv_loss
            out["pitch_voiced_loss"] = vq_output.pitch_voiced_loss
            out["pitch_contour_loss"] = vq_output.pitch_contour_loss

        return EncoderOutput(**out)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
