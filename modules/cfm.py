import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
import random
import torch.nn as nn
from abc import ABCMeta
from typing import Optional
from einops import rearrange
from torchdiffeq import odeint
from dataclasses import dataclass
from torch.nn import functional as F
from modules.Decoder import Transformer


@dataclass
class DiTOutput:
    loss: Optional[torch.Tensor] = None


@dataclass
class DiTConfig:
    audio_latent_dim: int
    unet_dim: int = 512
    unet_depth: int = 6
    unet_heads: int = 8
    unet_dropout_rate: float = 0.0
    use_conv_layer: bool = False
    use_sos_token: bool = False
    sigma: float = 1e-5
    expansion_factor: int = 1
    mel_channels: int = 100
    uncond_prob: float = 0.0
    learned_prior: bool = False
    use_vp_schedule: bool = False
    is_causal: bool = True
    # def __init__(
    #     self,
    #     audio_latent_dim: int,
    #     unet_dim: int = 512,
    #     unet_depth: int = 6,
    #     unet_heads: int = 8,
    #     unet_dropout_rate: float = 0.0,
    #     use_conv_layer: bool = False,
    #     use_sos_token: bool = False,
    #     sigma: float = 1e-5,
    #     expansion_factor: int = 1,
    #     mel_channels: int = 100,
    #     uncond_prob: float = 0.0,
    # ):
    #     """
    #     depformer_dim: int - Hidden dimension of the Depth Transformer
    #     depformer_dim_feedforward: int - Feedforward dimension of the Depth Transformer
    #     depformer_num_heads: int - Number of heads in the Depth Transformer
    #     depformer_num_layers: int - Number of layers in the Depth Transformer
    #     depformer_casual: bool - Whether the Depth Transformer is causal
    #     """

    #     # unet specific
    #     self.sigma = sigma
    #     self.unet_dim = unet_dim
    #     self.unet_heads = unet_heads
    #     self.unet_depth = unet_depth
    #     self.use_sos_token = use_sos_token
    #     self.use_conv_layer = use_conv_layer
    #     self.audio_latent_dim = audio_latent_dim
    #     self.unet_dropout_rate = unet_dropout_rate
    #     self.expansion_factor = expansion_factor
    #     self.mel_channels = mel_channels
    #     self.uncond_prob = uncond_prob


class DiT(torch.nn.Module):
    def __init__(
        self,
        config: DiTConfig,
    ):
        super().__init__()

        # inherit from config
        self.sigma = config.sigma
        self.unet_dim = config.unet_dim
        self.unet_heads = config.unet_heads
        self.unet_depth = config.unet_depth
        self.use_sos_token = config.use_sos_token
        self.use_conv_layer = config.use_conv_layer
        self.audio_latent_dim = config.audio_latent_dim
        self.expansion_factor = config.expansion_factor
        self.mel_channels = config.mel_channels
        self.uncond_prob = config.uncond_prob
        self.learned_prior = config.learned_prior
        self.is_causal = config.is_causal
        print(f"VAE is_causal: {self.is_causal}")
        self.context_vector_proj = nn.Sequential(
            nn.Linear(self.audio_latent_dim, self.unet_dim), nn.LayerNorm(self.unet_dim)
        )
        # hubert norm
        self.hubert_norm = nn.LayerNorm(self.audio_latent_dim)
        # hubert + context projection
        self.hubert_and_context_proj = nn.Sequential(
            nn.Linear(
                self.audio_latent_dim + self.audio_latent_dim, self.audio_latent_dim
            ),
            nn.LayerNorm(self.audio_latent_dim),
        )
        # noise projection
        if self.learned_prior:
            self.noise_proj = nn.Sequential(
                nn.Linear(self.mel_channels, self.unet_dim),
                nn.LayerNorm(self.unet_dim),
            )
            self.prior_proj = nn.Sequential(
                nn.Linear(self.audio_latent_dim, self.mel_channels),
                nn.LayerNorm(self.mel_channels),
            )

        else:
            self.noise_proj = nn.Sequential(
                nn.Linear(self.unet_dim + self.mel_channels, self.unet_dim),
                nn.LayerNorm(self.unet_dim),
            )

        # transformer
        self.transformer = Transformer(
            dim=self.unet_dim,
            depth=self.unet_depth,
            heads=self.unet_heads,
            use_conv_layer=self.use_conv_layer,
            audio_latent_dim=self.mel_channels,  # projection to mel
            is_causal=self.is_causal,
            attn_flash=True,
        )
        self.transformer.to(dtype=torch.bfloat16)

    def reparameterize(
        self, mu: torch.FloatTensor, std: Optional[float] = None
    ) -> torch.FloatTensor:
        eps = torch.randn_like(mu)
        if std is None:
            std = self.sample_scalar_std(mu)
        else:
            std = torch.ones_like(mu) * std
        while std.dim() < mu.dim():
            std = std.unsqueeze(-1)
        return mu + eps * std

    def sample_scalar_std(self, mu: torch.FloatTensor) -> torch.FloatTensor:
        return torch.randn(mu.shape[0], mu.shape[1], dtype=mu.dtype, device=mu.device)

    def handle_context_vector(
        self,
        context_vector: torch.FloatTensor,
        target: Optional[torch.FloatTensor] = None,
        hubert_guidance: Optional = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        frame_durations: Optional[torch.BoolTensor] = None,
    ):
        if hubert_guidance or frame_durations is None:
            context_vector = context_vector.repeat_interleave(
                self.expansion_factor, dim=1
            )
        else:
            if frame_durations is not None:
                durations = frame_durations * self.expansion_factor
                assert (
                    context_vector.shape[0] == durations.shape[0]
                ), "number of context vectors and durations must match"
                assert (
                    context_vector.shape[1] == durations.shape[1]
                ), "time dimension must match"
                context_vector_batch = []
                if padding_mask is not None:
                    padding_mask_batch = []
                for batch_idx in range(context_vector.shape[0]):
                    duration_sequence = durations[batch_idx]
                    context_vector_sequence = []
                    padding_mask_sequence = []
                    for idx, duration in enumerate(duration_sequence):
                        duration = round(duration.item())
                        if duration:
                            context_vector_sequence.append(
                                context_vector[batch_idx][idx].repeat(duration, 1)
                            )
                            if padding_mask is not None:
                                padding_mask_sequence.append(
                                    padding_mask[batch_idx][idx].repeat(duration)
                                )
                        else:
                            context_vector_sequence.append(
                                torch.zeros_like(context_vector[batch_idx][:1])
                            )
                            if padding_mask is not None:
                                padding_mask_sequence.append(
                                    torch.ones_like(padding_mask[batch_idx][:1])
                                )
                    context_vector_batch.append(
                        torch.cat(context_vector_sequence, dim=0)
                    )
                    if padding_mask is not None:
                        padding_mask_batch.append(
                            torch.cat(padding_mask_sequence, dim=0)
                        )
            else:
                durations = hubert_guidance.durations * self.expansion_factor
                assert (
                    context_vector.shape[0]
                    == hubert_guidance.semantic_embeddings.shape[0]
                    == durations.shape[0]
                ), "number of context vectors and semantic embeddings and durations must match"
                assert (
                    context_vector.shape[1]
                    == hubert_guidance.semantic_embeddings.shape[1]
                    == durations.shape[1]
                ), "time dimension must match"
                context_vector = self.hubert_and_context_proj(
                    torch.cat(
                        [
                            context_vector,
                            self.hubert_norm(hubert_guidance.semantic_embeddings),
                        ],
                        dim=-1,
                    )
                )
                context_vector_batch = []
                if padding_mask is not None:
                    padding_mask_batch = []
                for batch_idx in range(context_vector.shape[0]):
                    duration_sequence = durations[batch_idx]
                    context_vector_sequence = []
                    padding_mask_sequence = []
                    for idx, duration in enumerate(duration_sequence):
                        if duration:
                            context_vector_sequence.append(
                                context_vector[batch_idx][idx].repeat(duration, 1)
                            )
                            if padding_mask is not None:
                                padding_mask_sequence.append(
                                    padding_mask[batch_idx][idx].repeat(duration)
                                )
                        else:
                            context_vector_sequence.append(
                                torch.zeros_like(context_vector[batch_idx][:1])
                            )
                            if padding_mask is not None:
                                padding_mask_sequence.append(
                                    torch.ones_like(padding_mask[batch_idx][:1])
                                )

                    context_vector_batch.append(
                        torch.cat(context_vector_sequence, dim=0)
                    )
                    if padding_mask is not None:
                        padding_mask_batch.append(
                            torch.cat(padding_mask_sequence, dim=0)
                        )
            # pad the context vector sequence to the same length using orch.nn.utils.rnn.pad_sequence
            context_vector = torch.nn.utils.rnn.pad_sequence(
                context_vector_batch, batch_first=True, padding_value=0
            )
            if padding_mask is not None:
                padding_mask = torch.nn.utils.rnn.pad_sequence(
                    padding_mask_batch, batch_first=True, padding_value=False
                )
        if target is not None:
            min_length = min(context_vector.shape[1], target.shape[1])
            context_vector = context_vector[:, :min_length, :]
            target = target[:, :min_length, :]

        if self.learned_prior:
            context_vector = self.prior_proj(context_vector)
        else:
            context_vector = self.context_vector_proj(context_vector)

        return context_vector, target, padding_mask

    def prepare_flow(
        self,
        target: torch.FloatTensor,
        context_vector: torch.FloatTensor,
    ):
        if random.random() < self.uncond_prob:
            if self.learned_prior:
                context_vector = torch.randn_like(context_vector)
            else:
                context_vector = torch.zeros_like(context_vector)
        # We need times
        # We need times
        times = torch.rand(
            (target.shape[0],),
            dtype=context_vector.dtype,
            device=context_vector.device,
        )
        t = rearrange(times, "b -> b 1 1")
        # Now we need noise, x0 sampled from a normal distribution
        if self.learned_prior:
            x0 = context_vector
        else:
            x0 = torch.randn_like(target)
        # w is the noise signal that is transformed by the flow
        w = (1 - (1 - self.sigma) * t) * x0 + t * target
        # target is the original signal minus the noise
        target = target - (1 - self.sigma) * x0

        if not self.learned_prior:
            state = self.noise_proj(torch.cat([context_vector, w], dim=-1))
        else:
            state = self.noise_proj(w)
        return state, times, target

    def let_it_flow(
        self,
        times: torch.FloatTensor,
        state: torch.FloatTensor,
        target: Optional[torch.FloatTensor] = None,
        flow_mask: Optional[torch.BoolTensor] = None,
    ):
        mask_to_loss = ~flow_mask
        v = self.transformer(
            x=state,
            times=times,
            attention_mask=mask_to_loss,
        )
        loss = None
        if target is not None:
            v_to_loss = v[mask_to_loss].view(-1, self.mel_channels)
            target_to_loss = target[mask_to_loss].view(-1, self.mel_channels)
            # Compute loss in fp32 for numerical stability with fp16
            loss = F.mse_loss(v_to_loss.float(), target_to_loss.float()).to(v.dtype)

        return loss

    def forward(
        self,
        target: torch.FloatTensor,
        target_padding_mask: torch.BoolTensor,
        context_vector: torch.FloatTensor,
        **kwargs,
    ):
        assert not torch.isnan(context_vector).any(), "context vector contains nan"
        context_vector, target, target_padding_mask = self.handle_context_vector(
            context_vector,
            target,
            hubert_guidance=kwargs.get("hubert_guidance", None),
            frame_durations=kwargs.get("frame_durations", None),
            padding_mask=target_padding_mask,
        )

        # ---- flow ----
        state, times, v_target = self.prepare_flow(
            target=target,
            context_vector=context_vector,
        )
        target_padding_mask = target_padding_mask[:, : target.shape[1]]

        # ---- get the flow ----
        loss = self.let_it_flow(
            times=times, state=state, target=v_target, flow_mask=target_padding_mask
        )

        return DiTOutput(
            loss=loss,
        )

    def generate(
        self,
        num_steps: int,
        context_vector: torch.FloatTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        std: float = 1.0,
        hubert_guidance: Optional[torch.Tensor] = None,
        frame_durations: Optional[torch.Tensor] = None,
    ):
        cfg_scale = guidance_scale
        # ---- context vector z ----
        context_vector, _, padding_mask = self.handle_context_vector(
            context_vector,
            hubert_guidance=hubert_guidance,
            padding_mask=padding_mask,
            frame_durations=frame_durations,
        )
        if self.learned_prior:
            y0 = context_vector
        else:
            y0 = (
                torch.randn(
                    context_vector.shape[0],
                    context_vector.shape[1],
                    self.mel_channels,
                    dtype=context_vector.dtype,
                    device=context_vector.device,
                    generator=generator,
                )
                * temperature
            )

        B, T = context_vector.shape[:2]
        if padding_mask is None:
            if B == 1:
                padding_mask = torch.zeros(
                    (1, T),
                    device=context_vector.device,
                    dtype=torch.bool,
                )
            else:
                raise ValueError("Padding mask is required for batch size > 1")
        else:
            if hubert_guidance is None and frame_durations is None:
                padding_mask = padding_mask.repeat_interleave(
                    self.expansion_factor, dim=1
                )
        self.transformer.to(device=context_vector.device, dtype=context_vector.dtype)

        # ---- time span ----
        t_span = torch.linspace(
            0, 1, num_steps, device=context_vector.device, dtype=context_vector.dtype
        )
        # t_span = t_lin**gamma

        # ---- ODE ----
        def fn(t, state):
            features = self.cfg_forward(
                times=t,
                state=state,
                cfg_scale=cfg_scale,
                context_vector=context_vector,
                attention_mask=~padding_mask,
            )
            return features

        odeint_kwargs = dict(atol=1e-5, rtol=1e-5, method="euler")
        trajectory = odeint(fn, y0, t_span, **odeint_kwargs)

        generated_latents = trajectory[-1]

        return generated_latents.view(B, -1, self.mel_channels)

    def cfg_forward(
        self,
        times: torch.FloatTensor,
        state: torch.FloatTensor,
        cfg_scale: float,
        context_vector: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
    ):
        times = times.repeat(state.shape[0])
        cond_state = (
            self.noise_proj(state)
            if self.learned_prior
            else self.noise_proj(torch.cat([context_vector, state], dim=-1))
        )
        cond_out = self.transformer(
            x=cond_state,
            times=times,
            attention_mask=attention_mask,
        )
        if cfg_scale == 1.0:
            return cond_out
        uncond_state = (
            self.noise_proj(
                torch.cat([torch.zeros_like(context_vector), state], dim=-1)
            )
            if not self.learned_prior
            else self.noise_proj(torch.randn_like(state))
        )
        uncond_out = self.transformer(
            x=uncond_state,
            times=times,
            attention_mask=attention_mask,
        )

        final = (cfg_scale * cond_out + (1 - cfg_scale) * uncond_out).to(
            context_vector.dtype
        )

        return final
