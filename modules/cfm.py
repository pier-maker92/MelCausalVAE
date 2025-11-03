import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        # context vector projection
        self.context_vector_proj = nn.Linear(self.audio_latent_dim, self.unet_dim)
        # noise projection
        self.noise_proj = nn.Linear(self.unet_dim + self.mel_channels, self.unet_dim)

        # transformer
        self.transformer = Transformer(
            dim=self.unet_dim,
            depth=self.unet_depth,
            heads=self.unet_heads,
            use_conv_layer=self.use_conv_layer,
            audio_latent_dim=self.mel_channels,  # projection to mel
        )

    def handle_context_vector(self, context_vector: torch.FloatTensor):
        context_vector = self.context_vector_proj(context_vector)
        context_vector = context_vector.repeat_interleave(self.expansion_factor, dim=1)
        return context_vector

    def prepare_flow(
        self,
        target: torch.FloatTensor,
        context_vector: torch.FloatTensor,
    ):
        if context_vector.shape[1] != target.shape[1]:
            context_vector = context_vector[:, : target.shape[1], :]
        if random.random() < self.uncond_prob:
            context_vector = torch.zeros_like(context_vector)
        # We need times
        times = torch.rand(
            (target.shape[0],),
            dtype=context_vector.dtype,
            device=context_vector.device,
        )
        t = rearrange(times, "b -> b 1 1")
        # Now we need noise, x0 sampled from a normal distribution
        x0 = torch.randn_like(target)
        # w is the noise signal that is transformed by the flow
        w = (1 - (1 - self.sigma) * t) * x0 + t * target
        # target is the original signal minus the noise
        target = target - (1 - self.sigma) * x0

        state = self.noise_proj(torch.cat([context_vector, w], dim=-1))
        return state, times

    def let_it_flow(
        self,
        times: torch.FloatTensor,
        state: torch.FloatTensor,
        target: Optional[torch.FloatTensor] = None,
        flow_mask: Optional[torch.BoolTensor] = None,
        original_batch_size: Optional[int] = None,
    ):
        v = self.transformer(
            x=state,
            times=times,
        )
        loss = None
        mask_to_loss = ~flow_mask
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
        context_vector = self.handle_context_vector(context_vector)

        # ---- flow ----
        state, times = self.prepare_flow(
            target=target,
            context_vector=context_vector,
        )

        # ---- get the flow ----
        loss = self.let_it_flow(times=times, state=state, target=target, flow_mask=target_padding_mask)

        return DiTOutput(
            loss=loss,
        )

    def generate(
        self,
        num_steps: int,
        context_vector: torch.FloatTensor,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ):
        cfg_scale = guidance_scale
        # ---- context vector z ----
        context_vector = self.handle_context_vector(context_vector)
        B, T = context_vector.shape[:2]
        y0 = (
            torch.randn(
                (
                    B,
                    T,
                    self.mel_channels,
                ),
                device=context_vector.device,
                dtype=context_vector.dtype,
                generator=generator,
            )
            * temperature
        )  # .contiguous()
        self.transformer.to(device=context_vector.device, dtype=context_vector.dtype)

        # ---- time span ----
        t_span = torch.linspace(
            0,
            1,
            num_steps,
            device=context_vector.device,
            dtype=context_vector.dtype,
        )

        # ---- ODE ----
        def fn(t, state):
            features = self.cfg_forward(
                times=t,
                state=state,
                cfg_scale=cfg_scale,
                context_vector=context_vector,
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
    ):
        times = times.repeat(state.shape[0])
        cond_state = self.noise_proj(torch.cat([context_vector, state], dim=-1))
        cond_out = self.transformer(
            x=cond_state,
            times=times,
        )
        if cfg_scale == 1.0:
            return cond_out

        uncond_state = self.noise_proj(torch.cat([torch.zeros_like(context_vector), state], dim=-1))
        uncond_out = self.transformer(
            x=uncond_state,
            times=times,
        )

        final = (cfg_scale * cond_out + (1 - cfg_scale) * uncond_out).to(context_vector.dtype)

        return final
