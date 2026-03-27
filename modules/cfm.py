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
from modules.similarity import SimilarityUpsamplerBatch


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
    use_window_attention: bool = False
    window_attention_seconds: float = 3.0
    use_group_bidirectional: bool = False
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
        self.use_window_attention = config.use_window_attention
        self.use_group_bidirectional = config.use_group_bidirectional
        mel_fps = 93.75  # 24000 / 256
        self.window_size = int(config.window_attention_seconds * mel_fps) if config.use_window_attention else None
        print(f"VAE is_causal: {self.is_causal}")
        if self.window_size is not None:
            print(f"VAE window_attention: {config.window_attention_seconds}s = {self.window_size} mel frames")
        if self.use_group_bidirectional:
            print(f"VAE group_bidirectional: enabled (group_size will be set to expansion_factor)")
        # context vector projection
        self.context_vector_proj = nn.Sequential(
            nn.Linear(self.audio_latent_dim, self.unet_dim),
            #nn.LayerNorm(self.unet_dim)
        )
        # noise projection
        if self.learned_prior:
            self.noise_proj = nn.Sequential(nn.Linear(self.unet_dim, self.unet_dim), nn.LayerNorm(self.unet_dim))
            self.prior_proj = nn.Sequential(
                nn.Linear(self.audio_latent_dim, self.mel_channels), nn.LayerNorm(self.mel_channels)
            )

        else:
            self.noise_proj = nn.Sequential(
                nn.Linear(self.unet_dim + self.mel_channels, self.unet_dim), nn.LayerNorm(self.unet_dim)
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
            window_size=self.window_size,
        )
        self.transformer.to(dtype=torch.bfloat16)
        self.upsampler = SimilarityUpsamplerBatch()
        

    def reparameterize(self, mu: torch.FloatTensor, std: Optional[float] = None) -> torch.FloatTensor:
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
        durations: Optional[torch.LongTensor] = None,
        target: Optional[torch.FloatTensor] = None,
        temperature: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        std: Optional[float] = None,
    ):
        # Use durations if provided (similarity pooling case)
        upsampled_padding_mask = None
        if durations is not None:
            # Durations from similarity pooling are in latent-frame units.
            # Convert them to mel-frame units for CFM timeline restoration.
            expanded_durations = (durations * self.expansion_factor).long()
            # Compute target length from target or from durations
            if target is not None:
                target_T = torch.tensor(
                    [target.shape[1]] * target.shape[0],
                    device=context_vector.device,
                    dtype=torch.long,
                )
            else:
                target_T = expanded_durations.sum(dim=1)
            
            context_vector, upsampled_mask = self.upsampler(
                context_vector, expanded_durations, target_T=target_T
            )
            # Convert mask from 0/1 to bool (0=valid, 1=padding -> False=valid, True=padding)
            upsampled_padding_mask = upsampled_mask.bool()
            
            # Truncate to match target if needed
            if target is not None:
                min_length = min(context_vector.shape[1], target.shape[1])
                context_vector = context_vector[:, :min_length, :]
                target = target[:, :min_length, :]
                upsampled_padding_mask = upsampled_padding_mask[:, :min_length]
        else:
            # Original behavior: uniform repeat_interleave
            context_vector = context_vector.repeat_interleave(self.expansion_factor, dim=1)
            if target is not None:
                min_length = min(context_vector.shape[1], target.shape[1])
                context_vector = context_vector[:, :min_length, :]
                target = target[:, :min_length, :]

        if self.learned_prior:
            raise ValueError("Learned prior is not supported")
        else:
            temperature = temperature or 1.0
            prior = (
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
        return self.context_vector_proj(context_vector), prior, upsampled_padding_mask

    def prepare_flow(
        self,
        target: torch.FloatTensor,
        context_vector: torch.FloatTensor,
        x0: torch.FloatTensor,
    ):
        if random.random() < self.uncond_prob:
            if self.learned_prior:
                context_vector = torch.randn_like(context_vector)
            else:
                context_vector = torch.zeros_like(context_vector)
        # We need times
        times = torch.rand(
            (target.shape[0],),
            dtype=context_vector.dtype,
            device=context_vector.device,
        )
        t = rearrange(times, "b -> b 1 1")
        # Now we need noise, x0 sampled from a normal distribution
        x0 = torch.randn_like(target).to(context_vector.dtype)
        # w is the noise signal that is transformed by the flow
        w = ((1 - (1 - self.sigma) * t) * x0 + t * target).to(context_vector.dtype)
        # target is the original signal minus the noise
        target = (target - (1 - self.sigma) * x0).to(context_vector.dtype)
        state = self.noise_proj(torch.cat([context_vector, w], dim=-1))
        return state, times, target

    @property
    def _group_size(self) -> Optional[int]:
        if self.use_group_bidirectional and self.expansion_factor and self.expansion_factor > 1:
            return self.expansion_factor
        return None

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
            group_size=self._group_size,
        )
        loss = None
        if target is not None:
            v_to_loss = v[mask_to_loss].view(-1, self.mel_channels)
            target_to_loss = target[mask_to_loss].view(-1, self.mel_channels)
            loss = ((v_to_loss - target_to_loss) ** 2).mean()
            loss = loss.to(v.dtype)

        return loss

    def forward(
        self,
        target: torch.FloatTensor,
        target_padding_mask: torch.BoolTensor,
        context_vector: torch.FloatTensor,
        durations: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        context_vector, prior, _ = self.handle_context_vector(
            context_vector, durations=durations, target=target
        )

        # ---- flow ----
        state, times, v_target = self.prepare_flow(
            target=target,
            context_vector=context_vector,
            x0=prior,
        )

        # ---- get the flow ----
        loss = self.let_it_flow(times=times, state=state, target=v_target, flow_mask=target_padding_mask)

        return DiTOutput(
            loss=loss,
        )

    def generate(
        self,
        num_steps: int,
        context_vector: torch.FloatTensor,
        durations: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        std: float = 0.0,
    ):
        cfg_scale = guidance_scale
        # ---- context vector z ----
        context_vector, y0, upsampled_padding_mask = self.handle_context_vector(
            context_vector, durations=durations, temperature=temperature, generator=generator, std=std
        )
        B, T = context_vector.shape[:2]
        
        # Use upsampled padding mask if durations were provided
        if upsampled_padding_mask is not None:
            padding_mask = upsampled_padding_mask
        elif padding_mask is None:
            if B == 1:
                padding_mask = torch.zeros(
                    (1, T),
                    device=context_vector.device,
                    dtype=torch.bool,
                )
            else:
                raise ValueError("Padding mask is required for batch size > 1")
        else:
            # Only repeat_interleave if we didn't use durations
            if durations is None:
                padding_mask = padding_mask.repeat_interleave(self.expansion_factor, dim=1)
        
        # Ensure padding_mask matches the temporal dimension of y0 and context_vector
        if padding_mask.shape[1] != y0.shape[1]:
            # Adjust mask to match y0 length
            if padding_mask.shape[1] < y0.shape[1]:
                # Pad mask with False (valid)
                padding_extension = torch.zeros(
                    (B, y0.shape[1] - padding_mask.shape[1]),
                    device=padding_mask.device,
                    dtype=padding_mask.dtype,
                )
                padding_mask = torch.cat([padding_mask, padding_extension], dim=1)
            else:
                # Truncate mask
                padding_mask = padding_mask[:, :y0.shape[1]]
                
        self.transformer.to(device=context_vector.device, dtype=context_vector.dtype)

        # ---- time span ----
        t_span = torch.linspace(0, 1, num_steps, device=context_vector.device, dtype=context_vector.dtype)
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

        odeint_kwargs = dict(atol=1e-5, rtol=1e-5, method="midpoint")
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
            self.noise_proj(context_vector)
            if self.learned_prior
            else self.noise_proj(torch.cat([context_vector, state], dim=-1))
        )
        gs = self._group_size
        cond_out = self.transformer(
            x=cond_state,
            times=times,
            attention_mask=attention_mask,
            group_size=gs,
        )
        if cfg_scale == 1.0:
            return cond_out
        uncond_state = (
            self.noise_proj(torch.cat([torch.zeros_like(context_vector), state], dim=-1))
            if not self.learned_prior
            else self.noise_proj(torch.zeros_like(context_vector))
        )
        uncond_out = self.transformer(
            x=uncond_state,
            times=times,
            attention_mask=attention_mask,
            group_size=gs,
        )

        final = (cfg_scale * cond_out + (1 - cfg_scale) * uncond_out).to(context_vector.dtype)

        return final
