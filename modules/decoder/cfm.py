import torch
import random
import torch.nn as nn
from typing import Optional
from einops import rearrange
from torchdiffeq import odeint
from torch.nn import functional as F
from ..configs import DiTConfig
from .dit import Transformer
from ..output_dataclasses import DecoderOutput


class DiT(torch.nn.Module):
    def __init__(
        self,
        config: DiTConfig,
    ):
        super().__init__()

        # inherit from config
        self.sigma = config.sigma
        self.mel_dim = config.mel_dim
        self.dit_dim = config.dit_dim
        self.dit_heads = config.dit_heads
        self.dit_depth = config.dit_depth
        self.use_conv_layer = config.use_conv_layer
        self.audio_latent_dim = config.audio_latent_dim
        self.expansion_factor = config.expansion_factor
        self.uncond_prob = config.uncond_prob
        self.is_causal = config.is_causal
        self.use_window_attention = config.use_window_attention
        self.use_group_bidirectional = config.use_group_bidirectional
        mel_fps = 93.75  # 24000 / 256 #FIXME hardcoded to 24kHz dataset
        self.window_size = (
            int(config.window_attention_seconds * mel_fps)
            if config.use_window_attention
            else None
        )
        print(f"VAE is_causal: {self.is_causal}")
        if self.window_size is not None:
            print(
                f"VAE window_attention: {config.window_attention_seconds}s = {self.window_size} mel frames"
            )
        if self.use_group_bidirectional:
            print(
                f"VAE group_bidirectional: enabled (group_size will be set to expansion_factor)"
            )

        # context vector projection
        self.context_vector_proj = nn.Linear(self.audio_latent_dim, self.dit_dim)

        # noise projection
        self.noise_proj = nn.Sequential(
            nn.Linear(self.dit_dim + self.mel_dim, self.dit_dim),
            nn.LayerNorm(self.dit_dim),
        )

        # transformer
        # TODO add a dedicated config for Transformer
        self.transformer = Transformer(
            dim=self.dit_dim,
            depth=self.dit_depth,
            out_dim=self.mel_dim,
            heads=self.dit_heads,
            use_conv_layer=self.use_conv_layer,
            is_causal=self.is_causal,
            attn_flash=True,
            window_size=self.window_size,
        )

    def handle_context_vector(
        self,
        context_vector: torch.FloatTensor,
        target: Optional[torch.FloatTensor] = None,
        temperature: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
    ):

        # Original behavior: uniform repeat_interleave
        context_vector = context_vector.repeat_interleave(self.expansion_factor, dim=1)
        if padding_mask is not None:
            padding_mask = padding_mask.repeat_interleave(self.expansion_factor, dim=1)
        if target is not None:
            min_length = min(context_vector.shape[1], target.shape[1])
            context_vector = context_vector[:, :min_length, :]
            target = target[:, :min_length, :]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :min_length]

        temperature = temperature or 1.0
        x0 = (
            torch.randn(
                context_vector.shape[0],
                context_vector.shape[1],
                self.mel_dim,
                dtype=context_vector.dtype,
                device=context_vector.device,
                generator=generator,
            )
            * temperature
        )
        return self.context_vector_proj(context_vector), x0, padding_mask

    def prepare_flow(
        self,
        target: torch.FloatTensor,
        context_vector: torch.FloatTensor,
        x0: torch.FloatTensor,
    ):
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
        x0 = torch.randn_like(target).to(context_vector.dtype)
        # w is the noise signal that is transformed by the flow
        w = ((1 - (1 - self.sigma) * t) * x0 + t * target).to(context_vector.dtype)
        # target is the original signal minus the noise
        target = (target - (1 - self.sigma) * x0).to(context_vector.dtype)
        state = self.noise_proj(torch.cat([context_vector, w], dim=-1))
        return state, times, target

    @property
    def _group_size(self) -> Optional[int]:
        if (
            self.use_group_bidirectional
            and self.expansion_factor
            and self.expansion_factor > 1
        ):
            return self.expansion_factor
        return None

    def let_it_flow(
        self,
        times: torch.FloatTensor,
        state: torch.FloatTensor,
        target: torch.FloatTensor,
        flow_mask: torch.BoolTensor,
    ):
        mask_to_loss = ~flow_mask
        v = self.transformer(
            x=state,
            times=times,
            attention_mask=mask_to_loss,
            group_size=self._group_size,
        )

        v_to_loss = v[mask_to_loss].view(-1, self.mel_dim)
        target_to_loss = target[mask_to_loss].view(-1, self.mel_dim)
        loss = ((v_to_loss - target_to_loss) ** 2).mean()

        return loss

    def forward(
        self,
        target: torch.FloatTensor,
        target_padding_mask: torch.BoolTensor,
        context_vector: torch.FloatTensor,
        **kwargs,
    ):
        context_vector, prior, _ = self.handle_context_vector(
            context_vector, target=target
        )

        # ---- flow ----
        state, times, v_target = self.prepare_flow(
            target=target,
            context_vector=context_vector,
            x0=prior,
        )

        # ---- get the flow ----
        loss = self.let_it_flow(
            times=times, state=state, target=v_target, flow_mask=target_padding_mask
        )

        return DecoderOutput(
            loss=loss,
        )

    def generate(
        self,
        num_steps: int,
        context_vector: torch.FloatTensor,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        cfg_scale = guidance_scale
        # ---- context vector z ----
        context_vector, y0, upsampled_padding_mask = self.handle_context_vector(
            context_vector,
            temperature=temperature,
            generator=generator,
            padding_mask=padding_mask,
        )
        B, T = context_vector.shape[:2]

        # ---- time span ----
        t_span = torch.linspace(
            0, 1, num_steps, device=context_vector.device, dtype=context_vector.dtype
        )

        # ---- ODE ----
        def fn(t, state):
            features = self.cfg_forward(
                times=t,
                state=state,
                cfg_scale=cfg_scale,
                context_vector=context_vector,
                attention_mask=~upsampled_padding_mask,
            )
            return features

        odeint_kwargs = dict(atol=1e-5, rtol=1e-5, method="midpoint")
        trajectory = odeint(fn, y0, t_span, **odeint_kwargs)

        generated_latents = trajectory[-1]

        return DecoderOutput(
            audio_features=generated_latents.view(B, -1, self.mel_dim),
            padding_mask=upsampled_padding_mask,
        )

    def cfg_forward(
        self,
        times: torch.FloatTensor,
        state: torch.FloatTensor,
        cfg_scale: float,
        context_vector: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
    ):
        times = times.repeat(state.shape[0])
        cond_state = self.noise_proj(torch.cat([context_vector, state], dim=-1))
        gs = self._group_size
        cond_out = self.transformer(
            x=cond_state,
            times=times,
            attention_mask=attention_mask,
            group_size=gs,
        )
        if cfg_scale == 1.0:
            return cond_out

        uncond_state = self.noise_proj(
            torch.cat([torch.zeros_like(context_vector), state], dim=-1)
        )
        uncond_out = self.transformer(
            x=uncond_state,
            times=times,
            attention_mask=attention_mask,
            group_size=gs,
        )

        final = (cfg_scale * cond_out + (1 - cfg_scale) * uncond_out).to(
            context_vector.dtype
        )

        return final

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
