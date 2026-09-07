"""Small framewise conditional flow head, following hybrid_tts's MLP CFM."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .configs import DiffusionConfig
from .output_dataclasses import DiffusionOutput


class AdaptiveResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(conditioning).chunk(3, dim=-1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class DiffusionHead(nn.Module):
    def __init__(self, latent_dim: int, context_dim: int, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.frequencies = nn.Parameter(torch.randn(config.time_dim // 2))
        self.time_projection = nn.Linear(config.time_dim, config.hidden_dim)
        self.context_projection = nn.Linear(context_dim, config.hidden_dim)
        self.input_projection = nn.Linear(latent_dim, config.hidden_dim)
        self.blocks = nn.ModuleList([
            AdaptiveResidualBlock(config.hidden_dim) for _ in range(config.layers)
        ])
        self.output = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, latent_dim))

    def velocity(self, state: torch.Tensor, times: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        angles = times.unsqueeze(-1) * self.frequencies * (2 * math.pi)
        time_embedding = self.time_projection(torch.cat((angles.sin(), angles.cos()), dim=-1))
        conditioning = self.context_projection(context) + time_embedding
        x = self.input_projection(state)
        for block in self.blocks:
            x = block(x, conditioning)
        return self.output(x)

    def forward(self, target: torch.Tensor, context: torch.Tensor, valid: torch.Tensor) -> DiffusionOutput:
        # Operate only on valid shifted pairs. The head never mixes time positions.
        target, context = target[valid].detach(), context[valid]
        noise = torch.randn_like(target)
        times = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        t = times.unsqueeze(-1)
        noise_scale = 1 - self.config.sigma_min
        state = (1 - noise_scale * t) * noise + t * target
        target_velocity = target - noise_scale * noise
        velocity = self.velocity(state, times, context)
        return DiffusionOutput(F.mse_loss(velocity, target_velocity), velocity, target_velocity)

    @torch.no_grad()
    def sample(self, context: torch.Tensor, steps: int | None = None,
               temperature: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        steps = self.config.sampling_steps if steps is None else steps
        temperature = self.config.temperature if temperature is None else temperature
        if steps <= 0 or temperature < 0:
            raise ValueError("steps must be positive and temperature nonnegative.")
        state = torch.randn(*context.shape[:-1], self.latent_dim, device=context.device,
                            dtype=context.dtype, generator=generator) * temperature
        dt = 1 / steps
        for step in range(steps):
            times = context.new_full(context.shape[:-1], step * dt)
            velocity = self.velocity(state, times, context)
            midpoint = state + 0.5 * dt * velocity
            state = state + dt * self.velocity(midpoint, times + 0.5 * dt, context)
        return state
