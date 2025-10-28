import torch
import random
import torch.nn as nn
from abc import ABCMeta
from typing import Optional
from einops import rearrange
from torchdiffeq import odeint
from dataclasses import dataclass
from torch.nn import functional as F
from transformers import PretrainedConfig
from modules.Transformer import Transformer
from melspecEncoder import MelSpectrogramEncoder, MelSpectrogramConfig
from Encoder import ConvformerEncoder, ConvformerEncoderConfig


def build_audio_encoder():
    config = MelSpectrogramConfig()
    return MelSpectrogramEncoder(config=config)


@dataclass
class LightCFMTalkingHeadOutput:
    audio_loss: Optional[torch.Tensor] = None
    kl_loss: Optional[torch.Tensor] = None


class LightCFMTalkingHeadConfig:
    model_type = "cfm"

    def __init__(
        self,
        tempformer_dim: int = 2048,
        unet_dim: int = 512,
        unet_depth: int = 6,
        unet_heads: int = 8,
        unet_dropout_rate: float = 0.0,
        audio_latent_dim: int = 512,
        use_conv_layer: bool = False,
        use_sos_token: bool = False,
        sigma: float = 1e-5,
        encoder_config: ConvformerEncoderConfig = None,
    ):
        """
        tempformer_dim: int - Hidden dimension of the Temporal Transformer
        depformer_dim: int - Hidden dimension of the Depth Transformer
        depformer_dim_feedforward: int - Feedforward dimension of the Depth Transformer
        depformer_num_heads: int - Number of heads in the Depth Transformer
        depformer_num_layers: int - Number of layers in the Depth Transformer
        depformer_casual: bool - Whether the Depth Transformer is causal
        """

        self.tempformer_dim = tempformer_dim
        # unet specific
        self.sigma = sigma
        self.unet_dim = unet_dim
        self.unet_heads = unet_heads
        self.unet_depth = unet_depth
        self.use_sos_token = use_sos_token
        self.use_conv_layer = use_conv_layer
        self.audio_latent_dim = audio_latent_dim
        self.unet_dropout_rate = unet_dropout_rate


class LightTalkingHead(torch.nn.Module):
    _supports_flash_attn_2 = False
    _supports_sdpa = False

    def __init__(
        self,
        config: LightCFMTalkingHeadConfig,
        encoder_config: ConvformerEncoderConfig,
    ):
        super().__init__()

        # encoder
        self.encoder = ConvformerEncoder(encoder_config)


class LightCFMTalkingHead(LightTalkingHead):
    def __init__(
        self,
        config: LightCFMTalkingHeadConfig,
        encoder_config: ConvformerEncoderConfig,
    ):
        super().__init__(config, encoder_config)
        # audio encoder
        self.wav2mel = build_audio_encoder()

        # inherit from config
        self.sigma = config.sigma
        self.unet_dim = config.unet_dim
        self.unet_heads = config.unet_heads
        self.unet_depth = config.unet_depth
        self.use_sos_token = config.use_sos_token
        self.tempformer_dim = config.tempformer_dim
        self.use_conv_layer = config.use_conv_layer
        self.audio_latent_dim = config.audio_latent_dim
        self.depformer_expansion_factor = encoder_config.compress_factor_C

        # context vector projection
        self.context_vector_proj = nn.Linear(self.audio_latent_dim, self.unet_dim)
        # noise projection
        self.noise_proj = nn.Linear(self.unet_dim + self.audio_latent_dim, self.unet_dim)

        # transformer
        self.transformer = Transformer(
            dim=self.unet_dim,
            depth=self.unet_depth,
            heads=self.unet_heads,
            use_conv_layer=self.use_conv_layer,
            audio_latent_dim=self.audio_latent_dim,
        )

    def prepare_flow(
        self,
        target: torch.FloatTensor,
        context_vector: torch.FloatTensor,
    ):
        if context_vector.shape[1] != target.shape[1]:
            context_vector = context_vector[:, : target.shape[1], :]
        # We need times
        times = torch.rand(
            (B,),
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
        loss_first_frame = None
        mask_to_loss = ~flow_mask
        if target is not None:
            # if self.depformer_expansion_factor > 1:
            v_reshaped = v.view(original_batch_size, -1, self.audio_latent_dim)
            target_reshaped = target.view(original_batch_size, -1, self.audio_latent_dim)
            flow_mask_reshaped = mask_to_loss.view(original_batch_size, -1)
            v_to_loss = v_reshaped[flow_mask_reshaped].view(-1, self.audio_latent_dim)
            target = target_reshaped[flow_mask_reshaped].view(-1, self.audio_latent_dim)
            loss = F.mse_loss(v_to_loss, target)

        return loss

    def forward(
        self,
        audios_srs,
        **kwargs,
    ):
        encoded_audios = self.wav2mel(audios_srs)
        context_vector, kl_loss = self.encoder(encoded_audios.audio_features)
        context_vector = context_vector.repeat_interleave(self.depformer_expansion_factor, dim=1)

        target, target_padding_mask = (
            encoded_audios.audio_features,
            encoded_audios.padding_mask,
        )
        B, T = target.shape[:2]

        # ---- flow ----
        state, target = self.prepare_flow(
            target=target,
            context_vector=context_vector,
        )

        # ---- get the flow ----
        loss = self.let_it_flow(times=times, state=state, target=target, flow_mask=target_padding_mask)

        return LightCFMTalkingHeadOutput(
            audio_loss=loss,
            kl_loss=kl_loss,
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
        B, T = context_vector.shape[:2]
        y0 = (
            torch.randn(
                (
                    int(B * T),
                    self.depformer_expansion_factor,
                    self.audio_latent_dim,
                ),
                device=context_vector.device,
                dtype=context_vector.dtype,
                generator=generator,
            )
            * temperature
        )  # .contiguous()
        # ---- context vector z ----
        context_vector = self.handle_context_vector(context_vector)
        if self.use_sos_token:
            raise NotImplementedError("SOS token is not implemented")
            self.sos_token.to(context_vector.dtype)
            sos_token = self.sos_token.unsqueeze(0).repeat(B, 1, 1)
            context_vector[:, 0, :] += sos_token
        # context_vector = context_vector.repeat_interleave(
        #     self.depformer_expansion_factor, dim=1
        # )
        context_vector = context_vector.view(B * T, self.depformer_expansion_factor, self.unet_dim)

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

        return generated_latents.view(B, -1, self.audio_latent_dim)

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

        final = (cfg_scale * cond_out[:1] + (1 - cfg_scale) * cond_out[1:]).to(context_vector.dtype)

        return torch.cat([final, final], dim=0)
