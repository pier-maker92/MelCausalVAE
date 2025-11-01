import torch
from .cfm import DiT, DiTConfig
from dataclasses import dataclass, asdict
from .Encoder import ConvformerEncoderConfig, ConvformerEncoder
from .melspecEncoder import MelSpectrogramEncoder, MelSpectrogramConfig
from typing import Optional
import safetensors


@dataclass
class VAEOutput:
    audio_loss: torch.Tensor
    kl_loss: torch.Tensor


@dataclass
class VAEConfig:
    """Config for VAE model - needed for DeepSpeed compatibility"""

    encoder_config: ConvformerEncoderConfig
    decoder_config: DiTConfig
    mel_spec_config: MelSpectrogramConfig

    @property
    def hidden_size(self):
        """Return hidden size for DeepSpeed compatibility"""
        return 512

    def to_dict(self):
        """Convert config to dict for W&B logging compatibility"""
        return {
            "model_type": "VAE",
            "encoder_config": asdict(self.encoder_config),
            "decoder_config": asdict(self.decoder_config),
            "mel_spec_config": asdict(self.mel_spec_config),
        }


class VAE(torch.nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.decoder = DiT(config.decoder_config)
        self.encoder = ConvformerEncoder(config.encoder_config)
        self.wav2mel = MelSpectrogramEncoder(config.mel_spec_config)
        self.decoder.expansion_factor = config.encoder_config.compress_factor_C

    def set_device(self, device: torch.device):
        self.decoder.to(device=device)
        self.encoder.to(device=device)
        self.wav2mel.to(device=device)

    def set_dtype(self, dtype: torch.dtype):
        self.decoder.to(dtype=dtype)
        self.encoder.to(dtype=dtype)
        self.wav2mel.to(dtype=dtype)

    def from_pretrained(self, checkpoint_path: str):
        state_dict = safetensors.torch.load_file(checkpoint_path)
        self.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")

    def forward(self, audios_srs, **kwargs):
        encoded_audios = self.wav2mel(audios_srs)
        context_vector, kl_loss = self.encoder(
            x=encoded_audios.audio_features,
            padding_mask=encoded_audios.padding_mask,
            step=kwargs.get("training_step", None),
        )
        audio_loss = self.decoder(
            target=encoded_audios.audio_features,
            target_padding_mask=encoded_audios.padding_mask,
            context_vector=context_vector,
        ).loss

        return VAEOutput(
            audio_loss=audio_loss,
            kl_loss=kl_loss,
        )

    @torch.no_grad()
    def encode(self, audios_srs):
        encoded_audios = self.wav2mel(audios_srs)
        context_vector, _ = self.encoder(
            x=encoded_audios.audio_features,
            padding_mask=encoded_audios.padding_mask,
            step=None,
        )
        return context_vector

    @torch.no_grad()
    def encode_and_sample(
        self,
        audios_srs,
        num_steps: int = 50,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Encode audio to latent space and generate mel spectrogram.
        """

        # Encode audio to mel spectrogram
        encoded_audios = self.wav2mel(audios_srs)
        original_mel = encoded_audios.audio_features

        # Encode to latent space
        context_vector, _ = self.encoder(
            x=original_mel,
            padding_mask=encoded_audios.padding_mask,
            step=None,
        )

        # Generate mel spectrogram from latent
        reconstructed_mel = self.decoder.generate(
            num_steps=num_steps,
            context_vector=context_vector,
            temperature=temperature,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if self.config.mel_spec_config.normalize:
            original_mel = original_mel * self.wav2mel.std + self.wav2mel.mean
            reconstructed_mel = reconstructed_mel * self.wav2mel.std + self.wav2mel.mean

        return {
            "original_mel": original_mel,
            "reconstructed_mel": reconstructed_mel,
            "context_vector": context_vector,
            "padding_mask": encoded_audios.padding_mask,
        }
