import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import safetensors
from typing import Optional
from einops import rearrange
from .cfm import DiT, DiTConfig
from dataclasses import dataclass, asdict
from .semantic_module import SeamlessM4Tv2Encoder
from .semantic_mapper import SemanticMapperConfig, Z2YMapper
from .Encoder import ConvformerEncoderConfig, ConvformerEncoder
from .melspecEncoder import MelSpectrogramEncoder, MelSpectrogramConfig


@dataclass
class VAEOutput:
    audio_loss: torch.Tensor
    kl_loss: torch.Tensor
    mu_mean: Optional[torch.Tensor] = None
    mu_var: Optional[torch.Tensor] = None
    align_loss: Optional[torch.Tensor] = None


@dataclass
class VAEConfig:
    """Config for VAE model - needed for DeepSpeed compatibility"""

    encoder_config: ConvformerEncoderConfig
    decoder_config: DiTConfig
    mel_spec_config: MelSpectrogramConfig
    semantic_mapper_config: Optional[SemanticMapperConfig] = None
    add_semantic_distillation: bool = False
    add_semantic_mapper: bool = False

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
            "add_semantic_distillation": self.add_semantic_distillation,
            "add_semantic_mapper": self.add_semantic_mapper,
        }


class VAE(torch.nn.Module):
    def __init__(self, config: VAEConfig, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.decoder = DiT(config.decoder_config)
        # if config.encoder_config.compress_factor_C < 4:
        #     print(f"WARNING: compress_factor_C is less than 4, but the minimum downsampling factor is 4")
        #     print(f"Setting compress_factor_C to 4")
        #     config.encoder_config.compress_factor_C = 4
        self.encoder = ConvformerEncoder(config.encoder_config)
        self.wav2mel = MelSpectrogramEncoder(config.mel_spec_config)
        if config.add_semantic_distillation:
            self.semantic_module = SeamlessM4Tv2Encoder(dtype=dtype)
        if config.add_semantic_mapper:
            self.semantic_mapper = Z2YMapper(config.semantic_mapper_config)
        self.decoder.expansion_factor = config.encoder_config.compress_factor_C
        self.dtype = dtype
        self.set_dtype(dtype)

    def set_dtype(self, dtype: torch.dtype):
        self.dtype = dtype
        self.decoder.to(dtype=dtype)
        self.encoder.to(dtype=dtype)
        self.wav2mel.to(dtype=dtype)
        if self.config.add_semantic_distillation:
            self.semantic_module.set_dtype(dtype=dtype)
        if self.config.add_semantic_mapper:
            self.semantic_mapper.to(dtype=dtype)

    def set_device(self, device: torch.device):
        self.decoder.to(device=device)
        self.encoder.to(device=device)
        self.wav2mel.to(device=device)
        if self.config.add_semantic_distillation:
            self.semantic_module.set_device(device=device)
        if self.config.add_semantic_mapper:
            self.semantic_mapper.to(device=device)

    def from_pretrained(self, checkpoint_path: str):
        state_dict = safetensors.torch.load_file(checkpoint_path)
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    def repeat_tokens(self, z, durations):
        return torch.nn.utils.rnn.pad_sequence(
            [
                torch.repeat_interleave(z[i], d, dim=0)
                for i, d in enumerate(durations.round().long())
            ],
            batch_first=True,
        )

    def forward(self, audios_srs, **kwargs):
        encoded_audios = self.wav2mel(audios_srs)
        
        # Ensure input to encoder matches model dtype
        x = encoded_audios.audio_features.to(dtype=self.dtype)

        convformer_output = self.encoder(
            x=x,
            padding_mask=encoded_audios.padding_mask,
            step=kwargs.get("training_step", None),
            phonemes=kwargs.get("phonemes", None),
        )

        upsampled_z = self.repeat_tokens(
            convformer_output.z,
            convformer_output.durations * self.config.encoder_config.compress_factor_C,
        )
        breakpoint()

        audio_loss = self.decoder(
            target=x,
            target_padding_mask=encoded_audios.padding_mask,
            context_vector=upsampled_z,
        ).loss

        mu_mean = convformer_output.z[~convformer_output.padding_mask].mean()
        mu_var = convformer_output.z[~convformer_output.padding_mask].var()
        return VAEOutput(
            audio_loss=audio_loss,
            kl_loss=convformer_output.kl_loss,
            align_loss=convformer_output.align_loss,
            mu_mean=mu_mean,
            mu_var=mu_var,
        )

    @torch.no_grad()
    def denormalize_mel(self, mel: torch.Tensor):
        return mel * self.wav2mel.std + self.wav2mel.mean

    @torch.no_grad()
    def normalize_mel(self, mel: torch.Tensor):
        return (mel - self.wav2mel.mean) / self.wav2mel.std
    
    @torch.no_grad()
    def encode_and_sample(
        self,
        audios_srs,
        num_steps: int = 50,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        hubert_guidance: Optional[torch.Tensor] = None,
        phonemes: Optional[list] = None,
    ):
        """
        Encode audio to latent space and generate mel spectrogram.
        If transcriptions are provided, CTC boundaries are computed and returned for visualization.
        """

        # Encode audio to mel spectrogram
        encoded_audios = self.wav2mel(audios_srs)
        original_mel = encoded_audios.audio_features.to(dtype=self.dtype)

        convformer_output = self.encoder(
            x=original_mel,
            padding_mask=encoded_audios.padding_mask,
            step=None,
            phonemes=phonemes
        )

        upsampled_z = self.repeat_tokens(
            convformer_output.z,
            convformer_output.durations * self.config.encoder_config.compress_factor_C,
        )

        # Generate mel spectrogram from latent
        reconstructed_mel = self.decoder.generate(
            num_steps=num_steps,
            context_vector=upsampled_z,
            temperature=temperature,
            guidance_scale=guidance_scale,
            generator=generator,
            padding_mask=encoded_audios.padding_mask,
        )
        if self.config.mel_spec_config.normalize:
            original_mel = self.denormalize_mel(original_mel)
            reconstructed_mel = self.denormalize_mel(reconstructed_mel)

        result = {
            "original_mel": original_mel,
            "reconstructed_mel": reconstructed_mel,
            "durations": (convformer_output.durations * self.config.encoder_config.compress_factor_C).long()    ,
            "padding_mask": encoded_audios.padding_mask,
        }
        return result
