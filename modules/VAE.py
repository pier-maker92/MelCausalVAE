import torch
import safetensors
from typing import Optional
from .configs import VAEConfig
from .feature_extractor import FeatureExtractor
from .encoder.encoder import Encoder
from .decoder.cfm import DiT
from .utils import count_parameters_by_module


class VAE(torch.nn.Module):
    _keys_to_ignore_on_save = None

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(config.mel_spec_config)
        self.encoder = Encoder(config.encoder_config)
        self.decoder = DiT(config.decoder_config)

        count_parameters_by_module(self.encoder, "Encoder")
        count_parameters_by_module(self.decoder, "Decoder")

    def from_pretrained(self, checkpoint_path: str):
        state_dict = safetensors.torch.load_file(checkpoint_path)
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    @torch.no_grad()
    def extract_features(self, audios_srs, **kwargs):
        features_extractor_output = self.feature_extractor(audios_srs)
        features = features_extractor_output.audio_features
        padding_mask = features_extractor_output.padding_mask
        return features, padding_mask

    def encode(self, features, padding_mask, **kwargs):
        encoder_output = self.encoder(
            x=features,
            padding_mask=padding_mask,
            step=kwargs.get("training_step", None),
        )
        return encoder_output

    def forward(self, audios_srs, **kwargs):
        features, padding_mask = self.extract_features(audios_srs, **kwargs)
        encoder_output = self.encode(features, padding_mask, **kwargs)
        audio_loss = self.decoder(
            target=features,
            target_padding_mask=padding_mask,
            context_vector=encoder_output.z,
        ).loss
        mu_mean = encoder_output.mu[~encoder_output.padding_mask].mean()
        mu_var = encoder_output.mu[~encoder_output.padding_mask].var()
        vq_loss = getattr(encoder_output, "vq_loss", None)
        return {
            "audio_loss": audio_loss,
            "kl_loss": encoder_output.kl_loss,
            "vq_loss": vq_loss,
            "vq_perplexity": getattr(encoder_output, "vq_perplexity", None),
            "vq_codes_used": getattr(encoder_output, "vq_codes_used", None),
            "vq_codes_used_frac": getattr(encoder_output, "vq_codes_used_frac", None),
            "mu_mean": mu_mean,
            "mu_var": mu_var,
        }

    @torch.no_grad()
    def denormalize_mel(self, mel: torch.Tensor):
        if not self.config.mel_spec_config.normalize:
            return mel
        return mel * self.feature_extractor.std + self.feature_extractor.mean

    @torch.no_grad()
    def normalize_mel(self, mel: torch.Tensor):
        if not self.config.mel_spec_config.normalize:
            return mel
        return (mel - self.feature_extractor.mean) / self.feature_extractor.std

    def sample(
        self,
        num_steps: int = 4,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        z: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        assert z or mu, "Either z or mu must be provided"
        context_vector = z or mu
        decoder_output = self.decoder.generate(
            num_steps=num_steps,
            generator=generator,
            temperature=temperature,
            padding_mask=padding_mask,
            context_vector=context_vector,
            guidance_scale=guidance_scale,
        )
        reconstructed_mel = decoder_output.audio_features
        reconstructed_padding_mask = decoder_output.padding_mask
        if self.config.mel_spec_config.normalize:
            reconstructed_mel = self.denormalize_mel(reconstructed_mel)
        return reconstructed_mel, reconstructed_padding_mask

    @torch.no_grad()
    def encode_decode(
        self,
        audios_srs,
        num_steps: int = 50,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        """
        Encode audio to latent space and generate mel spectrogram.
        """

        # Encode audio to mel spectrogram
        features, padding_mask = self.extract_features(audios_srs, **kwargs)
        encoder_output = self.encode(features, padding_mask, **kwargs)

        reconstructed_mel, reconstructed_padding_mask = self.sample(
            num_steps=num_steps,
            temperature=temperature,
            guidance_scale=guidance_scale,
            z=encoder_output.z,
            generator=generator,
            padding_mask=padding_mask,
        )
        if self.config.mel_spec_config.normalize:
            features = self.denormalize_mel(features)

        return {
            "original_mel": features,
            "reconstructed_mel": reconstructed_mel,
            "context_vector": encoder_output.z,
            "padding_mask": reconstructed_padding_mask,
            "original_padding_mask": padding_mask,
        }

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
