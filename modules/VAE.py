import torch
import safetensors
from typing import Optional
from .cfm import DiT, DiTConfig
from dataclasses import dataclass, asdict
from .semantic_module import SeamlessM4Tv2Encoder
from .semantic_mapper import SemanticMapperConfig, Z2YMapper
from .Encoder import ConvformerEncoderConfig, ConvformerEncoder
from .encoder_1d import ConvformerEncoder1d
from .melspecEncoder import MelSpectrogramEncoder, MelSpectrogramConfig
from .ConvformerDecoder import ConvformerDecoder, ConvformerDecoderConfig

def count_parameters_by_module(model, title: str):
    table_width = 65
    print("=" * table_width)
    print(f"{title:^{table_width}}")
    print("-" * table_width)
    print(f"{'Module':<30} | {'Total Params':>15} | {'Trainable':>12}")
    print("-" * table_width)
    
    total_p = 0
    trainable_p = 0
    
    for name, module in model.named_children():
        t = sum(p.numel() for p in module.parameters())
        tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:<30} | {t:15,d} | {tr:12,d}")
        total_p += t
        trainable_p += tr
        
    print("-" * table_width)
    print(f"{'GRAND TOTAL':<30} | {total_p:15,d} | {trainable_p:12,d}")
    print("-" * table_width + "\n")

@dataclass
class VAEOutput:
    audio_loss: torch.Tensor
    kl_loss: torch.Tensor
    semantic_loss: Optional[torch.Tensor] = None
    mu_mean: Optional[torch.Tensor] = None
    mu_var: Optional[torch.Tensor] = None


@dataclass
class VAEConfig:
    """Config for VAE model - needed for DeepSpeed compatibility"""

    encoder_config: ConvformerEncoderConfig
    decoder_config: DiTConfig
    mel_spec_config: MelSpectrogramConfig
    add_semantic_distillation: bool = False
    add_semantic_mapper: bool = False
    use_classic_decoder: bool = False

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
            "use_classic_decoder": self.use_classic_decoder,
        }


class VAE(torch.nn.Module):
    _keys_to_ignore_on_save = None
    
    def __init__(self, config: VAEConfig, dtype: torch.dtype):
        super().__init__()
        self.config = config
        if config.use_classic_decoder:
            classic_dec_cfg = ConvformerDecoderConfig.from_encoder_config(config.encoder_config)
            self.decoder = ConvformerDecoder(classic_dec_cfg)
        else:
            self.decoder = DiT(config.decoder_config)
            self.decoder.expansion_factor = config.encoder_config.compress_factor_C
        if config.encoder_config.use_1d_encoder:
            self.encoder = ConvformerEncoder1d(config.encoder_config)
        else:
            self.encoder = ConvformerEncoder(config.encoder_config)
        self.wav2mel = MelSpectrogramEncoder(config.mel_spec_config)
        if config.add_semantic_distillation:
            self.semantic_module = SeamlessM4Tv2Encoder(dtype=dtype)
        if config.add_semantic_mapper:
            self.semantic_mapper = Z2YMapper(config.semantic_mapper_config)
        self.dtype = dtype
        self.set_dtype(dtype)
        count_parameters_by_module(self.encoder, "Encoder")
        count_parameters_by_module(self.decoder, "Decoder")

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

    def forward(self, audios_srs, **kwargs):
        encoded_audios = self.wav2mel(audios_srs)
        #corrupted_encoded_audios = self.wav2mel(kwargs.get("corrupted_audios_srs"))
        semantic_output = None
        if self.config.add_semantic_distillation:
            semantic_output = self.semantic_module(audios_srs)
        features = encoded_audios.audio_features.to(self.dtype)
        convformer_output = self.encoder(
            x=features,
            padding_mask=encoded_audios.padding_mask,
            step=kwargs.get("training_step", None),
            semantic_guidance=semantic_output,
        )
        audio_loss = self.decoder(
            target=encoded_audios.audio_features,
            target_padding_mask=encoded_audios.padding_mask,
            context_vector=convformer_output.z,
        ).loss
        mu_mean = convformer_output.mu[~convformer_output.padding_mask].mean()
        mu_var = convformer_output.mu[~convformer_output.padding_mask].var()
        return {
            "audio_loss": audio_loss,
            "kl_loss": convformer_output.kl_loss,
            "semantic_loss": None,  # convformer_output.semantic_loss * 0.1,
            "mu_mean": mu_mean,
            "mu_var": mu_var,
        }

    @torch.no_grad()
    def denormalize_mel(self, mel: torch.Tensor):
        if not self.config.mel_spec_config.normalize:
            return mel
        return mel * self.wav2mel.std + self.wav2mel.mean

    @torch.no_grad()
    def normalize_mel(self, mel: torch.Tensor):
        if not self.config.mel_spec_config.normalize:
            return mel
        return (mel - self.wav2mel.mean) / self.wav2mel.std

    @torch.no_grad()
    def encode(self, audios_srs, return_original_mel: bool = False):
        encoded_audios = self.wav2mel(audios_srs)
        convformer_output = self.encoder(
            x=encoded_audios.audio_features.to(self.dtype),
            padding_mask=encoded_audios.padding_mask,
            step=None,
        )
        if self.config.add_semantic_mapper:
            convformer_output.semantic_features = self.semantic_mapper(convformer_output.mu).y.to(
                convformer_output.mu.dtype
            )
        if not return_original_mel:
            return convformer_output
        else:
            encoded_audios.audio_features = self.denormalize_mel(encoded_audios.audio_features)
            return convformer_output, encoded_audios

    @torch.no_grad()
    def sample(
        self,
        num_steps: int = 4,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        z: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        std: float = 1.0,
    ):
        """
        Sample from the VAE.
        """
        # if z is not None:
        #     context_vector = z
        # elif µ is not None:
        #     context_vector = self.encoder.reparameterize(µ)
        # else:
        #     raise ValueError("Either z or µ must be provided")

        reconstructed_mel = self.decoder.generate(
            num_steps=num_steps,
            generator=generator,
            temperature=temperature,
            padding_mask=padding_mask,
            context_vector=z,
            guidance_scale=guidance_scale,
            std=std,
        )
        if self.config.mel_spec_config.normalize:
            reconstructed_mel = self.denormalize_mel(reconstructed_mel)
        return reconstructed_mel

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
        original_mel = encoded_audios.audio_features.to(self.dtype)

        # Encode to latent space
        convformer_output = self.encoder(
            x=original_mel,
            padding_mask=encoded_audios.padding_mask,
            step=None,
        )

        # Generate mel spectrogram from latent
        reconstructed_mel = self.decoder.generate(
            num_steps=num_steps,
            context_vector=convformer_output.z,  # z
            temperature=temperature,
            guidance_scale=guidance_scale,
            generator=generator,
            padding_mask=convformer_output.padding_mask,
        )
        if self.config.mel_spec_config.normalize:
            original_mel = self.denormalize_mel(original_mel)
            reconstructed_mel = self.denormalize_mel(reconstructed_mel)

        return {
            "original_mel": original_mel,
            "reconstructed_mel": reconstructed_mel,
            "context_vector": convformer_output.z,
            "padding_mask": encoded_audios.padding_mask,
        }
