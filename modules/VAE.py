import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import safetensors
from typing import Optional
from einops import rearrange
from .cfm import DiT, DiTConfig
from dataclasses import dataclass, asdict
from .Encoder import ConvformerEncoderConfig, ConvformerEncoder
from .melspecEncoder import MelSpectrogramEncoder, MelSpectrogramConfig
from .decoder_standard_vae import DecoderVAE, DecoderConfig


def count_parameters_by_module(model):
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:15s} total={total:12,}  trainable={trainable:12,}")


def durations_to_imv(durations, T_audio, text_mask):
    """
    Args:
        durations: [B, T_text]
        T_audio: int (lunghezza massima del batch audio)
        text_mask: [B, T_text] (1 per fonemi reali, 0 per padding)
    """
    # 1. Calcoliamo i confini
    boundaries = torch.cumsum(durations, dim=-1)  # [B, T_text]

    # 2. Griglia temporale
    q = torch.arange(T_audio, device=durations.device).view(1, -1)  # [1, T_audio]

    # 3. IMV grezzo
    # Confrontiamo ogni frame con ogni boundary: [B, T_audio, T_text]
    imv = (q.unsqueeze(-1) >= boundaries.unsqueeze(1)).sum(dim=-1).float()

    # 4. FIX PADDING: Ogni sequenza deve fermarsi al suo text_length - 1
    text_lengths = text_mask.sum(dim=-1).long()
    max_indices = (text_lengths - 1).view(-1, 1).expand(-1, T_audio)

    return torch.min(imv, max_indices.float())


def soft_upsampling_with_durations(
    z_phonemes, durations, mel_mask, text_mask, delta_base=10.0
):
    """
    Args:
        z_phonemes: [B, T_text, D]
        durations: [B, T_text]
        mel_mask: [B, T_audio] (maschera dei frame audio reali)
        text_mask: [B, T_text] (maschera dei fonemi reali)
    """
    B, T_text, D = z_phonemes.shape
    T_audio = mel_mask.size(1)
    device = z_phonemes.device

    # 1. IMV coerente con le lunghezze reali
    imv = durations_to_imv(durations, T_audio, text_mask)  # [B, T_audio]

    # 2. Matrice Gamma
    p = torch.arange(T_text, device=device).view(1, 1, T_text)
    dist_sq = (imv.unsqueeze(-1) - p) ** 2
    energies = -delta_base * dist_sq

    # 3. MASCHERA TESTO: Impediamo l'accesso ai fonemi di padding
    # Fondamentale per non avere pesi distribuiti sul nulla alla fine della sequenza
    energies = energies.masked_fill(text_mask.unsqueeze(1) == 0, -1e9)

    gamma = torch.softmax(energies, dim=-1)  # [B, T_audio, T_text]

    # 4. Ricostruzione
    upsampled = torch.bmm(gamma.to(z_phonemes.dtype), z_phonemes)

    # 5. MASCHERA AUDIO: Azzeriamo i frame di padding ricostruiti
    return upsampled * mel_mask.unsqueeze(-1).to(upsampled.dtype)


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
    use_aligner: bool = False

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
            "use_aligner": self.use_aligner,
        }


class VAE(torch.nn.Module):
    def __init__(self, config: VAEConfig, dtype: torch.dtype):
        super().__init__()
        self.config = config
        if config.decoder_config.decoder_type == "dit":
            self.decoder = DiT(config.decoder_config)
        elif config.decoder_config.decoder_type == "vae":
            self.decoder = DecoderVAE(config.decoder_config)
        else:
            raise ValueError("Decoder type not supported")
        self.encoder = ConvformerEncoder(config.encoder_config)
        self.wav2mel = MelSpectrogramEncoder(config.mel_spec_config)
        self.decoder.expansion_factor = config.encoder_config.compress_factor_C
        self.dtype = dtype
        self.set_dtype(dtype)
        count_parameters_by_module(self.encoder)

    def set_dtype(self, dtype: torch.dtype):
        self.dtype = dtype
        self.decoder.to(dtype=dtype)
        self.encoder.to(dtype=dtype)
        self.wav2mel.to(dtype=dtype)

    def set_device(self, device: torch.device):
        self.decoder.to(device=device)
        self.encoder.to(device=device)
        self.wav2mel.to(device=device)

    def from_pretrained(self, checkpoint_path: str):
        state_dict = safetensors.torch.load_file(checkpoint_path)
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    def forward(self, audios_srs, **kwargs):
        encoded_audios = self.wav2mel(audios_srs)

        # Ensure input to encoder matches model dtype
        x = encoded_audios.audio_features.to(dtype=self.dtype)

        convformer_output = self.encoder(
            x=x,
            padding_mask=encoded_audios.padding_mask,
            step=kwargs.get("training_step", None),
            phonemes=kwargs.get("phonemes", None),
            transcription=kwargs.get("transcription", None),
            audios_srs=audios_srs,
        )
        audio_loss = self.decoder(
            target=x,
            target_padding_mask=convformer_output.upsampled_padding_mask,
            context_vector=convformer_output.z_upsampled,
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
        **kwargs,
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
            phonemes=phonemes,
            inference=True,
            audios_srs=audios_srs,
            transcription=kwargs.get("transcription", None),
        )
        durations = convformer_output.durations
        # Generate mel spectrogram from latent
        reconstructed_mel = self.decoder.generate(
            num_steps=num_steps,
            context_vector=convformer_output.z_upsampled,
            temperature=temperature,
            guidance_scale=guidance_scale,
            generator=generator,
            padding_mask=convformer_output.upsampled_padding_mask,
        )
        if self.config.mel_spec_config.normalize:
            original_mel = self.denormalize_mel(original_mel)
            reconstructed_mel = self.denormalize_mel(reconstructed_mel)

        result = {
            "original_mel": original_mel,
            "reconstructed_mel": reconstructed_mel,
            "durations": durations,
            "padding_mask": encoded_audios.padding_mask,
            "segment_labels": convformer_output.segment_labels,
        }
        return result
