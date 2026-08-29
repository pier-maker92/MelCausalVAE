import logging
import torch
import torch.nn.functional as F
import safetensors.torch
from typing import Optional
from .decoder.cfm import DiT
from .configs import DicodecConfig
from .encoder.encoder import Encoder
from .utils import count_parameters_by_module
from .feature_extractor import FeatureExtractor, WavLMFeatureExtractor
from .speaker_encoder import WavLMSpeakerEncoder
from .output_dataclasses import DicodecOutput, DecoderOutput, FeatureExtractorOutput
from .lp_filter import LowPassFilter
from .output_dataclasses import AttributesOutput
from vocos import Vocos

logger = logging.getLogger(__name__)


class Dicodec(torch.nn.Module):
    _keys_to_ignore_on_save = None

    def __init__(self, config: DicodecConfig, **kwargs):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(config.mel_spectrogram_config)

        self.wavlm, self.wavlm_extractor, self.speaker_encoder = None, None, None
        if config.wavlm_module_config is not None:
            from transformers import WavLMModel

            model_name = config.wavlm_module_config.pretrained_model_name
            self.wavlm = WavLMModel.from_pretrained(
                model_name,
                use_safetensors=False,
            )
            self._freeze_wavlm()
            if config.wavlm_module_config.feature_extractor_config:
                self.wavlm_extractor = WavLMFeatureExtractor(
                    config.wavlm_module_config.feature_extractor_config,
                    wavlm=self.wavlm,
                )
            if config.wavlm_module_config.speaker_encoder_config:
                self.speaker_encoder = WavLMSpeakerEncoder(
                    config.wavlm_module_config.speaker_encoder_config,
                    wavlm=self.wavlm,
                )

        self.encoder = Encoder(config.encoder_config)
        self.decoder = DiT(config.decoder_config)
        self.lowpass_filter = LowPassFilter(
            cutoff_hz=config.lowpass_filter_config.cutoff_hz,
            sample_rate=config.lowpass_filter_config.sample_rate,
            order=config.lowpass_filter_config.order,
        )

        self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.vocoder.eval()
        for param in self.vocoder.parameters():
            param.requires_grad = False

        count_parameters_by_module(self.encoder, "Encoder")
        count_parameters_by_module(self.decoder, "Decoder")

    def get_feature_extractor(self):
        return self.feature_extractor

    def get_wavlm_extractor(self):
        return self.wavlm_extractor

    def get_speaker_encoder(self):
        return self.speaker_encoder

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_vocoder(self):
        return self.vocoder

    def _freeze_wavlm(self):
        if self.wavlm is None:
            return
        self.wavlm.eval()
        for parameter in self.wavlm.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_wavlm()
        return self

    def from_pretrained(self, checkpoint_path: str):
        import os

        if os.path.isdir(checkpoint_path):
            checkpoint_file = os.path.join(checkpoint_path, "model.safetensors")
        else:
            checkpoint_file = checkpoint_path

        state_dict = safetensors.torch.load_file(
            checkpoint_file, device=str(self.device)
        )
        print(f"Safetensors file loaded to {self.device}. Applying state dict...")
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_file}")

    @torch.no_grad()
    def extract_wavlm_features(
        self,
        target_length: int,
        audios_srs,
        audio_16khz=None,
        extractor: Optional[WavLMFeatureExtractor] = None,
    ):
        extractor = self.wavlm_extractor if extractor is None else extractor
        if extractor is None:
            return None, None

        wavlm_output = extractor(audios_srs, audio_16khz=audio_16khz)
        wavlm_features = wavlm_output.audio_features.to(self.dtype)
        wavlm_features = wavlm_features.repeat_interleave(2, dim=1)
        wavlm_features = (
            F.interpolate(
                wavlm_features.float().transpose(1, 2),
                size=target_length,
                mode="linear",
                align_corners=False,
            )
            .transpose(1, 2)
            .to(wavlm_features.dtype)
        )
        wavlm_padding_mask = (
            F.interpolate(
                wavlm_output.padding_mask.float().unsqueeze(1),
                size=target_length,
                mode="nearest",
            )
            .squeeze(1)
            .bool()
        )
        return wavlm_features, wavlm_padding_mask

    @torch.no_grad()
    def extract_features(self, audios_srs, target_audios_srs=None, **kwargs):
        target_audios_srs = (
            audios_srs if target_audios_srs is None else target_audios_srs
        )
        audio_16khz = kwargs.get("audio_16khz")

        target_output = self.feature_extractor(target_audios_srs)
        target_features = target_output.audio_features.to(self.dtype)
        target_padding_mask = target_output.padding_mask
        target_length = target_features.shape[1]

        wavlm_features, wavlm_padding_mask = self.extract_wavlm_features(
            target_length=target_length,
            audios_srs=audios_srs,
            audio_16khz=audio_16khz,
        )
        if wavlm_features is not None:
            return (
                wavlm_features,
                wavlm_padding_mask,
                target_features,
                target_padding_mask,
            )

        encoder_output = self.feature_extractor(audios_srs)
        return (
            encoder_output.audio_features.to(self.dtype),
            encoder_output.padding_mask,
            target_features,
            target_padding_mask,
        )

    def encode(self, features, padding_mask, **kwargs):
        encoder_output = self.encoder(
            x=features,
            padding_mask=padding_mask,
            step=kwargs.get("training_step", None),
        )
        return encoder_output

    @torch.no_grad()
    def encode_attributes(self, z, padding_mask=None, plot: bool = False):

        z_original = z

        if padding_mask is not None:
            if padding_mask.shape != z.shape[:2]:
                raise ValueError(
                    "padding_mask must have shape [batch, time], got "
                    f"{tuple(padding_mask.shape)} for z shape {tuple(z.shape)}."
                )
            valid_mask = (
                (~padding_mask).to(device=z.device, dtype=z.dtype).unsqueeze(-1)
            )
            valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            valid_mask = None
            valid_count = z.new_full((z.shape[0], 1, 1), z.shape[1])

        if valid_mask is None:
            z_mean = z.mean(dim=1, keepdim=True)
        else:
            z_mean = (z * valid_mask).sum(dim=1, keepdim=True) / valid_count

        z_centered = z - z_mean
        if valid_mask is not None:
            z_centered = z_centered * valid_mask

        z_lp = self.lowpass_filter(z_centered, valid_mask=valid_mask)
        z_hp = z_centered - z_lp

        dot_product = torch.sum(
            z_centered * z_lp,
            dim=1,
            keepdim=True,
        )

        norm_sq = torch.sum(
            z_lp.square(),
            dim=1,
            keepdim=True,
        )

        beta = dot_product / (norm_sq + 1e-8)

        z_pros = beta * z_lp
        z_res_centered = z_centered - z_pros
        if valid_mask is not None:
            z_hp = z_hp * valid_mask
            z_pros = z_pros * valid_mask
            z_res_centered = z_res_centered * valid_mask

        if plot:
            import matplotlib.pyplot as plt

            res_scale = 1.0
            z_reconstructed = z_pros + res_scale * z_res_centered + z_mean
            plt.figure(figsize=(14, 6))

            plt.plot(
                z_original[0, :, 0].detach().cpu().numpy(),
                label="z original",
                alpha=0.5,
            )

            plt.plot(
                z_centered[0, :, 0].detach().cpu().numpy(),
                label="z centered",
                alpha=0.5,
            )

            plt.plot(
                z_lp[0, :, 0].detach().cpu().numpy(),
                label="z_lp",
                alpha=0.5,
            )

            plt.plot(
                z_hp[0, :, 0].detach().cpu().numpy(),
                label="z_hp",
                linewidth=2,
            )

            plt.plot(
                z_pros[0, :, 0].detach().cpu().numpy(),
                label="z_pros (projected low-pass)",
                linewidth=2,
            )

            plt.plot(
                z_res_centered[0, :, 0].detach().cpu().numpy(),
                label="z_sem (high-pass)",
                alpha=0.8,
            )

            plt.plot(
                z_reconstructed[0, :, 0].detach().cpu().numpy(),
                label=f"output scale={res_scale}",
                linewidth=2,
            )

            plt.axhline(0, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig("plot_dc_removed_projection.png", dpi=150)
            plt.close()

        return AttributesOutput(
            z_sem=z_res_centered,
            z_pros=z_pros,
            z_mean=z_mean,
            z_lp=z_lp,
            z_hp=z_hp,
        )

    def extract_speaker_embedding(self, audios_srs):
        if self.speaker_encoder is None:
            return None
        self.speaker_encoder = self.speaker_encoder.to(device=self.device)
        return self.speaker_encoder(audios_srs).to(device=self.device, dtype=self.dtype)

    def decode(
        self,
        z: Optional[torch.Tensor],
        target_features: Optional[torch.Tensor],
        target_padding_mask: Optional[torch.BoolTensor],
        speaker_embedding: Optional[torch.FloatTensor] = None,
    ):
        decoder_output = self.decoder(
            target=target_features,
            target_padding_mask=target_padding_mask,
            context_vector=z,
            speaker_embedding=speaker_embedding,
        )
        return decoder_output

    def forward(self, audios_srs, **kwargs):
        # extract features
        (
            enc_features,
            enc_padding_mask,
            dec_features,
            dec_padding_mask,
        ) = self.extract_features(
            audios_srs,
            target_audios_srs=audios_srs,
            **kwargs,
        )
        # encode to latent space
        encoder_output = self.encode(enc_features, enc_padding_mask, **kwargs)
        speaker_embedding = kwargs.get("speaker_embedding")
        if speaker_embedding is None:
            speaker_embedding = self.extract_speaker_embedding(audios_srs)

        # decode from latent space
        decoder_output = self.decode(
            z=encoder_output.z,
            target_features=dec_features,
            target_padding_mask=dec_padding_mask,
            speaker_embedding=(
                speaker_embedding
                if speaker_embedding is not None
                else getattr(encoder_output, "speaker_embedding", None)
            ),
        )
        audio_loss = decoder_output.loss

        mu_mean = encoder_output.mu[
            ~encoder_output.padding_mask
        ].mean()  # whatever is not quantized
        mu_var = encoder_output.mu[
            ~encoder_output.padding_mask
        ].var()  # whatever is not quantized
        out = {
            "audio_loss": audio_loss,
            "kl_loss": encoder_output.kl_loss,
            "vq_loss": encoder_output.vq_loss,
            "vq_perplexity": encoder_output.vq_perplexity,
            "vq_codebook_usage": encoder_output.vq_codebook_usage,
            "mu_mean": mu_mean,
            "mu_var": mu_var,
        }

        return DicodecOutput(**out)

    @torch.no_grad()
    def denormalize_mel(self, mel: torch.Tensor):
        if not self.config.mel_spectrogram_config.normalize:
            return mel
        return mel * self.feature_extractor.std + self.feature_extractor.mean

    @torch.no_grad()
    def normalize_mel(self, mel: torch.Tensor):
        if not self.config.mel_spectrogram_config.normalize:
            return mel
        return (mel - self.feature_extractor.mean) / self.feature_extractor.std

    def sample(
        self,
        num_steps: int = 4,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        z: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        speaker_embedding: Optional[torch.FloatTensor] = None,
        guide_only_speaker: bool = False,
        **kwargs,
    ):
        decoder_output = self.decoder.generate(
            num_steps=num_steps,
            generator=generator,
            temperature=temperature,
            padding_mask=padding_mask,
            context_vector=z,
            guidance_scale=guidance_scale,
            speaker_embedding=speaker_embedding,
            guide_only_speaker=guide_only_speaker,
        )
        reconstructed_mel = decoder_output.audio_features
        reconstructed_padding_mask = decoder_output.padding_mask
        if self.config.mel_spectrogram_config.normalize:
            reconstructed_mel = self.denormalize_mel(reconstructed_mel)
        return reconstructed_mel, reconstructed_padding_mask

    @torch.no_grad()
    def apply_kmeans(self, z, padding_mask=None, chunk_size=16384):
        if not hasattr(self, "kmeans_codebook") or self.kmeans_codebook is None:
            return z

        attrs = self.encode_attributes(z, padding_mask=padding_mask)
        z_sem = attrs.z_sem

        centroids = self.kmeans_codebook["centroids"].to(device=z.device, dtype=z.dtype)

        if padding_mask is not None:
            valid_mask = ~padding_mask
            X = z_sem[valid_mask]
        else:
            X = z_sem.view(-1, z_sem.shape[-1])

        distances = []
        for i in range(0, X.shape[0], chunk_size):
            chunk = X[i : i + chunk_size]
            dist = torch.cdist(chunk, centroids)
            distances.append(dist)
        distances = torch.cat(distances, dim=0)

        indices = torch.argmin(distances, dim=-1)
        quantized = centroids[indices]

        z_sem_q = torch.zeros_like(z_sem)
        if padding_mask is not None:
            z_sem_q[valid_mask] = quantized
        else:
            z_sem_q = quantized.view(*z_sem.shape)

        # Reconstruct z with quantized semantic part
        z_reconstructed = z_sem_q + attrs.z_pros + attrs.z_mean
        breakpoint()
        return z_reconstructed

    @torch.no_grad()
    def apply_external_quantizer(
        self,
        z,
        padding_mask=None,
        quantizer_name: Optional[str] = None,
        residual: bool = False,
    ):
        quantizer = None
        external_quantizers = getattr(self, "external_quantizers", {}) or {}
        if quantizer_name is not None:
            quantizer_str = str(quantizer_name).strip()
            quantizer = external_quantizers.get(quantizer_str)
            if quantizer is None:
                for k, v in external_quantizers.items():
                    if (
                        k == quantizer_str
                        or k == f"z_sem_vq_ema_{quantizer_str}"
                        or k.endswith(f"_{quantizer_str}")
                    ):
                        quantizer = v
                        break
            if (
                quantizer is None
                and getattr(self, "external_quantizer", None) is not None
            ):
                ext_name = getattr(self, "external_quantizer_name", "")
                if quantizer_str in ext_name or ext_name.endswith(f"_{quantizer_str}"):
                    quantizer = self.external_quantizer
            if quantizer is None:
                available = ", ".join(sorted(external_quantizers)) or "<none>"
                raise ValueError(
                    f"Unknown external quantizer '{quantizer_name}'. Available: {available}"
                )
        else:
            quantizer = getattr(self, "external_quantizer", None)

        if quantizer is None:
            raise ValueError(
                "Quantized z_sem inference requested, but no external quantizer is attached to the model."
            )
        quantizer.eval()

        quantizer = quantizer.to(device=z.device, dtype=z.dtype)
        attrs = self.encode_attributes(z, padding_mask=padding_mask)
        vq_out = quantizer(attrs.z_sem, padding_mask=padding_mask)
        z_sem = attrs.z_sem - vq_out.quantized if residual else vq_out.quantized
        return z_sem + attrs.z_pros + attrs.z_mean

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
        enc_features, enc_padding_mask, dec_features, dec_padding_mask = (
            self.extract_features(
                audios_srs,
                target_audios_srs=audios_srs,
                **kwargs,
            )
        )
        encoder_output = self.encode(enc_features, enc_padding_mask, **kwargs)

        # speaker embedding
        speaker_embedding = kwargs.get("speaker_embedding")
        if speaker_embedding is None:
            speaker_embedding = self.extract_speaker_embedding(audios_srs)
        if kwargs.get("zero_speaker", False) and speaker_embedding is not None:
            speaker_embedding = torch.zeros_like(speaker_embedding)

        z = encoder_output.z

        # if kwargs.get("quantized", False) or kwargs.get("residual", False):
        #     z = self.apply_external_quantizer(
        #         z,
        #         padding_mask=encoder_output.padding_mask,
        #         quantizer_name=kwargs.get("quantizer_name"),
        #         residual=kwargs.get("residual", False),
        #     )
        # # Apply kmeans if loaded
        # elif hasattr(self, "kmeans_codebook") and self.kmeans_codebook is not None:
        #     z = self.apply_kmeans(z, padding_mask=encoder_output.padding_mask)

        reconstructed_mel, reconstructed_padding_mask = self.sample(
            num_steps=num_steps,
            temperature=temperature,
            guidance_scale=guidance_scale,
            z=z,
            generator=generator,
            padding_mask=encoder_output.padding_mask,
            speaker_embedding=speaker_embedding,
            guide_only_speaker=kwargs.get("guide_only_speaker", False),
        )

        # Vocode the mel spectrogram
        # reconstructed_mel is [B, T, F] -> [B, F, T] for vocos
        audio = self.vocoder.decode(reconstructed_mel.permute(0, 2, 1))
        # normalize waveform
        audio = audio / (audio.abs().max(dim=-1, keepdim=True)[0] + 1e-8)

        if self.config.mel_spectrogram_config.normalize:
            dec_features = self.denormalize_mel(dec_features)

        return {
            "decoder_output": DecoderOutput(
                audio_features=reconstructed_mel,
                padding_mask=reconstructed_padding_mask,
            ),
            "encoder_output": encoder_output,
            "feature_extractor_output": FeatureExtractorOutput(
                audio_features=dec_features,
                padding_mask=dec_padding_mask,
            ),
            "audio_waveform": audio,
        }

    @property
    def dtype(self):
        # WavLM is frozen fp32 and registered first, so skip it
        return next(self.encoder.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
