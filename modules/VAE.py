import logging
import torch
import torch.nn.functional as F
import safetensors.torch
from typing import Optional
from .decoder.cfm import DiT
from .configs import VAEConfig
from .encoder.encoder import Encoder
from .utils import count_parameters_by_module
from .feature_extractor import FeatureExtractor, WavLMFeatureExtractor
from .output_dataclasses import VAEOutput, DecoderOutput, FeatureExtractorOutput

logger = logging.getLogger(__name__)


class VAE(torch.nn.Module):
    _keys_to_ignore_on_save = None

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(config.mel_spectrogram_config)

        self.wavlm_extractor = None
        if getattr(config, "wavlm_config", None) is not None:
            self.wavlm_extractor = WavLMFeatureExtractor(config.wavlm_config)

        self.distill_wavlm_extractor = None
        sem_cfg = getattr(config.encoder_config, "semantic_distillation_config", None)
        if sem_cfg is not None:
            from .configs import WavLMConfig

            self.distill_wavlm_extractor = WavLMFeatureExtractor(
                WavLMConfig(layer=sem_cfg.wavlm_layer)
            )
            dim_to_quantize = (
                config.encoder_config.vq_config.dim_to_quantize
                if config.encoder_config.vq_config
                else config.encoder_config.latent_dim
            )
            self.distill_proj_head = torch.nn.Linear(dim_to_quantize, 1024)

        self.encoder = Encoder(config.encoder_config)
        self.decoder = DiT(config.decoder_config)

        self.semantic_downsample_factor = getattr(config.encoder_config, "semantic_downsample_factor", 1)
        if self.semantic_downsample_factor > 1:
            from .encoder.utils import TimeCausalConv1d
            qd = config.encoder_config.vq_config.dim_to_quantize if config.encoder_config.vq_config else config.encoder_config.latent_dim
            # Upsampler: casual Conv1d. We will use repeat_interleave and then a causal Conv1d for smoothing, 
            # or just a causal Conv1d with stride=1 to smooth after repeat.
            self.semantic_upsampler_conv = TimeCausalConv1d(
                qd, qd, k=self.semantic_downsample_factor * 2, d=1, s=1
            )

        count_parameters_by_module(self.encoder, "Encoder")
        count_parameters_by_module(self.decoder, "Decoder")

    def semantic_upsample(self, z_semantic):
        if self.semantic_downsample_factor <= 1 or z_semantic is None:
            return z_semantic
        # z_semantic is [B, T/factor, qd]
        z_semantic = z_semantic.repeat_interleave(self.semantic_downsample_factor, dim=1)
        z_semantic = z_semantic.transpose(1, 2)
        z_semantic = self.semantic_upsampler_conv(z_semantic)
        z_semantic = z_semantic.transpose(1, 2)
        return z_semantic

    def from_pretrained(self, checkpoint_path: str):
        state_dict = safetensors.torch.load_file(
            checkpoint_path, device=str(self.device)
        )
        print(f"Safetensors file loaded to {self.device}. Applying state dict...")
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    @torch.no_grad()
    def extract_features(self, audios_srs, **kwargs):
        features_extractor_output = self.feature_extractor(audios_srs)
        features = features_extractor_output.audio_features.to(self.dtype)
        padding_mask = features_extractor_output.padding_mask

        distill_features = None
        if self.distill_wavlm_extractor is not None:
            distill_features = self.distill_wavlm_extractor(audios_srs).audio_features.to(self.dtype)

        if self.wavlm_extractor is not None:
            wavlm_output = self.wavlm_extractor(audios_srs)
            wavlm_feats = wavlm_output.audio_features.to(self.dtype)  # [B, T_w, 1024]
            T_mel = features.shape[1]
            # Causal upsample ×2 (repeat), then interpolate to exact mel length
            wavlm_feats = wavlm_feats.repeat_interleave(2, dim=1)    # [B, 2*T_w, 1024]
            wavlm_feats = F.interpolate(
                wavlm_feats.float().transpose(1, 2),
                size=T_mel,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).to(wavlm_feats.dtype)                   # [B, T_mel, 1024]
            enc_padding_mask = (
                F.interpolate(
                    wavlm_output.padding_mask.float().unsqueeze(1),
                    size=T_mel,
                    mode="nearest",
                )
                .squeeze(1)
                .bool()
            )
            return (
                wavlm_feats,
                enc_padding_mask,
                features,
                padding_mask,
                distill_features,
            )

        return features, padding_mask, features, padding_mask, distill_features

    def encode(self, features, padding_mask, **kwargs):
        encoder_output = self.encoder(
            x=features,
            padding_mask=padding_mask,
            step=kwargs.get("training_step", None),
        )
        return encoder_output

    def decode(
        self,
        z_semantic: Optional[torch.Tensor],
        z_acoustic: Optional[torch.Tensor],
        target_features: Optional[torch.Tensor],
        target_padding_mask: Optional[torch.BoolTensor],
        speaker_embedding: Optional[torch.FloatTensor] = None,
    ):
        z_semantic = self.semantic_upsample(z_semantic)
        if z_semantic is not None and z_acoustic is not None:
            # We must ensure they have exactly the same length if padding or rounding caused differences
            min_len = min(z_semantic.shape[1], z_acoustic.shape[1])
            z_semantic = z_semantic[:, :min_len, :]
            z_acoustic = z_acoustic[:, :min_len, :]
            z = torch.cat([z_semantic, z_acoustic], dim=-1)
        else:
            z = z_acoustic if z_semantic is None else z_semantic

        if target_features is not None and target_padding_mask is not None:
            # truncate target to match z
            target_features = target_features[:, :z.shape[1], :]
            target_padding_mask = target_padding_mask[:, :z.shape[1]]
            
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
            distill_features,
        ) = self.extract_features(audios_srs, **kwargs)
        # encode to latent space
        encoder_output = self.encode(enc_features, enc_padding_mask, **kwargs)

        # decode from latent space
        decoder_output = self.decode(
            z_semantic=encoder_output.z_semantic,
            z_acoustic=encoder_output.z_acoustic,
            target_features=dec_features,
            target_padding_mask=dec_padding_mask,
            speaker_embedding=getattr(encoder_output, "speaker_embedding", None),
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
            "mu_mean": mu_mean,
            "mu_var": mu_var,
        }
        vq_stats = getattr(encoder_output, "vq_stats", None)
        if vq_stats is not None:
            out.update({"vq_loss": encoder_output.vq_loss, "vq_stats": vq_stats})

        if distill_features is not None:
            distill_cosine_loss = self._compute_distillation_losses(
                encoder_output, distill_features
            )
            out["distill_cosine_loss"] = distill_cosine_loss

        if getattr(encoder_output, "ortho_loss", None) is not None:
            out["distill_ortho_loss"] = encoder_output.ortho_loss

        return VAEOutput(**out)

    def _compute_distillation_losses(self, encoder_output, distill_features):
        mu_pre_vq = encoder_output.mu_pre_vq
        B, T_mu, D_mu = mu_pre_vq.shape

        aligned_wavlm = F.interpolate(
            distill_features.transpose(1, 2),
            size=T_mu,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        mask = ~encoder_output.padding_mask
        qd = (
            self.config.encoder_config.vq_config.dim_to_quantize
            if getattr(self.config.encoder_config, "vq_config", None)
            else self.config.encoder_config.latent_dim
        )

        mu_head = mu_pre_vq[..., :qd]
        projected_mu_head = self.distill_proj_head(mu_head)

        projected_mu_head_masked = projected_mu_head[mask]
        aligned_wavlm_masked = aligned_wavlm[mask]

        distill_cosine_loss = (
            1.0
            - F.cosine_similarity(
                projected_mu_head_masked, aligned_wavlm_masked, dim=-1
            ).mean()
        )

        return distill_cosine_loss

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
        z_semantic: Optional[torch.Tensor] = None,
        z_acoustic: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        speaker_embedding: Optional[torch.FloatTensor] = None,
        guide_only_speaker: bool = False,
        **kwargs,
    ):
        assert z_semantic is not None or z_acoustic is not None, "Either z_semantic or z_acoustic must be provided"
        
        z_semantic = self.semantic_upsample(z_semantic)
        if z_semantic is not None and z_acoustic is not None:
            min_len = min(z_semantic.shape[1], z_acoustic.shape[1])
            z_semantic = z_semantic[:, :min_len, :]
            z_acoustic = z_acoustic[:, :min_len, :]
            z = torch.cat([z_semantic, z_acoustic], dim=-1)
        else:
            z = z_acoustic if z_semantic is None else z_semantic

        context_vector = z
        decoder_output = self.decoder.generate(
            num_steps=num_steps,
            generator=generator,
            temperature=temperature,
            padding_mask=padding_mask,
            context_vector=context_vector,
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
        enc_features, enc_padding_mask, dec_features, dec_padding_mask, _ = (
            self.extract_features(audios_srs, **kwargs)
        )
        encoder_output = self.encode(enc_features, enc_padding_mask, **kwargs)

        # Handle cases where VQ is disabled or parts are missing
        if (
            encoder_output.quantized is None
            and encoder_output.residual is None
            and encoder_output.tail is None
        ):
            vq_flags = [k for k in ("quantized", "residual", "tail") if kwargs.get(k) is not None]
            if vq_flags:
                logger.warning(
                    "VQ flags %s were passed but this model has no VQ — "
                    "using encoder_output.z directly (flags have no effect).",
                    vq_flags,
                )
            context_vector = encoder_output.z.clone()
            chunk_idx = kwargs.get("chunk")
            exclude_chunk_idx = kwargs.get("exclude_chunk")
            exclude_start = kwargs.get("exclude_start_chunk")
            exclude_end = kwargs.get("exclude_end_chunk")
            if chunk_idx is not None or exclude_chunk_idx is not None or (exclude_start is not None and exclude_end is not None):
                chunk_size = 2
                if kwargs.get("chunk_size") is not None:
                    chunk_size = kwargs.get("chunk_size")
                elif hasattr(self.config.encoder_config, "kl_chunk_regularizer_config") and self.config.encoder_config.kl_chunk_regularizer_config is not None:
                    chunk_size = self.config.encoder_config.kl_chunk_regularizer_config.chunk_size
                
                mask = torch.ones_like(context_vector)
                if chunk_idx is not None:
                    keep_dim = chunk_idx * chunk_size
                    if context_vector.shape[-1] > keep_dim:
                        mask[..., keep_dim:] = 0.0
                        
                if exclude_chunk_idx is not None:
                    exclude_dim = exclude_chunk_idx * chunk_size
                    if context_vector.shape[-1] >= exclude_dim:
                        mask[..., :exclude_dim] = 0.0
                    else:
                        mask[..., :] = 0.0
                
                if exclude_start is not None and exclude_end is not None:
                    start_dim = exclude_start * chunk_size
                    end_dim = exclude_end * chunk_size
                    if context_vector.shape[-1] >= end_dim:
                        mask[..., start_dim:end_dim] = 0.0
                    elif context_vector.shape[-1] > start_dim:
                        mask[..., start_dim:] = 0.0
                        
                context_vector = context_vector * mask
        else:
            context_vector = torch.zeros_like(encoder_output.z)
            qd = self.encoder._qd
            if kwargs.get("quantized", True) and encoder_output.quantized is not None:
                context_vector[..., :qd] += encoder_output.quantized
            if kwargs.get("residual", True) and encoder_output.residual is not None:
                context_vector[..., :qd] += encoder_output.residual
            if kwargs.get("tail", True) and encoder_output.tail is not None:
                tail = encoder_output.tail
                chunk_idx = kwargs.get("chunk")
                exclude_chunk_idx = kwargs.get("exclude_chunk")
                exclude_start = kwargs.get("exclude_start_chunk")
                exclude_end = kwargs.get("exclude_end_chunk")
                if chunk_idx is not None or exclude_chunk_idx is not None or (exclude_start is not None and exclude_end is not None):
                    chunk_size = 2
                    if kwargs.get("chunk_size") is not None:
                        chunk_size = kwargs.get("chunk_size")
                    elif hasattr(self.config.encoder_config, "kl_chunk_regularizer_config") and self.config.encoder_config.kl_chunk_regularizer_config is not None:
                        chunk_size = self.config.encoder_config.kl_chunk_regularizer_config.chunk_size
                    
                    tail_mask = torch.ones_like(tail)
                    if chunk_idx is not None:
                        keep_dim = chunk_idx * chunk_size
                        if tail.shape[-1] > keep_dim:
                            tail_mask[..., keep_dim:] = 0.0
                            
                    if exclude_chunk_idx is not None:
                        exclude_dim = exclude_chunk_idx * chunk_size
                        if tail.shape[-1] >= exclude_dim:
                            tail_mask[..., :exclude_dim] = 0.0
                        else:
                            tail_mask[..., :] = 0.0
                            
                    if exclude_start is not None and exclude_end is not None:
                        start_dim = exclude_start * chunk_size
                        end_dim = exclude_end * chunk_size
                        if tail.shape[-1] >= end_dim:
                            tail_mask[..., start_dim:end_dim] = 0.0
                        elif tail.shape[-1] > start_dim:
                            tail_mask[..., start_dim:] = 0.0
                            
                    tail = tail * tail_mask
                context_vector[..., qd:] += tail

        speaker_embedding = getattr(encoder_output, "speaker_embedding", None)
        if kwargs.get("zero_speaker", False) and speaker_embedding is not None:
            speaker_embedding = torch.zeros_like(speaker_embedding)

        reconstructed_mel, reconstructed_padding_mask = self.sample(
            num_steps=num_steps,
            temperature=temperature,
            guidance_scale=guidance_scale,
            z=context_vector,
            generator=generator,
            padding_mask=encoder_output.padding_mask,
            speaker_embedding=speaker_embedding,
            guide_only_speaker=kwargs.get("guide_only_speaker", False),
        )
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
        }

    @property
    def dtype(self):
        # WavLM is frozen fp32 and registered first, so skip it
        return next(self.encoder.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
