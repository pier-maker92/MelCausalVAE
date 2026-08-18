import logging
import inspect
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import safetensors.torch
from typing import Optional
from .decoder.cfm import DiT
from .configs import VAEConfig
from .encoder.encoder import Encoder
from .utils import count_parameters_by_module
from .feature_extractor import FeatureExtractor, WavLMFeatureExtractor
from .speaker_encoder import WavLMSpeakerEncoder
from .output_dataclasses import VAEOutput, DecoderOutput, FeatureExtractorOutput

logger = logging.getLogger(__name__)


class VAE(torch.nn.Module):
    _keys_to_ignore_on_save = None

    def __init__(self, config: VAEConfig, **kwargs):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(config.mel_spectrogram_config)

        self.wavlm_extractor = None
        if getattr(config, "wavlm_config", None) is not None:
            self.wavlm_extractor = WavLMFeatureExtractor(config.wavlm_config)

        self.speaker_encoder = None
        self.speaker_encoder_type = None
        speaker_cfg = getattr(config, "speaker_encoder_config", None)
        if speaker_cfg is not None:
            self.speaker_encoder_type = speaker_cfg.encoder_type
            if self.speaker_encoder_type == "wavlm":
                self.speaker_encoder = WavLMSpeakerEncoder(speaker_cfg)
            elif self.speaker_encoder_type != "ecapa":
                raise ValueError(
                    "speaker_encoder_config.encoder_type must be 'ecapa' or 'wavlm'."
                )

        if speaker_cfg is not None and self.speaker_encoder_type == "ecapa":
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                import huggingface_hub
            except ImportError as exc:
                raise ImportError(
                    "ECAPA speaker conditioning requires speechbrain. "
                    "Install it with `pip install speechbrain`."
                ) from exc

            hf_download_sig = inspect.signature(huggingface_hub.hf_hub_download)
            if "use_auth_token" not in hf_download_sig.parameters:
                original_hf_download = huggingface_hub.hf_hub_download

                def _hf_hub_download_compat(*args, use_auth_token=None, **kwargs):
                    if use_auth_token is not None and "token" not in kwargs:
                        kwargs["token"] = use_auth_token
                    return original_hf_download(*args, **kwargs)

                huggingface_hub.hf_hub_download = _hf_hub_download_compat

            source = speaker_cfg.pretrained_model_name
            source_path = Path(source)
            from_hparams_kwargs = {}
            if source_path.is_dir() and (source_path / "hyperparams.yaml").exists():
                # Force local-only loading path when local files are available.
                from_hparams_kwargs["savedir"] = str(source_path)

            self.speaker_encoder = EncoderClassifier.from_hparams(
                source=source,
                **from_hparams_kwargs,
            )
            for parameter in self.speaker_encoder.parameters():
                parameter.requires_grad = False
            self.speaker_encoder.eval()

        self.distill_wavlm_extractor = None
        sem_cfg = getattr(config.encoder_config, "semantic_distillation_config", None)
        if sem_cfg is not None:
            from .configs import WavLMConfig

            self.distill_wavlm_extractor = WavLMFeatureExtractor(
                WavLMConfig(layer=sem_cfg.wavlm_layer)
            )
            self.distill_proj_head = torch.nn.Linear(
                config.encoder_config.latent_dim, 1024
            )

        self.encoder = Encoder(config.encoder_config)
        self.decoder = DiT(config.decoder_config)

        self.semantic_downsample_factor = getattr(
            config.encoder_config, "semantic_downsample_factor", 1
        )
        if self.semantic_downsample_factor > 1:
            raise NotImplementedError("Semantic downsampling is not implemented yet.")
          
        if kwargs.get("train_only_vq"):
            for name, param in self.named_parameters():
                if "vq" not in name:
                    param.requires_grad = False
            self.train_only_vq = True
        else:
            self.train_only_vq = False


        count_parameters_by_module(self.encoder, "Encoder")
        count_parameters_by_module(self.decoder, "Decoder")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.speaker_encoder is not None and self.speaker_encoder_type == "ecapa":
            self.speaker_encoder.eval()
        return self

    def semantic_upsample(self, z_semantic):
        raise NotImplementedError("Semantic upsampling is not implemented yet.")

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
    def extract_features(self, encoder_audios_srs, target_audios_srs=None, **kwargs):
        if target_audios_srs is None:
            target_audios_srs = encoder_audios_srs

        target_features_output = self.feature_extractor(target_audios_srs)
        target_features = target_features_output.audio_features.to(self.dtype)
        target_padding_mask = target_features_output.padding_mask

        distill_features = None
        if self.distill_wavlm_extractor is not None:
            distill_features = self.distill_wavlm_extractor(
                encoder_audios_srs
            ).audio_features.to(self.dtype)

        if self.wavlm_extractor is not None:
            wavlm_output = self.wavlm_extractor(encoder_audios_srs)
            wavlm_feats = wavlm_output.audio_features.to(self.dtype)  # [B, T_w, 1024]
            T_mel = target_features.shape[1]
            # Causal upsample ×2 (repeat), then interpolate to exact mel length
            wavlm_feats = wavlm_feats.repeat_interleave(2, dim=1)  # [B, 2*T_w, 1024]
            wavlm_feats = (
                F.interpolate(
                    wavlm_feats.float().transpose(1, 2),
                    size=T_mel,
                    mode="linear",
                    align_corners=False,
                )
                .transpose(1, 2)
                .to(wavlm_feats.dtype)
            )  # [B, T_mel, 1024]
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
                target_features,
                target_padding_mask,
                distill_features,
            )

        encoder_features_output = self.feature_extractor(encoder_audios_srs)
        encoder_features = encoder_features_output.audio_features.to(self.dtype)
        encoder_padding_mask = encoder_features_output.padding_mask
        return (
            encoder_features,
            encoder_padding_mask,
            target_features,
            target_padding_mask,
            distill_features,
        )

    def encode(self, features, padding_mask, **kwargs):
        encoder_output = self.encoder(
            x=features,
            padding_mask=padding_mask,
            step=kwargs.get("training_step", None),
        )
        return encoder_output

    def extract_speaker_embedding(self, audios_srs):
        if self.speaker_encoder is None:
            return None
        if self.speaker_encoder_type == "wavlm":
            self.speaker_encoder = self.speaker_encoder.to(device=self.device)
            return self.speaker_encoder(audios_srs).to(
                device=self.device, dtype=self.dtype
            )

        # Keep ECAPA frozen in fp32 for numerical stability and dtype consistency.
        self.speaker_encoder = self.speaker_encoder.to(
            device=self.device, dtype=torch.float32
        )
        speaker_param = next(self.speaker_encoder.parameters())
        speaker_device = speaker_param.device
        speaker_dtype = speaker_param.dtype

        speaker_cfg = self.config.speaker_encoder_config
        waveforms = []
        lengths = []
        for audio, sample_rate in audios_srs:
            waveform = audio.to(device=speaker_device, dtype=torch.float32)
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=0)
            if sample_rate != speaker_cfg.sampling_rate:
                waveform = AF.resample(
                    waveform, sample_rate, speaker_cfg.sampling_rate
                )
            waveform = waveform.to(device=speaker_device, dtype=torch.float32)
            waveforms.append(waveform)
            lengths.append(waveform.numel())

        padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True).to(
            device=speaker_device,
            dtype=speaker_dtype
        )
        relative_lengths = torch.tensor(
            lengths, device=padded.device, dtype=padded.dtype
        ) / padded.shape[1]
        with torch.no_grad(), torch.autocast(
            device_type=speaker_device.type, enabled=False
        ):
            # Use SpeechBrain submodules directly to avoid internal device moves in
            # encode_batch that can desynchronize input/peso device under Trainer.
            mods = self.speaker_encoder.mods
            wavs = padded.float()
            wav_lens = relative_lengths.float()

            feats = mods.compute_features(wavs)
            feats = mods.mean_var_norm(feats, wav_lens)
            try:
                embedding = mods.embedding_model(feats, wav_lens)
            except TypeError:
                embedding = mods.embedding_model(feats)

            if hasattr(mods, "mean_var_norm_emb"):
                ones = torch.ones(
                    embedding.shape[0], device=embedding.device, dtype=wav_lens.dtype
                )
                embedding = mods.mean_var_norm_emb(embedding, ones)

        if embedding.ndim == 3 and embedding.shape[1] == 1:
            embedding = embedding.squeeze(1)
        return embedding.to(device=self.device, dtype=self.dtype)

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
        feature_audios_srs = kwargs.get("feature_audios_srs", audios_srs)

        # extract features
        (
            enc_features,
            enc_padding_mask,
            dec_features,
            dec_padding_mask,
            distill_features,
        ) = self.extract_features(
            feature_audios_srs,
            target_audios_srs=audios_srs,
            **kwargs,
        )
        # encode to latent space
        encoder_output = self.encode(enc_features, enc_padding_mask, **kwargs)
        speaker_embedding = kwargs.get("speaker_embedding")
        if speaker_embedding is None:
            speaker_embedding = self.extract_speaker_embedding(audios_srs)
        if self.train_only_vq:
            # If training only VQ, we don't compute the decoder loss, but we still return the encoder output for VQ loss.
            out = {
                "audio_loss": torch.tensor(0.0, device=encoder_output.z.device),
                "kl_loss": encoder_output.kl_loss,
                "mu_mean": encoder_output.mu[~encoder_output.padding_mask].mean(),
                "mu_var": encoder_output.mu[~encoder_output.padding_mask].var(),
            }
            vq_stats = getattr(encoder_output, "vq_stats", None)
            if vq_stats is not None:
                out.update({"vq_loss": encoder_output.vq_loss, "vq_stats": vq_stats})
            return VAEOutput(**out)

        # decode from latent space
        decoder_output = self.decode(
            z=encoder_output.z,
            target_features=dec_features,
            target_padding_mask=dec_padding_mask,
            speaker_embedding=speaker_embedding
            if speaker_embedding is not None
            else getattr(encoder_output, "speaker_embedding", None),
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
        projected_mu_head = self.distill_proj_head(mu_pre_vq)

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

    def _apply_kmeans_codebook(
        self,
        z: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor],
        kmeans_codebook: dict,
        chunk_size: int,
    ) -> torch.Tensor:
        centroids = kmeans_codebook["centroids"].to(device=z.device, dtype=torch.float32)
        selection = kmeans_codebook.get("latent_selection")
        if selection is None:
            selection = {
                "indices": None,
                "start": 0,
                "end": int(kmeans_codebook["feature_dims"]),
            }

        indices = selection.get("indices")
        if indices is not None:
            dim_index = torch.as_tensor(indices, device=z.device, dtype=torch.long)
            if dim_index.numel() == 0:
                raise ValueError("K-means latent selection has no dimensions.")
            if dim_index.min().item() < 0 or dim_index.max().item() >= z.shape[-1]:
                raise ValueError(
                    f"K-means latent indices {indices} are incompatible with "
                    f"latent dim {z.shape[-1]}."
                )
            selected = z.index_select(dim=-1, index=dim_index)
        else:
            start = int(selection.get("start", 0))
            end = int(selection["end"])
            if start < 0 or end > z.shape[-1] or end <= start:
                raise ValueError(
                    f"K-means latent slice [{start}:{end}] is incompatible with "
                    f"latent dim {z.shape[-1]}."
                )
            selected = z[..., start:end]

        if selected.shape[-1] != centroids.shape[-1]:
            raise ValueError(
                f"K-means centroids have dim {centroids.shape[-1]}, "
                f"but selected latent has dim {selected.shape[-1]}."
            )

        if padding_mask is None:
            valid_mask = torch.ones(z.shape[:2], device=z.device, dtype=torch.bool)
        else:
            valid_mask = ~padding_mask

        selected_valid = selected[valid_mask].to(dtype=torch.float32)
        quantized_chunks = []
        for selected_chunk in selected_valid.split(chunk_size):
            distances = (
                selected_chunk[:, None, :] - centroids[None, :, :]
            ).square().sum(dim=-1)
            nearest = distances.argmin(dim=1)
            quantized_chunks.append(centroids.index_select(0, nearest))

        if not quantized_chunks:
            return z

        quantized = torch.cat(quantized_chunks, dim=0).to(dtype=z.dtype)
        out = z.clone()
        out_flat = out.reshape(-1, out.shape[-1])
        valid_positions = valid_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        valid_values = out_flat.index_select(0, valid_positions)
        if indices is not None:
            valid_values[:, dim_index] = quantized
        else:
            valid_values[:, start:end] = quantized
        out_flat.index_copy_(0, valid_positions, valid_values)
        return out_flat.reshape_as(out)

    def _kmeans_encode(
        self,
        z: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor],
        kmeans_codebook: dict,
        chunk_size: int,
    ) -> dict[str, torch.Tensor]:
        centroids = kmeans_codebook["centroids"].to(device=z.device, dtype=torch.float32)
        selection = kmeans_codebook.get("latent_selection")
        if selection is None:
            selection = {
                "indices": None,
                "start": 0,
                "end": int(kmeans_codebook["feature_dims"]),
            }

        indices = selection.get("indices")
        if indices is not None:
            dim_index = torch.as_tensor(indices, device=z.device, dtype=torch.long)
            if dim_index.numel() == 0:
                raise ValueError("K-means latent selection has no dimensions.")
            if dim_index.min().item() < 0 or dim_index.max().item() >= z.shape[-1]:
                raise ValueError(
                    f"K-means latent indices {indices} are incompatible with "
                    f"latent dim {z.shape[-1]}."
                )
            selected = z.index_select(dim=-1, index=dim_index)
            tail_mask = torch.ones(z.shape[-1], device=z.device, dtype=torch.bool)
            tail_mask[dim_index] = False
            tail = z[..., tail_mask]
        else:
            start = int(selection.get("start", 0))
            end = int(selection["end"])
            if start < 0 or end > z.shape[-1] or end <= start:
                raise ValueError(
                    f"K-means latent slice [{start}:{end}] is incompatible with "
                    f"latent dim {z.shape[-1]}."
                )
            selected = z[..., start:end]
            tail = torch.cat((z[..., :start], z[..., end:]), dim=-1)

        if selected.shape[-1] != centroids.shape[-1]:
            raise ValueError(
                f"K-means centroids have dim {centroids.shape[-1]}, "
                f"but selected latent has dim {selected.shape[-1]}."
            )

        if padding_mask is None:
            valid_mask = torch.ones(z.shape[:2], device=z.device, dtype=torch.bool)
        else:
            valid_mask = ~padding_mask

        selected_valid = selected[valid_mask].to(dtype=torch.float32)
        index_chunks = []
        quantized_chunks = []
        for selected_chunk in selected_valid.split(chunk_size):
            distances = (
                selected_chunk[:, None, :] - centroids[None, :, :]
            ).square().sum(dim=-1)
            nearest = distances.argmin(dim=1)
            index_chunks.append(nearest)
            quantized_chunks.append(centroids.index_select(0, nearest))

        flat_shape = z.shape[0] * z.shape[1]
        out_indices = torch.full(
            (flat_shape,),
            -1,
            device=z.device,
            dtype=torch.long,
        )
        out_quantized = torch.zeros(
            (flat_shape, centroids.shape[-1]),
            device=z.device,
            dtype=z.dtype,
        )
        if index_chunks:
            valid_positions = valid_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            nearest = torch.cat(index_chunks, dim=0)
            quantized = torch.cat(quantized_chunks, dim=0).to(dtype=z.dtype)
            out_indices.index_copy_(0, valid_positions, nearest)
            out_quantized.index_copy_(0, valid_positions, quantized)

        return {
            "indices": out_indices.reshape(z.shape[:2]),
            "quantized": out_quantized.reshape(*z.shape[:2], centroids.shape[-1]),
            "tail": tail,
        }

    @torch.no_grad()
    def encode_audio(
        self,
        audios_srs,
        kmeans_codebook: dict,
        kmeans_chunk_size: int = 16384,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        enc_features, enc_padding_mask, _, _, _ = self.extract_features(
            audios_srs,
            target_audios_srs=audios_srs,
            **kwargs,
        )
        encoder_output = self.encode(enc_features, enc_padding_mask, **kwargs)
        return self._kmeans_encode(
            z=encoder_output.z,
            padding_mask=encoder_output.padding_mask,
            kmeans_codebook=kmeans_codebook,
            chunk_size=int(kmeans_chunk_size),
        )

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
        if speaker_embedding is None:
            speaker_embedding = getattr(encoder_output, "speaker_embedding", None)
        if kwargs.get("zero_speaker", False) and speaker_embedding is not None:
            speaker_embedding = torch.zeros_like(speaker_embedding)

        use_quantized = kwargs.get("quantized", False)
        use_residual = kwargs.get("residual", False)
        if use_quantized or use_residual:
            quantized = getattr(encoder_output, "quantized", None)
            residual = getattr(encoder_output, "residual", None)
            if use_quantized and quantized is None:
                raise ValueError("Quantized inference requires an encoder quantizer.")
            if use_residual and residual is None:
                raise ValueError("Residual inference requires an encoder quantizer.")

            if use_quantized and use_residual:
                z = quantized + residual
            elif use_quantized:
                z = quantized
            else:
                z = residual
        else:
            z = encoder_output.z
        z = z.clone()

        chunk_size = kwargs.get("chunk_size", None)
        chunk = kwargs.get("chunk", None)
        exclude_start_chunk = kwargs.get("exclude_start_chunk", None)

        if chunk_size and chunk:
            keep_len = chunk * chunk_size
            z[..., keep_len:] = 0
        if chunk_size and exclude_start_chunk:
            zero_len = exclude_start_chunk * chunk_size
            z[..., :zero_len] = 0

        kmeans_codebook = kwargs.get("kmeans_codebook", None)
        if kmeans_codebook is not None:
            z = self._apply_kmeans_codebook(
                z=z,
                padding_mask=encoder_output.padding_mask,
                kmeans_codebook=kmeans_codebook,
                chunk_size=int(kwargs.get("kmeans_chunk_size", 16384)),
            )

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
