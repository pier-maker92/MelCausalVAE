import os
import sys
import torch
import torchaudio
import logging
from typing import Tuple, Optional
import huggingface_hub

# Monkey-patch hf_hub_download to handle use_auth_token -> token rename if necessary
# This fixes compatibility between some versions of SpeechBrain and huggingface_hub
original_hf_hub_download = huggingface_hub.hf_hub_download

def patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return original_hf_hub_download(*args, **kwargs)

huggingface_hub.hf_hub_download = patched_hf_hub_download

logger = logging.getLogger(__name__)

class BaselineVocoder:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        vocoder_dir: Optional[str] = None
    ):
        self.model_name = model_name.lower()
        self.device = device
        self.vocoder_dir = vocoder_dir

        logger.info(f"Initializing BaselineVocoder: {self.model_name}")

        if self.model_name == "bigvgan":
            if vocoder_dir not in sys.path and vocoder_dir is not None:
                sys.path.insert(0, vocoder_dir)
            
            try:
                import bigvgan
                from meldataset import get_mel_spectrogram
            except ImportError as e:
                logger.error(f"Failed to import bigvgan or get_mel_spectrogram from {vocoder_dir}. Ensure the repository is valid.")
                raise e
                
            model_path_or_name = vocoder_dir
            logger.info(f"Loading BigVGAN from {model_path_or_name}")
            
            config_file = os.path.join(model_path_or_name, "config.json")
            if os.path.exists(config_file):
                h = bigvgan.load_hparams_from_json(config_file)
                self.model = bigvgan.BigVGAN(h, use_cuda_kernel=False)
                model_file = os.path.join(model_path_or_name, "bigvgan_generator.pt")
                checkpoint_dict = torch.load(model_file, map_location=device, weights_only=False)
                try:
                    self.model.load_state_dict(checkpoint_dict["generator"])
                except RuntimeError:
                    self.model.remove_weight_norm()
                    self.model.load_state_dict(checkpoint_dict["generator"])
            else:
                self.model = bigvgan.BigVGAN.from_pretrained(model_path_or_name, use_cuda_kernel=False)
            
            self.model.remove_weight_norm()
            self.model = self.model.eval().to(device)
            self.h = self.model.h
            self.get_mel_spectrogram_fn = get_mel_spectrogram
            self.target_sr = self.h.sampling_rate

        elif self.model_name == "vocos":
            from vocos import Vocos
            from MelCausalVAE.modules.melspecEncoder import MelSpectrogramEncoder, MelSpectrogramConfig
            
            self.model = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
            config = MelSpectrogramConfig(normalize=False) 
            self.mel_encoder = MelSpectrogramEncoder(config).to(device)
            self.target_sr = 24000
            
        elif self.model_name == "hifigan":
            from speechbrain.inference.vocoders import HIFIGAN
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models", "tts-hifigan-libritts-22050Hz")
            self.model = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir=save_dir, run_opts={"device": str(device)})
            self.target_sr = self.model.hparams.sample_rate if hasattr(self.model.hparams, "sample_rate") else 22050
            
        else:
            raise ValueError(f"Unknown vocoder model: {self.model_name}")

    def reconstruct(self, audio: torch.Tensor, original_sr: int) -> Tuple[torch.Tensor, int]:
        """
        Extracts Mel spectrogram and generates waveform.
        Args:
            audio: [channels, time]
            original_sr: sample rate of the input audio
        Returns:
            reconstructed_audio: [time] down-mixed to mono 1D
            output_sr: sample rate of the output audio
        """
        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if original_sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, original_sr, self.target_sr)
            
        audio = audio.squeeze(0).to(self.device).to(torch.get_default_dtype())
        
        # Prevent clipping safely
        if audio.abs().max() > 0:
            audio = audio / (audio.abs().max() + 1e-8)

        with torch.no_grad():
            if self.model_name == "bigvgan":
                wav_tensor = audio.unsqueeze(0).float()
                mel = self.get_mel_spectrogram_fn(wav_tensor, self.h).to(torch.get_default_dtype())
                wav_gen = self.model(mel)
                return wav_gen.squeeze().cpu(), self.target_sr
                
            elif self.model_name == "vocos":
                out = self.mel_encoder.forward([(audio.cpu(), self.target_sr)])
                mel = out.audio_features.to(self.device).to(torch.get_default_dtype()) # [B, T, C]
                mel = mel.permute(0, 2, 1) # [B, C, T]
                wav_gen = self.model.decode(mel)
                return wav_gen.squeeze().cpu(), self.target_sr
                
            elif self.model_name == "hifigan":
                # Speechbrain mel_spectogram requires cpu or specific tensor processing, so putting back to CPU if needed
                from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
                
                spectrogram, _ = mel_spectogram(
                    audio=audio.unsqueeze(0).cpu(),
                    sample_rate=self.target_sr,
                    hop_length=256,
                    win_length=1024,
                    n_mels=80,
                    n_fft=1024,
                    f_min=0.0,
                    f_max=8000.0,
                    power=1,
                    normalized=False,
                    min_max_energy_norm=True,
                    norm="slaney",
                    mel_scale="slaney",
                    compression=True
                )
                
                spectrogram = spectrogram.to(self.device).to(torch.get_default_dtype())
                waveforms = self.model.decode_batch(spectrogram)
                return waveforms.squeeze().cpu(), self.target_sr

        raise NotImplementedError(f"Reconstruction logic for {self.model_name} is missing.")