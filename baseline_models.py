import torch
import torchaudio
import logging
from typing import Tuple, Dict, Any, Optional
import sys

logger = logging.getLogger(__name__)


class BaselineAudioCodec:
    """Wrapper for various baseline audio codecs for evaluation."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        baseline_hz: Optional[int] = None,
        baseline_tau: Optional[float] = None,
    ):
        self.model_name = model_name.lower()
        self.device = device
        self.baseline_hz = baseline_hz
        self.baseline_tau = baseline_tau

        logger.info(f"Initializing BaselineAudioCodec: {self.model_name}")

        if self.model_name == "encodec":
            from transformers import EncodecModel, AutoProcessor

            self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
            self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
            self.target_sr = self.processor.sampling_rate

        elif self.model_name == "mimi":
            from transformers import MimiModel, AutoFeatureExtractor

            self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
            self.model = MimiModel.from_pretrained("kyutai/mimi").to(device)
            self.target_sr = self.feature_extractor.sampling_rate

        elif self.model_name == "dualcodec":
            import dualcodec

            if self.baseline_hz not in [12, 25]:
                raise ValueError("dualcodec requires --baseline-hz to be 12 or 25")
            model_id = f"{self.baseline_hz}hz_v1"
            logger.info(f"Using dualcodec model_id: {model_id}")
            self.dualcodec_model = dualcodec.get_model(model_id)
            self.model = dualcodec.Inference(dualcodec_model=self.dualcodec_model, device=str(device))
            self.target_sr = 24000

        elif self.model_name == "xytokenizer":
            from transformers import AutoFeatureExtractor, AutoModel

            model_id = "OpenMOSS-Team/XY_Tokenizer_TTSD_V0_hf"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, ignore_mismatched_sizes=True).eval().to(device)
            self.target_sr = 16000

        elif self.model_name == "flexicodec":
            try:
                from flexicodec.infer import prepare_model
            except ImportError as e:
                logger.error("flexicodec module not found. Did you clone and install it?")
                raise e

            if self.baseline_tau is None:
                raise ValueError("flexicodec requires --baseline-tau (e.g. 1.0, 0.91, 0.867)")
            self.model_dict = prepare_model()
            # We will use this parameter at inference time
            self.target_sr = 16000  # Flexicodec uses 16kHz audio internally/output

        else:
            raise ValueError(f"Unknown baseline model: {self.model_name}")

    def reconstruct(self, audio: torch.Tensor, original_sr: int) -> Tuple[torch.Tensor, int]:
        """
        Encodes and decodes the audio using the selected baseline codec.
        Args:
            audio: [channels, time]
            original_sr: sample rate of the input audio
        Returns:
            reconstructed_audio: [channels, time] down-mixed to mono, 1D or [1, time]
            output_sr: sample rate of the output audio
        """
        # Downmix to mono if stereo
        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device).float()

        if self.model_name == "encodec":
            if original_sr != self.target_sr:
                audio = torchaudio.functional.resample(audio, original_sr, self.target_sr)
            # HF API expects numpy
            # raw_audio will be 1D if single sample, or 2D [C, T]
            inputs = self.processor(
                raw_audio=audio.squeeze(0).cpu().numpy(),
                sampling_rate=self.target_sr,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                encoder_outputs = self.model.encode(inputs["input_values"], inputs["padding_mask"])
                audio_values = self.model.decode(
                    encoder_outputs.audio_codes,
                    encoder_outputs.audio_scales,
                    inputs["padding_mask"]
                )[0]
            
            return audio_values.squeeze(), 24000

        elif self.model_name == "mimi":
            if original_sr != self.target_sr:
                audio = torchaudio.functional.resample(audio, original_sr, self.target_sr)
            
            # Mimi's conv-based decoder has strict residual shape requirements
            # Pad the audio length to a multiple of 1920 to prevent shape mismatches
            pad_len = 1920 - (audio.shape[-1] % 1920)
            if pad_len != 1920:
                audio = torch.nn.functional.pad(audio, (0, pad_len))

            with torch.no_grad():
                # bypass AutoFeatureExtractor to avoid internal padding bugs
                encoder_outputs = self.model.encode(audio.unsqueeze(1))
                audio_values = self.model.decode(encoder_outputs.audio_codes)[0]
            
            # mimi outputs 24kHz
            return audio_values.squeeze(), 24000

        elif self.model_name == "dualcodec":
            if original_sr != self.target_sr:
                audio = torchaudio.functional.resample(audio, original_sr, self.target_sr)
            
            # shape must be [B, 1, T]
            audio = audio.unsqueeze(0)
            
            with torch.no_grad():
                # using 8 quantizers as default based on example
                semantic_codes, acoustic_codes = self.model.encode(audio, n_quantizers=8)
                out_audio = self.model.decode(semantic_codes, acoustic_codes)
            
            return out_audio.squeeze(), 24000

        elif self.model_name == "xytokenizer":
            if original_sr != self.target_sr:
                audio = torchaudio.functional.resample(audio, original_sr, self.target_sr)
            
            inputs = self.feature_extractor(
                audio.squeeze().cpu().numpy(),
                sampling_rate=self.target_sr,
                return_attention_mask=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                code = self.model.encode(inputs)
                output_wav = self.model.decode(code["audio_codes"], overlap_seconds=10)
            
            # XY outputs 24kHz
            return output_wav["audio_values"][0].squeeze(), 24000

        elif self.model_name == "flexicodec":
            from flexicodec.infer import encode_flexicodec
            
            # Ensure audio is CPU for encode_flexicodec if required, but the library accepts Tensors.
            audio = audio.cpu()
            with torch.no_grad():
                encoded_output = encode_flexicodec(
                    audio, 
                    self.model_dict, 
                    original_sr, 
                    num_quantizers=8, 
                    merging_threshold=self.baseline_tau
                )
                
                reconstructed_audio = self.model_dict['model'].decode_from_codes(
                    semantic_codes=encoded_output['semantic_codes'].to(self.device),
                    acoustic_codes=encoded_output['acoustic_codes'].to(self.device),
                    token_lengths=encoded_output['token_lengths'].to(self.device),
                )
            
            return reconstructed_audio.squeeze(), 16000

        raise NotImplementedError(f"Reconstruction logic for {self.model_name} is missing.")
