import inspect
import torch
import torchaudio
from torchmetrics.text import CharErrorRate, WordErrorRate
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import yaml

# Import your custom modules
from modules.VAE import VAE, VAEConfig
from modules.Encoder import ConvformerEncoderConfig
from modules.cfm import DiTConfig
from modules.decoder_standard_vae import DecoderConfig
from modules.melspecEncoder import MelSpectrogramConfig

# External dependencies for evaluation
from vocos import Vocos
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class SpeechEvaluation:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.cer_metric = CharErrorRate().to(self.device)
        self.wer_metric = WordErrorRate().to(self.device)
        
        self.vae_model = None
        self.vocos = None
        self.asr_processor = None
        self.asr_model = None

    def _filter_kwargs(self, cls, kwargs_dict):
        """Filters a dictionary to only include keys accepted by the class's __init__ method."""
        sig = inspect.signature(cls.__init__)
        valid_keys = [
            p.name for p in sig.parameters.values() 
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        # If the function accepts **kwargs, return the whole dict
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs_dict
            
        return {k: v for k, v in kwargs_dict.items() if k in valid_keys}

    def load_models(self, exp_config_path: str, model_checkpoint: str = None):
        print(f"Loading models to {self.device}...")
        
        # 1. Load VAE Config and Model
        with open(exp_config_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        convformer_cfg = cfg.get("convformer", {})
        cfm_cfg = cfg.get("cfm", {})
        
        # Filter kwargs dynamically so we don't pass unexpected arguments like 'unet_dim'
        encoder_config = ConvformerEncoderConfig(**self._filter_kwargs(ConvformerEncoderConfig, convformer_cfg))
        
        if cfm_cfg.get("decoder_type") == "dit":
            decoder_config = DiTConfig(**self._filter_kwargs(DiTConfig, cfm_cfg))
        else:
            decoder_config = DecoderConfig(**self._filter_kwargs(DecoderConfig, cfm_cfg))
            
        self.vae_model = VAE(
            config=VAEConfig(
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                mel_spec_config=MelSpectrogramConfig(),
            ),
            dtype=torch.float32
        ).to(self.device)
        
        # Load weights only if a checkpoint path is provided
        if model_checkpoint:
            print(f"Loading checkpoint from {model_checkpoint}...")
            self.vae_model.from_pretrained(model_checkpoint)
        else:
            print("No model checkpoint provided. Using randomly initialized weights.")
            
        self.vae_model.eval()

        # 2. Load Vocos Vocoder
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device)

        # 3. Load Whisper for ASR Evaluation
        print("Loading Whisper ASR model...")
        self.asr_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(self.device)
        self.asr_model.eval()

    def download_dataset(self, dataset="LibriSpeech"):
        print("Downloading/Loading dataset...")
        if dataset == "LibriSpeech":
            test_dataset = torchaudio.datasets.LIBRISPEECH(
                root="clean_cache_dir", 
                url="test-clean", 
                download=True
            )
            return test_dataset
        raise ValueError(f"Dataset {dataset} not supported.")

    def transcribe_audio(self, audio_waveform, sr=16000):
        """Transcribe audio waveform using Whisper."""
        inputs = self.asr_processor(
            audio_waveform.squeeze().cpu().numpy(), 
            sampling_rate=sr, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        with torch.no_grad():
            predicted_ids = self.asr_model.generate(inputs)
            
        transcription = self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip().lower()

    def evaluate_dataset(self, test_dataset, max_samples=None, use_hubert=False):
        print("Starting evaluation...")
        predictions = []
        references = []
        
        # Define resamplers
        resample_to_24k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000).to(self.device)
        resample_to_16k = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000).to(self.device)

        for i, data in enumerate(tqdm(test_dataset, desc="Evaluating")):
            if max_samples and i >= max_samples:
                break
                
            waveform, sample_rate, transcript, _, _, _ = data
            waveform = waveform.to(self.device)
            
            #Resample to 24kHz for VAE
            
            if sample_rate != 24000:
                waveform_24k = resample_to_24k(waveform)
            else:
                waveform_24k = waveform

            # --- ADD THIS: Match audio dtype to the VAE model's dtype ---
            model_dtype = next(self.vae_model.parameters()).dtype
            waveform_24k = waveform_24k.to(model_dtype)

            # Prepare inputs for the model
            audios_srs = [(waveform_24k, 24000)]
            
            hubert_guidance = None 
            phonemes = None

            #Encode and Sample (Reconstruct Mel)
            with torch.no_grad():
                results = self.vae_model.encode_and_sample(
                    audios_srs=audios_srs,
                    num_steps=16,
                    temperature=1.0,
                    guidance_scale=1.5,
                    hubert_guidance=hubert_guidance,
                    phonemes=phonemes
                )
                
                reconstructed_mel = results["reconstructed_mel"][0]
                padding_mask = results["padding_mask"][0]
                
                #Filter out padding
                valid_len = (~padding_mask).sum()
                features = reconstructed_mel[:valid_len].unsqueeze(0).permute(0, 2, 1).to(self.device)

                #Decode Mel to Audio using Vocos
                reconstructed_wav = self.vocos.decode(features).float().detach()

            #Resample back to 16kHz for Whisper
            reconstructed_wav_16k = resample_to_16k(reconstructed_wav)

            #Transcribe
            predicted_text = self.transcribe_audio(reconstructed_wav_16k, sr=16000)
            
            predictions.append(predicted_text)
            references.append(transcript.lower())

        # 6. Calculate Metrics
        return self.evaluate_text_mode(predictions, references)

    def evaluate_text_mode(self, predictions, references, metrics=['cer', 'wer']):
        results = {}
        if 'cer' in metrics:
            results['cer'] = self.cer_metric(predictions, references).item()
        if 'wer' in metrics:
            results['wer'] = self.wer_metric(predictions, references).item()

        self.cer_metric.reset()
        self.wer_metric.reset()
        
        return results


if __name__ == "__main__":
    evaluator = SpeechEvaluation()
    
    # 1. Setup Models
    evaluator.load_models(
        exp_config_path="./configs/settings/boh.yaml", 
        model_checkpoint=None # Make sure to point this to your actual weights file when ready!
    )
    
    # 2. Get Dataset
    test_dataset = evaluator.download_dataset("LibriSpeech")
    
    # 3. Run Evaluation
    results = evaluator.evaluate_dataset(test_dataset, max_samples=100)
    print(f"Evaluation Results: {results}")