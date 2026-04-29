from data.audio_dataset import TestDatasetWrapper
import os
import wandb
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from vocos import Vocos
<<<<<<< HEAD
from typing import Dict, List
import matplotlib.pyplot as plt
from modules.VAE import VAE, VAEConfig
from modules.feature_extractor import MelSpectrogramConfig
=======
from pathlib import Path
from einops import rearrange
from typing import Dict, List
import matplotlib.pyplot as plt
from modules.cfm import DiTConfig
from modules.VAE import VAE, VAEConfig
from modules.similarity import plot_durations_on_mel
from modules.Encoder import ConvformerEncoderConfig
from modules.decoder_standard_vae import DecoderConfig
from modules.melspecEncoder import MelSpectrogramConfig
>>>>>>> origin/main
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
)
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR

# data
from data.mls import MLSDataset
<<<<<<< HEAD
import torch.distributed as dist
from data.libri_tts import LibriTTS
from data.audio_dataset import DataCollator
from data.audio_dataset import TrainDatasetWrapper
from data.librispeech_align import LibriSpeechAlignDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from modules.configs import (
    EncoderConfig,
    VQConfig,
    DropoutConfig,
    KLChunkRegularizer,
    DiTConfig,
)
=======
from data.libri_tts import LibriTTS
from data.librispeechHubert import LibriSpeech100h
from data.audio_dataset import TrainDatasetWrapper
from data.audio_dataset import DataCollator, HubertDatasetWrapper

>>>>>>> origin/main

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

<<<<<<< HEAD
=======

def get_phonemes(phonemes: List[str], parsing_mode: str = "phoneme"):
    """
    Convert transcripts to phoneme tokens.
    Returns: List[B] of List[phoneme]
    """
    phonemes_batch = []
    for phoneme_str in phonemes:
        phoneme_str = f"<sil> {phoneme_str} <sil>"

        if parsing_mode == "phoneme":
            tokens = phoneme_str.split()
        elif parsing_mode == "char":
            tokens = []
            for p in phoneme_str.split():
                if p == "<sil>":
                    tokens.append(p)
                else:
                    tokens.extend(list(p))
        else:
            raise ValueError(f"Unknown parsing mode: {parsing_mode}")

        phonemes_batch.append(tokens)
    return phonemes_batch


def get_cosine_schedule_with_warmup_and_min_lr(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    lr_min: float = 0.0,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function
    between the initial lr set in the optimizer to `lr_min`, after a warmup period during which it
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of waves in the cosine schedule (default: 0.5).
        last_epoch: The index of the last epoch when resuming training (default: -1).
        lr_min: The minimum learning rate (default: 0.0).

    Returns:
        A LambdaLR scheduler.
    """
    # Get the initial learning rate from the optimizer at creation time
    initial_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # Cosine annealing with minimum learning rate
        cosine_value = 0.5 * (
            1.0
            + torch.cos(torch.tensor(num_cycles * 2.0 * 3.141592653589793 * progress))
        )
        # Scale cosine from [lr_min, initial_lr] instead of [0, initial_lr]
        return (lr_min / initial_lr) + (1.0 - lr_min / initial_lr) * cosine_value

    return LambdaLR(optimizer, lr_lambda, last_epoch)

>>>>>>> origin/main

class AddGranularLossesToTrainerState(TrainerCallback):
    """Callback to add granular losses to trainer state"""

    def __init__(self, granular_losses: List[str]):
        self.granular_losses = granular_losses

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        control.granular_losses = {
            k: torch.tensor(0.0).to(args.device) for k in self.granular_losses
        }
        return control


class VAEtrainer(Trainer):
<<<<<<< HEAD
    def __init__(self, min_learning_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.min_learning_rate = min_learning_rate
        # Register granular losses
        granular_losses = [
            "audio_loss",
            "kl_loss",
            "mu_mean",
            "mu_var",
        ]
        if getattr(self.model.encoder.config, "vq_config", None) is not None:
            granular_losses.extend(
                [
                    "vq_loss",
                    "vq_perplexity",
                    "vq_codes_used",
                    "vq_codes_used_frac",
                ]
            )
        self.add_callback(AddGranularLossesToTrainerState(granular_losses))

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        self.lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=self.min_learning_rate,
        )
        return self.lr_scheduler

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)

        # Save VAEConfig alongside the model
        if output_dir is None:
            output_dir = self.args.output_dir

        if output_dir is not None:
            import os
            import json
            import dataclasses

            config_path = os.path.join(output_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(dataclasses.asdict(self.model.config), f, indent=4)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if hasattr(self.control, "granular_losses") and model.training:
            audios_srs = inputs["output_audios_srs"]
            output = model(
                audios_srs=audios_srs,
                training_step=self.state.global_step,
                phoneme_alignments=inputs["phoneme_alignments"],
            )
            audio_loss = output.audio_loss
            kl_loss = output.kl_loss
            vq_loss = getattr(output, "vq_loss", None)
            vq_stats = getattr(output, "vq_stats", None)
            loss = audio_loss + kl_loss + (vq_loss if vq_loss is not None else 0.0)

            # Accumulate granular losses
            flat_metrics = {
                "audio_loss": audio_loss,
                "kl_loss": kl_loss,
                "mu_mean": getattr(output, "mu_mean", None),
                "mu_var": getattr(output, "mu_var", None),
                "vq_loss": vq_loss,
            }
            if vq_stats is not None:
                flat_metrics["vq_perplexity"] = vq_stats.perplexity
                flat_metrics["vq_codes_used"] = vq_stats.codes_used
                flat_metrics["vq_codes_used_frac"] = vq_stats.codes_used_frac

            for key in self.control.granular_losses:
                if flat_metrics.get(key) is not None:
                    val = flat_metrics[key].detach().float()
                    if self.args.n_gpu > 1 and val.dim() > 0:
                        val = val.mean()
                    self.control.granular_losses[key] += (
                        val.to(self.control.granular_losses[key].dtype)
                        / self.args.gradient_accumulation_steps
                    )
            return (loss, output) if return_outputs else loss

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        tr_loss = args[0]
        grad_norm = args[1]
        model = args[2]
        trial = args[3]
        epoch = args[4]
        ignore_keys_for_eval = args[5]
=======
    """Custom trainer for VAE"""

    def __init__(self, phonemes=False, **kwargs):
        super().__init__(**kwargs)
        self.phonemes = phonemes
        # Register granular losses
        granular_losses = ["audio_loss", "kl_loss", "mu_mean", "mu_var", "align_loss"]
        try:
            if getattr(self.model.encoder.config, "semantic_regulation", False):
                granular_losses.append("semantic_loss")
        except Exception:
            pass
        # Add CTC loss if phonemes are enabled
        if phonemes:
            granular_losses.append("ctc_loss")
        self.add_callback(AddGranularLossesToTrainerState(granular_losses))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute the loss for the VAE"""
        if hasattr(self.control, "granular_losses") and model.training:
            audios_srs = inputs["output_audios_srs"]
            hubert_guidance = inputs.get("hubert_guidance", None)
            phonemes = inputs.get("phonemes", None)

            # Forward pass
            outputs = model(
                audios_srs=audios_srs,
                training_step=self.state.global_step,
                hubert_guidance=hubert_guidance,
                phonemes=phonemes,
            )
            audio_loss = outputs.audio_loss
            kl_loss = outputs.kl_loss
            semantic_loss = getattr(outputs, "semantic_loss", None)
            ctc_loss = getattr(outputs, "ctc_loss", None)
            align_loss = getattr(outputs, "align_loss", None)
            mu_mean = getattr(outputs, "mu_mean")
            mu_var = getattr(outputs, "mu_var")
            loss = (
                audio_loss
                + kl_loss
                + (semantic_loss if semantic_loss is not None else 0.0)
                + (ctc_loss if ctc_loss is not None else 0.0)
                + (align_loss if align_loss is not None else 0.0)
            )

            # Accumulate granular losses
            if self.args.n_gpu > 1:
                audio_loss = audio_loss.mean()
                kl_loss = kl_loss.mean()
                align_loss = align_loss.mean()
            self.control.granular_losses["audio_loss"] += (
                audio_loss.detach() / self.args.gradient_accumulation_steps
            )
            self.control.granular_losses["kl_loss"] += (
                kl_loss.detach() / self.args.gradient_accumulation_steps
            )
            if align_loss is not None:
                self.control.granular_losses["align_loss"] += (
                    align_loss.detach() / self.args.gradient_accumulation_steps
                )
            if semantic_loss is not None:
                if self.args.n_gpu > 1:
                    semantic_loss = semantic_loss.mean()
                self.control.granular_losses["semantic_loss"] += (
                    semantic_loss.detach() / self.args.gradient_accumulation_steps
                )
            if ctc_loss is not None:
                if self.args.n_gpu > 1:
                    ctc_loss = ctc_loss.mean()
                self.control.granular_losses["ctc_loss"] += (
                    ctc_loss.detach() / self.args.gradient_accumulation_steps
                )
            if mu_mean is not None:
                val = mu_mean.detach().float()
                if val.dim() > 0:
                    val = val.mean()
                self.control.granular_losses["mu_mean"] += (
                    val.to(self.control.granular_losses["mu_mean"].dtype)
                    / self.args.gradient_accumulation_steps
                )
            if mu_var is not None:
                val = mu_var.detach().float()
                if val.dim() > 0:
                    val = val.mean()
                self.control.granular_losses["mu_var"] += (
                    val.to(self.control.granular_losses["mu_var"].dtype)
                    / self.args.gradient_accumulation_steps
                )

            return (loss, outputs) if return_outputs else loss
        else:
            # Fallback for eval mode or before callback is initialized
            audios_srs = inputs["output_audios_srs"]
            outputs = model(
                audios_srs=audios_srs,
                training_step=self.state.global_step,
            )
            audio_loss = outputs.audio_loss
            kl_loss = outputs.kl_loss
            semantic_loss = getattr(outputs, "semantic_loss", None)
            align_loss = getattr(outputs, "align_loss", None)
            loss = (
                audio_loss
                + kl_loss
                + (semantic_loss if semantic_loss is not None else 0.0)
                + (ctc_loss if ctc_loss is not None else 0.0)
                + (align_loss if align_loss is not None else 0.0)
            )
            return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Override scheduler creation to use custom cosine scheduler with min_lr support.
        """
        # Check if lr_min is specified in lr_scheduler_kwargs
        lr_scheduler_kwargs = getattr(self.args, "lr_scheduler_kwargs", {}) or {}
        lr_min = lr_scheduler_kwargs.get("lr_min", None)

        # If lr_min is specified and scheduler type is cosine, use custom scheduler
        if lr_min is not None and self.args.lr_scheduler_type == "cosine":
            if optimizer is None:
                optimizer = self.optimizer

            num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
            num_cycles = lr_scheduler_kwargs.get("num_cycles", 0.5)

            self.lr_scheduler = get_cosine_schedule_with_warmup_and_min_lr(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
                lr_min=float(lr_min),
            )
            logger.info(f"Using custom cosine scheduler with min_lr={lr_min}")
        else:
            # Use default scheduler creation
            super().create_scheduler(num_training_steps, optimizer)

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
>>>>>>> origin/main
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )

            # Add granular losses
            if hasattr(self.control, "granular_losses"):
                for k, v in self.control.granular_losses.items():
                    logs[k] = self._nested_gather(v).mean().item()
                    # reset the loss
                    self.control.granular_losses[k] -= self.control.granular_losses[k]

                    avg_val = logs[k] / (
                        self.state.global_step - self._globalstep_last_logged
                    )
                    if k in ("mu_mean", "mu_var"):
                        logs[k] = round(avg_val, 8)
                    else:
                        logs[k] = round(avg_val, 4)

            logs["learning_rate"] = self._get_learning_rate()

            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm if isinstance(grad_norm, float) else grad_norm.item()
                )

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time=start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
<<<<<<< HEAD
            self._save_checkpoint(model, trial)  # metrics=metrics
=======
            self._save_checkpoint(
                model,
                trial,
            )  # metrics=metrics
>>>>>>> origin/main
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and generate sample reconstructions.
        """
        metrics = {}
        # Set model to eval mode for sample generation
        self.model.eval()
        
        # Generate samples and log to wandb
        # All processes must call this to avoid deadlock in distributed training
        self._generate_and_log_samples()
        
        # Set model back to train mode
        self.model.train()

        return metrics

    def _generate_and_log_samples(self):
        logger.info("Generating reconstruction samples...")

<<<<<<< HEAD
        if getattr(self.model.config.mel_spectrogram_config, "use_bigvgan_mel", False):
            try:
                import sys
                import os

                bigvgan_path = (
                    "/home/ec2-user/MelCausalVAE/bigvgan/bigvgan_v2_24khz_100band_256x"
                )
                if bigvgan_path not in sys.path:
                    sys.path.append(bigvgan_path)
                import bigvgan

                vocoder = bigvgan.BigVGAN.from_pretrained(
                    bigvgan_path, use_cuda_kernel=False
                )
                vocoder_type = "bigvgan"
            except Exception as e:
                logger.error(
                    f"Failed to load BigVGAN vocoder: {e}. Falling back to Vocos."
                )
                vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
                vocoder_type = "vocos"
        else:
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            vocoder_type = "vocos"

        vocoder.to(self.args.device)

        # Get some samples from the eval dataset using its dataloader
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

        need_aligner = getattr(self.model.config.encoder_config, "use_aligner", False)
        max_batches = (
            len(eval_dataloader) if hasattr(eval_dataloader, "__len__") else 10_000
        )

        audios_srs = None
        phoneme_alignments = None
        for i, batch in enumerate(eval_dataloader):
            if i >= max_batches:
                break
            if "output_audios_srs" not in batch:
                continue
            pas = batch.get("phoneme_alignments")
            if need_aligner and (pas is None or any(p is None for p in pas)):
                continue
            audios_srs = batch["output_audios_srs"]
            audios_srs = [
                (audio.to(self.args.device).to(self.model.dtype), sr)
                for audio, sr in audios_srs
            ]
            phoneme_alignments = pas
            break

        if audios_srs is None:
            logger.warning(
                "Skipping reconstruction logging: no eval batch with "
                f"output audio{' and non-null phoneme_alignments' if need_aligner else ''}"
            )
            return

        # Generate reconstructions
        try:
            with torch.no_grad():
                results = self.model.encode_decode(
                    audios_srs=audios_srs,
                    num_steps=16,
                    temperature=0.2,
                    guidance_scale=1.3,
                    phoneme_alignments=phoneme_alignments,
                )
            # Create visualizations
            images = []
            # Resolve device id safely for distributed/non-distributed
            if dist.is_available() and dist.is_initialized():
                device_id = dist.get_rank()
            else:
                device_id = 0

            audios = []
            audio_paths = []
            segmentation_plots = []
            for idx in range(len(audios_srs)):
                fig = self._create_mel_comparison_plot(
                    original=results["original_mel"][idx],
                    reconstructed=results["reconstructed_mel"][idx],
                    original_padding_mask=results["original_padding_mask"][idx],
                    reconstructed_padding_mask=results["padding_mask"][idx],
                    sample_idx=idx,
=======
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.args.device)
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        for batch in eval_dataloader:
            if "output_audios_srs" in batch:
                audios_srs = [
                    (a.to(self.args.device, torch.float32), sr)
                    for a, sr in batch["output_audios_srs"]
                ]
                break

        hubert_guidance = batch.get("hubert_guidance", None)
        phonemes = batch.get("phonemes", None) if self.phonemes else None

        with torch.no_grad():
            results = self.model.encode_and_sample(
                audios_srs=audios_srs,
                num_steps=16,
                temperature=1.0,
                guidance_scale=1.5,
                hubert_guidance=hubert_guidance,
                phonemes=phonemes,
            )

        if phonemes is not None:
            phonemes = get_phonemes(phonemes)

        device_id = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        boundaries_images, images, audios = [], [], []

        for idx in range(len(audios_srs)):
            pad_mask = results["padding_mask"][idx]
            valid_len = (~pad_mask).sum()

            fig = self._create_mel_comparison_plot(
                original=results["original_mel"][idx],
                reconstructed=results["reconstructed_mel"][idx],
                padding_mask=pad_mask,
                sample_idx=idx,
                device_id=device_id,
            )
            images.append(
                wandb.Image(
                    fig,
                    caption=f"Sample {idx} - Step {self.state.global_step} - Device {device_id}",
                )
            )
            plt.close(fig)

            if results["durations"] is not None:
                fig = plot_durations_on_mel(
                    batch_idx=idx,
                    step=self.state.global_step,
                    mels=results["original_mel"],
                    durations=results["durations"],
                    text_length=[len(p) for p in phonemes],
                    labels=phonemes[idx],
>>>>>>> origin/main
                    device_id=device_id,
                    mel_mask=results["padding_mask"],
                )
<<<<<<< HEAD

                # Convert matplotlib figure to wandb Image
                images.append(
                    wandb.Image(
                        fig,
                        caption=f"Sample {idx} - Step {self.state.global_step} - Device {device_id}",
                    )
                )
                plt.close(fig)

                # Decode reconstructed mel to audio
                mel = results["reconstructed_mel"][idx]  # [T, F]
                pad_mask = results["padding_mask"][idx]  # [T] True = padded
                T = min(mel.shape[0], pad_mask.shape[0])
                mel = mel[:T]
                pad_mask = pad_mask[:T]
                valid_mel = mel[~pad_mask]

                # Shape for Vocos/BigVGAN: [B, F, T]
                features = (
                    valid_mel.unsqueeze(0)
                    .permute(0, 2, 1)
                    .float()
                    .to(self.args.device)
                )

                if vocoder_type == "bigvgan":
                    # BigVGAN expects exp(mel)
                    # features = torch.exp(features.float())
                    waveform = vocoder(features)
                else:
                    waveform = vocoder.decode(features)  # [1, samples]

                waveform = waveform.float().squeeze().detach().cpu()
                # normalize waveform to -1 to 1
                waveform = waveform / (waveform.abs().max() + 1e-8)
                sr = audios_srs[idx][1]
                # Log as wandb audio as well
                audios.append(
                    wandb.Audio(
                        waveform.numpy(),
                        sample_rate=sr,
=======
                boundaries_images.append(
                    wandb.Image(
                        fig,
>>>>>>> origin/main
                        caption=f"Sample {idx} - Step {self.state.global_step} - Device {device_id}",
                    )
                )
                plt.close(fig)

<<<<<<< HEAD
            # Log to wandb as a gallery
            if wandb.run is not None:
                log_dict = {
                    "reconstructions": images,
                    "reconstructions_audio": audios,
                    "reconstructions_audio_paths": audio_paths,
                    "step": self.state.global_step,
                }
                if segmentation_plots:
                    log_dict["segmentation_plots"] = segmentation_plots
                wandb.log(log_dict)
                logger.info(
                    f"Successfully logged {len(images)} reconstruction samples to wandb"
=======
            for mel_data, label in [
                (results["reconstructed_mel"][idx], "Reconstructed"),
                (results["original_mel"][idx], "Original"),
            ]:
                features = (
                    mel_data[:valid_len]
                    .unsqueeze(0)
                    .permute(0, 2, 1)
                    .to(torch.bfloat16)
                    .to(self.args.device)
                )
                wav = vocos.decode(features).float().squeeze(0).detach().cpu()
                wav = wav / (wav.abs().max() + 1e-8)
                audios.append(
                    wandb.Audio(
                        wav.numpy(),
                        sample_rate=audios_srs[idx][1],
                        caption=f"{label} - Sample {idx} - Step {self.state.global_step} - Device {device_id}",
                    )
>>>>>>> origin/main
                )

        if wandb.run is not None:
            to_log = {
                "reconstructions": images,
                "reconstructions_audio": audios,
                "step": self.state.global_step,
            }
            if boundaries_images:
                to_log["boundaries"] = boundaries_images
            wandb.log(to_log)
            logger.info(f"Logged {len(images)} reconstruction samples to wandb")

    def _create_mel_comparison_plot(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        original_padding_mask: torch.Tensor,
        reconstructed_padding_mask: torch.Tensor,
        sample_idx: int,
        device_id: int,
        boundaries_mel=None,
    ):
<<<<<<< HEAD
        """
        Create a side-by-side comparison plot of original and reconstructed mel spectrograms.

        Args:
            original: Original mel spectrogram [T, F]
            reconstructed: Reconstructed mel spectrogram [T, F]
            original_padding_mask: True where original mel is padded [T_orig]
            reconstructed_padding_mask: True where reconstructed mel is padded [T_recon]
            sample_idx: Index of the sample

        Returns:
            matplotlib figure
        """
        # Move to CPU and convert to numpy
=======
>>>>>>> origin/main
        original = original.float().detach().cpu().numpy()
        reconstructed = reconstructed.float().detach().cpu().numpy()
        om = original_padding_mask.detach().cpu().numpy().astype(bool)
        rm = reconstructed_padding_mask.detach().cpu().numpy().astype(bool)

<<<<<<< HEAD
        # Each spectrogram uses its own mask (lengths may differ after pooling / decoder).
        To, Fo = original.shape[0], original.shape[1]
        Tr, Fr = reconstructed.shape[0], reconstructed.shape[1]
        om = om[:To]
        rm = rm[:Tr]
        original = original[~om]
        reconstructed = reconstructed[~rm]
=======
        min_len = min(original.shape[0], reconstructed.shape[0])
        mask = ~padding_mask[:min_len]
        original = original[:min_len][mask]
        reconstructed = reconstructed[:min_len][mask]
        n_frames = original.shape[0]

        has_phonemes = boundaries_mel is not None and len(boundaries_mel) > 0
        fig, (ax_orig, ax_recon) = plt.subplots(
            2, 1, figsize=(14, 10 if has_phonemes else 8)
        )
>>>>>>> origin/main

        # Original mel
        ax_orig.imshow(
            original.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
        )
        ax_orig.set_title(f"Original - Sample {sample_idx} - Device {device_id}")
        ax_orig.set_ylabel("Mel Bin")

<<<<<<< HEAD
        # Plot original
        im1 = axes[0].imshow(
            original.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
        )
        axes[0].set_title(
            f"Original Mel Spectrogram - Sample {sample_idx} - Device {device_id}"
        )
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Mel Frequency")
        plt.colorbar(im1, ax=axes[0])

        # Plot reconstructed
        im2 = axes[1].imshow(
=======
        if has_phonemes:
            trans = ax_orig.get_xaxis_transform()
            for start, end, ph in boundaries_mel:
                start, end = int(start), min(int(end), n_frames)
                if start >= n_frames:
                    continue
                ax_orig.axvline(x=start, color="red", linewidth=1.0, alpha=0.6)
                mid = (start + end) / 2.0
                label = ph if len(ph) <= 6 else ph[:5] + "…"
                ax_orig.text(
                    mid,
                    -0.05,
                    label,
                    transform=trans,
                    ha="center",
                    va="top",
                    fontsize=5,
                    clip_on=False,
                )
            ax_orig.set_xlim(0, n_frames)

        # Reconstructed mel
        ax_recon.imshow(
>>>>>>> origin/main
            reconstructed.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
        )
<<<<<<< HEAD
        axes[1].set_title(
            f"Reconstructed Mel Spectrogram - Sample {sample_idx} - Device {device_id}"
        )
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Mel Frequency")
        plt.colorbar(im2, ax=axes[1])
=======
        ax_recon.set_title(f"Reconstructed - Sample {sample_idx} - Device {device_id}")
        ax_recon.set_xlabel("Time (frames)")
        ax_recon.set_ylabel("Mel Bin")
>>>>>>> origin/main

        plt.tight_layout()
        return fig


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    # Convert OmegaConf DictConfig to standard python dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

<<<<<<< HEAD
    training_cfg = cfg_dict.get("training", {})
    encoder_cfg = cfg_dict.get("encoder", {})
    decoder_cfg = cfg_dict.get("decoder", {})
=======
    # Data arguments
    parser.add_argument(
        "--exp-config",
        dest="exp_config_path",
        type=Path,
        required=True,
        help="Path to experiment YAML overriding defaults in configs/defaults",
    )

    # DeepSpeed config
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load defaults + experiment overrides ---
    def deep_update(base: dict, override: dict) -> dict:
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = deep_update(base[k], v)
            else:
                base[k] = v
        return base

    def load_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        return data

    cfg_root = Path(__file__).resolve().parent / "configs"
    defaults = {
        "training": load_yaml(cfg_root / "defaults" / "train.yaml").get("training", {}),
        "convformer": load_yaml(cfg_root / "defaults" / "convformer.yaml").get(
            "convformer", {}
        ),
        "cfm": load_yaml(cfg_root / "defaults" / "cfm.yaml").get("cfm", {}),
    }
    custom = load_yaml(args.exp_config_path)
    merged = deep_update(defaults, custom)
    training_cfg = merged.get("training", {})
    convformer_cfg = merged.get("convformer", {})
    cfm_cfg = merged.get("cfm", {})
>>>>>>> origin/main

    # Set seed for reproducibility
    set_seed(training_cfg.get("seed", 42))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create AudioDataset
    dataset_name = training_cfg.pop("dataset_name", None)
    if dataset_name == "mls":
        dataset = MLSDataset()
    elif dataset_name == "libritts":
        dataset = LibriTTS()
<<<<<<< HEAD
    elif dataset_name == "librispeech_aligned":
        dataset = LibriSpeechAlignDataset()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    train_dataset = TrainDatasetWrapper(dataset, "train")
    test_dataset = TestDatasetWrapper(dataset, "test")
=======
    elif dataset_name == "librispeech100h":
        phoneme_parsing_mode = training_cfg.pop("phoneme_parsing_mode", "phoneme")
        vocab_path = training_cfg.pop("vocab_path", "data/vocab.json")
        dataset = LibriSpeech100h(
            phoneme_parsing_mode=phoneme_parsing_mode, vocab_path=vocab_path
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    hubert_guidance = training_cfg.pop("hubert_guidance", False)
    phonemes = training_cfg.pop("phonemes", False)

    # Inject into convformer config for the model
    convformer_cfg["phoneme_parsing_mode"] = phoneme_parsing_mode
    convformer_cfg["vocab_path"] = vocab_path

    train_dataset = TrainDatasetWrapper(
        dataset, "train", hubert_guidance=hubert_guidance, phonemes=phonemes
    )
    test_dataset = TrainDatasetWrapper(
        dataset, "train", hubert_guidance=hubert_guidance, phonemes=phonemes
    )

>>>>>>> origin/main
    # handle wandb - only initialize on main process (rank 0)
    wandb_project = training_cfg.pop("wandb_project", None)
    wandb_run_name = training_cfg.pop("wandb_run_name", None)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if training_cfg.get("report_to", "none") == "wandb" and (
        local_rank == -1 or local_rank == 0
    ):
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
        )
        logger.info(f"Initialized W&B on rank {local_rank}")

<<<<<<< HEAD
    decoder_config = DiTConfig(**decoder_cfg)

    vq_dict = encoder_cfg.pop("vq_config", None)
    vq_config = VQConfig(**vq_dict) if vq_dict else None

    dropout_dict = encoder_cfg.pop("dropout_regularizer_config", None)
    dropout_config = DropoutConfig(**dropout_dict) if dropout_dict else None

    kl_dict = encoder_cfg.pop("kl_chunk_regularizer_config", None)
    kl_config = KLChunkRegularizer(**kl_dict) if kl_dict else None

    use_bigvgan_mel = cfg_dict.get("use_bigvgan_mel", False)

    encoder_config = EncoderConfig(
        vq_config=vq_config,
        dropout_regularizer_config=dropout_config,
        kl_chunk_regularizer_config=kl_config,
        **encoder_cfg,
=======
    # Create model config
    # Build model configs from merged YAML
    if cfm_cfg.get("decoder_type") == "dit":
        decoder_config = DiTConfig(**cfm_cfg)
    elif cfm_cfg.get("decoder_type") == "Convformer":
        decoder_config = DecoderConfig(**cfm_cfg)
    else:
        raise ValueError(f"Decoder type {cfm_cfg.get('decoder_type')} not supported")
    encoder_config = ConvformerEncoderConfig(**convformer_cfg)
    # Create model
    logger.info("Creating VAE model...")
    model = VAE(
        config=VAEConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            mel_spec_config=MelSpectrogramConfig(),
        ),
        dtype=dtype,
>>>>>>> origin/main
    )

    logger.info("Creating VAE model...")
    mel_spec_config = MelSpectrogramConfig(
        use_bigvgan_mel=use_bigvgan_mel,
    )
    if mel_spec_config.use_bigvgan_mel:
        logger.info("Using BigVGAN-compatible mel spectrogram")

<<<<<<< HEAD
    # Initialize VAEConfig with global parameters (falling back to defaults)
    vae_config = VAEConfig(
        mel_dim=cfg_dict.get("mel_dim", 100),
        latent_dim=cfg_dict.get("latent_dim", 64),
        sample_rate=cfg_dict.get("sample_rate", 24000),
        compress_factor=cfg_dict.get("compress_factor", 4),
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        mel_spectrogram_config=mel_spec_config,
    )

    model = VAE(config=vae_config)

    training_cfg["learning_rate"] = float(training_cfg.get("learning_rate"))
    min_learning_rate = float(training_cfg.pop("min_learning_rate", 0.0))

    # Check for DeepSpeed config in training_cfg
    if "deepspeed" in training_cfg and training_cfg["deepspeed"]:
        logger.info(f"Using DeepSpeed config: {training_cfg['deepspeed']}")
=======
    # Extract min_learning_rate and set it in lr_scheduler_kwargs
    min_learning_rate = training_cfg.pop("min_learning_rate", None)
    if min_learning_rate is not None:
        # Initialize lr_scheduler_kwargs if it doesn't exist
        if "lr_scheduler_kwargs" not in training_cfg:
            training_cfg["lr_scheduler_kwargs"] = {}
        # Set lr_min for cosine scheduler
        training_cfg["lr_scheduler_kwargs"]["lr_min"] = float(min_learning_rate)
        logger.info(
            f"Setting minimum learning rate to {min_learning_rate} in scheduler kwargs"
        )

    # Add DeepSpeed config if provided
    if args.deepspeed:
        training_cfg["deepspeed"] = args.deepspeed
        logger.info(f"Using DeepSpeed config: {args.deepspeed}")
>>>>>>> origin/main

    from_pretrained = training_cfg.pop("from_pretrained", None)
    if from_pretrained:
        model.from_pretrained(from_pretrained)
        logger.info(f"Loaded pretrained model from {from_pretrained}")

    # Warm-up phonemizer in the main process so espeak is loaded once (evita "failed to find
    # espeak library" in evaluation o in step successivi per differenze di ambiente/fork).
    if phonemes and hasattr(model.encoder, "ctc"):
        try:
            model.encoder.ctc.get_phonemes(["test"])
        except RuntimeError as e:
            if "espeak" in str(e).lower():
                logger.error(str(e))
                raise
        logger.info("Phonemizer (espeak) warm-up OK")

    # Setup training arguments
    training_args = TrainingArguments(
        remove_unused_columns=False,  # Don't let Trainer auto-remove columns
        **training_cfg,
    )

    # Create trainer
    data_collator = DataCollator()
    trainer = VAEtrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
<<<<<<< HEAD
        min_learning_rate=min_learning_rate,
=======
        phonemes=phonemes,
>>>>>>> origin/main
    )

    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save the final model
    trainer.save_model()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
