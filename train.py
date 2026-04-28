import os
import yaml
import wandb
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from vocos import Vocos
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from modules.decoder.cfm import DiTConfig
from modules.VAE import VAE, VAEConfig
from modules.Encoder import ConvformerEncoderConfig
from modules.feature_extractor import MelSpectrogramConfig
from modules.similarity import plot_durations_on_mel
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
)

# data
from data.mls import MLSDataset
from data.libri_tts import LibriTTS
from data.audio_dataset import DataCollator
from data.audio_dataset import TrainDatasetWrapper
from data.librispeech_align import LibriSpeechAlignDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
    """Custom trainer for VAE"""

    def __init__(self, min_learning_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.min_learning_rate = min_learning_rate
        # Register granular losses
        granular_losses = [
            "audio_loss",
            "kl_loss",
            "mu_mean",
            "mu_var",
            "z_pooled_fps",
        ]
        try:
            if getattr(self.model.encoder.config, "semantic_regulation", False):
                granular_losses.append("semantic_loss")
            if getattr(self.model.encoder.config, "use_vq", False):
                granular_losses.extend(
                    [
                        "vq_loss",
                        "vq_perplexity",
                        "vq_codes_used",
                        "vq_codes_used_frac",
                    ]
                )
        except Exception:
            pass
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute the loss for the VAE"""
        if hasattr(self.control, "granular_losses") and model.training:
            audios_srs = inputs["output_audios_srs"]
            # corrupted_audios_srs = inputs["corrupted_audios_srs"]
            # Forward pass
            outputs = model(
                audios_srs=audios_srs,
                training_step=self.state.global_step,
                phoneme_alignments=inputs["phoneme_alignments"],
                # corrupted_audios_srs=corrupted_audios_srs,
            )
            audio_loss = outputs["audio_loss"]
            kl_loss = outputs["kl_loss"]
            vq_loss = outputs.get("vq_loss", None)
            vq_perplexity = outputs.get("vq_perplexity", None)
            vq_codes_used = outputs.get("vq_codes_used", None)
            vq_codes_used_frac = outputs.get("vq_codes_used_frac", None)
            semantic_loss = outputs.get("semantic_loss", None)
            mu_mean = outputs.get("mu_mean")
            mu_var = outputs.get("mu_var")
            z_pooled_fps = outputs.get("z_pooled_fps")
            loss = (
                audio_loss
                + kl_loss
                + (vq_loss if vq_loss is not None else 0.0)
                + (semantic_loss if semantic_loss is not None else 0.0)
            )

            # Accumulate granular losses
            if self.args.n_gpu > 1:
                audio_loss = audio_loss.mean()
                kl_loss = kl_loss.mean()
                if vq_loss is not None:
                    vq_loss = vq_loss.mean()
                if vq_perplexity is not None:
                    vq_perplexity = vq_perplexity.mean()
                if vq_codes_used is not None:
                    vq_codes_used = vq_codes_used.mean()
                if vq_codes_used_frac is not None:
                    vq_codes_used_frac = vq_codes_used_frac.mean()

            self.control.granular_losses["audio_loss"] += (
                audio_loss.detach() / self.args.gradient_accumulation_steps
            )
            self.control.granular_losses["kl_loss"] += (
                kl_loss.detach() / self.args.gradient_accumulation_steps
            )
            if vq_loss is not None:
                self.control.granular_losses["vq_loss"] += (
                    vq_loss.detach() / self.args.gradient_accumulation_steps
                )
            if vq_perplexity is not None:
                self.control.granular_losses["vq_perplexity"] += (
                    vq_perplexity.detach().float()
                    / self.args.gradient_accumulation_steps
                )
            if vq_codes_used is not None:
                self.control.granular_losses["vq_codes_used"] += (
                    vq_codes_used.detach().float()
                    / self.args.gradient_accumulation_steps
                )
            if vq_codes_used_frac is not None:
                self.control.granular_losses["vq_codes_used_frac"] += (
                    vq_codes_used_frac.detach().float()
                    / self.args.gradient_accumulation_steps
                )
            if semantic_loss is not None:
                if self.args.n_gpu > 1:
                    semantic_loss = semantic_loss.mean()
                self.control.granular_losses["semantic_loss"] += (
                    semantic_loss.detach() / self.args.gradient_accumulation_steps
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
            if z_pooled_fps is not None:
                val = z_pooled_fps.detach().float()
                if val.dim() > 0:
                    val = val.mean()
                self.control.granular_losses["z_pooled_fps"] += (
                    val.to(self.control.granular_losses["z_pooled_fps"].dtype)
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
            loss = (
                audio_loss
                + kl_loss
                + (semantic_loss if semantic_loss is not None else 0.0)
            )
            return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        tr_loss = args[0]
        grad_norm = args[1]
        model = args[2]
        trial = args[3]
        epoch = args[4]
        ignore_keys_for_eval = args[5]
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

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial)  # metrics=metrics
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
        # Generate samples and log to wandb
        # All processes must call this to avoid deadlock in distributed training
        self._generate_and_log_samples()

        return metrics

    def _generate_and_log_samples(self):
        """
        Generate mel spectrogram reconstructions and log to wandb.
        """
        logger.info(f"Generating reconstruction samples...")

        if getattr(self.model.config.mel_spec_config, "use_bigvgan_mel", False):
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
                (audio.to(self.args.device).to(torch.bfloat16), sr)
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
                results = self.model.encode_and_sample(
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
                    device_id=device_id,
                )

                # Convert matplotlib figure to wandb Image
                images.append(
                    wandb.Image(
                        fig,
                        caption=f"Sample {idx} - Step {self.state.global_step} - Device {device_id}",
                    )
                )
                plt.close(fig)

                # Plot durations on mel if available
                if (
                    results.get("durations") is not None
                    and results["durations"][idx] is not None
                ):
                    durations = results["durations"][idx]

                    # Use original_padding_mask for the original mel (before upsampling)
                    mel_mask = results.get(
                        "original_padding_mask", results["padding_mask"]
                    )[idx].unsqueeze(0)

                    # Get compress_factor_C from model config
                    compress_factor_C = (
                        self.model.config.encoder_config.compress_factor_C
                    )

                    seg_fig = plot_durations_on_mel(
                        mels=results["original_mel"][idx].unsqueeze(0),
                        durations=durations.unsqueeze(0),
                        mel_mask=mel_mask,
                        compress_factor_C=compress_factor_C,
                        batch_idx=0,
                        step=self.state.global_step,
                        labels=None,
                        device_id=device_id,
                    )
                    segmentation_plots.append(
                        wandb.Image(
                            seg_fig,
                            caption=f"Segmentation Sample {idx} - Step {self.state.global_step} - Device {device_id}",
                        )
                    )
                    plt.close(seg_fig)

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
                    .to(torch.bfloat16)
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
                        caption=f"Sample {idx} - Step {self.state.global_step} - Device {device_id}",
                    )
                )

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
                )

        except Exception as e:
            logger.error(f"Failed to generate samples: {e}", exc_info=True)

    def _create_mel_comparison_plot(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        original_padding_mask: torch.Tensor,
        reconstructed_padding_mask: torch.Tensor,
        sample_idx: int,
        device_id: int,
    ):
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
        original = original.float().detach().cpu().numpy()
        reconstructed = reconstructed.float().detach().cpu().numpy()
        om = original_padding_mask.detach().cpu().numpy().astype(bool)
        rm = reconstructed_padding_mask.detach().cpu().numpy().astype(bool)

        # Each spectrogram uses its own mask (lengths may differ after pooling / decoder).
        To, Fo = original.shape[0], original.shape[1]
        Tr, Fr = reconstructed.shape[0], reconstructed.shape[1]
        om = om[:To]
        rm = rm[:Tr]
        original = original[~om]
        reconstructed = reconstructed[~rm]

        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

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
            reconstructed.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
        )
        axes[1].set_title(
            f"Reconstructed Mel Spectrogram - Sample {sample_idx} - Device {device_id}"
        )
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Mel Frequency")
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        return fig


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    # Convert OmegaConf DictConfig to standard python dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    training_cfg = cfg_dict.get("training", {})
    convformer_cfg = cfg_dict.get("convformer", {})
    cfm_cfg = cfg_dict.get("cfm", {})

    # Set seed for reproducibility
    set_seed(training_cfg.get("seed", 42))

    # Set default dtype
    if training_cfg.get("bf16", False):
        torch.set_default_dtype(torch.bfloat16)
        dtype = torch.bfloat16
    elif training_cfg.get("fp16", False):
        torch.set_default_dtype(torch.float16)
        dtype = torch.float16
    else:
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create AudioDataset
    dataset_name = training_cfg.pop("dataset_name", None)
    if dataset_name == "mls":
        dataset = MLSDataset()
    elif dataset_name == "libritts":
        dataset = LibriTTS()
    elif dataset_name == "librispeech_aligned":
        dataset = LibriSpeechAlignDataset()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    train_dataset = TrainDatasetWrapper(dataset, "train")
    test_dataset = TrainDatasetWrapper(dataset, "train")

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

    # Create model config
    # Build model configs from merged YAML
    decoder_config = DiTConfig(**cfm_cfg)
    use_classic_decoder = convformer_cfg.pop("use_classic_decoder", False)
    encoder_config = ConvformerEncoderConfig(**convformer_cfg)
    # Create model
    logger.info("Creating VAE model...")
    mel_spec_config = MelSpectrogramConfig(
        use_bigvgan_mel=convformer_cfg.get("use_bigvgan_mel", False),
    )
    if mel_spec_config.use_bigvgan_mel:
        logger.info("Using BigVGAN-compatible mel spectrogram")

    model = VAE(
        config=VAEConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            mel_spec_config=mel_spec_config,
            use_classic_decoder=use_classic_decoder,
        ),
        dtype=dtype,
    )

    training_cfg["learning_rate"] = float(training_cfg.get("learning_rate"))
    min_learning_rate = float(training_cfg.pop("min_learning_rate", 0.0))

    # Check for DeepSpeed config in training_cfg
    if "deepspeed" in training_cfg and training_cfg["deepspeed"]:
        logger.info(f"Using DeepSpeed config: {training_cfg['deepspeed']}")

    from_pretrained = training_cfg.pop("from_pretrained", None)
    if from_pretrained:
        model.from_pretrained(from_pretrained)
        logger.info(f"Loaded pretrained model from {from_pretrained}")

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
        min_learning_rate=min_learning_rate,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save the final model
    trainer.save_model()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
