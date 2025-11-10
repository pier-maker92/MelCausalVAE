import os
import yaml
import wandb
import torch
import logging
import argparse
from vocos import Vocos
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from .modules.cfm import DiTConfig
from .modules.VAE import VAE, VAEConfig
from .modules.Encoder import ConvformerEncoderConfig
from .modules.melspecEncoder import MelSpectrogramConfig
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
    is_torch_tpu_available,
)

# data
from .data.audio_dataset import DataCollator
from .data.libri_tts import LibriTTS
from .data.audio_dataset import TrainDatasetWrapper


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm


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
        control.granular_losses = {k: torch.tensor(0.0).to(args.device) for k in self.granular_losses}
        return control


class VAEtrainer(Trainer):
    """Custom trainer for VAE"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Register granular losses
        granular_losses = ["audio_loss", "kl_loss"]
        self.add_callback(AddGranularLossesToTrainerState(granular_losses))

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss for the VAE"""
        if hasattr(self.control, "granular_losses") and model.training:
            audios_srs = inputs["output_audios_srs"]
            # Forward pass
            outputs = model(
                audios_srs=audios_srs,
                training_step=self.state.global_step,
            )
            audio_loss = outputs.audio_loss
            kl_loss = outputs.kl_loss
            loss = audio_loss + kl_loss

            # Accumulate granular losses
            if self.args.n_gpu > 1:
                audio_loss = audio_loss.mean()
                kl_loss = kl_loss.mean()

            self.control.granular_losses["audio_loss"] += audio_loss.detach() / self.args.gradient_accumulation_steps
            self.control.granular_losses["kl_loss"] += kl_loss.detach() / self.args.gradient_accumulation_steps

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
            loss = audio_loss + kl_loss
            return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged),
                4,
            )

            # Add granular losses
            if hasattr(self.control, "granular_losses"):
                for k, v in self.control.granular_losses.items():
                    logs[k] = self._nested_gather(v).mean().item()
                    # reset the loss
                    self.control.granular_losses[k] -= self.control.granular_losses[k]

                    logs[k] = round(
                        logs[k] / (self.state.global_step - self._globalstep_last_logged),
                        4,
                    )

            logs["learning_rate"] = self._get_learning_rate()

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm if isinstance(grad_norm, float) else grad_norm.item()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

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

        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        vocos.to(self.args.device)

        # Get some samples from the eval dataset using its dataloader
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

        for i, batch in enumerate(eval_dataloader):
            if "output_audios_srs" in batch:
                audios_srs = batch["output_audios_srs"]
                audios_srs = [(audio.to(self.args.device).to(torch.bfloat16), sr) for audio, sr in audios_srs]
                break
        # Generate reconstructions
        try:
            with torch.no_grad():
                results = self.model.encode_and_sample(
                    audios_srs=audios_srs,
                    num_steps=8,
                    temperature=0.8,
                    guidance_scale=1.5,
                )
            # Create visualizations
            images = []
            # Resolve device id safely for distributed/non-distributed
            device_id = torch.distributed.get_rank()

            audios = []
            audio_paths = []
            for idx in range(len(audios_srs)):
                fig = self._create_mel_comparison_plot(
                    original=results["original_mel"][idx],
                    reconstructed=results["reconstructed_mel"][idx],
                    padding_mask=results["padding_mask"][idx],
                    sample_idx=idx,
                    device_id=device_id,
                )

                # Convert matplotlib figure to wandb Image
                images.append(
                    wandb.Image(fig, caption=f"Sample {idx} - Step {self.state.global_step} - Device {device_id}")
                )
                plt.close(fig)

                # Decode reconstructed mel to audio with Vocos
                mel = results["reconstructed_mel"][idx]  # [T, F]
                pad_mask = results["padding_mask"][idx]  # [T]
                valid_mel = mel[: (~pad_mask).sum()]

                # Shape for Vocos: [B, F, T]
                features = valid_mel.unsqueeze(0).permute(0, 2, 1).to(torch.bfloat16).to(self.args.device)
                waveform = vocos.decode(features)  # [1, samples]
                waveform = waveform.float().squeeze(0).detach().cpu()
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
                wandb.log(
                    {
                        "reconstructions": images,
                        "reconstructions_audio": audios,
                        "reconstructions_audio_paths": audio_paths,
                        "step": self.state.global_step,
                    }
                )
                logger.info(f"Successfully logged {len(images)} reconstruction samples to wandb")

        except Exception as e:
            logger.error(f"Failed to generate samples: {e}", exc_info=True)

    def _create_mel_comparison_plot(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        padding_mask: torch.Tensor,
        sample_idx: int,
        device_id: int,
    ):
        """
        Create a side-by-side comparison plot of original and reconstructed mel spectrograms.

        Args:
            original: Original mel spectrogram [T, F]
            reconstructed: Reconstructed mel spectrogram [T, F]
            padding_mask: Padding mask [T]
            sample_idx: Index of the sample

        Returns:
            matplotlib figure
        """
        # Move to CPU and convert to numpy
        original = original.float().detach().cpu().numpy()
        reconstructed = reconstructed.float().detach().cpu().numpy()
        padding_mask = padding_mask.detach().cpu().numpy()

        min_length = min(original.shape[0], reconstructed.shape[0])
        original = original[:min_length]
        reconstructed = reconstructed[:min_length]
        padding_mask = padding_mask[:min_length]

        # Mask padded regions
        original = original.copy()[~padding_mask]
        reconstructed = reconstructed.copy()[~padding_mask]

        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot original
        im1 = axes[0].imshow(original.T, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
        axes[0].set_title(f"Original Mel Spectrogram - Sample {sample_idx} - Device {device_id}")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Mel Frequency")
        plt.colorbar(im1, ax=axes[0])

        # Plot reconstructed
        im2 = axes[1].imshow(reconstructed.T, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
        axes[1].set_title(f"Reconstructed Mel Spectrogram - Sample {sample_idx} - Device {device_id}")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Mel Frequency")
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE")

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
        "convformer": load_yaml(cfg_root / "defaults" / "convformer.yaml").get("convformer", {}),
        "cfm": load_yaml(cfg_root / "defaults" / "cfm.yaml").get("cfm", {}),
    }
    custom = load_yaml(args.exp_config_path)
    merged = deep_update(defaults, custom)
    training_cfg = merged.get("training", {})
    convformer_cfg = merged.get("convformer", {})
    cfm_cfg = merged.get("cfm", {})

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
    train_dataset = TrainDatasetWrapper(LibriTTS(), "train")
    test_dataset = TrainDatasetWrapper(LibriTTS(), "test")

    # handle wandb - only initialize on main process (rank 0)
    wandb_project = training_cfg.pop("wandb_project", None)
    wandb_run_name = training_cfg.pop("wandb_run_name", None)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if training_cfg.get("report_to", "none") == "wandb" and (local_rank == -1 or local_rank == 0):
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
        )
        logger.info(f"Initialized W&B on rank {local_rank}")

    # Create model config
    # Build model configs from merged YAML
    decoder_config = DiTConfig(**cfm_cfg)
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
    )

    training_cfg["learning_rate"] = float(training_cfg.get("learning_rate"))

    # Add DeepSpeed config if provided
    if args.deepspeed:
        training_cfg["deepspeed"] = args.deepspeed
        logger.info(f"Using DeepSpeed config: {args.deepspeed}")

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
    )

    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save the final model
    trainer.save_model()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
