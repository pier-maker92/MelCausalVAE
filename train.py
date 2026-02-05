import os
import yaml
import wandb
import torch
import logging
import argparse
from vocos import Vocos
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from modules.cfm import DiTConfig
from modules.VAE import VAE, VAEConfig
from modules.Encoder import ConvformerEncoderConfig
from modules.melspecEncoder import MelSpectrogramConfig
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
from data.libri_tts import LibriTTS
from data.librispeechHubert import LibriSpeech100h
from data.audio_dataset import TrainDatasetWrapper
from data.audio_dataset import DataCollator, HubertDatasetWrapper

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # Cosine annealing with minimum learning rate
        cosine_value = 0.5 * (1.0 + torch.cos(torch.tensor(num_cycles * 2.0 * 3.141592653589793 * progress)))
        # Scale cosine from [lr_min, initial_lr] instead of [0, initial_lr]
        return (lr_min / initial_lr) + (1.0 - lr_min / initial_lr) * cosine_value

    return LambdaLR(optimizer, lr_lambda, last_epoch)


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
        granular_losses = ["audio_loss", "kl_loss", "mu_mean", "mu_var"]
        try:
            if getattr(self.model.encoder.config, "semantic_regulation", False):
                granular_losses.append("semantic_loss")
        except Exception:
            pass
        self.add_callback(AddGranularLossesToTrainerState(granular_losses))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute the loss for the VAE"""
        if hasattr(self.control, "granular_losses") and model.training:
            audios_srs = inputs["output_audios_srs"]
            hubert_guidance = inputs.get("hubert_guidance", None)

            # Forward pass
            outputs = model(
                audios_srs=audios_srs,
                training_step=self.state.global_step,
                hubert_guidance=hubert_guidance,
            )
            audio_loss = outputs.audio_loss
            kl_loss = outputs.kl_loss
            semantic_loss = getattr(outputs, "semantic_loss", None)
            mu_mean = getattr(outputs, "mu_mean")
            mu_var = getattr(outputs, "mu_var")
            loss = audio_loss + kl_loss + (semantic_loss if semantic_loss is not None else 0.0)

            # Accumulate granular losses
            if self.args.n_gpu > 1:
                audio_loss = audio_loss.mean()
                kl_loss = kl_loss.mean()

            self.control.granular_losses["audio_loss"] += audio_loss.detach() / self.args.gradient_accumulation_steps
            self.control.granular_losses["kl_loss"] += kl_loss.detach() / self.args.gradient_accumulation_steps
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
                    val.to(self.control.granular_losses["mu_mean"].dtype) / self.args.gradient_accumulation_steps
                )
            if mu_var is not None:
                val = mu_var.detach().float()
                if val.dim() > 0:
                    val = val.mean()
                self.control.granular_losses["mu_var"] += (
                    val.to(self.control.granular_losses["mu_var"].dtype) / self.args.gradient_accumulation_steps
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
            loss = audio_loss + kl_loss + (semantic_loss if semantic_loss is not None else 0.0)
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
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:

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

                    avg_val = logs[k] / (self.state.global_step - self._globalstep_last_logged)
                    if k in ("mu_mean", "mu_var"):
                        logs[k] = round(avg_val, 8)
                    else:
                        logs[k] = round(avg_val, 4)

            logs["learning_rate"] = self._get_learning_rate()

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm if isinstance(grad_norm, float) else grad_norm.item()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time=start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(
                model,
                trial,
            )  # metrics=metrics
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
        hubert_guidance = batch.get("hubert_guidance", None)

        # Generate reconstructions
        with torch.no_grad():
            results = self.model.encode_and_sample(
                audios_srs=audios_srs,
                num_steps=8,
                temperature=0.8,
                guidance_scale=1.3,
                hubert_guidance=hubert_guidance,
            )
        # Create visualizations
        images = []
        # Resolve device id safely for distributed/non-distributed
        device_id = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

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
            # now original audio decoded with Vocos
            original_mel = results["original_mel"][idx]
            original_mel = original_mel[: (~pad_mask).sum()]
            original_mel = original_mel.unsqueeze(0).permute(0, 2, 1).to(torch.bfloat16).to(self.args.device)
            original_waveform = vocos.decode(original_mel)
            original_waveform = original_waveform.float().squeeze(0).detach().cpu()
            original_waveform = original_waveform / (original_waveform.abs().max() + 1e-8)
            audios.append(
                wandb.Audio(
                    original_waveform.numpy(),
                    sample_rate=sr,
                    caption=f"Original Audio - Sample {idx} - Step {self.state.global_step} - Device {device_id}",
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
    dataset_name = training_cfg.pop("dataset_name", None)
    if dataset_name == "mls":
        dataset = MLSDataset()
    elif dataset_name == "libritts":
        dataset = LibriTTS()
    elif dataset_name == "librispeech100h":
        dataset = LibriSpeech100h()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    hubert_guidance = training_cfg.pop("hubert_guidance", False)
    train_dataset = TrainDatasetWrapper(dataset, "train", hubert_guidance=hubert_guidance)
    test_dataset = TrainDatasetWrapper(dataset, "test", hubert_guidance=hubert_guidance)

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

    # Extract min_learning_rate and set it in lr_scheduler_kwargs
    min_learning_rate = training_cfg.pop("min_learning_rate", None)
    if min_learning_rate is not None:
        # Initialize lr_scheduler_kwargs if it doesn't exist
        if "lr_scheduler_kwargs" not in training_cfg:
            training_cfg["lr_scheduler_kwargs"] = {}
        # Set lr_min for cosine scheduler
        training_cfg["lr_scheduler_kwargs"]["lr_min"] = float(min_learning_rate)
        logger.info(f"Setting minimum learning rate to {min_learning_rate} in scheduler kwargs")

    # Add DeepSpeed config if provided
    if args.deepspeed:
        training_cfg["deepspeed"] = args.deepspeed
        logger.info(f"Using DeepSpeed config: {args.deepspeed}")

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
    )

    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save the final model
    trainer.save_model()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
