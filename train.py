from data.audio_dataset import TestDatasetWrapper
import os
import wandb
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from vocos import Vocos
from typing import Dict, List
import matplotlib.pyplot as plt
from modules.VAE import VAE, VAEConfig
from modules.feature_extractor import MelSpectrogramConfig
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
    encoder_cfg = cfg_dict.get("encoder", {})
    decoder_cfg = cfg_dict.get("decoder", {})

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
    elif dataset_name == "librispeech_aligned":
        dataset = LibriSpeechAlignDataset()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    train_dataset = TrainDatasetWrapper(dataset, "train")
    test_dataset = TestDatasetWrapper(dataset, "test")
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
    )

    logger.info("Creating VAE model...")
    mel_spec_config = MelSpectrogramConfig(
        use_bigvgan_mel=use_bigvgan_mel,
    )
    if mel_spec_config.use_bigvgan_mel:
        logger.info("Using BigVGAN-compatible mel spectrogram")

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
