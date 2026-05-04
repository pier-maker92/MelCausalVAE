from data.audio_dataset import TestDatasetWrapper
import os
import wandb
import torch
import datetime
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from vocos import Vocos
from typing import Dict, List
import matplotlib.pyplot as plt
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
    EarlyStoppingCallback,
)

# data
import torch.distributed as dist
from data.audio_dataset import DataCollator
from data.audio_dataset import TrainDatasetWrapper
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from evaluate import run_evaluation

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
    def __init__(self, dataset_name: str = "dataset", **kwargs):
        self.generate_and_log_samples = kwargs.pop("generate_and_log_samples", False)
        self.min_learning_rate = kwargs.pop("min_learning_rate", 0.0)
        eval_num_samples = kwargs.pop("eval_num_samples", 100)
        self.eval_num_samples = (
            eval_num_samples if eval_num_samples is not None else float("inf")
        )
        self.run_id = kwargs.pop("run_id", "default_run")
        self.encoder_lr = kwargs.pop("encoder_lr", None)
        self.decoder_lr = kwargs.pop("decoder_lr", None)
        self.encoder_warmup_ratio = kwargs.pop("encoder_warmup_ratio", None)
        self.decoder_warmup_ratio = kwargs.pop("decoder_warmup_ratio", None)
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self._vocoder = None
        self._vocoder_type = None
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

    def create_optimizer(self):
        """
        Setup the optimizer with different learning rates for encoder and decoder if specified.
        """
        if self.optimizer is None:
            # Use specific LRs if provided, otherwise fallback to the global learning_rate
            encoder_lr = self.encoder_lr if self.encoder_lr is not None else self.args.learning_rate
            decoder_lr = self.decoder_lr if self.decoder_lr is not None else self.args.learning_rate

            logger.info(f"Setting up optimizer with encoder_lr: {encoder_lr}, decoder_lr: {decoder_lr}")

            # Define parameter groups
            # We group encoder and feature_extractor together, and decoder separately.
            encoder_params = []
            decoder_params = []
            
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if "encoder" in n or "feature_extractor" in n:
                    encoder_params.append(p)
                elif "decoder" in n:
                    decoder_params.append(p)
                else:
                    # Fallback for any other parameters (e.g. at root level)
                    encoder_params.append(p)

            optimizer_grouped_parameters = [
                {"params": encoder_params, "lr": encoder_lr},
                {"params": decoder_params, "lr": decoder_lr},
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Setup the scheduler. Support for differential warmup ratios for encoder and decoder.
        """
        if self.encoder_warmup_ratio is not None or self.decoder_warmup_ratio is not None:
            logger.info(
                f"Setting up differential warmup scheduler: "
                f"encoder_warmup_ratio: {self.encoder_warmup_ratio}, "
                f"decoder_warmup_ratio: {self.decoder_warmup_ratio}"
            )
            
            enc_warmup_steps = int(num_training_steps * (self.encoder_warmup_ratio or 0.0))
            dec_warmup_steps = int(num_training_steps * (self.decoder_warmup_ratio or 0.0))

            def enc_lambda(current_step):
                if current_step < enc_warmup_steps:
                    return float(current_step) / float(max(1, enc_warmup_steps))
                return 1.0

            def dec_lambda(current_step):
                if current_step < dec_warmup_steps:
                    return float(current_step) / float(max(1, dec_warmup_steps))
                return 1.0

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, [enc_lambda, dec_lambda]
            )
            return self.lr_scheduler

        # Default to Trainer's native scheduler (which will be 'constant' by default now)
        # unless 'cosine' is explicitly requested.
        if self.args.lr_scheduler_type != "cosine":
            return super().create_scheduler(num_training_steps, optimizer)

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
                phoneme_alignments=inputs.get("phoneme_alignments", None),
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

            # Log separate learning rates for encoder and decoder
            if self.optimizer is not None:
                for i, group in enumerate(self.optimizer.param_groups):
                    name = "lr_encoder" if i == 0 else "lr_decoder"
                    logs[name] = group["lr"]
            else:
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

        # 1. Original reconstruction samples (Mels + Audio)
        # 2. Run detailed metrics (UTMOS, WER, CER) on 100 samples
        # Only run on main process to avoid redundant computation and file conflicts
        if self.args.process_index == 0:
            if self.generate_and_log_samples:
                self._generate_and_log_samples()

            eval_dataloader = self.get_eval_dataloader(
                eval_dataset or self.eval_dataset
            )

            # We need the vocoder for evaluate.py
            # Reusing the loading logic from _generate_and_log_samples or assuming it's available
            vocoder, vocoder_type = self._get_vocoder()

            eval_metrics = run_evaluation(
                model=self.model,
                vocoder=vocoder,
                vocoder_type=vocoder_type,
                eval_dataloader=eval_dataloader,
                device=self.args.device,
                step=self.state.global_step,
                dataset_name=self.dataset_name,
                num_samples=self.eval_num_samples,
                run_id=getattr(self, "run_id", "default_run"),
            )
            metrics.update(eval_metrics)

        # Broadcast metrics from rank 0 to all other ranks in distributed setup
        if dist.is_available() and dist.is_initialized():
            broadcast_list = [metrics]
            dist.broadcast_object_list(broadcast_list, src=0)
            metrics = broadcast_list[0]

        return metrics

    def _get_vocoder(self):
        """Helper to get vocoder and its type with caching."""
        if self._vocoder is not None:
            return self._vocoder, self._vocoder_type

        if getattr(self.model.config.mel_spectrogram_config, "use_bigvgan_mel", False):
            import sys
            import os

            # Try to find bigvgan in the current workspace first
            current_dir = os.path.dirname(os.path.abspath(__file__))
            bigvgan_path = os.path.join(
                current_dir, "bigvgan/bigvgan_v2_24khz_100band_256x"
            )

            # Fallback to the hardcoded path if it doesn't exist locally
            if not os.path.exists(bigvgan_path):
                bigvgan_path = "/home/piermel/links/scratch/MelCausalVAE/bigvgan/bigvgan_v2_24khz_100band_256x"

            if bigvgan_path not in sys.path:
                sys.path.append(bigvgan_path)
            import bigvgan

            vocoder = bigvgan.BigVGAN.from_pretrained(
                bigvgan_path, use_cuda_kernel=False
            )
            vocoder_type = "bigvgan"
        else:
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            vocoder_type = "vocos"

        vocoder.to(self.args.device)
        vocoder.eval()
        self._vocoder = vocoder
        self._vocoder_type = vocoder_type
        return vocoder, vocoder_type

    def _generate_and_log_samples(self):
        """
        Generate mel spectrogram reconstructions and log to wandb.
        """
        logger.info(f"Generating reconstruction samples...")
        vocoder, vocoder_type = self._get_vocoder()

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
                original=results["feature_extractor_output"].audio_features[idx],
                reconstructed=results["decoder_output"].audio_features[idx],
                original_padding_mask=results["feature_extractor_output"].padding_mask[
                    idx
                ],
                reconstructed_padding_mask=results["decoder_output"].padding_mask[idx],
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
            mel = results["decoder_output"].audio_features[idx]  # [T, F]
            pad_mask = results["decoder_output"].padding_mask[idx]  # [T] True = padded
            T = min(mel.shape[0], pad_mask.shape[0])
            mel = mel[:T]
            pad_mask = pad_mask[:T]
            valid_mel = mel[~pad_mask]

            # Shape for Vocos/BigVGAN: [B, F, T]
            features = (
                valid_mel.unsqueeze(0).permute(0, 2, 1).float().to(self.args.device)
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

        # Log to wandb as a table for better visualization
        # Log to wandb as simple lists (reverting to what worked before but with correct data)
        if wandb.run is not None:
            wandb.log(
                {
                    "reconstructions": images,
                    "reconstructions_audio": audios,
                },
                step=self.state.global_step,
            )

            logger.info(
                f"Successfully logged {len(images)} reconstruction samples to wandb"
            )

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

    accelerator = Accelerator()
    logger.info(f"Using device: {accelerator.device}")
    logger.info(f"Mixed precision: {accelerator.state.mixed_precision}")

    # Create AudioDataset
    dataset_name = training_cfg.pop("dataset_name", None)
    if dataset_name == "mls":
        from data.mls import MLSDataset

        dataset = MLSDataset()
    elif dataset_name == "libritts":
        from data.libri_tts import LibriTTS

        dataset = LibriTTS()
    elif dataset_name in ["librispeech_aligned", "librispeech-aligned"]:
        from data.librispeech_align import LibriSpeechAlignDataset

        dataset = LibriSpeechAlignDataset()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    train_dataset = TrainDatasetWrapper(dataset, "train")
    test_dataset = TestDatasetWrapper(dataset, "test")
    # handle wandb - only initialize on main process
    wandb_project = training_cfg.pop("wandb_project", None)
    wandb_run_name = training_cfg.pop("wandb_run_name", None)
    wandb_id = training_cfg.pop("wandb_id", None)
    if training_cfg.get("report_to", "none") == "wandb" and accelerator.is_main_process:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume="allow" if wandb_id else None,
        )
        logger.info(f"Initialized W&B on main process (id: {wandb_id})")

    from modules.builder import build_model

    logger.info("Creating VAE model...")
    model = build_model(cfg_dict)
    if getattr(model.config.mel_spectrogram_config, "use_bigvgan_mel", False):
        logger.info("Using BigVGAN-compatible mel spectrogram")

    training_cfg["learning_rate"] = float(training_cfg.get("learning_rate"))
    min_learning_rate = float(training_cfg.pop("min_learning_rate", 0.0))
    early_stopping_patience = training_cfg.pop("early_stopping_patience", None)
    generate_and_log_samples = training_cfg.pop("generate_and_log_samples", True)
    eval_num_samples = training_cfg.pop("eval_num_samples", 100)

    # Check for DeepSpeed config in training_cfg
    if "deepspeed" in training_cfg and training_cfg["deepspeed"]:
        logger.info(f"Using DeepSpeed config: {training_cfg['deepspeed']}")

    from_pretrained = training_cfg.pop("from_pretrained", None)
    if from_pretrained:
        model.from_pretrained(from_pretrained)
        logger.info(f"Loaded pretrained model from {from_pretrained}")

    if "lr_scheduler_type" not in training_cfg:
        training_cfg["lr_scheduler_type"] = "constant"

    # Extract optional differential learning rates and warmup ratios
    encoder_lr = training_cfg.pop("encoder_lr", None)
    decoder_lr = training_cfg.pop("decoder_lr", None)
    encoder_warmup_ratio = training_cfg.pop("encoder_warmup_ratio", None)
    decoder_warmup_ratio = training_cfg.pop("decoder_warmup_ratio", None)
    
    if encoder_lr is not None: encoder_lr = float(encoder_lr)
    if decoder_lr is not None: decoder_lr = float(decoder_lr)
    if encoder_warmup_ratio is not None: encoder_warmup_ratio = float(encoder_warmup_ratio)
    if decoder_warmup_ratio is not None: decoder_warmup_ratio = float(decoder_warmup_ratio)

    # Create unique run ID for evaluation outputs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = wandb_run_name or "run"
    run_id = f"{run_name}_{timestamp}"

    # Setup training arguments
    training_args = TrainingArguments(
        remove_unused_columns=False,  # Don't let Trainer auto-remove columns
        **training_cfg,
    )
    logger.info(f"TrainingArgs bf16 enabled: {training_args.bf16}")

    # Create trainer
    data_collator = DataCollator()
    trainer = VAEtrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        min_learning_rate=min_learning_rate,
        dataset_name=dataset_name or "librispeech",
        generate_and_log_samples=generate_and_log_samples,
        eval_num_samples=eval_num_samples,
        run_id=run_id,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        encoder_warmup_ratio=encoder_warmup_ratio,
        decoder_warmup_ratio=decoder_warmup_ratio,
    )

    # Add Early Stopping if enabled
    if training_args.load_best_model_at_end and early_stopping_patience is not None:
        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )
        logger.info(f"Enabled Early Stopping with patience {early_stopping_patience}")

    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save the final model
    trainer.save_model()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

# python train.py -m train settings=exps/vq.yaml
