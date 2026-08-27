import os
import math
import wandb
import torch
import hydra
import logging
import datetime
from typing import Dict, List
from omegaconf import DictConfig, OmegaConf
from accelerate import InitProcessGroupKwargs
from dicodec.data.audio_dataset import TrainDatasetWrapper, TestDatasetWrapper
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
)


# data
import torch.distributed as dist
from accelerate import Accelerator
from eval import run_evaluation
from dicodec.modules.builder import build_model
from dicodec.data.audio_dataset import DataCollator
from dicodec.data.audio_dataset import TrainDatasetWrapper

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


class KLWarmupRatioCallback(TrainerCallback):
    """Resolves kl_loss_warmup_ratio into kl_loss_warmup_steps at training start."""

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        model = kwargs.get("model")
        if model is not None and hasattr(model, "module"):
            model = model.module
        if model is None or not hasattr(model, "encoder"):
            return control

        enc_cfg = model.encoder.config
        ratio = getattr(enc_cfg, "kl_loss_warmup_ratio", None)
        if ratio is not None:
            warmup_steps = int(state.max_steps * ratio)
            enc_cfg.kl_loss_warmup_steps = warmup_steps
            logger.info(
                f"KLWarmupRatioCallback: kl_loss_warmup_ratio={ratio} "
                f"-> kl_loss_warmup_steps={warmup_steps} (max_steps={state.max_steps})"
            )
        return control


class Dicodectrainer(Trainer):
    def __init__(self, dataset_name: str = "dataset", **kwargs):
        self.min_learning_rate = kwargs.pop("min_learning_rate")
        eval_num_samples = kwargs.pop("eval_num_samples")
        self.eval_num_samples = (
            eval_num_samples if eval_num_samples is not None else float("inf")
        )
        self.run_id = kwargs.pop("run_id")
        self.encoder_lr = kwargs.pop("encoder_lr")
        self.decoder_lr = kwargs.pop("decoder_lr")
        self.encoder_min_lr = kwargs.pop("encoder_min_lr")
        self.decoder_min_lr = kwargs.pop("decoder_min_lr")
        self.encoder_warmup_ratio = kwargs.pop("encoder_warmup_ratio")
        self.decoder_warmup_ratio = kwargs.pop("decoder_warmup_ratio")

        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        # Register granular losses
        granular_losses = [
            "audio_loss",
            "kl_loss",
            "mu_mean",
            "mu_var",
        ]

        self.add_callback(AddGranularLossesToTrainerState(granular_losses))
        self.add_callback(KLWarmupRatioCallback())

    def create_optimizer(self):
        """
        Setup the optimizer with different learning rates for encoder and decoder if specified.
        """
        if self.optimizer is None:
            # Use specific LRs if provided, otherwise fallback to the global learning_rate
            encoder_lr = (
                self.encoder_lr
                if self.encoder_lr is not None
                else self.args.learning_rate
            )
            decoder_lr = (
                self.decoder_lr
                if self.decoder_lr is not None
                else self.args.learning_rate
            )

            logger.info(
                f"Setting up optimizer with encoder_lr: {encoder_lr}, decoder_lr: {decoder_lr}"
            )

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

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Setup the scheduler. Support for differential warmup ratios and cosine annealing
        with different minimum learning rates for encoder and decoder.
        """
        if optimizer is None:
            optimizer = self.optimizer

        use_cosine = self.args.lr_scheduler_type == "cosine"
        has_warmup = (
            self.encoder_warmup_ratio is not None
            or self.decoder_warmup_ratio is not None
        )

        if has_warmup or use_cosine:
            logger.info(
                f"Setting up differential LambdaLR scheduler (cosine: {use_cosine}): "
                f"encoder_warmup: {self.encoder_warmup_ratio}, "
                f"decoder_warmup: {self.decoder_warmup_ratio}"
            )

            enc_warmup_steps = int(
                num_training_steps * (self.encoder_warmup_ratio or 0.0)
            )
            dec_warmup_steps = int(
                num_training_steps * (self.decoder_warmup_ratio or 0.0)
            )

            encoder_lr = (
                self.encoder_lr
                if self.encoder_lr is not None
                else self.args.learning_rate
            )
            decoder_lr = (
                self.decoder_lr
                if self.decoder_lr is not None
                else self.args.learning_rate
            )

            enc_min_lr = (
                self.encoder_min_lr
                if self.encoder_min_lr is not None
                else self.min_learning_rate
            )
            dec_min_lr = (
                self.decoder_min_lr
                if self.decoder_min_lr is not None
                else self.min_learning_rate
            )

            def get_lr_lambda(current_step, warmup_steps, initial_lr, min_lr):
                # 1. Warmup phase
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))

                # 2. Constant phase if no cosine decay
                if not use_cosine:
                    return 1.0

                # 3. Cosine annealing phase
                progress = float(current_step - warmup_steps) / float(
                    max(1, num_training_steps - warmup_steps)
                )
                progress = min(1.0, max(0.0, progress))

                min_lr_ratio = min_lr / initial_lr if initial_lr > 0 else 0.0
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

            enc_lambda = lambda step: get_lr_lambda(
                step, enc_warmup_steps, encoder_lr, enc_min_lr
            )
            dec_lambda = lambda step: get_lr_lambda(
                step, dec_warmup_steps, decoder_lr, dec_min_lr
            )

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, [enc_lambda, dec_lambda]
            )
            return self.lr_scheduler

        # Default fallback
        return super().create_scheduler(num_training_steps, optimizer)

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)

        # Solo il processo principale (Rank 0) deve salvare il file di configurazione
        if not self.args.should_save:
            return

        # Save DicodecConfig alongside the model
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
            audios_srs = inputs["audio_input_srs"]
            output = model(
                audios_srs=audios_srs,
                audio_16khz=inputs["16k_audio"],
                training_step=self.state.global_step,
            )
            audio_loss = output.audio_loss
            kl_loss = output.kl_loss
            loss = audio_loss + kl_loss

            # Accumulate granular losses
            flat_metrics = {
                "audio_loss": audio_loss,
                "kl_loss": kl_loss,
                "mu_mean": getattr(output, "mu_mean", None),
                "mu_var": getattr(output, "mu_var", None),
            }

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
            self.log(metrics)

            # Determine if this is the new best model
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial
            )

            # If we are saving only the best, update should_save
            if getattr(self.args, "save_strategy", None) == "best":
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)

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

            eval_dataloader = self.get_eval_dataloader(
                eval_dataset or self.eval_dataset
            )

            eval_metrics = run_evaluation(
                model=self.model,
                eval_dataloader=eval_dataloader,
                device=self.args.device,
                step=self.state.global_step,
                dataset_name=self.dataset_name,
                num_samples=self.eval_num_samples,
                run_id=getattr(self, "run_id", "default_run"),
            )
            # Add prefix for Trainer's best model logic
            eval_metrics = {
                f"{metric_key_prefix}_{k}": v for k, v in eval_metrics.items()
            }
            metrics.update(eval_metrics)

        # Broadcast metrics from rank 0 to all other ranks in distributed setup
        if dist.is_available() and dist.is_initialized():
            broadcast_list = [metrics]
            dist.broadcast_object_list(broadcast_list, src=0)
            metrics = broadcast_list[0]

        return metrics


def get_dataset(training_cfg):
    dataset_name = training_cfg.pop("dataset_name")
    if dataset_name == "mls":
        from dicodec.data.mls import MLSDataset

        dataset = MLSDataset()
    elif dataset_name == "librispeech":
        from dicodec.data.librispeech import LibriSpeechDataset

        dataset = LibriSpeechDataset()
    elif dataset_name == "libritts-r":
        from dicodec.data.libri_tts_r import LibriTTSR

        dataset = LibriTTSR()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    train_dataset = TrainDatasetWrapper(dataset, "train")
    test_dataset = TestDatasetWrapper(dataset, "test")
    return dataset_name, train_dataset, test_dataset


def maybe_init_wandb(training_cfg, accelerator):
    wandb_project = training_cfg.pop("wandb_project")
    wandb_run_name = training_cfg.pop("wandb_run_name")
    wandb_id = training_cfg.pop("wandb_id")
    if training_cfg["report_to"] == "wandb" and accelerator.is_main_process:
        import wandb

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume="allow" if wandb_id else None,
        )
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Initialized W&B on main process (id: {wandb_id})")
    return wandb_run_name


def get_config(training_cfg):
    training_cfg["learning_rate"] = float(training_cfg.get("learning_rate", 1e-4))

    min_learning_rate = float(training_cfg.pop("min_learning_rate"))
    eval_num_samples = training_cfg.pop("eval_num_samples")
    from_pretrained = training_cfg.pop("from_pretrained")

    def parse_float_opt(key):
        val = training_cfg.pop(key)
        return float(val) if val is not None else None

    encoder_lr = parse_float_opt("encoder_lr")
    decoder_lr = parse_float_opt("decoder_lr")
    encoder_min_lr = parse_float_opt("encoder_min_lr")
    decoder_min_lr = parse_float_opt("decoder_min_lr")
    encoder_warmup_ratio = parse_float_opt("encoder_warmup_ratio")
    decoder_warmup_ratio = parse_float_opt("decoder_warmup_ratio")

    return (
        min_learning_rate,
        eval_num_samples,
        from_pretrained,
        encoder_lr,
        decoder_lr,
        encoder_min_lr,
        decoder_min_lr,
        encoder_warmup_ratio,
        decoder_warmup_ratio,
    )


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    # Convert OmegaConf DictConfig to standard python dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    training_cfg = cfg_dict["training"]

    # Increase timeout to 2 hours for lengthy evaluation on Rank 0
    kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    logger.info(f"Using device: {accelerator.device}")
    logger.info(f"Mixed precision: {accelerator.state.mixed_precision}")

    # Set seed for reproducibility
    set_seed(training_cfg["seed"])

    dataset_name, train_dataset, test_dataset = get_dataset(training_cfg)
    wandb_run_name = maybe_init_wandb(training_cfg, accelerator)

    logger.info("Creating Dicodec model...")
    model = build_model(cfg_dict)


    (
        min_learning_rate,
        eval_num_samples,
        from_pretrained,
        encoder_lr,
        decoder_lr,
        encoder_min_lr,
        decoder_min_lr,
        encoder_warmup_ratio,
        decoder_warmup_ratio,
    ) = get_config(training_cfg)

    # Check for DeepSpeed config in training_cfg
    if "deepspeed" in training_cfg and training_cfg["deepspeed"]:
        logger.info(f"Using DeepSpeed config: {training_cfg['deepspeed']}")

    if from_pretrained:
        model.from_pretrained(from_pretrained)
        logger.info(f"Loaded pretrained model from {from_pretrained}")

    # Create unique run ID for evaluation outputs
    # If run_id is provided via command line (run_job.sh), use it.
    run_id = training_cfg.pop("run_id")
    if run_id is None:
        date_dir = datetime.datetime.now().strftime("%d-%B-%Y")
        time_dir = datetime.datetime.now().strftime("%H:%M:%S")
        run_name = wandb_run_name or "run"
        run_id = f"{date_dir}/{time_dir}/{run_name}"

    # Setup training arguments
    training_args = TrainingArguments(
        remove_unused_columns=False,  # Don't let Trainer auto-remove columns
        ddp_timeout=7200,  # 2 hours timeout for long evaluation on Rank 0
        **training_cfg,
    )
    logger.info(f"TrainingArgs bf16 enabled: {training_args.bf16}")

    # Create trainer
    data_collator = DataCollator()
    trainer = Dicodectrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        min_learning_rate=min_learning_rate,
        dataset_name=dataset_name or "librispeech",
        eval_num_samples=eval_num_samples,
        run_id=run_id,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        encoder_min_lr=encoder_min_lr,
        decoder_min_lr=decoder_min_lr,
        encoder_warmup_ratio=encoder_warmup_ratio,
        decoder_warmup_ratio=decoder_warmup_ratio,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save the final model
    trainer.save_model()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

# python train.py -m train settings=exps/vq.yaml
