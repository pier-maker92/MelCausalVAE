"""
Training script for Semantic Mapper

Trains a semantic mapper to align VAE latent representations with semantic features
from SeamlessM4Tv2Encoder. The VAE is kept frozen during training.
"""

import os
import yaml
import wandb
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict
from .modules.VAE import VAE, VAEConfig
from .modules.Encoder import ConvformerEncoderConfig
from .modules.melspecEncoder import MelSpectrogramConfig
from .modules.cfm import DiTConfig
from .modules.semantic_module import SeamlessM4Tv2Encoder
from .modules.semantic_mapper import Z2YMapper, SemanticMapperConfig
from .modules.regulator import InterpolateRegulator
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
from .data.mls import MLSDataset
from .data.libri_tts import LibriTTS
from .data.audio_dataset import DataCollator
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

    def __init__(self):
        self.granular_losses = ["semantic_loss", "cons_loss"]

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        control.granular_losses = {k: torch.tensor(0.0).to(args.device) for k in self.granular_losses}
        logger.info(f"Initialized granular losses: {list(control.granular_losses.keys())}")
        return control


class SemanticMapperWrapper(torch.nn.Module):
    """
    Wrapper that combines frozen VAE encoder and trainable Z2Y mapper.

    Flow:
        1. VAE encoder (frozen) -> z
        2. Z2YMapper (trainable) -> y
        3. SeamlessM4Tv2 (frozen) -> semantic_features
        4. InterpolateRegulator -> semantic_loss (cosine similarity between y and semantic)
        5. Total loss = semantic_loss + cons_loss(y, z)
    """

    def __init__(
        self,
        vae: VAE,
        semantic_encoder: SeamlessM4Tv2Encoder,
        mapper: Z2YMapper,
        regulator: InterpolateRegulator,
        lambda_cons: float = 0.2,
    ):
        super().__init__()
        self.vae = vae
        self.semantic_encoder = semantic_encoder
        self.mapper = mapper
        self.regulator = regulator
        self.lambda_cons = lambda_cons

        # Add config for DeepSpeed ZeRO-3 compatibility
        self.config = mapper.config

        # Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False

        # Freeze semantic encoder
        for param in self.semantic_encoder.parameters():
            param.requires_grad = False

        # Freeze regulator (only used for computing loss)
        for param in self.regulator.parameters():
            param.requires_grad = False

        # Only mapper parameters are trainable
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.mapper.parameters() if p.requires_grad):,}")
        logger.info(f"Frozen VAE parameters: {sum(p.numel() for p in self.vae.parameters()):,}")
        logger.info(
            f"Frozen semantic encoder parameters: {sum(p.numel() for p in self.semantic_encoder.parameters()):,}"
        )
        logger.info(f"Frozen regulator parameters: {sum(p.numel() for p in self.regulator.parameters()):,}")

    def forward(self, audios_srs):
        """
        Forward pass:
        1. Extract VAE latents z (frozen)
        2. Map z -> y through trainable mapper
        3. Extract semantic features (frozen)
        4. Compute semantic_loss via InterpolateRegulator(y, semantic)
        5. Compute cons_loss = MSE(y, z)
        """
        # Get VAE latents (frozen)
        with torch.no_grad():
            encoded_audios = self.vae.wav2mel(audios_srs)
            convformer_output = self.vae.encoder(
                x=encoded_audios.audio_features,
                padding_mask=encoded_audios.padding_mask,
                step=None,
            )
            z = convformer_output.z  # Use z as the latent
            z_padding_mask = convformer_output.padding_mask

            # Get semantic features (frozen)
            semantic_output = self.semantic_encoder(audios_srs)
            semantic_features = semantic_output.semantic_features
            semantic_padding_mask = semantic_output.padding_mask

        # Map z -> y (trainable)
        mapper_output = self.mapper(z=z)
        y = mapper_output.y
        # cons_loss = mapper_output.cons_loss

        # Compute semantic distillation loss via regulator
        semantic_loss = self.regulator(
            guidance=semantic_features,  # [B, T_s, 1024]
            guidance_mask=semantic_padding_mask,  # [B, T_s] (1=pad, 0=real)
            target=y,  # [B, T_z, 64]
            target_padding_mask=z_padding_mask,  # [B, T_z] (1=pad, 0=real)
        )

        # Total loss
        total_loss = semantic_loss  # + cons_loss

        # Return output in expected format
        class Output:
            def __init__(self, loss, semantic_loss):
                self.loss = loss
                self.semantic_loss = semantic_loss
                # self.cons_loss = cons_loss

        return Output(loss=total_loss, semantic_loss=semantic_loss)  # , cons_loss=cons_loss)


class SemanticMapperTrainer(Trainer):
    """Custom trainer for semantic mapper"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_callback(AddGranularLossesToTrainerState())
        logger.info("Added granular losses callback to trainer")

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss for the semantic mapper"""
        audios_srs = inputs["output_audios_srs"]

        # Forward pass - always returns loss, semantic_loss, cons_loss
        outputs = model(audios_srs=audios_srs)

        loss = outputs.loss
        semantic_loss = outputs.semantic_loss
        # cons_loss = outputs.cons_loss

        # Accumulate granular losses if training and callback initialized
        if hasattr(self.control, "granular_losses"):
            # Handle multi-GPU
            if self.args.n_gpu > 1:
                semantic_loss_detached = semantic_loss.mean().detach()
                # cons_loss_detached = cons_loss.mean().detach()
            else:
                semantic_loss_detached = semantic_loss.detach()
                # cons_loss_detached = cons_loss.detach()

            self.control.granular_losses["semantic_loss"] += (
                semantic_loss_detached / self.args.gradient_accumulation_steps
            )
            # self.control.granular_losses["cons_loss"] += cons_loss_detached / self.args.gradient_accumulation_steps

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

                    avg_val = logs[k] / (self.state.global_step - self._globalstep_last_logged)
                    logs[k] = round(avg_val, 4)

            logs["learning_rate"] = self._get_learning_rate()

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm if isinstance(grad_norm, float) else grad_norm.item()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        # Skip evaluation completely
        # if self.control.should_evaluate:
        #     metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
        #     self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Semantic Mapper")

    # Config arguments
    parser.add_argument(
        "--exp-config",
        dest="exp_config_path",
        type=Path,
        required=True,
        help="Path to experiment YAML config",
    )

    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained VAE checkpoint",
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

    # --- Load configuration ---
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
        "semantic_mapper": {},
    }
    custom = load_yaml(args.exp_config_path)
    merged = deep_update(defaults, custom)
    training_cfg = merged.get("training", {})
    convformer_cfg = merged.get("convformer", {})
    cfm_cfg = merged.get("cfm", {})
    mapper_cfg = merged.get("semantic_mapper", {})

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

    # Create dataset
    dataset_name = training_cfg.pop("dataset_name", None)
    if dataset_name == "mls":
        dataset = MLSDataset()
    elif dataset_name == "libritts":
        dataset = LibriTTS()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    train_dataset = TrainDatasetWrapper(dataset, "train")

    # Handle wandb
    wandb_project = training_cfg.pop("wandb_project", None)
    wandb_run_name = training_cfg.pop("wandb_run_name", None)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if training_cfg.get("report_to", "none") == "wandb" and (local_rank == -1 or local_rank == 0):
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
        )
        logger.info(f"Initialized W&B on rank {local_rank}")

    # Load pretrained VAE (frozen)
    logger.info(f"Loading pretrained VAE from {args.vae_checkpoint}...")
    decoder_config = DiTConfig(**cfm_cfg)
    encoder_config = ConvformerEncoderConfig(**convformer_cfg)
    vae = VAE(
        config=VAEConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            mel_spec_config=MelSpectrogramConfig(),
            add_semantic_distillation=False,
            add_semantic_mapper=False,
            semantic_mapper_config=None,
        ),
        dtype=dtype,
    )
    vae.from_pretrained(args.vae_checkpoint)
    vae.eval()
    logger.info("✓ Loaded and froze VAE")

    # Initialize semantic encoder (frozen)
    logger.info("Initializing semantic encoder...")
    semantic_encoder = SeamlessM4Tv2Encoder(dtype=dtype)
    semantic_encoder.eval()
    logger.info("✓ Initialized and froze semantic encoder")

    # Initialize Z2Y mapper (trainable)
    logger.info("Initializing Z2Y mapper...")
    mapper_config = SemanticMapperConfig(
        z_dim=encoder_config.latent_dim,
        n_layers=mapper_cfg.get("n_layers", 6),
        hidden_dim=mapper_cfg.get("hidden_dim", 128),
    )
    mapper = Z2YMapper(mapper_config)
    logger.info("✓ Initialized Z2Y mapper")

    # Initialize InterpolateRegulator for semantic distillation (frozen)
    logger.info("Initializing InterpolateRegulator...")
    regulator = InterpolateRegulator(
        depth=2,
        in_channels=1024,  # SeamlessM4Tv2 output dimension
        channels=256,
        out_channels=encoder_config.latent_dim,  # z_dim
        groups=1,
        is_causal=True,
    )
    regulator.eval()
    logger.info("✓ Initialized and froze InterpolateRegulator")

    # Create wrapper model
    model = SemanticMapperWrapper(
        vae=vae,
        semantic_encoder=semantic_encoder,
        mapper=mapper,
        regulator=regulator,
        lambda_cons=mapper_cfg.get("lambda_cons", 0.2),
    )

    training_cfg["learning_rate"] = float(training_cfg.get("learning_rate", 1e-4))

    # Add DeepSpeed config if provided
    if args.deepspeed:
        training_cfg["deepspeed"] = args.deepspeed
        logger.info(f"Using DeepSpeed config: {args.deepspeed}")

    # Setup training arguments
    # FIXME this is just hardcoded for now
    from_pretrained = training_cfg.pop("from_pretrained", None)
    training_args = TrainingArguments(
        remove_unused_columns=False,
        **training_cfg,
    )

    # Create trainer
    data_collator = DataCollator()
    trainer = SemanticMapperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # No eval dataset
        data_collator=data_collator,
    )

    # Start training
    logger.info("Starting semantic mapper training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save the final mapper
    logger.info("Saving semantic mapper...")
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save only the mapper weights
    mapper_path = output_dir / "semantic_mapper.pt"
    torch.save(
        {
            "mapper_state_dict": model.mapper.state_dict(),
            "mapper_config": mapper_config.__dict__,
        },
        mapper_path,
    )
    logger.info(f"✓ Saved semantic mapper to {mapper_path}")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
