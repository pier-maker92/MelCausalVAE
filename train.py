import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Sequence
import yaml

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    set_seed,
)

from .data.libri_tts import LibriTTS
from .data.audio_dataset import DataCollator
from .modules.cfm import LightCFMTalkingHead, LightCFMTalkingHeadConfig


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class LightCFMTrainer(Trainer):
    """Custom trainer for Light CFM Talking Head"""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss for the Light CFM model"""
        audios_srs = inputs["output_audios_srs"]
        # Forward pass
        outputs = model(
            audios_srs=audios_srs,
        )
        audio_loss = outputs.audio_loss
        kl_loss = outputs.kl_loss
        loss = audio_loss + kl_loss
        # log both total loss and kl loss
        try:
            self.log(
                {
                    "audio_loss": float(audio_loss.detach().mean().cpu().item()),
                    "kl_loss": float(kl_loss.detach().mean().cpu().item()),
                    "total_loss": float(loss.detach().mean().cpu().item()),
                }
            )
        except Exception:
            raise Exception("Failed to log loss and kl loss")
        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Light CFM Talking Head")

    # Data arguments
    parser.add_argument(
        "--exp-config",
        dest="exp_config_path",
        type=Path,
        required=True,
        help="Path to experiment YAML overriding defaults in configs/defaults",
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
    elif training_cfg.get("fp16", False):
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(torch.float32)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create AudioDataset
    train_dataset = LibriTTS(
        dataset=datasets.train_dataset,
        tokenizer=tokenizer,
        data_args=data_args,
    )

    eval_dataset = None

    # Create model config
    # Build model configs from merged YAML
    encoder_config = ConvformerEncoderConfig(**convformer_cfg)
    model_config = LightCFMTalkingHeadConfig(
        **cfm_cfg,
    )

    # Create model
    logger.info("Creating Light CFM Talking Head model...")
    model = LightCFMTalkingHead(
        config=model_config,
        encoder_config=encoder_config,
    )

    # Setup training arguments
    training_args = TrainingArguments(
        **training_cfg,
    )

    # Create trainer
    data_collator = DataCollator()
    trainer = LightCFMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    trainer.save_model()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
