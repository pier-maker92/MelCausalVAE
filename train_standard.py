"""
Training script for VAEWithStandardDecoder (CNN decoder + multi-scale GAN discriminator).

Extends VAEtrainer with:
  - Separate discriminator optimizer (AdamW, betas=(0.8, 0.99))
  - Alternating D/G updates inside training_step
  - Granular logging: audio_loss, adv_loss, fm_loss, d_loss, kl_loss, [vq_*]
"""
import os
import math
import wandb
import torch

# --- WORKAROUND FOR FAIRSEQ/HYDRA ON PYTHON 3.11 ---
import dataclasses
_orig_get_field = dataclasses._get_field

def _patched_get_field(cls, a_name, a_type, *args, **kwargs):
    try:
        return _orig_get_field(cls, a_name, a_type, *args, **kwargs)
    except ValueError as e:
        if "mutable default" in str(e):
            default = getattr(cls, a_name, dataclasses.MISSING)
            actual_default = default.default if isinstance(default, dataclasses.Field) else default
            if actual_default is not dataclasses.MISSING:
                default_cls = actual_default.__class__
                orig_hash = getattr(default_cls, '__hash__', None)
                try:
                    default_cls.__hash__ = lambda self: id(self)
                except TypeError:
                    pass
                try:
                    return _orig_get_field(cls, a_name, a_type, *args, **kwargs)
                finally:
                    try:
                        if orig_hash is None:
                            default_cls.__hash__ = None
                        else:
                            default_cls.__hash__ = orig_hash
                    except TypeError:
                        pass
        raise

dataclasses._get_field = _patched_get_field
# ---------------------------------------------------

# --- WORKAROUND FOR PYTORCH 2.6 WEIGHTS_ONLY ---
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
# -----------------------------------------------

import hydra
import hydra.experimental
hydra.experimental.initialize = hydra.initialize
hydra.experimental.initialize_config_module = hydra.initialize_config_module
hydra.experimental.initialize_config_dir = hydra.initialize_config_dir
hydra.experimental.compose = hydra.compose

import hydra.experimental.initialize as _hydra_exp_init
_hydra_exp_init.initialize = hydra.initialize
_hydra_exp_init.initialize_config_module = hydra.initialize_config_module
_hydra_exp_init.initialize_config_dir = hydra.initialize_config_dir

import hydra.experimental.compose as _hydra_exp_compose
_hydra_exp_compose.compose = hydra.compose

import logging
import datetime
from vocos import Vocos
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from accelerate import InitProcessGroupKwargs
from data.audio_dataset import TestDatasetWrapper, TrainDatasetWrapper
from transformers import (
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
    EarlyStoppingCallback,
)

import torch.distributed as dist
from accelerate import Accelerator
from evaluate import run_evaluation
from modules.builder import build_standard_model
from modules.decoder.discriminator import (
    discriminator_hinge_loss,
    generator_hinge_loss,
    feature_matching_loss,
)
from data.audio_dataset import DataCollator
from train import VAEtrainer, AddGranularLossesToTrainerState, KLWarmupRatioCallback

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class VAEStandardTrainer(VAEtrainer):
    """
    Trainer for VAEWithStandardDecoder.

    Overrides:
      - create_optimizer: adds a separate discrim_optimizer
      - training_step:    alternating D update (every step) + G update (accumulated)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.discrim_optimizer: Optional[torch.optim.Optimizer] = None

    def create_optimizer(self):
        if self.optimizer is None:
            actual_model = self.model.module if hasattr(self.model, "module") else self.model

            encoder_lr = self.encoder_lr if self.encoder_lr is not None else self.args.learning_rate
            decoder_lr = self.decoder_lr if self.decoder_lr is not None else self.args.learning_rate
            disc_cfg = actual_model.config.discriminator_config

            encoder_params, decoder_params = [], []
            for n, p in actual_model.named_parameters():
                if not p.requires_grad or "discriminator" in n:
                    continue
                if "encoder" in n or "feature_extractor" in n:
                    encoder_params.append(p)
                elif "decoder" in n:
                    decoder_params.append(p)
                else:
                    encoder_params.append(p)

            optimizer_cls, optimizer_kwargs = VAEtrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(
                [
                    {"params": encoder_params, "lr": encoder_lr},
                    {"params": decoder_params, "lr": decoder_lr},
                ],
                **optimizer_kwargs,
            )

            self.discrim_optimizer = torch.optim.AdamW(
                actual_model.discriminator.parameters(),
                lr=disc_cfg.discrim_lr,
                betas=(0.8, 0.99),
                weight_decay=0.0,
            )
            logger.info(
                f"Standard trainer: encoder_lr={encoder_lr}, decoder_lr={decoder_lr}, "
                f"discrim_lr={disc_cfg.discrim_lr}"
            )

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        # Discriminator uses a fixed LR; only G scheduler is managed by Trainer
        return super().create_scheduler(num_training_steps, optimizer)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        audios_srs = inputs["output_audios_srs"]
        actual_model = model.module if hasattr(model, "module") else model
        disc_cfg = actual_model.config.discriminator_config

        with self.compute_loss_context_manager():
            # ── Feature extraction (no grad) ────────────────────────────────
            with torch.no_grad():
                (
                    enc_features, enc_mask,
                    dec_features, dec_mask,
                    _distill,
                ) = actual_model.extract_features(audios_srs)

            # ── Encoder forward ─────────────────────────────────────────────
            enc_output = actual_model.encode(
                enc_features, enc_mask, training_step=self.state.global_step
            )

            # ── Decoder forward ─────────────────────────────────────────────
            dec_output = actual_model.decoder(
                context_vector=enc_output.z,
                target=dec_features,
                target_padding_mask=dec_mask,
            )

        mel_pred = dec_output.audio_features   # [B, T, mel_dim] — grad through encoder+decoder
        mel_target = dec_features              # [B, T, mel_dim] — no grad (original length)

        # mel_pred_gan: decoder-only gradients (context_vector detached).
        # This prevents adversarial gradients from flowing through the VQ
        # straight-through estimator into the encoder, which would destabilize
        # the KL/VQ losses.
        with self.compute_loss_context_manager():
            mel_pred_gan = actual_model.decoder.generate(
                enc_output.z.detach(),
                padding_mask=enc_output.padding_mask,
            ).audio_features   # [B, T', mel_dim] — grad through decoder only
        # Trim mel_target to match mel_pred_gan length (may differ by ±1 due to
        # CausalConv downsampling rounding before 8× upsample)
        T_gan = mel_pred_gan.shape[1]
        mel_target_gan = mel_target[:, :T_gan, :]

        # ── Discriminator update (D) ─────────────────────────────────────────
        with self.compute_loss_context_manager():
            real_logits, _ = actual_model.discriminator(mel_target_gan.detach())
            fake_logits, _ = actual_model.discriminator(mel_pred_gan.detach())
            d_loss = discriminator_hinge_loss(real_logits, fake_logits)

        if self.discrim_optimizer is not None:
            self.discrim_optimizer.zero_grad()
            self.accelerator.backward(d_loss)
            if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(
                    actual_model.discriminator.parameters(),
                    self.args.max_grad_norm,
                )
            self.discrim_optimizer.step()

        # ── Generator loss (G) ──────────────────────────────────────────────
        with self.compute_loss_context_manager():
            # Freeze D so its params get no gradient from G backward
            for p in actual_model.discriminator.parameters():
                p.requires_grad_(False)

            fake_logits_g, fake_feats_g = actual_model.discriminator(mel_pred_gan)
            _, real_feats_g = actual_model.discriminator(mel_target_gan)

            for p in actual_model.discriminator.parameters():
                p.requires_grad_(True)

            adv_loss = generator_hinge_loss(fake_logits_g)
            fm_loss = feature_matching_loss(real_feats_g, fake_feats_g)
            recon_loss = dec_output.loss

            kl_loss = enc_output.kl_loss if enc_output.kl_loss is not None else 0.0
            vq_loss = (
                enc_output.vq_loss
                if getattr(enc_output, "vq_loss", None) is not None
                else 0.0
            )

            sem_cfg = getattr(actual_model.encoder.config, "semantic_distillation_config", None)
            distill_loss = 0.0
            if _distill is not None and sem_cfg is not None:
                cosine_loss = actual_model._compute_distillation_losses(enc_output, _distill)
                distill_loss = cosine_loss * sem_cfg.cosine_loss_weight
            if getattr(enc_output, "ortho_loss", None) is not None and sem_cfg is not None:
                distill_loss = distill_loss + enc_output.ortho_loss * sem_cfg.ortho_loss_weight

            g_loss = (
                recon_loss * disc_cfg.recon_loss_weight
                + adv_loss * disc_cfg.adv_loss_weight
                + fm_loss * disc_cfg.fm_loss_weight
                + kl_loss
                + vq_loss
                + distill_loss
            )

            if self.args.n_gpu > 1:
                g_loss = g_loss.mean()

        self.accelerator.backward(g_loss)

        # ── Accumulate granular metrics ──────────────────────────────────────
        if hasattr(self.control, "granular_losses") and model.training:
            vq_stats = getattr(enc_output, "vq_stats", None)
            flat = {
                "audio_loss": recon_loss,
                "adv_loss": adv_loss,
                "fm_loss": fm_loss,
                "d_loss": d_loss,
                "kl_loss": enc_output.kl_loss,
                "mu_mean": enc_output.mu[~enc_output.padding_mask].mean(),
                "mu_var": enc_output.mu[~enc_output.padding_mask].var(),
            }
            if vq_stats is not None:
                flat["vq_loss"] = getattr(enc_output, "vq_loss", None)
                flat["vq_perplexity"] = vq_stats.perplexity
                flat["vq_codes_used"] = vq_stats.codes_used
                flat["vq_codes_used_frac"] = vq_stats.codes_used_frac
            for key in self.control.granular_losses:
                if flat.get(key) is not None:
                    val = flat[key].detach().float()
                    if self.args.n_gpu > 1 and val.dim() > 0:
                        val = val.mean()
                    self.control.granular_losses[key] += (
                        val.to(self.control.granular_losses[key].dtype)
                        / self.args.gradient_accumulation_steps
                    )

        return g_loss.detach() / self.args.gradient_accumulation_steps


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    training_cfg = cfg_dict.get("training", {})

    kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    logger.info(f"Device: {accelerator.device} | Mixed precision: {accelerator.state.mixed_precision}")

    set_seed(training_cfg.get("seed", 42))

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

    logger.info("Building VAEWithStandardDecoder...")
    model = build_standard_model(cfg_dict)

    training_cfg["learning_rate"] = float(training_cfg.get("learning_rate"))
    min_learning_rate = float(training_cfg.pop("min_learning_rate", 0.0))
    early_stopping_patience = training_cfg.pop("early_stopping_patience", None)
    generate_and_log_samples = training_cfg.pop("generate_and_log_samples", True)
    eval_num_samples = training_cfg.pop("eval_num_samples", 10)
    vq_collapse_threshold = float(training_cfg.pop("vq_collapse_threshold", 0.5))
    vq_collapse_patience = training_cfg.pop("vq_collapse_patience", None)
    if vq_collapse_patience is not None:
        vq_collapse_patience = int(vq_collapse_patience)

    from_pretrained = training_cfg.pop("from_pretrained", None)
    if from_pretrained:
        model.from_pretrained(from_pretrained)
        logger.info(f"Loaded pretrained model from {from_pretrained}")

    if "lr_scheduler_type" not in training_cfg:
        training_cfg["lr_scheduler_type"] = "constant"

    encoder_lr = training_cfg.pop("encoder_lr", None)
    decoder_lr = training_cfg.pop("decoder_lr", None)
    encoder_min_lr = training_cfg.pop("encoder_min_lr", None)
    decoder_min_lr = training_cfg.pop("decoder_min_lr", None)
    encoder_warmup_ratio = training_cfg.pop("encoder_warmup_ratio", None)
    decoder_warmup_ratio = training_cfg.pop("decoder_warmup_ratio", None)

    def _to_float(v):
        return float(v) if v is not None else None

    encoder_lr = _to_float(encoder_lr)
    decoder_lr = _to_float(decoder_lr)
    encoder_min_lr = _to_float(encoder_min_lr)
    decoder_min_lr = _to_float(decoder_min_lr)
    encoder_warmup_ratio = _to_float(encoder_warmup_ratio)
    decoder_warmup_ratio = _to_float(decoder_warmup_ratio)

    run_id = training_cfg.pop("run_id", None)
    if run_id is None:
        date_dir = datetime.datetime.now().strftime("%d-%B-%Y")
        time_dir = datetime.datetime.now().strftime("%H:%M:%S")
        run_id = f"{date_dir}/{time_dir}/{wandb_run_name or 'run'}"

    training_args = TrainingArguments(
        remove_unused_columns=False,
        ddp_timeout=7200,
        **training_cfg,
    )

    # Granular losses for the standard trainer
    granular_losses = [
        "audio_loss", "adv_loss", "fm_loss", "d_loss",
        "kl_loss", "mu_mean", "mu_var",
    ]
    has_vq = getattr(model.encoder.config, "vq_config", None) is not None
    if has_vq:
        granular_losses.extend(["vq_loss", "vq_perplexity", "vq_codes_used", "vq_codes_used_frac"])

    data_collator = DataCollator()
    trainer = VAEStandardTrainer(
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
        encoder_min_lr=encoder_min_lr,
        decoder_min_lr=decoder_min_lr,
        encoder_warmup_ratio=encoder_warmup_ratio,
        decoder_warmup_ratio=decoder_warmup_ratio,
        vq_collapse_threshold=vq_collapse_threshold,
        vq_collapse_patience=vq_collapse_patience,
    )
    # Replace the granular loss tracker set by VAEtrainer.__init__
    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if not isinstance(cb, AddGranularLossesToTrainerState)
    ]
    trainer.add_callback(AddGranularLossesToTrainerState(granular_losses))

    if training_args.load_best_model_at_end and early_stopping_patience is not None:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    logger.info("Starting training (standard CNN decoder + GAN discriminator)...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
