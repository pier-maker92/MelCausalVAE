"""Single-device training, validation and resumable checkpoints."""

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .configs import Config, from_dict, load_config
from .data import LatentCollator, build_dataset
from .model import SMQuantizer
from .tracking import init_wandb
from .wer import word_error_rate


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, config: Config, epoch: int, shuffle: bool) -> DataLoader:
    # Separate deterministic sampler RNG from the model's diffusion/dropout RNG.
    generator = torch.Generator().manual_seed(config.training.seed + epoch)
    return DataLoader(
        dataset, batch_size=config.training.batch_size, shuffle=shuffle,
        num_workers=config.training.num_workers, generator=generator,
        collate_fn=LatentCollator(config.model.latent_dim, config.data.max_frames, config.data.key,
                                 config.data.input, config.data.target, config.model.asr,
                                 config.data.text_key, config.model.language_modeling),
    )


def train_step(model, optimizer, batch, device, grad_clip: float):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = batch.to(device)
    output = model(batch.inputs, batch.valid_mask, target=batch.targets,
                   text_targets=batch.text_targets, text_lengths=batch.text_lengths)
    if not torch.isfinite(output.loss):
        raise FloatingPointError("Nonfinite training loss.")
    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip if grad_clip > 0 else float("inf"), error_if_nonfinite=True,
    )
    optimizer.step()
    return output.metrics()


@torch.no_grad()
def validate(model, loader, device) -> dict[str, float]:
    model.eval()
    sums = {name: 0.0 for name in ("flow_loss", "reconstruction_l1", "reconstruction_l2",
                                  "commitment_loss", "bsq_regularization_loss", "asr_loss")}
    frames = pairs = 0
    batch_statistics = {"perplexity": 0.0, "codebook_utilization_pct": 0.0}
    num_batches = 0
    examples = 0
    wer_errors = wer_words = 0
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.inputs, batch.valid_mask, target=batch.targets,
                       text_targets=batch.text_targets, text_lengths=batch.text_lengths)
        examples += batch.inputs.shape[0]
        if output.wer_errors is not None:
            wer_errors += output.wer_errors
            wer_words += output.wer_words
        num_batches += 1
        for name in batch_statistics:
            batch_statistics[name] += getattr(output, name).item()
        n_frames, n_pairs = int(batch.valid_mask.sum()), int(output.next_frame_mask.sum())
        frames += n_frames
        pairs += n_pairs
        for name in sums:
            count = batch.inputs.shape[0] if name == "asr_loss" else n_pairs if name == "flow_loss" else n_frames
            sums[name] += getattr(output, name).item() * count
    metrics = {name: total / max(1, examples if name == "asr_loss" else pairs if name == "flow_loss" else frames)
               for name, total in sums.items()}
    # Epoch summary is the mean of batch metrics, not a dataset-wide histogram.
    metrics.update({name: total / num_batches for name, total in batch_statistics.items()})
    if model.config.asr.enabled and model.config.loss.asr > 0:
        metrics["wer"] = word_error_rate(wer_errors, wer_words)
    weights = model.config.loss
    metrics["loss"] = (weights.flow * metrics["flow_loss"]
                       + weights.reconstruction_l1 * metrics["reconstruction_l1"]
                       + weights.reconstruction_l2 * metrics["reconstruction_l2"]
                       + weights.commitment * metrics["commitment_loss"]
                       + weights.bsq_regularization * metrics["bsq_regularization_loss"]
                       + weights.asr * metrics["asr_loss"])
    if model.config.quantizer.type != "bsq":
        metrics.pop("bsq_regularization_loss")
    return metrics


def save_checkpoint(path, model, optimizer, config, epoch, batch_index, step):
    checkpoint = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": asdict(config),
        "epoch": epoch, "batch_index": batch_index, "step": step,
        "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, config) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # Populate added fields with their legacy-compatible defaults before comparing.
    old = asdict(from_dict(Config, checkpoint["config"]))
    if old["model"] != asdict(config.model):
        raise ValueError("Resume requires the same model configuration.")
    if old["data"] != asdict(config.data) or any(
        old["training"][key] != getattr(config.training, key) for key in ("batch_size", "seed")
    ):
        raise ValueError("Resume requires the same data, batch_size and seed.")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"])
    random.setstate(checkpoint["python_rng"])
    if torch.cuda.is_available() and checkpoint["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if config.training.wandb_id is None:
        config.training.wandb_id = old["training"]["wandb_id"]
    return checkpoint["epoch"], checkpoint["batch_index"], checkpoint["step"]


def retain_step_checkpoint(path: Path, step: int, limit: int):
    """Archive the completed last.pt, then prune only our numbered snapshots."""
    destination = path.parent / f"step_{step}.pt"
    temporary = destination.with_suffix(".tmp")
    shutil.copyfile(path, temporary)
    temporary.replace(destination)
    snapshots = [p for p in path.parent.glob("step_*.pt") if p.stem[5:].isdigit()]
    snapshots.sort(key=lambda p: int(p.stem[5:]))
    for obsolete in snapshots[:-limit]:
        obsolete.unlink()


def log_metrics(path: Path, split: str, step: int, metrics: dict, run=None):
    line = json.dumps({"split": split, "step": step, **metrics})
    print(line, flush=True)
    with path.open("a") as handle:
        handle.write(line + "\n")
    if run is not None:
        run.log({"global_step": step, **{f"{split}/{name}": value for name, value in metrics.items()}})


def train(config: Config):
    settings = config.training
    seed_everything(settings.seed)
    device = resolve_device(settings.device)
    train_data = build_dataset(config.data, asr_enabled=config.model.asr.enabled)
    validation_data = build_dataset(config.data, validation=True, asr_enabled=config.model.asr.enabled)
    model = SMQuantizer(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    epoch, batch_index, step = (0, 0, 0)
    if settings.resume:
        epoch, batch_index, step = restore_checkpoint(settings.resume, model, optimizer, config)
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with init_wandb(config) as run:
        with (output_dir / "config.yaml").open("w") as handle:
            yaml.safe_dump(asdict(config), handle, sort_keys=False)
        metrics_path = output_dir / "metrics.jsonl"
        checkpoint_path = output_dir / "last.pt"
        if settings.max_steps is not None and step >= settings.max_steps:
            return
        for current_epoch in range(epoch, settings.epochs):
            loader = make_loader(train_data, config, current_epoch, shuffle=True)
            print(f"Epoch {current_epoch + 1}/{settings.epochs} (global step {step})", flush=True)
            for index, batch in enumerate(loader):
                if current_epoch == epoch and index < batch_index:
                    continue
                metrics = train_step(model, optimizer, batch, device, settings.grad_clip)
                metrics["epoch"] = current_epoch + 1
                metrics["total_epochs"] = settings.epochs
                step += 1
                if index == 0 or (current_epoch == epoch and index == batch_index) or step % settings.log_every == 0:
                    log_metrics(metrics_path, "train", step, metrics, run)
                stopping = settings.max_steps is not None and step >= settings.max_steps
                periodic_save = settings.save_every_steps is not None and step % settings.save_every_steps == 0
                # At epoch end, save after validation instead of writing twice.
                if stopping or (periodic_save and index + 1 < len(loader)):
                    save_checkpoint(checkpoint_path, model, optimizer, config, current_epoch, index + 1, step)
                    if periodic_save:
                        retain_step_checkpoint(checkpoint_path, step, settings.max_save_limit)
                if stopping:
                    return
            if validation_data is not None:
                validation_loader = make_loader(validation_data, config, current_epoch, shuffle=False)
                metrics = validate(model, validation_loader, device)
                metrics["epoch"] = current_epoch + 1
                metrics["total_epochs"] = settings.epochs
                log_metrics(metrics_path, "validation", step, metrics, run)
            save_checkpoint(checkpoint_path, model, optimizer, config, current_epoch + 1, 0, step)
            if settings.save_every_steps is not None and step % settings.save_every_steps == 0:
                retain_step_checkpoint(checkpoint_path, step, settings.max_save_limit)


def main():
    if any(argument.startswith("settings=") for argument in sys.argv[1:]):
        from .hydra_training import main as hydra_main

        return hydra_main()
    parser = argparse.ArgumentParser(description="Train an online speech quantizer by next-latent flow matching.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("default.yaml"))
    parser.add_argument("--resume", type=str, help="Override training.resume from YAML.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.resume:
        config.training.resume = args.resume
    train(config)
