"""Optional WandB initialization from the serialized training configuration."""

from contextlib import nullcontext
from dataclasses import asdict

from .configs import Config


def init_wandb(config: Config):
    settings = config.training
    if settings.wandb_mode == "disabled":
        return nullcontext(None)
    import wandb

    run = wandb.init(
        project=settings.wandb_project, name=settings.wandb_run_name,
        id=settings.wandb_id, mode=settings.wandb_mode,
        resume="allow" if settings.wandb_id and settings.wandb_mode == "online" else None,
        config=asdict(config), dir=settings.output_dir,
    )
    settings.wandb_id = run.id
    run.define_metric("global_step")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("validation/*", step_metric="global_step")
    return run
