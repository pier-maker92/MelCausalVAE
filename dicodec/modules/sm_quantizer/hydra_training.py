"""Bridge the shared run_job Hydra arguments to the standalone SM trainer."""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .configs import Config, from_dict
from .training import train


def build_training_config(config: DictConfig) -> Config:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("SM quantizer training requires num_gpus: 1 (EMA is not distributed).")
    values = OmegaConf.to_container(config.sm_quantizer, resolve=True, throw_on_missing=True)
    return from_dict(Config, values)


@hydra.main(version_base=None, config_path=str(Path(__file__).resolve().parents[3] / "configs"),
            config_name="main")
def main(config: DictConfig):
    train(build_training_config(config))
