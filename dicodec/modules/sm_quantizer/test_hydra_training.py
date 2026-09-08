"""Exercise the exact CLI convention emitted by Narval run_job.sh locally."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import yaml

from .test_sm_quantizer import row, write_shard


class HydraLauncherTests(unittest.TestCase):
    def test_launcher_cli_trains_on_selected_staged_partitions(self):
        repo = Path(__file__).resolve().parents[3]
        experiment = yaml.safe_load((Path(__file__).parent / "experiments" /
                                    "quantize-ls-v2-25-ema1024-zsem.yaml").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "datasets" / experiment["dataset_path"]
            for partition in ("train_clean_100", "train_clean_360"):
                write_shard(data / partition / "shard.parquet", [row(4)])
            output = root / "checkpoint"
            command = [sys.executable, str(repo / experiment["train_script"]),
                       f"settings={experiment['hydra_settings'][0]}",
                       f"training.dataset_name={experiment['dataset_name']}",
                       "sm_quantizer.data.train_partitions=[train_clean_100,train_clean_360]",
                       f"hydra.run.dir={root / 'hydra'}", "training.run_id=smoke",
                       "training.wandb_run_name=smoke", f"training.output_dir={output}",
                       "sm_quantizer.training.device=cpu", "sm_quantizer.training.max_steps=1",
                       "sm_quantizer.training.wandb_mode=offline",
                       "sm_quantizer.model.latent_dim=4", "sm_quantizer.model.projection_hidden_dim=8",
                       "sm_quantizer.model.transformer.dim=8", "sm_quantizer.model.transformer.heads=2",
                       "sm_quantizer.model.transformer.layers=1", "sm_quantizer.model.transformer.ff_dim=16",
                       "sm_quantizer.model.diffusion.hidden_dim=8", "sm_quantizer.model.diffusion.layers=1"]
            environment = dict(os.environ, SLURM_TMPDIR=str(root), WORLD_SIZE="1", OMP_NUM_THREADS="1")
            result = subprocess.run(command, cwd=repo, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            config = checkpoint["config"]
            self.assertEqual(checkpoint["step"], 1)
            self.assertEqual(config["data"]["input"], "z_sem")
            self.assertEqual(config["data"]["target"], "z_sem")
            self.assertEqual(config["data"]["train_partitions"], ["train_clean_100", "train_clean_360"])
            self.assertEqual(config["data"]["train_path"], str(data))
            self.assertEqual(config["training"]["output_dir"], str(output))
            self.assertEqual(config["model"]["quantizer"]["type"], "vq_ema")
            self.assertEqual(config["model"]["quantizer"]["codebook_size"], 1024)
            self.assertEqual(config["training"]["wandb_run_name"], "smoke")
            self.assertTrue(config["training"]["wandb_id"])
            self.assertTrue(list((output / "wandb").glob("offline-run-*/run-*.wandb")))


if __name__ == "__main__":
    unittest.main()
