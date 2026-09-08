"""Local regression tests, including Arrow fixtures and deterministic resume."""

import copy
import io
import itertools
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from .configs import Config, DataConfig, QuantizerConfig, from_dict, load_config
from .data import LatentCollator, build_dataset, parquet_files
from .fsq_levels import FSQ_LEVELS
from .model import SMQuantizer
from .quantizer import OnlineQuantizer
from .training import train


def tiny_config(kind="vq_ema"):
    return from_dict(Config, {
        "model": {
            "latent_dim": 4, "projection_hidden_dim": 8,
            "quantizer": {"type": kind, "dim": 4, "codebook_size": 512, "reset_every_forward": 2},
            "transformer": {"dim": 8, "heads": 2, "layers": 1, "ff_dim": 16,
                            "dropout": 0.0, "max_length": 8},
            "diffusion": {"hidden_dim": 8, "layers": 1, "time_dim": 4, "sampling_steps": 2},
        },
        "data": {"max_frames": 8},
        "training": {"batch_size": 2, "epochs": 2, "device": "cpu", "log_every": 10},
    })


def row(length=5):
    return {"z": torch.randn(length, 4).tolist(), "attributes": {
        "z_sem": (torch.randn(length, 4) + 2).tolist(),
        "z_pros": torch.zeros(length, 4).tolist(), "z_mean": torch.zeros(1, 4).tolist(),
    }}


def write_shard(path, rows):
    matrix = pa.list_(pa.list_(pa.float32()))
    schema = pa.schema([("z", matrix), ("attributes", pa.struct([
        ("z_sem", matrix), ("z_pros", matrix), ("z_mean", matrix),
    ]))])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


class SMQuantizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_all_quantizers_and_source_pairs(self):
        items = [row(5), row(3)]
        for kind, source, target in itertools.product(("vq_ema", "bsq", "fsq"), ("z", "z_sem"), ("z", "z_sem")):
            with self.subTest(kind=kind, source=source, target=target):
                config = tiny_config(kind)
                batch = LatentCollator(4, 8, input=source, target=target)(items)
                model = SMQuantizer(config.model)
                captured = []
                hook = model.diffusion_head.register_forward_pre_hook(lambda module, args: captured.append(args))
                out = model(batch.inputs, batch.valid_mask, target=batch.targets)
                hook.remove()
                torch.testing.assert_close(captured[0][0], batch.targets[:, 1:])
                torch.testing.assert_close(out.reconstruction_l1,
                    (out.reconstruction[batch.valid_mask] - batch.targets[batch.valid_mask]).abs().mean())
                torch.testing.assert_close(out.reconstruction_l2,
                    (out.reconstruction[batch.valid_mask] - batch.targets[batch.valid_mask]).square().mean())
                self.assertEqual(out.indices[~batch.valid_mask].tolist(), [-1, -1])
                self.assertTrue(((out.indices[batch.valid_mask] >= 0) & (out.indices[batch.valid_mask] < 512)).all())
                # Isolate the next-frame task: task gradients must pass through quantization.
                out.flow_loss.backward()
                self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)
                self.assertEqual(model.quantizer.dim, 9 if kind == "bsq" else 4)
                if kind != "vq_ema":
                    self.assertEqual(out.commitment_loss.item(), 0)

    def test_fsq_presets_and_tanh_gradient(self):
        for size, levels in FSQ_LEVELS.items():
            with self.subTest(size=size):
                quantizer = OnlineQuantizer(QuantizerConfig(type="fsq", codebook_size=size))
                self.assertEqual(tuple(quantizer.codebook.levels_list), levels)
                self.assertEqual(math.prod(levels), size)
                x = torch.tensor([[[-10., -.4, .7, 10.]]], requires_grad=True)
                out = quantizer(x, torch.ones(1, 1, dtype=torch.bool))
                out.codes.sum().backward()
                torch.testing.assert_close(x.grad, 1 - x.detach().tanh().square())
                self.assertTrue(((out.indices >= 0) & (out.indices < size)).all())
        with self.assertRaises(ValueError):
            QuantizerConfig(type="fsq", codebook_size=1000)

    def test_bsq_regularization(self):
        q = OnlineQuantizer(QuantizerConfig(type="bsq", codebook_size=512))
        x = torch.randn(2, 4, 9, requires_grad=True)
        valid = torch.tensor([[True]*4, [True, True, False, False]])
        out = q(x, valid)
        probs = torch.sigmoid(x[valid])
        avg = probs.mean(0)
        centered = probs - avg
        cov = centered.T @ centered / len(probs)
        expected = (avg * (avg + 1e-5).log() + (1-avg) * (1-avg+1e-5).log()).mean()
        expected += (cov - torch.diag_embed(cov.diagonal())).square().mean()
        torch.testing.assert_close(out.bsq_regularization_loss, expected)
        out.bsq_regularization_loss.backward()
        self.assertGreater(x.grad[valid].abs().sum().item(), 0)
        self.assertEqual(x.grad[~valid].abs().sum().item(), 0)

    def test_padding_causality_and_frozen_inference(self):
        batch = LatentCollator(4, 8)([row(5), row(3)])
        for kind in ("vq_ema", "bsq", "fsq"):
            with self.subTest(kind=kind):
                model = SMQuantizer(tiny_config(kind).model)
                other = copy.deepcopy(model)
                corrupted = batch.inputs.clone()
                corrupted[~batch.valid_mask] = float("nan")
                torch.manual_seed(2)
                out = model(batch.inputs, batch.valid_mask)
                torch.manual_seed(2)
                alt = other(corrupted, batch.valid_mask)
                torch.testing.assert_close(out.loss, alt.loss)
                out.loss.backward()
                alt.loss.backward()
                torch.testing.assert_close(model.encoder[0].weight.grad, other.encoder[0].weight.grad)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, other.state_dict()[key])
                model.eval()
                before = copy.deepcopy(model.state_dict())
                changed = batch.inputs.clone()
                changed[:, 3:] += 100
                torch.testing.assert_close(model(batch.inputs).context[:, :3], model(changed).context[:, :3])
                codes = torch.randn(2, 5, model.quantizer.dim)
                changed_codes = codes.clone()
                changed_codes[:, 3:] += 100
                mask = torch.ones(2, 5, dtype=torch.bool)
                torch.testing.assert_close(model.decoder(codes, mask)[:, :3], model.decoder(changed_codes, mask)[:, :3])
                prediction = model.predict_next(batch.inputs, batch.valid_mask, temperature=0)
                self.assertEqual(prediction.shape, (2, 4))
                self.assertTrue(torch.isfinite(prediction).all())
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[key])

    def test_parquet_partitions_and_source_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [row(5), row(3)]
            write_shard(root / "train_clean_100" / "shard.parquet", rows)
            write_shard(root / "train_clean_360" / "shard.parquet", [row(4)])
            (root / "train_clean_100" / "._junk.parquet").write_bytes(b"ignore")
            for source, target in itertools.product(("z", "z_sem"), repeat=2):
                config = DataConfig(format="parquet", train_path=str(root),
                    train_partitions=["train_clean_100"], input=source, target=target,
                    cache_dir=str(root / "cache"))
                dataset = build_dataset(config)
                self.assertEqual(len(dataset), 2)
                batch = LatentCollator(4, 4, input=source, target=target)([dataset[0], dataset[1]])
                for name, chosen in (("inputs", source), ("targets", target)):
                    expected = rows[0]["z"] if chosen == "z" else rows[0]["attributes"]["z_sem"]
                    torch.testing.assert_close(getattr(batch, name)[0], torch.tensor(expected[:4]))
                self.assertEqual(batch.valid_mask.tolist(), [[True]*4, [True, True, True, False]])
                self.assertIsNone(build_dataset(config, validation=True))
            self.assertEqual(len(parquet_files(str(root), ["train_clean_100", "train_clean_360"])), 2)
            with self.assertRaises(FileNotFoundError):
                parquet_files(str(root), ["missing"])
            with self.assertRaises(FileNotFoundError):
                parquet_files(str(root), [])

    def test_invalid_pairs_and_masks(self):
        item = row(5)
        item["attributes"]["z_sem"] = item["attributes"]["z_sem"][:4]
        with self.assertRaises(ValueError):
            LatentCollator(4, 2, input="z", target="z_sem")([item])
        with self.assertRaises(ValueError):
            LatentCollator(4, 8, input="z", target="z_sem")([torch.randn(4, 4)])
        model = SMQuantizer(tiny_config().model)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 1, 4))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 4), torch.tensor([[True, False, True]]))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 4), target=torch.randn(1, 2, 4))

    def test_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            config = tiny_config()
            path.write_text(yaml.safe_dump(asdict(config)))
            self.assertEqual(load_config(path), config)
        with self.assertRaises(ValueError):
            from_dict(Config, {"model": {"typo": 10}})

    def test_training_resume_and_legacy_checkpoint(self):
        for kind in ("vq_ema", "bsq", "fsq"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                root = Path(directory)
                data = root / "data"
                data.mkdir()
                for index in range(6):
                    torch.save(row(3 + index % 3), data / f"{index}.pt")
                config = tiny_config(kind)
                config.data.train_path = str(data)
                config.data.validation_path = str(data)
                config.data.target = "z" if kind == "vq_ema" else "z_sem"
                config.training.output_dir = str(root / "full")
                train(config)
                config.training.output_dir = str(root / "resumed")
                config.training.max_steps = 2
                train(config)
                checkpoint_path = root / "resumed" / "last.pt"
                if kind == "vq_ema":
                    checkpoint = torch.load(checkpoint_path, weights_only=True)
                    saved = checkpoint["config"]
                    for key in ("type", "entropy_temperature"):
                        saved["model"]["quantizer"].pop(key)
                    saved["model"]["loss"].pop("bsq_regularization")
                    for key in ("format", "input", "target", "train_partitions", "validation_partitions", "cache_dir"):
                        saved["data"].pop(key)
                    torch.save(checkpoint, checkpoint_path)
                config.training.resume = str(checkpoint_path)
                config.training.max_steps = None
                rejected = copy.deepcopy(config)
                rejected.data.target = "z_sem" if config.data.target == "z" else "z"
                with self.assertRaises(ValueError):
                    train(rejected)
                train(config)
                actual = torch.load(checkpoint_path, weights_only=True)
                expected = torch.load(root / "full" / "last.pt", weights_only=True)
                self.assertEqual(actual["step"], 6)
                for name, value in actual["model"].items():
                    torch.testing.assert_close(value, expected["model"][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
