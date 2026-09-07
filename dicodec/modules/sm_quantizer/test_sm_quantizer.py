"""Run with python -m unittest dicodec.modules.sm_quantizer.test_sm_quantizer."""

import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from .configs import Config, from_dict, load_config
from .model import SMQuantizer
from .training import train


def tiny_config() -> Config:
    return from_dict(Config, {
        "model": {
            "latent_dim": 4, "projection_hidden_dim": 8,
            "quantizer": {"dim": 4, "codebook_size": 8, "reset_every_forward": 2},
            "transformer": {"dim": 8, "heads": 2, "layers": 1, "ff_dim": 16,
                            "dropout": 0.0, "max_length": 8},
            "diffusion": {"hidden_dim": 8, "layers": 1, "time_dim": 4, "sampling_steps": 2},
        },
        "data": {"max_frames": 8},
        "training": {"batch_size": 2, "epochs": 2, "device": "cpu", "log_every": 1},
    })


class SMQuantizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.config = tiny_config()
        self.z = torch.randn(2, 5, 4)
        self.valid = torch.tensor([[True] * 5, [True, True, True, False, False]])

    def test_flow_gradient_reaches_encoder(self):
        self.config.model.loss.reconstruction_l1 = 0
        self.config.model.loss.reconstruction_l2 = 0
        self.config.model.loss.commitment = 0
        model = SMQuantizer(self.config.model)
        out = model(self.z, self.valid)
        out.loss.backward()
        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.decoder.input_projection.weight.grad.abs().sum().item(), 0)
        self.assertEqual(out.indices[~self.valid].tolist(), [-1, -1])

    def test_padding_never_changes_losses_or_ema(self):
        model = SMQuantizer(self.config.model)
        other = copy.deepcopy(model)
        altered = self.z.clone()
        altered[~self.valid] = float("nan")
        torch.manual_seed(1)
        first = model(self.z, self.valid)
        torch.manual_seed(1)
        second = other(altered, self.valid)
        torch.testing.assert_close(first.loss, second.loss)
        torch.testing.assert_close(model.quantizer.codebook.embedding, other.quantizer.codebook.embedding)
        first.loss.backward()
        second.loss.backward()
        torch.testing.assert_close(model.encoder[0].weight.grad, other.encoder[0].weight.grad)

    def test_causal_prefix_and_shifted_targets(self):
        model = SMQuantizer(self.config.model).eval()
        changed = self.z.clone()
        changed[:, 3:] += 100
        first = model(self.z)
        second = model(changed)
        torch.testing.assert_close(first.context[:, :3], second.context[:, :3])
        torch.testing.assert_close(first.reconstruction[:, :3], second.reconstruction[:, :3])
        captured = []
        hook = model.diffusion_head.register_forward_pre_hook(lambda module, args: captured.append(args))
        model(self.z, self.valid)
        hook.remove()
        torch.testing.assert_close(captured[0][0], self.z[:, 1:])
        torch.testing.assert_close(captured[0][2], self.valid[:, :-1] & self.valid[:, 1:])
        # Exercise attention itself independently of possibly identical VQ codes.
        codes = torch.randn(2, 5, 4)
        modified = codes.clone()
        modified[:, 3:] += 100
        valid = torch.ones(2, 5, dtype=torch.bool)
        torch.testing.assert_close(model.decoder(codes, valid)[:, :3], model.decoder(modified, valid)[:, :3])

    def test_eval_freezes_codebook_and_sampling(self):
        model = SMQuantizer(self.config.model).eval()
        before = copy.deepcopy(model.state_dict())
        prediction = model.predict_next(self.z, self.valid, temperature=0)
        self.assertEqual(prediction.shape, (2, 4))
        self.assertTrue(torch.isfinite(prediction).all())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        single = model.predict_next(self.z[:, :1], temperature=0)
        self.assertEqual(single.shape, (2, 4))

    def test_loss_formula_and_invalid_masks(self):
        model = SMQuantizer(self.config.model)
        out = model(self.z, self.valid)
        torch.testing.assert_close(out.loss, out.flow_loss + out.reconstruction_l1
                                   + out.reconstruction_l2 + 0.25 * out.commitment_loss)
        for mask in (torch.zeros_like(self.valid), ~self.valid, self.valid.float()):
            with self.assertRaises(ValueError):
                model(self.z, mask)
        with self.assertRaises(ValueError):
            model(self.z[:, :1])

    def test_yaml_round_trip_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(asdict(self.config)))
            self.assertEqual(load_config(path), self.config)
        with self.assertRaises(ValueError):
            from_dict(Config, {"model": {"typo": 10}})

    def test_training_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for index in range(6):
                torch.save({"z": torch.randn(3 + index % 3, 4)}, data / f"{index}.pt")
            full = copy.deepcopy(self.config)
            full.data.train_path = str(data)
            full.data.validation_path = str(data)
            full.training.output_dir = str(root / "full")
            train(full)
            resumed = copy.deepcopy(full)
            resumed.training.output_dir = str(root / "resumed")
            resumed.training.max_steps = 2
            train(resumed)
            checkpoint = root / "resumed" / "last.pt"
            resumed.training.resume = str(checkpoint)
            resumed.training.max_steps = None
            train(resumed)
            actual = torch.load(checkpoint, weights_only=True)
            expected = torch.load(root / "full" / "last.pt", weights_only=True)
            self.assertEqual(actual["step"], 6)
            for name, value in actual["model"].items():
                torch.testing.assert_close(value, expected["model"][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
