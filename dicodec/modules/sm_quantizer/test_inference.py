import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from .inference import load_quantizer, parse_args, reconstruct, select_latents, QUANTIZER_CHECKPOINT


class InferenceTests(unittest.TestCase):
    def test_real_fsq_checkpoint_reconstructs_semantic_dimension(self):
        if not QUANTIZER_CHECKPOINT.exists():
            self.skipTest("Local trained checkpoint is not present.")
        quantizer = load_quantizer(QUANTIZER_CHECKPOINT, torch.device("cpu"))
        with torch.inference_mode():
            z = torch.randn(1, 5, 64)
            codes = quantizer.encode(z).codes
            q_sem = quantizer.reconstruction_head(codes)
        self.assertEqual(codes.shape, (1, 5, 4))
        self.assertEqual(q_sem.shape, z.shape)
        self.assertTrue(torch.isfinite(q_sem).all())
        self.assertFalse(quantizer.training)

    def test_modes_and_speaker_conditioning(self):
        z_sem = torch.ones(1, 3, 4)
        attributes = SimpleNamespace(z_sem=z_sem, z_pros=2 * z_sem, z_mean=torch.ones(1, 1, 4))
        z = z_sem + attributes.z_pros + attributes.z_mean
        q_sem = z_sem * .25
        torch.testing.assert_close(select_latents(z, q_sem, attributes, "quantized"), q_sem)
        torch.testing.assert_close(select_latents(z, q_sem, attributes, "residual") + q_sem, z)
        torch.testing.assert_close(select_latents(z, q_sem, attributes, "full"), q_sem + 3)
        padding = torch.tensor([[False, False, True]])
        waveform, target = torch.randn(24), torch.randn(48)
        for mode in ("full", "quantized", "residual"):
            for speaker_target in (None, target):
                model = Mock()
                model.config.sample_rate = 24000
                model.extract_features.return_value = (z, padding, None, None)
                model.encode.return_value = SimpleNamespace(z=z, attributes=attributes, padding_mask=padding)
                model.extract_speaker_embedding.return_value = torch.ones(1, 8)
                model.sample.return_value = (torch.ones(1, 6, 100), None)
                model.vocoder.decode.return_value = torch.ones(1, 24)
                quantizer = Mock()
                quantizer.encode.return_value = SimpleNamespace(codes=q_sem)
                quantizer.reconstruction_head.return_value = q_sem
                audio = reconstruct(model, quantizer, waveform, speaker_target, mode)
                self.assertEqual(audio.shape, (1, 24))
                args = quantizer.encode.call_args.args
                torch.testing.assert_close(args[0], z_sem)
                torch.testing.assert_close(args[1], ~padding)
                expected = select_latents(z, q_sem, attributes, mode).masked_fill(padding.unsqueeze(-1), 0)
                torch.testing.assert_close(model.sample.call_args.kwargs["z"], expected)
                torch.testing.assert_close(model.sample.call_args.kwargs["speaker_embedding"], torch.ones(1, 8))
                speaker_audio = model.extract_speaker_embedding.call_args.args[0][0][0]
                self.assertIs(speaker_audio, waveform if speaker_target is None else target)

    def test_cli(self):
        args = parse_args(["-i", "source.wav", "-ta", "speaker.wav", "-r"])
        self.assertTrue(args.r)
        self.assertFalse(args.q)
        self.assertEqual(args.type, "fsq")
        for kind in ("vq_ema", "fsq", "bsq"):
            self.assertEqual(parse_args(["--type", kind]).type, kind)
        with self.assertRaises(SystemExit):
            parse_args(["-q", "-r"])


if __name__ == "__main__":
    unittest.main()
