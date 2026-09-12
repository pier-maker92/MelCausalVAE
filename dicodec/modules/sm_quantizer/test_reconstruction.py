import unittest
from dataclasses import asdict

import torch

from .configs import Config, from_dict
from .model import SMQuantizer
from .test_sm_quantizer import tiny_config


class ReconstructionTests(unittest.TestCase):
    def test_only_reconstruction_all_backends(self):
        torch.set_num_threads(1)
        for kind in ('vq_ema', 'bsq', 'fsq'):
            with self.subTest(kind=kind):
                values = asdict(tiny_config(kind))
                values['model']['simple_reconstruction'] = True
                values['model']['asr']['enabled'] = True
                config = from_dict(Config, values)
                model = SMQuantizer(config.model)
                self.assertIsNone(model.decoder)
                self.assertIsNone(model.diffusion_head)
                self.assertIsNone(model.asr_head)
                z = torch.randn(2, 3, 4)
                valid = torch.tensor([[True, True, True], [True, False, False]])
                z[~valid] = float('nan')
                target = torch.randn_like(z)
                output = model(z, valid, target=target)
                expected = output.reconstruction[valid] - target[valid]
                torch.testing.assert_close(output.loss, expected.abs().mean() + expected.square().mean())
                self.assertNotIn('wer', output.metrics())
                output.loss.backward()
                self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)
                self.assertEqual(config, from_dict(Config, asdict(config)))

    def test_four_hydra_presets_all_quantizers(self):
        from pathlib import Path
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from .data import LatentCollator
        from .test_sm_quantizer import row

        torch.set_num_threads(1)
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[3] / 'configs'), version_base=None):
            for name in ('recon', 'lm', 'asr', 'lm_and_asr'):
                preset = compose(config_name='main', overrides=[f'settings=dicodec/quantize/{name}'])
                selected = OmegaConf.to_container(preset.sm_quantizer.model)
                for kind in ('vq_ema', 'bsq', 'fsq'):
                    with self.subTest(preset=name, kind=kind):
                        values = asdict(tiny_config(kind))
                        for key in ('simple_reconstruction', 'language_modeling', 'loss'):
                            values['model'][key] = selected[key]
                        values['model']['asr'].update(enabled=selected['asr']['enabled'],
                            hidden_size=8, embedding_dim=8, layers=1, dropout=0)
                        config = from_dict(Config, values)
                        model = SMQuantizer(config.model)
                        batch = LatentCollator(4, 8, asr_config=config.model.asr,
                            language_modeling=config.model.language_modeling)([dict(row(5), transcript='ab')])
                        out = model(batch.inputs, batch.valid_mask, text_targets=batch.text_targets,
                                    text_lengths=batch.text_lengths)
                        self.assertEqual(model.reconstruction_head is not None, name == 'recon')
                        self.assertEqual(out.reconstruction is not None, name == 'recon')
                        self.assertEqual(any(k.startswith('reconstruction_head.') for k in model.state_dict()), name == 'recon')
                        if name != 'recon':
                            self.assertEqual(out.reconstruction_l1.item(), 0)
                            self.assertEqual(out.reconstruction_l2.item(), 0)
                        self.assertEqual(model.decoder is not None, name in ('lm', 'lm_and_asr'))
                        self.assertEqual(model.asr_head is not None, name in ('asr', 'lm_and_asr'))
                        out.loss.backward()
                        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)
