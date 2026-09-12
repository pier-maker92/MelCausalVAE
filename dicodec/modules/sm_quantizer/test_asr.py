"""Auxiliary task, transcript alignment and tokenizer-gradient regressions."""
import copy
import io
import itertools
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .configs import ASRConfig, EncoderConfig
from .asr import TextTokenizer
from .data import LatentCollator, build_dataset
from .model import SMQuantizer
from .test_sm_quantizer import tiny_config, row
from .training import asr_curriculum_ratio, asr_curriculum_reconstruction_weight, train


def config_for(kind='vq_ema', lm=False):
    config = tiny_config(kind)
    config.model.language_modeling = lm
    config.model.asr = ASRConfig(enabled=True, hidden_size=8, embedding_dim=8, layers=1, dropout=0)
    return config


def collate(config, items=None):
    return LatentCollator(4, 8, asr_config=config.model.asr,
                         language_modeling=config.model.language_modeling)(
        items or [dict(row(6), transcript='AB'), dict(row(4), transcript='a')])


def forward(model, batch):
    return model(batch.inputs, batch.valid_mask, target=batch.targets,
                 text_targets=batch.text_targets, text_lengths=batch.text_lengths)


class ASRTests(unittest.TestCase):
    def test_sentencepiece(self):
        import sentencepiece as spm

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / 'text.txt'
            corpus.write_text('a small speech corpus\na second sample sentence\n')
            spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(root / 'tokens'),
                                          vocab_size=24, model_type='bpe', minloglevel=2)
            config = ASRConfig(enabled=True, tokenizer='sentencepiece', vocab_size=24,
                               tokenizer_model=str(root / 'tokens.model'))
            tokenizer = TextTokenizer(config)
            text = 'a small speech corpus'
            self.assertEqual(tokenizer.encode(text).tolist(), tokenizer.sentencepiece.encode(text))
            config.vocab_size = 25
            with self.assertRaisesRegex(ValueError, 'vocab_size'):
                TextTokenizer(config)

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_switches_and_asr_gradients_all_quantizers(self):
        for kind, lm, asr in itertools.product(('vq_ema', 'bsq', 'fsq'), (False, True), (False, True)):
            with self.subTest(kind=kind, lm=lm, asr=asr):
                config = config_for(kind, lm)
                config.model.asr.enabled = asr
                model = SMQuantizer(config.model)
                output = forward(model, collate(config))
                self.assertEqual(model.decoder is not None, lm)
                self.assertEqual(model.diffusion_head is not None, lm)
                self.assertEqual(model.asr_head is not None, asr)
                self.assertEqual(output.context is not None, lm)
                self.assertEqual(output.flow_loss.item() == 0, not lm)
                self.assertEqual(output.asr_loss.item() == 0, not asr)
                if asr:
                    output.asr_loss.backward()
                    self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)
                if not lm:
                    with self.assertRaises(RuntimeError):
                        model.eval().predict_next(torch.randn(1, 2, 4))

    def test_asr_curriculum_bypasses_quantization_only_when_enabled(self):
        config = config_for('fsq')
        config.model.asr.curriculum = True
        config.model.loss.reconstruction_l1 = 0.0
        config.model.loss.reconstruction_l2 = 0.0
        model = SMQuantizer(config.model)
        batch = collate(config)
        captured = []
        hook = model.asr_head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        output = model(batch.inputs, batch.valid_mask, target=batch.targets,
                       text_targets=batch.text_targets, text_lengths=batch.text_lengths,
                       asr_curriculum_ratio=1.0,
                       asr_curriculum_reconstruction_weight=1.0)
        hook.remove()
        encoded = model.encoder(batch.inputs.detach().masked_fill(~batch.valid_mask.unsqueeze(-1), 0))
        torch.testing.assert_close(captured[0], encoded)
        self.assertEqual(output.metrics()['asr_curriculum_pct'], 100.0)
        self.assertEqual(output.metrics()['asr_curriculum_reconstruction_weight'], 1.0)
        expected_loss = output.asr_loss + output.reconstruction_l1 + output.reconstruction_l2
        torch.testing.assert_close(output.loss, expected_loss)

        config.model.asr.curriculum = False
        model = SMQuantizer(config.model)
        captured = []
        hook = model.asr_head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        output = model(batch.inputs, batch.valid_mask, target=batch.targets,
                       text_targets=batch.text_targets, text_lengths=batch.text_lengths,
                       asr_curriculum_ratio=1.0)
        hook.remove()
        torch.testing.assert_close(captured[0], output.quantized)
        self.assertNotIn('asr_curriculum_pct', output.metrics())

    def test_asr_curriculum_schedule(self):
        config = config_for('fsq')
        config.model.asr.curriculum = True
        config.model.asr.curriculum_start_pct = 70.0
        config.model.asr.curriculum_end_pct = 0.0
        config.model.asr.curriculum_reconstruction_start_weight = 1.0
        config.model.asr.curriculum_reconstruction_end_weight = 0.0
        config.training.epochs = 2
        self.assertEqual(asr_curriculum_ratio(config, 0, 0, 3), 0.7)
        self.assertEqual(asr_curriculum_ratio(config, 1, 2, 3), 0.0)
        self.assertAlmostEqual(asr_curriculum_ratio(config, 0, 1, 3), 0.56)
        self.assertEqual(asr_curriculum_reconstruction_weight(config, 0, 0, 3), 1.0)
        self.assertEqual(asr_curriculum_reconstruction_weight(config, 1, 2, 3), 0.0)
        self.assertAlmostEqual(asr_curriculum_reconstruction_weight(config, 0, 1, 3), 0.8)
        config.model.asr.enabled = False
        self.assertEqual(asr_curriculum_ratio(config, 0, 0, 3), 0.0)
        self.assertEqual(asr_curriculum_reconstruction_weight(config, 0, 0, 3), 0.0)

    def test_bilstm_padding_and_encoder_causality(self):
        for encoder_type in ('mlp', 'causal_conv'):
            config = config_for('fsq')
            config.model.encoder = EncoderConfig(type=encoder_type, hidden_dim=16, layers=3)
            model = SMQuantizer(config.model).eval()
            batch = collate(config)
            out = forward(model, batch)
            altered = copy.deepcopy(batch)
            altered.inputs[~altered.valid_mask] = float('nan')
            altered.targets[~altered.valid_mask] = float('nan')
            altered.text_targets[1, 1:] = 999
            alt = forward(model, altered)
            torch.testing.assert_close(out.asr_loss, alt.asr_loss)
            torch.testing.assert_close(out.asr_logits, alt.asr_logits)
            single = collate(config, [dict(z=batch.inputs[1, :4], transcript='a')])
            torch.testing.assert_close(out.asr_logits[1, :4], forward(model, single).asr_logits[0])
            z = torch.randn(1, 6, 4)
            changed = z.clone()
            changed[:, 3:] += 50
            torch.testing.assert_close(model.encoder(z)[:, :3], model.encoder(changed)[:, :3])
            out.asr_loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_invalid_transcripts_and_ctc_alignment(self):
        config = config_for()
        with self.assertRaisesRegex(ValueError, 'transcript'):
            collate(config, [row(4)])
        with self.assertRaisesRegex(ValueError, 'truncate'):
            collate(config, [dict(row(9), transcript='a')])
        batch = collate(config, [dict(row(2), transcript='aa')])
        with self.assertRaisesRegex(ValueError, 'alignment impossible'):
            forward(SMQuantizer(config.model), batch)
        config.model.asr.upsample_factor = 2
        self.assertTrue(torch.isfinite(forward(SMQuantizer(config.model), batch).asr_loss))
        batch = collate(config, [dict(row(1), transcript='a')])
        self.assertTrue(torch.isfinite(forward(SMQuantizer(config.model), batch).loss))

    def test_transcript_parquet_and_resume(self):
        for kind in ('vq_ema', 'fsq', 'bsq'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                root = Path(directory)
                partition = root / 'train_clean_100'
                partition.mkdir()
                pq.write_table(pa.Table.from_pylist([dict(row(5), transcript='ab'), dict(row(4), transcript='a')]),
                               partition / 'data.parquet')
                config = config_for(kind, lm=kind == 'bsq')
                config.data.format = 'parquet'
                config.data.train_path = str(root)
                config.data.train_partitions = ['train_clean_100']
                config.data.cache_dir = str(root / 'cache')
                config.data.validation_path = str(root)
                config.data.validation_partitions = ['train_clean_100']
                dataset = build_dataset(config.data, asr_enabled=True)
                self.assertEqual(dataset[0]['transcript'], 'ab')
                config.training.output_dir = str(root / 'full')
                train(config)
                config.training.output_dir = str(root / 'resume')
                config.training.max_steps = 1
                train(config)
                config.training.resume = str(root / 'resume' / 'last.pt')
                config.training.max_steps = None
                train(config)
                actual = torch.load(config.training.resume, weights_only=True)
                expected = torch.load(root / 'full' / 'last.pt', weights_only=True)
                for name, value in actual['model'].items():
                    torch.testing.assert_close(value, expected['model'][name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
