import unittest
from unittest.mock import patch

import torch

from .asr import ASRHead
from .configs import ASRConfig
from .model import SMQuantizer
from .test_asr import config_for, collate, forward
from .training import validate
from .wer import word_error_counts, word_error_rate


class WERTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_edits_and_normalization(self):
        for reference, hypothesis, expected in [
            ('A b', 'a b', (0, 2)), ('a b', 'a c', (1, 2)),
            ('a b', 'a', (1, 2)), ('a', 'a b c', (2, 1)),
            ('a b', '', (2, 2)), ('', '', (0, 0)), ('', 'a', (1, 0)),
        ]:
            self.assertEqual(word_error_counts(reference, hypothesis), expected)
        self.assertEqual(word_error_rate(2, 1), 2.0)

    def test_ctc_collapse_blank_and_padding(self):
        config = ASRConfig(enabled=True, characters='ab ', hidden_size=4, layers=1, embedding_dim=4)
        head = ASRHead(4, config)
        # a, a, blank, a, space, b -> "aa b"; padded last token must be ignored.
        ids = torch.tensor([[0, 0, 3, 0, 2, 1, 0], [0, 3, 3, 3, 3, 3, 1]])
        logits = torch.nn.functional.one_hot(ids, 4).float()
        targets = torch.tensor([[0, 0, 2, 1], [0, -1, 999, -1]])
        self.assertEqual(head.word_error_counts(logits, torch.tensor([6, 1]),
                                               targets, torch.tensor([4, 1])), (0, 3))
        logits[0, 5] = torch.tensor([1., 0, 0, 0])
        self.assertEqual(head.word_error_counts(logits, torch.tensor([6, 1]),
                                               targets, torch.tensor([4, 1])), (1, 3))

    def test_metric_visibility_and_corpus_aggregation(self):
        for kind in ('vq_ema', 'fsq', 'bsq'):
            for enabled, weight in ((False, 1), (True, 0), (True, 1)):
                config = config_for(kind)
                config.model.asr.enabled = enabled
                config.model.loss.asr = weight
                model = SMQuantizer(config.model)
                batch = collate(config)
                output = forward(model, batch)
                self.assertEqual('wer' in output.metrics(), enabled and weight > 0)
                self.assertEqual('bsq_regularization_loss' in output.metrics(), kind == 'bsq')
                metrics = validate(model, [batch], 'cpu')
                self.assertEqual('wer' in metrics, enabled and weight > 0)
                self.assertEqual('bsq_regularization_loss' in metrics, kind == 'bsq')
        model = SMQuantizer(config_for().model)
        batch = collate(config_for())
        with patch.object(model.asr_head, 'word_error_counts', side_effect=[(1, 1), (0, 9)]):
            self.assertEqual(validate(model, [batch, batch], 'cpu')['wer'], 0.1)


if __name__ == '__main__':
    unittest.main()
