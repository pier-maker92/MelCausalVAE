import unittest
from types import SimpleNamespace

import torch

from dicodec.modules.dicodec import Dicodec


def normalize(features, mask=None):
    return Dicodec._normalize_ssl_features(None, features, padding_mask=mask)


class SSLNormalizationTest(unittest.TestCase):
    def test_valid_frames_do_not_depend_on_padding_or_batch(self):
        torch.manual_seed(11)
        short = torch.randn(1, 5, 4)
        batch = torch.randn(2, 12, 4)
        batch[0, :5] = short[0]
        batch[0, 5:] = float('nan')
        mask = torch.zeros(2, 12, dtype=torch.bool)
        mask[0, 5:] = True
        result = normalize(batch, mask)
        torch.testing.assert_close(result[0, :5], normalize(short)[0])
        torch.testing.assert_close(result[0, 5:], torch.zeros(7, 4))
        torch.testing.assert_close(result[1], normalize(batch[1:])[0])

    def test_unpadded_matches_original_sample_std(self):
        torch.manual_seed(12)
        values = torch.randn(3, 7, 4, dtype=torch.float64)
        expected = (values - values.mean(1, keepdim=True)) / (values.std(1, keepdim=True) + 1e-8)
        torch.testing.assert_close(normalize(values), expected)
        torch.testing.assert_close(normalize(values, torch.zeros(3, 7, dtype=torch.bool)), expected)

    def test_non_contiguous_mask_and_gradients(self):
        torch.manual_seed(13)
        values = torch.randn(1, 6, 3, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[False, True, False, True, False, True]])
        valid = values[:, [0, 2, 4]].detach().clone().requires_grad_()
        result = normalize(values, mask)
        reference = normalize(valid)
        torch.testing.assert_close(result[:, [0, 2, 4]], reference)
        weights = torch.randn_like(reference)
        (result[:, [0, 2, 4]] * weights).sum().backward()
        (reference * weights).sum().backward()
        torch.testing.assert_close(values.grad[:, [0, 2, 4]], valid.grad)
        torch.testing.assert_close(values.grad[:, [1, 3, 5]], torch.zeros_like(valid))

    def test_empty_single_frame_and_constant_sequences_have_finite_gradients(self):
        values = torch.full((3, 5, 4), 2.0, requires_grad=True)
        mask = torch.tensor([[True] * 5, [False, True, True, True, True], [False] * 5])
        result = normalize(values, mask)
        torch.testing.assert_close(result, torch.zeros_like(result))
        result.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        torch.testing.assert_close(values.grad[mask], torch.zeros_like(values.grad[mask]))

    def test_low_precision_uses_float32_statistics(self):
        for dtype in (torch.float16, torch.bfloat16):
            values = torch.arange(32).reshape(1, 8, 4).to(dtype)
            mask = torch.tensor([[False] * 5 + [True] * 3])
            result = normalize(values, mask)
            self.assertEqual(result.dtype, dtype)
            torch.testing.assert_close(result, normalize(values.float(), mask).to(dtype))

    def test_encode_passes_mask_to_normalization(self):
        features = torch.tensor([[[1.0], [3.0], [999.0]]])
        mask = torch.tensor([[False, False, True]])
        captured = {}
        def encoder(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()
        model = SimpleNamespace(
            encoder=encoder, external_semantic_quantizer=None,
            _normalize_ssl_features=lambda features, **kw: normalize(features, kw.get('padding_mask')),
        )
        Dicodec.encode(model, features, mask)
        torch.testing.assert_close(captured['x'], normalize(features, mask))
        self.assertIs(captured['padding_mask'], mask)

    def test_invalid_mask_is_rejected(self):
        for mask in (torch.zeros(1, 3), torch.zeros(1, 4, dtype=torch.bool)):
            with self.assertRaisesRegex(ValueError, 'padding_mask'):
                normalize(torch.randn(1, 3, 2), mask)


if __name__ == '__main__':
    unittest.main()
