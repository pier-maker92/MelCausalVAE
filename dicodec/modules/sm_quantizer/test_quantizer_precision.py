import unittest

import torch

from .configs import QuantizerConfig
from .quantizer import OnlineQuantizer


class QuantizerPrecisionTests(unittest.TestCase):
    def test_fsq_bf16_preserves_unsaturated_float32_gradient(self):
        q = OnlineQuantizer(QuantizerConfig(type='fsq', codebook_size=1024))
        x = torch.full((1, 2, 4), 4., dtype=torch.bfloat16, requires_grad=True)
        mask = torch.tensor([[True, False]])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = q(x, mask)
        result.codes.sum().backward()
        expected = (1 - x.detach().float().tanh().square()).to(x.dtype)
        expected[~mask] = 0
        torch.testing.assert_close(x.grad, expected)
        self.assertGreater(x.grad[0, 0, 0].item(), 0)
        self.assertEqual(result.codes.dtype, x.dtype)

    def test_ema_autocast_does_not_change_assignments_or_updates(self):
        import copy
        torch.manual_seed(42)
        q = OnlineQuantizer(QuantizerConfig(type='vq_ema', codebook_size=1024))
        q.codebook.embedding.normal_()
        other = copy.deepcopy(q)
        x = torch.randn(2, 50, 64)
        valid = torch.ones(2, 50, dtype=torch.bool)
        expected = q(x, valid)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = other(x, valid)
        torch.testing.assert_close(expected.indices, actual.indices)
        for name, value in q.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)
