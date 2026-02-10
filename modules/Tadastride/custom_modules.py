import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm1d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, T, C) -> LN(C) -> (B, C, T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)

class Conv1d(nn.Conv1d):
    def __init__(self, *args, causal=False, **kwargs):
        self.is_causal = causal
        if causal and kwargs.get('padding', 0) > 0:
            # Convert symmetric padding to left-only (causal) padding
            self._causal_padding = kwargs['padding'] * 2
            kwargs['padding'] = 0
        else:
            self._causal_padding = 0
        super(Conv1d, self).__init__(*args, **kwargs)

    def forward(self, x, x_mask=None):
        if self.is_causal and self._causal_padding > 0:
            x = F.pad(x, (self._causal_padding, 0))

        if x_mask is None:
            output = super(Conv1d, self).forward(x)
        else:
            output = super(Conv1d, self).forward(x) * x_mask.unsqueeze(1)
            
        return output
