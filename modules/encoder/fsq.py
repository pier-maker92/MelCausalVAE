import torch
import torch.nn as nn
from typing import Tuple, List

__all__ = ["FiniteScalarQuantizer"]

FSQ_LEVELS = {
    128:  [4, 4, 4, 2],
    256:  [4, 4, 4, 4],
    512:  [8, 4, 4, 4],
    1024: [8, 8, 4, 4],
    2048: [8, 8, 8, 4],
    4096: [8, 8, 8, 8],
    8192: [16, 8, 8, 8],
}

class FiniteScalarQuantizer(nn.Module):
    """
    Finite Scalar Quantizer (FSQ)
    Reference: "Finite Scalar Quantization: VQ-VAE Made Simple" (https://arxiv.org/abs/2309.15505)
    """
    def __init__(self, codebook_size: int = 256):
        super().__init__()
        levels = FSQ_LEVELS.get(codebook_size, None)
        if levels is None:
            raise ValueError(f"Unsupported number of embeddings: {codebook_size}. Supported values: {list(FSQ_LEVELS.keys())}")
        self.levels_list = levels
        self.dim = len(levels)
        
        _levels = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("_levels", _levels, persistent=False)
        
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int32), dim=0)
        self.register_buffer("_basis", _basis, persistent=False)
        
        self.codebook_size = int(_levels.prod().item())

    def forward(self, lats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        lats:
            Input latents of shape (..., dim).
            
        Returns
        -------
            - Output tokens of shape (...);
            - output codes (i.e. quantized latents) of shape (..., dim).
        """
        codes = self.lats_to_codes(lats)
        toks = self.codes_to_toks(codes)
        return toks, codes

    def lats_to_codes(self, lats: torch.Tensor) -> torch.Tensor:
        """
        Transform continuous latents into discrete codes.
        Applies bounding (tanh) and then quantization to the specified levels.
        """
        z = torch.tanh(lats)
        half_l = (self._levels - 1) * 0.5
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        
        z = z * half_l - offset
        z_q = torch.round(z)
        
        z_q = z_q + offset
        z_q = z_q / half_l
        return z_q

    def codes_to_toks(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Transform continuous codes back into scalar integer tokens.
        """
        half_l = (self._levels - 1) * 0.5
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        
        # Recover rounded integers
        z_hat = torch.round(codes * half_l - offset)
        
        # Shift to start at 0
        z_hat = z_hat + half_l + offset
        
        # Combine into single integer
        return (z_hat * self._basis).sum(dim=-1).long()
        
    def lats_to_toks(self, lats: torch.Tensor) -> torch.Tensor:
        codes = self.lats_to_codes(lats)
        return self.codes_to_toks(codes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(levels={self.levels_list}, codebook_size={self.codebook_size})"
