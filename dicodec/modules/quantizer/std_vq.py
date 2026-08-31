import torch
import torch.nn as nn
from typing import Tuple

__all__ = ["StandardVectorQuantizer"]

class StandardVectorQuantizer(nn.Module):
    """
    Standard Vector Quantizer with learned embeddings.
    """
    def __init__(self, dim: int, codebook_size: int = 256):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        
        self.embedding = nn.Embedding(self.codebook_size, self.dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        
    def forward(self, lats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lats: [..., D]
        Returns:
            toks: [...]
            codes: [..., D]
        """
        flatten = lats.reshape(-1, self.dim)
        
        # Calculate distances
        distances = (
            torch.sum(flatten**2, dim=1, keepdim=True) 
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flatten, self.embedding.weight.t())
        )
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        toks = encoding_indices.view(*lats.shape[:-1])
        codes = self.embedding(toks)
        
        return toks, codes
