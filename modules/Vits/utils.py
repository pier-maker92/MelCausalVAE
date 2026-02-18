import torch


# Utility maschera (spesso chiamata commons.sequence_mask nelle repo VITS)
class commons:
    @staticmethod
    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)
