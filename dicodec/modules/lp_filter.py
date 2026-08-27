import torch
import torch.nn as nn
import torch.nn.functional as F

from .output_dataclasses import LowPassFilterOutput


class LowPassFilter(nn.Module):
    """
    FIR low-pass filter for temporal latent sequences.

    The filter is designed using the windowed-sinc method with a Hann
    window. It attenuates temporal frequency components above the
    specified cutoff frequency while preserving lower-frequency
    components.

    The same filter is applied independently to each latent channel
    along the temporal dimension.

    Args:
        cutoff_hz: Cutoff frequency in Hz.
        sample_rate: Sampling rate of the latent sequence in Hz.
        order: Order of the FIR filter. The kernel size is order + 1.

    Input:
        z: Latent sequence of shape [B, T, D].
        valid_mask: Optional mask of shape [B, T] or [B, T, 1].

    Output:
        Low-pass filtered sequence of shape [B, T, D].
    """

    def __init__(
        self,
        cutoff_hz: float,
        sample_rate: float,
        order: int = 20,
    ):
        super().__init__()

        if order <= 0:
            raise ValueError("order must be > 0.")

        if order % 2 != 0:
            raise ValueError(
                "order must be even so that kernel_size = order + 1 is odd."
            )

        if not 0.0 < cutoff_hz < sample_rate / 2:
            raise ValueError(
                "cutoff_hz must be between 0 and Nyquist "
                f"({sample_rate / 2:.2f} Hz)."
            )

        self.cutoff_hz = float(cutoff_hz)
        self.sample_rate = float(sample_rate)
        self.order = order
        self.kernel_size = order + 1

        kernel = self._build_kernel(
            cutoff_hz=self.cutoff_hz,
            sample_rate=self.sample_rate,
            kernel_size=self.kernel_size,
        )

        self.register_buffer("kernel", kernel)

    @staticmethod
    def _build_kernel(
        cutoff_hz: float,
        sample_rate: float,
        kernel_size: int,
    ) -> torch.Tensor:
        n = torch.arange(kernel_size, dtype=torch.float32)
        n = n - (kernel_size - 1) / 2

        fc = cutoff_hz / sample_rate
        kernel = 2.0 * fc * torch.sinc(2.0 * fc * n)

        window = torch.hann_window(
            kernel_size,
            periodic=False,
            dtype=torch.float32,
        )

        kernel = kernel * window
        kernel = kernel / kernel.sum()

        return kernel

    def _filter_sequence(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the FIR filter to a sequence.

        Args:
            x: [B, D, T]

        Returns:
            Filtered sequence [B, D, T].
        """

        _, dim, length = x.shape

        kernel = self.kernel.to(
            device=x.device,
            dtype=x.dtype,
        )

        weight = kernel.view(1, 1, -1).expand(
            dim,
            1,
            self.kernel_size,
        )

        padding = self.order // 2

        if length > padding:
            x = F.pad(
                x,
                (padding, padding),
                mode="reflect",
            )
        else:
            x = F.pad(
                x,
                (padding, padding),
                mode="replicate",
            )

        return F.conv1d(
            x,
            weight,
            stride=1,
            padding=0,
            groups=dim,
        )

    def forward(
        self,
        z: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.dim() != 3:
            raise ValueError(f"z must have shape [B, T, D], got {tuple(z.shape)}.")

        x = z.transpose(1, 2)

        if valid_mask is None:
            z_lp = self._filter_sequence(x)
            return z_lp.transpose(1, 2)

        if valid_mask.ndim == 3:
            valid_mask = valid_mask.squeeze(-1)

        if valid_mask.ndim != 2:
            raise ValueError(
                "valid_mask must have shape [B, T] or [B, T, 1]."
            )

        if valid_mask.shape != z.shape[:2]:
            raise ValueError(
                "valid_mask must have shape [B, T] or [B, T, 1], got "
                f"{tuple(valid_mask.shape)} for z shape {tuple(z.shape)}."
            )

        batch_size, _, _ = z.shape
        output = torch.zeros_like(z)

        for b in range(batch_size):
            length = int(valid_mask[b].sum().item())

            if length == 0:
                continue

            xb = x[b : b + 1, :, :length]
            yb = self._filter_sequence(xb)
            output[b, :length] = yb[0].transpose(0, 1)

        return output

    def decompose(
        self,
        z: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> LowPassFilterOutput:
        z_lp = self(z, valid_mask=valid_mask)
        z_hp = z - z_lp

        if valid_mask is not None:
            if valid_mask.ndim == 3:
                valid_mask = valid_mask.squeeze(-1)
            z_hp = z_hp * valid_mask.to(device=z.device, dtype=z.dtype).unsqueeze(-1)

        return LowPassFilterOutput(z_lp=z_lp, z_hp=z_hp)
