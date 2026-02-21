import torch
import torch.nn as nn
from typing import Optional


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        n_layers,
        p_dropout=0.0,
        dilation_rate: Optional[int] = None,
    ):
        super().__init__()
        self.half_channels = channels // 2

        # Rete non causale (WaveNet-like) che trasforma metà dei canali
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = nn.ModuleList()
        self.dilations = [
            1
        ] * n_layers  # Semplificazione: dilatations fisse o crescenti

        for i in range(n_layers):
            self.enc.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        dilation=self.dilations[i],
                        padding=(kernel_size - 1) * self.dilations[i] // 2,
                    ),
                    nn.GELU(),
                    nn.Dropout(p_dropout),
                )
            )

        # Proietta indietro per ottenere shift (m) e scale (logs)
        # Nota: l'output è doppio perché predice m e logs per l'altra metà
        self.post = nn.Conv1d(hidden_channels, self.half_channels * 2, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, reverse=False):
        # Split canali: [x0, x1]
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        # x0 passa attraverso la rete per trasformare x1
        h = self.pre(x0) * x_mask
        for layer in self.enc:
            h = layer(h) + h  # Residual connection interna
            h = h * x_mask

        stats = self.post(h) * x_mask
        m, logs = torch.split(stats, [self.half_channels] * 2, 1)

        if not reverse:
            # Forward (Normalizing): Audio complesso -> Gaussiana semplice
            # Trasformazione affine: x1_new = (m - x1) * exp(logs)
            # Jacobiano: df/dx1 = -exp(logs), log|det J| = sum(logs)
            x1 = (m - x1) * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs * x_mask, [1, 2])
            return x, logdet
        else:
            # Reverse (Generative): Gaussiana semplice -> Audio complesso
            # Inversione: x1 = m - x1_new * exp(-logs)
            x1 = (m - x1 * torch.exp(-logs)) * x_mask
            x = torch.cat([x0, x1], 1)
            return x
