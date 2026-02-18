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
        dilation_rate: Optional[int] = None,  # FIXME why this is not used?
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
            # Forward: Training (Normalizing: Audio complesso -> Gaussiana semplice)
            # x1 = (x1 - m) * exp(-logs)
            x1 = (
                (m - x1) * torch.exp(logs) * x_mask
            )  # Implementazione VITS standard varia leggermente nel segno
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            # Reverse: Inference (Generating: Gaussiana semplice -> Audio complesso)
            # x1 = x1 * exp(-logs) + m (inversione della formula sopra)
            # Nota: L'algebra esatta dipende dalla convenzione affine.
            # Qui usiamo: x1_new = m - x1_old * exp(-logs)
            x1 = m - x1 * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x
