import torch
import numpy as np
import torch.nn as nn
from . import monotonic_align
from typing import List
from dataclasses import dataclass
from .text_encoder import TextEncoder
from .flow_model import ResidualCouplingLayer


def search_path(z_p, m_p, logs_p, x_mask, y_mask):
    with torch.no_grad():
        b, c, t_y = z_p.shape
        t_x = m_p.shape[2]

        # z_p: [B, C, T_audio] (Variabili latenti audio)
        # m_p: [B, C, T_text] (Media predetta dal testo)
        # logs_p: [B, C, T_text] (Log-varianza predetta dal testo)

        # 1. Calcoliamo l'inverso della varianza (precisione)
        # 1 / sigma^2 = exp(-2 * logs)
        z_p_sq = z_p**2
        o_p = torch.exp(-2 * logs_p)  # [B, C, T_x]

        # Dobbiamo calcolare: sum_channels( (z - m)^2 / sigma^2 )
        # Espandendo il quadrato: (z^2)/sigma^2 - (2zm)/sigma^2 + (m^2)/sigma^2

        # Termine A: sum(z^2 * (1/sigma^2))
        # Usiamo Matmul: [B, T_x, C] * [B, C, T_y] -> [B, T_x, T_y]
        term_a = torch.matmul(o_p.transpose(1, 2), z_p_sq)

        # Termine B: -2 * sum(z * m * (1/sigma^2))
        # Weighted mean: m_p * o_p
        term_b = -2 * torch.matmul((m_p * o_p).transpose(1, 2), z_p)

        # Termine C: sum(m^2 * (1/sigma^2))
        # Questo è costante per ogni frame audio, dipende solo dal testo
        term_c = torch.sum(m_p**2 * o_p, dim=1, keepdim=True).transpose(
            1, 2
        )  # [B, T_x, 1]

        # Termine D: Log Determinante (Parte della normalizzazione della Gaussiana)
        # sum(log(2*pi) + log(sigma^2)) -> ignoriamo costanti, teniamo 2 * logs_p
        term_d = torch.sum(2 * logs_p, dim=1, keepdim=True).transpose(
            1, 2
        )  # [B, T_x, 1]

        # Somma totale per la Log Likelihood Negativa (pesata)
        # log_p = -0.5 * ( (z-m)^2/var + log_var )
        log_p = -0.5 * (term_a + term_b + term_c + term_d)  # [B, T_x, T_y]

        # 2. Preparazione per Cython
        # Cython implementation expects [T_y, T_x] but we have [T_x, T_y]
        # We need to transpose inputs and outputs [B, T_y, T_x]
        log_p_transposed = log_p.transpose(1, 2).contiguous()

        # Path buffer [B, T_y, T_x]
        path = torch.zeros(b, t_y, t_x, dtype=torch.int32).to(device=z_p.device)

        # Ensure contiguous memory layout for Cython
        log_p_cpu = log_p_transposed.data.cpu().numpy().astype(np.float32)
        path_cpu = path.data.cpu().numpy().astype(np.int32)
        t_x_len = x_mask.sum([1, 2]).data.cpu().numpy().astype(np.int32)
        t_y_len = y_mask.sum([1, 2]).data.cpu().numpy().astype(np.int32)

        # Chiamata Cython
        monotonic_align.maximum_path_c(path_cpu, log_p_cpu, t_y_len, t_x_len)

        # Ritorna il path come tensore float (0.0 o 1.0)
        # Converti e transponi back a [B, T_x, T_y]
        path = torch.from_numpy(path_cpu).to(device=z_p.device, dtype=torch.float32)
        path = path.transpose(1, 2)
        return path


class VitsKLProcessor(nn.Module):
    def __init__(self, text_encoder, flow_model):
        super().__init__()
        self.text_encoder = text_encoder
        self.flow = flow_model

    def forward(self, z_spec, y_mask, phonemes, phoneme_lengths):
        """
        z_spec: [B, C, T_audio] (Output del Posterior Encoder - campionato con reparam trick)
        y_mask: [B, 1, T_audio] (Mask dell'audio)
        phonemes: [B, T_text]
        phoneme_lengths: [B]
        """

        # 1. Text Encoding (Prior)
        # m_p, logs_p: statistiche previste dal testo [B, C, T_text]
        m_p, logs_p, x_mask = self.text_encoder(phonemes, phoneme_lengths)

        # 2. Flow Model (Trasformazione)
        # Trasforma z_spec (complesso) verso z_flow (semplice)
        z_flow, logdet_flow = self.flow(z_spec, y_mask, reverse=False)

        # 3. Monotonic Alignment Search (MAS)
        # Trova la mappa ottimale tra z_flow (audio trasformato) e m_p (testo)
        # Nota: MAS usa z_flow, non z_spec, perché z_flow è comparabile al testo
        mas_mask = search_path(z_flow, m_p, logs_p, x_mask, y_mask)
        # mas_mask shape: [B, T_text, T_audio]

        # 4. Espansione delle statistiche del testo
        # Moltiplica Prior per Allineamento: [B, C, T_text] x [B, T_text, T_audio] = [B, C, T_audio]
        m_p_expanded = torch.matmul(m_p, mas_mask)
        logs_p_expanded = torch.matmul(logs_p, mas_mask)

        # 5. Calcolo KL Loss
        # La formula è: -LogLikelihood(z_flow | N(m_p_exp, logs_p_exp)) - log_det_jacobian

        # Log Normal PDF
        # Costante -0.5 * log(2pi) omessa perché costante durante l'ottimizzazione
        kl_loss_elements = logs_p_expanded + 0.5 * torch.exp(-2 * logs_p_expanded) * (
            (z_flow - m_p_expanded) ** 2
        )

        # Somma su canali e tempo, media sul batch
        kl_loss_raw = torch.sum(kl_loss_elements * y_mask) - torch.sum(logdet_flow)

        # Normalizzazione per batch e time (o per numero di elementi totali mascherati)
        # In VITS solitamente si normalizza per numero totale di frame nel batch
        total_frames = torch.sum(y_mask)
        loss = kl_loss_raw / total_frames

        return loss, mas_mask, z_flow


@dataclass
class AlignerConfig:
    z_dim: int = 64
    hidden_dim: int = 256
    text_encoder_n_layers: int = 4
    text_encoder_n_heads: int = 4
    flow_model_n_layers: int = 4
    flow_model_kernel_size: int = 5


class Aligner(nn.Module):
    def __init__(self, config: AlignerConfig):
        super().__init__()
        self.text_encoder = TextEncoder(
            hidden_channels=config.hidden_dim,
            n_heads=config.text_encoder_n_heads,
            n_layers=config.text_encoder_n_layers,
            kernel_size=config.flow_model_kernel_size,
            output_dim=config.z_dim,
            p_dropout=0.1,
            vocab_path="data/vocab.json",
            parsing_mode="phoneme",
        )
        self.flow = ResidualCouplingLayer(
            channels=config.z_dim,
            hidden_channels=config.hidden_dim,
            n_layers=config.flow_model_n_layers,
            kernel_size=config.flow_model_kernel_size,
        )

    def forward(
        self,
        z_spec: torch.FloatTensor,
        y_mask: torch.BoolTensor,
        phonemes: List[str],
    ):
        """
        z_spec: [B, C, T_audio] (Output del Posterior Encoder - campionato con reparam trick)
        y_mask: [B, 1, T_audio] (Mask dell'audio)
        phonemes: List[str]
        """

        # 1. Text Encoding (Prior)
        # m_p, logs_p: statistiche previste dal testo [B, C, T_text]
        m_p, logs_p, x_mask = self.text_encoder(phonemes)

        # 2. Flow Model (Trasformazione)
        # Trasforma z_spec (complesso) verso z_flow (semplice)
        z_flow, logdet_flow = self.flow(z_spec, y_mask, reverse=False)

        # 3. Monotonic Alignment Search (MAS)
        # Trova la mappa ottimale tra z_flow (audio trasformato) e m_p (testo)
        # Nota: MAS usa z_flow, non z_spec, perché z_flow è comparabile al testo
        mas_mask = search_path(z_flow, m_p, logs_p, x_mask, y_mask)
        # mas_mask shape: [B, T_text, T_audio]

        # 4. Espansione delle statistiche del testo
        # Moltiplica Prior per Allineamento: [B, C, T_text] x [B, T_text, T_audio] = [B, C, T_audio]
        m_p_expanded = torch.matmul(m_p, mas_mask)
        logs_p_expanded = torch.matmul(logs_p, mas_mask)

        # 5. Calcolo KL Loss
        # La formula è: -LogLikelihood(z_flow | N(m_p_exp, logs_p_exp)) - log_det_jacobian

        # Log Normal PDF
        # Costante -0.5 * log(2pi) omessa perché costante durante l'ottimizzazione
        kl_loss_elements = logs_p_expanded + 0.5 * torch.exp(-2 * logs_p_expanded) * (
            (z_flow - m_p_expanded) ** 2
        )

        # Somma su canali e tempo, media sul batch
        kl_loss_raw = torch.sum(kl_loss_elements * y_mask) - torch.sum(logdet_flow)

        # Normalizzazione per batch e time (o per numero di elementi totali mascherati)
        # In VITS solitamente si normalizza per numero totale di frame nel batch
        total_frames = torch.sum(y_mask)
        loss = kl_loss_raw / total_frames

        durations = mas_mask.sum(dim=-1)

        z_pooled = torch.bmm(mas_mask, z_spec.permute(0, 2, 1)) / x_mask.sum(
            -1
        ).unsqueeze(-1)

        return z_pooled, durations, loss, (~x_mask).squeeze(1)


# # Esempio di utilizzo fittizio
# if __name__ == "__main__":
#     # Config
#     vocab_size = 100
#     channels = 192
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     config = AlignerConfig()
#     config.z_dim = channels
#     aligner = Aligner(config)
#     aligner.to(device)

#     # Input Finti
#     B = 24
#     T_audio = 1300
#     T_text = 240

#     # z dal posterior encoder (già campionato)
#     z_from_posterior = torch.randn(B, channels, T_audio).to(device)
#     y_mask = torch.ones(B, 1, T_audio).to(device)

#     # generate random chars for phonemes batched
#     phonemes = [
#         [
#             "a",
#             "b",
#         ]
#         * T_text
#         for _ in range(B)
#     ]

#     # Calcolo
#     loss, alignment, z_transformed = aligner(z_from_posterior, y_mask, phonemes)

#     print(f"KL Loss: {loss.item()}")
#     print(f"Alignment Shape: {alignment.shape}")  # Dovrebbe essere [2, 10, 50]

#     # save a figure with  the alignement
#     import matplotlib.pyplot as plt

#     plt.imshow(alignment[0].cpu().detach().numpy(), aspect="auto")
#     plt.savefig("alignment.png")
