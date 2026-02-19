import torch
import numpy as np
import torch.nn as nn
from typing import List
from dataclasses import dataclass

# Assumiamo che questi moduli siano disponibili nel tuo environment
from . import monotonic_align
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
        term_a = torch.matmul(o_p.transpose(1, 2), z_p_sq)

        # Termine B: -2 * sum(z * m * (1/sigma^2))
        term_b = -2 * torch.matmul((m_p * o_p).transpose(1, 2), z_p)

        # Termine C: sum(m^2 * (1/sigma^2))
        term_c = torch.sum(m_p**2 * o_p, dim=1, keepdim=True).transpose(
            1, 2
        )  # [B, T_x, 1]

        # Termine D: Log Determinante (Parte della normalizzazione della Gaussiana)
        term_d = torch.sum(2 * logs_p, dim=1, keepdim=True).transpose(
            1, 2
        )  # [B, T_x, 1]

        # Somma totale per la Log Likelihood Negativa (pesata)
        log_p = -0.5 * (term_a + term_b + term_c + term_d)  # [B, T_x, T_y]

        # 2. Preparazione per Cython
        # Cython implementation expects [T_y, T_x] but we have [T_x, T_y]
        log_p_transposed = log_p.transpose(1, 2).contiguous()

        # Path buffer [B, T_y, T_x]
        path = torch.zeros(b, t_y, t_x, dtype=torch.int32).to(device=z_p.device)

        # Ensure contiguous memory layout for Cython
        log_p_cpu = log_p_transposed.data.float().cpu().numpy().astype(np.float32)
        path_cpu = path.data.cpu().numpy().astype(np.int32)
        t_x_len = x_mask.sum([1, 2]).data.cpu().numpy().astype(np.int32)
        t_y_len = y_mask.sum([1, 2]).data.cpu().numpy().astype(np.int32)

        # Chiamata Cython
        monotonic_align.maximum_path_c(path_cpu, log_p_cpu, t_y_len, t_x_len)

        # Ritorna il path come tensore float (0.0 o 1.0)
        path = torch.from_numpy(path_cpu).to(device=z_p.device, dtype=torch.float32)
        path = path.transpose(1, 2)
        return path


@dataclass
class AlignerConfig:
    z_dim: int = 64
    hidden_dim: int = 256
    text_encoder_n_layers: int = 4
    text_encoder_n_heads: int = 4
    flow_model_n_layers: int = 4
    flow_model_kernel_size: int = 5


import math
import torch
import torch.nn as nn
from typing import List


class Aligner(nn.Module):
    def __init__(self, config):
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
        z_spec: torch.FloatTensor,  # [B, C, T_audio] sample dal posterior encoder (o output che vuoi allineare)
        y_mask: torch.BoolTensor,  # [B, 1, T_audio]
        phonemes: List[str],
    ):
        """
        Loss usata per MAS / prior: massimizza log p(z | text, A)
        (eventualmente con change-of-variables del flow).
        """

        # 1) Prior da testo
        m_p, logs_p, x_mask = self.text_encoder(
            phonemes
        )  # [B,C,T_text], [B,C,T_text], [B,1,T_text]

        # 2) Flow: porta z_spec nello spazio del prior
        y_mask_f = y_mask.to(dtype=z_spec.dtype)
        z_flow, logdet_flow = self.flow(
            z_spec, y_mask_f, reverse=False
        )  # z_flow: [B,C,T_audio]

        # 3) MAS in spazio prior
        mas_mask = search_path(
            z_flow, m_p, logs_p, x_mask, y_mask
        )  # [B,T_text,T_audio]

        # 4) Espandi statistiche del prior su frame audio
        m_p_exp = torch.matmul(m_p.to(dtype=mas_mask.dtype), mas_mask)  # [B,C,T_audio]
        logs_p_exp = torch.matmul(
            logs_p.to(dtype=mas_mask.dtype), mas_mask
        )  # [B,C,T_audio]

        # 5) Negative log-likelihood del prior su z_flow
        # log p(z_flow | text,A) = sum_{c,t} -0.5[(z-m)^2/sigma^2 + 2log sigma + log 2pi]
        # NLL = -log p = 0.5[(z-m)^2/sigma^2 + 2log sigma + log 2pi]
        inv_var = torch.exp(-2.0 * logs_p_exp)
        nll = 0.5 * (
            (z_flow - m_p_exp) ** 2 * inv_var
            + 2.0 * logs_p_exp
            + math.log(2.0 * math.pi)
        )
        nll = nll * y_mask_f  # maschera audio

        nll_prior = torch.sum(nll)

        # 6) Change-of-variables del flow:
        # log p(z_spec) = log p(z_flow) + log|det J|
        # quindi NLL(z_spec) = NLL(z_flow) - log|det J|
        # Se il tuo logdet_flow è già +log|det J|, allora:
        loss_raw = nll_prior - torch.sum(logdet_flow)

        # Se invece la tua implementazione ritorna logdet con segno opposto, usa:
        # loss_raw = nll_prior + torch.sum(logdet_flow)

        # normalizzazione (consiglio: per frame*canali, così è stabile)
        B, C, T = z_spec.shape
        denom = (torch.sum(y_mask_f) * C).clamp_min(1.0)
        loss = loss_raw / denom

        # durate e pooling token-level
        durations = mas_mask.sum(dim=-1)  # [B,T_text]

        z_pooled = torch.bmm(
            mas_mask.to(dtype=z_spec.dtype), z_spec.permute(0, 2, 1)
        )  # [B,T_text,C]
        dur = durations.to(dtype=z_spec.dtype).clamp_min(1.0).unsqueeze(-1)
        z_pooled = z_pooled / dur

        text_mask = (~x_mask.bool()).squeeze(1)  # [B,T_text]

        return z_pooled, durations.long(), loss, text_mask


# # ==========================================
# # Esempio di utilizzo fittizio
# # ==========================================
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     # Config
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

#     # z e logs_q dal posterior encoder
#     z_from_posterior = torch.randn(B, channels, T_audio).to(device)

#     # Simuliamo la log-varianza in uscita dal posterior encoder.
#     # Spesso viene inizializzata vicino a 0 (che significa varianza = 1)
#     logs_q_from_posterior = torch.zeros(B, channels, T_audio).to(device)

#     y_mask = torch.ones(B, 1, T_audio).to(device)

#     # Generiamo caratteri random per simulare i fonemi
#     phonemes = [
#         [
#             "a",
#             "b",
#         ]
#         * T_text
#         for _ in range(B)
#     ]

#     # Esecuzione
#     z_pooled, durations, loss, text_mask = aligner(
#         z_from_posterior,
#         logs_q_from_posterior,  # Passiamo il nuovo parametro
#         y_mask,
#         phonemes,
#     )

#     print(f"KL Loss corretta: {loss.item()}")
#     print(f"Z Pooled Shape: {z_pooled.shape}")

#     # Salva una figura con l'allineamento (per il batch 0)
#     # Ricalcoliamo mas_mask per il plot visto che non è più restituita dal forward
#     with torch.no_grad():
#         m_p, logs_p, x_mask = aligner.text_encoder(phonemes)
#         z_flow, _ = aligner.flow(z_from_posterior, y_mask, reverse=False)
#         alignment = search_path(z_flow, m_p, logs_p, x_mask, y_mask)

#     plt.imshow(alignment[0].cpu().detach().numpy(), aspect="auto", origin="lower")
#     plt.title("MAS Alignment")
#     plt.xlabel("Audio Frames")
#     plt.ylabel("Text Phonemes")
#     plt.savefig("alignment.png")
#     print("Allineamento salvato in 'alignment.png'")
