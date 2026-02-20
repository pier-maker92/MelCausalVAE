import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from dataclasses import dataclass

# Assumiamo che questi moduli siano disponibili nel tuo environment
from . import monotonic_align
from .text_encoder import TextEncoder


def search_path_cosine(z_proj, m_p, x_mask, y_mask, temperature=0.1):
    """
    Versione del MAS che usa la Cosine Similarity (coerente con InfoNCE Loss).
    """
    with torch.no_grad():
        b, c, t_y = z_proj.shape
        t_x = m_p.shape[2]

        # 1. Normalizziamo i vettori (esattamente come nella Loss)
        z_proj_norm = F.normalize(z_proj, p=2, dim=1) # [B, C, T_y]
        m_p_norm = F.normalize(m_p, p=2, dim=1)       # [B, C, T_x]

        # 2. Calcoliamo la Cosine Similarity (Prodotto scalare dei vettori normalizzati)
        # [B, T_x, C] x [B, C, T_y] -> [B, T_x, T_y]
        sim = torch.matmul(m_p_norm.transpose(1, 2), z_proj_norm) 

        # 3. Scaliamo con la temperatura (Logits)
        # Questo passaggio è vitale per il MAS: la similarità è tra -1 e 1. 
        # Dividendola per 0.1 la portiamo tra -10 e 10. 
        # Questo dà al MAS una "pendenza" molto più netta per decidere i confini.
        log_p = sim / temperature 

        # 4. Preparazione per Cython (Identica a prima)
        log_p_transposed = log_p.transpose(1, 2).contiguous()
        path = torch.zeros(b, t_y, t_x, dtype=torch.int32).to(device=z_proj.device)

        log_p_cpu = log_p_transposed.data.float().cpu().numpy().astype(np.float32)
        path_cpu = path.data.cpu().numpy().astype(np.int32)
        t_x_len = x_mask.sum([1, 2]).data.cpu().numpy().astype(np.int32)
        t_y_len = y_mask.sum([1, 2]).data.cpu().numpy().astype(np.int32)

        monotonic_align.maximum_path_c(path_cpu, log_p_cpu, t_y_len, t_x_len)

        path = torch.from_numpy(path_cpu).to(device=z_proj.device, dtype=torch.float32)
        path = path.transpose(1, 2)  
        return path

def search_path_l2(z_proj, m_p, x_mask, y_mask):
    """
    Versione semplificata del MAS che usa la distanza L2 negativa
    invece della Log-Likelihood Gaussiana complessa.
    """
    with torch.no_grad():
        b, c, t_y = z_proj.shape
        t_x = m_p.shape[2]

        # Calcoliamo la distanza L2 quadratica espansa: (z - m)^2 = z^2 + m^2 - 2zm
        z_sq = torch.sum(z_proj**2, dim=1, keepdim=True)  # [B, 1, T_y]
        m_sq = torch.sum(m_p**2, dim=1, keepdim=True).transpose(1, 2)  # [B, T_x, 1]
        z_m = torch.matmul(m_p.transpose(1, 2), z_proj)  # [B, T_x, T_y]

        # Vogliamo MASSIMIZZARE la similarità, quindi usiamo la distanza negativa
        log_p = -(z_sq + m_sq - 2 * z_m)  # [B, T_x, T_y]

        # Preparazione per Cython (Mantenuta identica alla tua logica)
        log_p_transposed = log_p.transpose(1, 2).contiguous()
        path = torch.zeros(b, t_y, t_x, dtype=torch.int32).to(device=z_proj.device)

        log_p_cpu = log_p_transposed.data.float().cpu().numpy().astype(np.float32)
        path_cpu = path.data.cpu().numpy().astype(np.int32)
        t_x_len = x_mask.sum([1, 2]).data.cpu().numpy().astype(np.int32)
        t_y_len = y_mask.sum([1, 2]).data.cpu().numpy().astype(np.int32)

        monotonic_align.maximum_path_c(path_cpu, log_p_cpu, t_y_len, t_x_len)

        path = torch.from_numpy(path_cpu).to(device=z_proj.device, dtype=torch.float32)
        path = path.transpose(1, 2)  # Ritorna [B, T_x, T_y]
        return path


@dataclass
class AlignerConfig:
    z_dim: int = 192               
    proj_dim: int = 256            
    text_encoder_n_layers: int = 4
    text_encoder_n_heads: int = 4
    
    # Parametri per la Projection Head Profonda
    proj_kernel_size: int = 5      
    proj_n_layers: int = 3         
    proj_dropout: float = 0.1      


class Aligner(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.text_encoder = TextEncoder(
            hidden_channels=config.proj_dim, 
            n_heads=config.text_encoder_n_heads,
            n_layers=config.text_encoder_n_layers,
            kernel_size=config.proj_kernel_size,
            output_dim=config.proj_dim, 
            p_dropout=0.1,
            vocab_path="data/vocab.json",
            parsing_mode="phoneme",
        )
        
        # --- NUOVA PROJECTION HEAD PROFONDA ---
        layers = []
        in_ch = config.z_dim
        
        # Strati nascosti con Attivazione e Dropout
        for _ in range(config.proj_n_layers - 1):
            layers.append(
                nn.Conv1d(
                    in_channels=in_ch, 
                    out_channels=config.proj_dim, 
                    kernel_size=config.proj_kernel_size, 
                    padding=config.proj_kernel_size // 2
                )
            )
            layers.append(nn.GELU()) 
            layers.append(nn.Dropout(config.proj_dropout))
            in_ch = config.proj_dim
            
        # Ultimo strato di proiezione puro (Lineare, kernel size 1, senza attivazioni)
        layers.append(nn.Conv1d(in_ch, config.proj_dim, kernel_size=1))
        
        self.proj_head = nn.Sequential(*layers)
        # --------------------------------------

    def forward(
        self,
        z_spec: torch.FloatTensor,  # [B, z_dim, T_audio] dal posterior encoder
        y_mask: torch.BoolTensor,   # [B, 1, T_audio]
        phonemes: List[str],
    ):
        y_mask_f = y_mask.to(dtype=z_spec.dtype)

        # 1) Prior da testo
        out = self.text_encoder(phonemes)
        m_p, x_mask = out[0], out[-1] # [B, proj_dim, T_text], [B, 1, T_text]

        # 2) Proiezione profonda
        z_proj = self.proj_head(z_spec) * y_mask_f # [B, proj_dim, T_audio]

        # 3) MAS nello spazio proiettato 
        #mas_mask = search_path_l2(z_proj, m_p, x_mask, y_mask)  # [B, T_text, T_audio]
        # 3) MAS nello spazio proiettato (usando Cosine Similarity)
        mas_mask = search_path_cosine(z_proj, m_p, x_mask, y_mask, temperature=0.1)

        # 4) Contrastive Loss (InfoNCE)
        # Sostituisce la L2 loss per evitare il Dimensional Collapse
        
        # Normalizza i vettori per calcolare la Cosine Similarity
        z_proj_norm = F.normalize(z_proj, p=2, dim=1) # [B, C, T_audio]
        m_p_norm = F.normalize(m_p, p=2, dim=1)       # [B, C, T_text]

        # Ottieni il target espanso
        m_p_exp_norm = torch.matmul(m_p_norm.to(dtype=mas_mask.dtype), mas_mask) # [B, C, T_audio]

        # Calcola la similarità per i POSITIVI (Audio vs Fonema corretto)
        sim_pos = torch.sum(z_proj_norm * m_p_exp_norm, dim=1) # [B, T_audio]

        # Calcola la similarità per i NEGATIVI (Audio vs TUTTI i fonemi del batch)
        # [B, T_audio, C] x [B, C, T_text] -> [B, T_audio, T_text]
        sim_all = torch.bmm(z_proj_norm.transpose(1, 2), m_p_norm) 

        # Contrastive Loss (CrossEntropy mascherata)
        temperature = 0.1
        logits = sim_all / temperature
        targets = mas_mask.transpose(1, 2).argmax(dim=-1) # Indice del fonema corretto

        # Flattening per la CrossEntropy
        logits = logits.view(-1, logits.size(-1)) # [B * T_audio, T_text]
        targets = targets.view(-1)                # [B * T_audio]
        mask_flat = y_mask.view(-1)

        # Calcola la loss solo sui frame audio validi
        loss_contrastive = F.cross_entropy(logits[mask_flat], targets[mask_flat])


        # 5) Pooling token-level (fatto su z_spec originale)
        durations = mas_mask.sum(dim=-1)  # [B, T_text]

        z_pooled = torch.bmm(
            mas_mask.to(dtype=z_spec.dtype), z_spec.permute(0, 2, 1)
        )
        
        # Media
        dur = durations.to(dtype=z_spec.dtype).clamp_min(1.0).unsqueeze(-1)
        z_pooled = z_pooled / dur

        text_mask = (~x_mask.bool()).squeeze(1)  # [B, T_text]

        return z_pooled, durations.long(), loss_contrastive, text_mask