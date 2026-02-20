import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityUpsamplerBatch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pooled_x, durations, target_T=None):
        """
        Args:
            pooled_x: [B, T_new, C] - Feature poolate
            durations: [B, T_new] - Durate di ogni segmento (0 per padding)
            target_T: [B] (Tensor) - Lunghezza totale originale per ogni sample
        Returns:
            x_restored: [B, max(target_T), C]
            new_mask: [B, max(target_T)] - 0 per valid, 1 per padded
        """
        B, T_reduced, C = pooled_x.shape
        device = pooled_x.device

        # Se target_T non è fornito, lo calcoliamo come somma delle durate
        if target_T is None:
            target_T = durations.sum(dim=1)

        max_len = int(target_T.max().item())

        # Inizializziamo i tensor di output
        x_restored = torch.zeros((B, max_len, C), device=device)
        # Inizializziamo la maschera a 1 (tutto padding)
        new_mask = torch.ones((B, max_len), device=device, dtype=torch.long)

        for b in range(B):
            # 1. Recuperiamo le durate valide (maggiori di 0)
            valid_durs = durations[b][durations[b] > 0]

            # 2. Selezioniamo solo i segmenti validi in pooled_x
            # Nota: il numero di segmenti validi corrisponde a quanti elementi in valid_durs
            num_valid_segments = valid_durs.size(0)
            valid_feats = pooled_x[b, :num_valid_segments]

            # 3. Espansione (repeat_interleave)
            res = torch.repeat_interleave(valid_feats, valid_durs, dim=0)

            # 4. Inserimento nel batch restaurato
            # Usiamo min() per sicurezza nel caso target_T[b] sia più piccolo della somma delle durate
            curr_len = min(res.shape[0], int(target_T[b].item()))
            x_restored[b, :curr_len] = res[:curr_len]

            # 5. Aggiornamento maschera: i primi curr_len frame sono validi (0)
            new_mask[b, :curr_len] = 0

        return x_restored, new_mask


class SimilarityPoolingBatch(nn.Module):
    def __init__(self, threshold=0.9, **kwargs):
        super().__init__()
        self.threshold = threshold
        self.conv1d = nn.Conv1d(
            kwargs.get("input_dim", 64),
            kwargs.get("hidden_dim", 512),
            kernel_size=3,
            padding=1,
        )


        #let the conv1d be initialized using a gaussian distribution with mean 0 and std 0.02 (like hubert)
        nn.init.normal_(self.conv1d.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.conv1d.bias, 0.0)


        self.out_proj = nn.Linear(kwargs.get("hidden_dim", 512), kwargs.get("input_dim", 64))

    def forward(self, x, mask):
        """
        Args:
            x: [B, T, C] - Feature di input
            mask: [B, T] - 0 per valid, 1 per padded
        Returns:
            pooled_x: [B, T_new, C] - Feature poolate (con padding)
            durations: [B, T_new] - Durata di ogni segmento (0 se padding)
            new_mask: [B, T_new] - Nuova maschera per i dati poolati
        """
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype

        # 0. apply convolution
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.out_proj(x)
        
        # 1. Calcolo similarità (solo sui frame validi)
        x_norm = F.normalize(x, p=2, dim=-1)
        # sim[B, T-1]: similarità tra t e t+1
        sim = torch.sum(x_norm[:, :-1] * x_norm[:, 1:], dim=-1)

        # 2. Identificazione confini
        # Un confine esiste se: sim < threshold E entrambi i frame sono validi
        valid_pairs = (mask[:, :-1] == 0) & (mask[:, 1:] == 0)
        is_boundary = (sim < self.threshold) & valid_pairs

        # Il primo frame di ogni sequenza valida è sempre l'inizio di un segmento
        starts = torch.cat([(mask[:, :1] == 0), is_boundary], dim=1)

        # 3. Creazione Segment IDs (locali per ogni riga)
        segment_ids = torch.cumsum(starts.long(), dim=1) - 1
        # Mettiamo a -1 i segmenti di padding per ignorarli nello scatter
        segment_ids[mask == 1] = -1

        # 4. Calcolo del numero massimo di segmenti nel batch
        num_segments_per_batch = segment_ids.max(dim=1).values + 1
        max_segments = num_segments_per_batch.max().item()

        # 5. Pooling tramite scatter_reduce (Media dei segmenti)
        pooled_x = torch.zeros((B, max_segments, C), device=device, dtype=dtype)
        counts = torch.zeros((B, max_segments), device=device, dtype=torch.long)

        # Preparazione indici per scatter
        # Trasformiamo segment_ids in indici globali per il batch [B*T]
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * max_segments
        flat_segment_ids = (segment_ids + batch_offsets).view(-1)

        # Maschera per ignorare i frame di padding originali e segmenti -1
        valid_flat_mask = segment_ids.view(-1) >= 0

        # Scatter add
        pooled_x.view(-1, C).index_add_(
            0, flat_segment_ids[valid_flat_mask], x.view(-1, C)[valid_flat_mask]
        )
        counts.view(-1).index_add_(
            0,
            flat_segment_ids[valid_flat_mask],
            torch.ones_like(flat_segment_ids[valid_flat_mask], dtype=torch.long),
        )

        # Evitiamo divisioni per zero e calcoliamo la media
        safe_counts = counts.unsqueeze(-1).clamp(min=1)
        pooled_x = pooled_x / safe_counts

        # 6. Preparazione output
        durations = counts.long()
        new_mask = (counts == 0).long()  # 1 dove non ci sono stati segmenti (padding)

        return pooled_x, durations, new_mask


class Aligner(nn.Module):
    def __init__(self, threshold=0.9, **kwargs):
        super().__init__()
        self.pooler = SimilarityPoolingBatch(threshold=threshold, **kwargs)

    def forward(self, mels, padding_mask, target_T=None):
        pooled_x, durations, new_mask = self.pooler(mels, padding_mask)
        return pooled_x, durations, new_mask
