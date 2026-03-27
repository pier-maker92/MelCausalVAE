import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_durations_on_mel(
    mels,
    durations,
    mel_mask,
    compress_factor_C=8,
    batch_idx=0,
    step=0,
    labels=None,
    device_id=0,
):
    """
    Plot mel spectrogram with segmentation boundaries from similarity pooling.
    
    Args:
        mels: [B, T, F] mel spectrograms
        durations: [B, T_pooled] durations for each pooled segment (in latent space frames)
        mel_mask: [B, T] padding mask (True for padding)
        compress_factor_C: compression factor to convert latent frames to mel frames
        batch_idx: which sample to plot
        step: training step
        labels: optional labels for segments
        device_id: device id for title
    """
    # mel_mask is True for padding. We want valid length.
    valid_len = (~mel_mask[batch_idx]).long().sum().item()

    mel = mels[batch_idx, :valid_len].detach().float().cpu().numpy().T
    
    # Get valid durations (non-zero, non-padding)
    dur_tensor = durations[batch_idx]
    valid_mask = dur_tensor > 0
    dur = dur_tensor[valid_mask].detach().float().cpu().numpy()
    
    # Log first 10 durations for debugging
    print(f"[Step {step}] Sample {batch_idx} durations (first 10): {dur[:10]}")
    print(f"[Step {step}] Total segments: {len(dur)}, mel frames: {valid_len}, latent frames: {valid_len // compress_factor_C}")
    
    # Scale durations to mel space (they are in latent compressed space)
    dur_mel_space = dur * compress_factor_C
    positions = dur_mel_space.cumsum()
    
    fig, (ax_mel, ax_dur) = plt.subplots(2, 1, figsize=(16, 6))

    ax_mel.imshow(mel, origin="lower", aspect="auto")
    ax_mel.set_xlim(0, mel.shape[1])
    for pos in positions:
        ax_mel.axvline(pos, color="white", linestyle="--", linewidth=0.8, alpha=0.7)

    ax_mel.set_ylabel("Mel bin")
    ax_mel.set_title(f"Sample {batch_idx} - Step {step} - Device {device_id} - {len(dur)} segments")
    ax_mel.set_xticks([])

    # Plot durations (in mel space for clarity)
    norm_dur = (dur_mel_space - dur_mel_space.min()) / (dur_mel_space.max() - dur_mel_space.min() + 1e-8) * 0.7 + 0.3
    ax_dur.bar(
        range(len(dur_mel_space)),
        dur_mel_space,
        color=plt.cm.Blues(norm_dur),
        edgecolor="black",
        linewidth=0.5,
    )
    ax_dur.set_xlabel("Segment index")
    ax_dur.set_ylabel("Duration (mel frames)")
    ax_dur.set_xlim(-0.5, len(dur_mel_space) - 0.5)
    
    # Only show some x-ticks if too many segments
    if len(dur_mel_space) > 20:
        step_size = max(1, len(dur_mel_space) // 20)
        tick_positions = range(0, len(dur_mel_space), step_size)
        ax_dur.set_xticks(tick_positions)
        ax_dur.set_xticklabels(
            [labels[i] if labels and i < len(labels) else str(i) for i in tick_positions],
            rotation=45, ha="right"
        )
    else:
        ax_dur.set_xticks(range(len(dur_mel_space)))
        ax_dur.set_xticklabels(
            labels[:len(dur_mel_space)] if labels else range(len(dur_mel_space)), 
            rotation=0, ha="right"
        )

    plt.tight_layout()
    return fig


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
    def __init__(self, threshold=0.9, threshold_in_01: bool = False, **kwargs):
        super().__init__()
        self.threshold = threshold
        self.threshold_in_01 = threshold_in_01
        self.last_below_threshold_pct = None

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

        # 1. Calcolo similarità (solo sui frame validi)
        x_norm = F.normalize(x, p=2, dim=-1)
        # sim_raw[B, T-1]: cosine similarity in [-1, 1]
        sim_raw = torch.sum(x_norm[:, :-1] * x_norm[:, 1:], dim=-1)
        # Optionally map to [0, 1] for more intuitive thresholds
        sim = (sim_raw + 1.0) / 2.0 if self.threshold_in_01 else sim_raw

        # 2. Identificazione confini
        # Un confine esiste se: sim < threshold E entrambi i frame sono validi
        valid_pairs = (mask[:, :-1] == 0) & (mask[:, 1:] == 0)
        below_threshold = sim < self.threshold
        is_boundary = below_threshold & valid_pairs
        valid_pairs_count = valid_pairs.sum()
        if valid_pairs_count.item() > 0:
            self.last_below_threshold_pct = (
                (below_threshold & valid_pairs).sum().float() / valid_pairs_count.float()
            ) * 100.0
        else:
            self.last_below_threshold_pct = torch.tensor(
                0.0, device=x.device, dtype=torch.float32
            )

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
        flat_segment_ids = (segment_ids + batch_offsets).reshape(-1)

        # Maschera per ignorare i frame di padding originali e segmenti -1
        valid_flat_mask = segment_ids.reshape(-1) >= 0

        # Scatter add
        pooled_x.reshape(-1, C).index_add_(
            0, flat_segment_ids[valid_flat_mask], x.reshape(-1, C)[valid_flat_mask]
        )
        counts.reshape(-1).index_add_(
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