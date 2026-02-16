import json
import math
import torch
import torch.nn as nn
from typing import List
import matplotlib.pyplot as plt
import torch.nn.functional as F


def compute_sma_loss(alpha, imv, mel_mask, text_mask):
    """
    Calcola la Soft Monotonic Alignment Loss.

    Args:
        alpha: Matrice di attenzione [B, T_text, T_audio]
        imv: Index Mapping Vector [B, T_audio]
        mel_mask: Maschera audio [B, T_audio] (1.0 valid, 0.0 padding)
        text_mask: Maschera testo [B, T_text] (1.0 valid, 0.0 padding)
    """
    B, T1, T2 = alpha.shape
    dtype = alpha.dtype
    device = alpha.device

    # 1. Normalizzazione degli indici dei fonemi (0 -> 1)
    # Usiamo (T1 - 1) per mappare l'ultimo fonema a 1.0
    denom = (text_mask.sum(dim=-1) - 1).clamp(min=1.0).view(B, 1, 1)

    # Indici fonemi normalizzati: [B, T1, 1]
    p_idx = torch.arange(T1, device=device, dtype=dtype).view(1, T1, 1)
    p_idx_norm = p_idx / denom

    # 2. IMV normalizzato: [B, 1, T2]
    # L'imv è già stato scalato tra 0 e T1-1 nell'IMV_generator
    imv_norm = imv.unsqueeze(1) / denom

    # 3. Distanza quadratica: quanto l'attenzione è lontana dal "sentiero" dell'IMV
    # dist: [B, T1, T2]
    dist = (p_idx_norm - imv_norm) ** 2

    # 4. Pesatura con l'attenzione e mascheramento
    sma_map = alpha * dist

    # Maschera combinata: ignora padding sia di testo che di audio
    combined_mask = text_mask.unsqueeze(-1) * mel_mask.unsqueeze(1)
    sma_map = sma_map * combined_mask.to(dtype)

    # 5. Riduzione: Media dell'errore per frame audio reale
    loss_per_frame = sma_map.sum(dim=1)  # Somma sui fonemi
    total_loss = loss_per_frame.sum() / mel_mask.sum().clamp(min=1.0)

    return total_loss


def plot_durations_on_mel(
    mels,
    durations,
    mel_mask,
    text_length,
    batch_idx=0,
    step=0,
    labels=None,
    device_id=0,
):
    # mel_mask is True for padding. We want valid length.
    valid_len = (~mel_mask[batch_idx]).long().sum().item()

    mel = mels[batch_idx, :valid_len].detach().float().cpu().numpy().T
    dur = durations[batch_idx, : text_length[batch_idx]].detach().float().cpu().numpy()

    positions = dur.cumsum()
    fig, (ax_mel, ax_dur) = plt.subplots(2, 1, figsize=(16, 6))

    ax_mel.imshow(mel, origin="lower", aspect="auto")
    ax_mel.set_xlim(0, mel.shape[1])
    for pos in positions:
        ax_mel.axvline(pos, color="white", linestyle="--", linewidth=0.8, alpha=0.7)

    ax_mel.set_ylabel("Mel bin")
    ax_mel.set_title(f"Sample {batch_idx} - Step {step} - Device {device_id}")
    ax_mel.set_xticks([])

    norm_dur = (dur - dur.min()) / (dur.max() - dur.min() + 1e-8) * 0.7 + 0.3
    ax_dur.bar(
        range(len(dur)),
        dur,
        color=plt.cm.Blues(norm_dur),
        edgecolor="black",
        linewidth=0.5,
    )
    ax_dur.set_xlabel("z position")
    ax_dur.set_ylabel("Duration (frames)")
    ax_dur.set_xlim(-0.5, len(dur) - 0.5)
    ax_dur.set_xticks(range(len(dur)))
    ax_dur.set_xticklabels(
        labels[: len(dur)] if labels else range(len(dur)), rotation=0, ha="right"
    )

    plt.tight_layout()
    return fig


class PhonemesEmbeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(PhonemesEmbeddings, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, phonemes):
        """
        Args:
            phonemes: [Batch, T_phonemes]
        Returns:
            phoneme_embeddings: [Batch, T_phonemes, embedding_dim]
        """
        return self.embedding(phonemes)


class EfficientAlphaAttention(nn.Module):
    def __init__(self, audio_dim, text_dim, attn_dim):
        super(EfficientAlphaAttention, self).__init__()
        # QUERY = Audio (Frames)
        self.query_proj = nn.Linear(audio_dim, attn_dim)
        # KEY = Testo (Fonemi)
        self.key_proj = nn.Linear(text_dim, attn_dim)
        self.scale = 1.0 / math.sqrt(attn_dim)

    def forward(
        self,
        frames: torch.FloatTensor,
        phonemes_embeddings: torch.FloatTensor,
        audio_mask: torch.BoolTensor = None,
        text_mask: torch.BoolTensor = None,
    ):
        """
        Args:
            frames: [Batch, T_frames, audio_dim]
            phonemes_embeddings: [Batch, T_phonemes, embedding_dim]
            audio_mask: [Batch, T_frames]
            text_mask: [Batch, T_phonemes]

        Returns:
            alpha: [Batch, T_frames, T_phonemes]
            (Nota: le dimensioni sono invertite rispetto a prima: Frame x Fonemi)
        """
        B, T_frames, _ = frames.shape
        T_P = phonemes_embeddings.shape[1]

        # 1. Proiezioni (Q=Audio, K=Text)
        Q = self.query_proj(frames)  # [B, T_frames, attn_dim]
        K = self.key_proj(phonemes_embeddings)  # [B, T_phonemes, attn_dim]

        # 2. Score [B, T_frames, T_phonemes]
        # "Per ogni frame, quanto è simile a ogni fonema?"
        scores = torch.matmul(Q, K.transpose(1, 2)) * self.scale

        # 3. Maschera TESTO (Prima della Softmax)
        # Non vogliamo che l'audio punti a fonemi di padding
        if text_mask is not None:
            # text_mask: [B, T_P] -> unsqueeze -> [B, 1, T_P]
            scores = scores.masked_fill(text_mask.unsqueeze(1) == 0, -1e9)

        # 4. Softmax sui FONEMI (dim=-1)
        # Ogni frame deve scegliere una distribuzione di probabilità sui fonemi
        alpha = F.softmax(scores, dim=-1)  # [B, T_frames, T_phonemes]

        # 5. Maschera AUDIO (Dopo la Softmax)
        # Azzeriamo l'attenzione per i frame di padding (per pulizia)
        if audio_mask is not None:
            # audio_mask: [B, T_frames] -> unsqueeze -> [B, T_frames, 1]
            alpha = alpha * audio_mask.unsqueeze(2).to(alpha.dtype)

        return alpha


class IMV(nn.Module):
    """
    - gamma: [B, T2, T1]  (per-frame distribution over phonemes, softmax on T1)
    - durations: [B, T1]  (soft mass per phoneme = sum over frames)
    - aligned_mels: [B, T1, F] (frames "collassati" sui fonemi via pooling token->frame)
    """

    def __init__(
        self, delta: float = 10.0, gate_floor: float = 0.05, eps: float = 1e-8
    ):
        super().__init__()
        self.delta_base = float(delta)
        self.gate_floor = float(gate_floor)
        self.eps = float(eps)

    def generate_index_vector(self, text_mask, dtype=torch.float32):
        """
        p: [B, T1] = 0..T1-1 (NON moltiplicato per mask; la mask si applica nei logits)
        """
        T1 = text_mask.size(-1)
        p = torch.arange(0, T1, device=text_mask.device, dtype=dtype)
        return p.unsqueeze(0).expand(text_mask.size(0), -1)

    @staticmethod
    def _safe_renorm_lastdim(x, mask=None, eps=1e-8):
        """
        Renormalize over last dim so that sums are 1 where possible.
        If mask is provided, x is zeroed on masked-out positions before renorm.
        """
        if mask is not None:
            x = x * mask.to(dtype=x.dtype)
        s = x.sum(dim=-1, keepdim=True)
        return x / (s + eps)

    def compute_similarity_aware_gamma(
        self,
        z_audio,
        alpha,
        p,
        audio_mask,
        text_mask,
        delta_base=None,
        threshold=0.98,  # Soglia alta per essere molto selettivi
        steepness=30.0,
    ):
        audio_mask = audio_mask.to(torch.bool)
        text_mask = text_mask.to(torch.bool)
        B, T2, D = z_audio.shape
        T1 = p.size(-1)

        # ---- 1) Similarity Gate (Activity Detection) ----
        z_norm = F.normalize(z_audio, p=2, dim=-1)
        # Calcoliamo la similarità tra frame
        cosine_sim = torch.sum(z_norm[:, :-1] * z_norm[:, 1:], dim=-1)  # [B, T2-1]

        # ACTIVITY: 1 se il frame è DIVERSO dal precedente, 0 se è SIMILE
        # Se cosine_sim > threshold, activity -> 0 (Accorpiamo)
        activity = torch.sigmoid(steepness * (threshold - cosine_sim))
        activity = torch.cat(
            [torch.ones(B, 1, device=z_audio.device), activity], dim=-1
        )
        activity = activity * audio_mask.to(activity.dtype)

        # ---- 2) IMV modulato dall'attività ----
        # Proposta dell'attenzione: dove "vorrebbe" andare l'audio nel testo
        imv_proposal = torch.bmm(alpha, p.unsqueeze(-1)).squeeze(-1)

        delta_prop = torch.zeros_like(imv_proposal)
        delta_prop[:, 1:] = imv_proposal[:, 1:] - imv_proposal[:, :-1]

        # TRUCCO: L'avanzamento nel testo (delta) avviene SOLO se c'è attività acustica.
        # Se l'audio è simile (silenzio o fonema costante), eff_delta sarà quasi 0.
        # Questo costringe tutti i frame simili a puntare allo stesso indice p.
        eff_delta = torch.clamp(delta_prop * activity, min=0.0, max=1.0)

        imv = torch.cumsum(eff_delta, dim=-1) * audio_mask.to(eff_delta.dtype)

        # ---- 3) Normalizzazione e Gamma ----
        # (Mantieni la tua logica di normalizzazione per raggiungere l'ultimo fonema)
        last_idx = (audio_mask.sum(dim=-1).long() - 1).clamp(min=0)
        last_val = (
            torch.gather(imv, 1, last_idx.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
        )
        text_len = text_mask.sum(dim=-1).to(imv.dtype)
        scale = ((text_len - 1.0).clamp(min=0.0) / last_val).unsqueeze(-1)
        imv = imv * scale * audio_mask.to(imv.dtype)

        dist_sq = (imv.unsqueeze(-1) - p.unsqueeze(1)) ** 2
        logits = -float(delta_base if delta_base else self.delta_base) * dist_sq
        logits = logits.masked_fill(~text_mask.unsqueeze(1), -1e9)

        gamma = torch.softmax(logits, dim=-1)
        return gamma * audio_mask.unsqueeze(-1), activity, imv

    def forward(self, mels, alpha, mel_mask, text_mask):
        """
        mels:      [B, T2, F]
        alpha:     [B, T2, T1]
        mel_mask:  [B, T2]  (0/1 or bool)
        text_mask: [B, T1]  (0/1 or bool)
        """
        mel_mask_bool = mel_mask.to(torch.bool)
        text_mask_bool = text_mask.to(torch.bool)

        p = self.generate_index_vector(text_mask_bool, dtype=alpha.dtype)  # [B,T1]

        gamma, sim_gate, imv = self.compute_similarity_aware_gamma(
            z_audio=mels.to(alpha.dtype),
            alpha=alpha,
            p=p,
            audio_mask=mel_mask_bool,
            text_mask=text_mask_bool,
            threshold=0.9,
            steepness=10.0,
        )  # gamma: [B,T2,T1]

        # ---- durations (soft, in frames) ----
        durations = (gamma * mel_mask_bool.unsqueeze(-1).to(gamma.dtype)).sum(
            dim=1
        )  # [B,T1]
        durations = durations * text_mask_bool.to(durations.dtype)

        # ---- "collassare" frames -> fonemi: need token->frame weights (sum over T2 = 1 per token) ----
        # gamma is frame->token (sum over T1 = 1 per frame). Convert to token->frame by renorm on T2.
        w = gamma.permute(0, 2, 1)  # [B,T1,T2]
        w = w * mel_mask_bool.unsqueeze(1).to(w.dtype)
        w = w / (
            w.sum(dim=-1, keepdim=True) + self.eps
        )  # now sum over T2 = 1 per token (where possible)

        aligned_mels = torch.bmm(w, mels.to(w.dtype))  # [B,T1,F]
        aligned_mels = aligned_mels * text_mask_bool.unsqueeze(-1).to(
            aligned_mels.dtype
        )

        return aligned_mels, durations, imv


class PhonemeVocab:
    def __init__(self, path_to_vocab: str, parsing_mode: str = "phoneme"):
        self.vocab = self._load_vocab(path_to_vocab)
        self.parsing_mode = parsing_mode
        assert "<pad>" in self.vocab
        assert "<sil>" in self.vocab
        assert "<unk>" in self.vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _load_vocab(self, path_to_vocab: str):
        # load from json
        with open(path_to_vocab, "r") as f:
            vocab = json.load(f)
        return vocab

    def token2id(self, token: str):
        token_id = self.vocab.get(token, self.vocab["<unk>"])
        if token_id == self.vocab["<unk>"]:
            # print(f"Warning: token {token} not found in vocab")
            pass
        return token_id

    def _get_phonemes(self, phonemes: List[str]):
        """
        Convert transcripts to phoneme tokens.
        Returns: List[B] of List[phoneme]
        """
        phonemes_batch = []
        for phoneme_str in phonemes:
            phoneme_str = f"<sil> {phoneme_str} <sil>"

            if self.parsing_mode == "phoneme":
                tokens = phoneme_str.split()
            elif self.parsing_mode == "char":
                tokens = []
                for p in phoneme_str.split():
                    if p == "<sil>":
                        tokens.append(p)
                    else:
                        tokens.extend(list(p))
            else:
                raise ValueError(f"Unknown parsing mode: {self.parsing_mode}")

            phonemes_batch.append(
                torch.tensor([self.token2id(token) for token in tokens]).long()
            )
        return phonemes_batch

    def __call__(self, phonemes: List[str], device: torch.device):
        phoneme_ids = self._get_phonemes(phonemes)
        phoneme_mask = [torch.ones(len(ph)) for ph in phoneme_ids]
        # phoneme ids padded
        phoneme_ids = (
            torch.nn.utils.rnn.pad_sequence(
                phoneme_ids, batch_first=True, padding_value=self.vocab["<pad>"]
            )
            .to(device)
            .long()
        )
        # phoneme mask --> 1 means valid token, 0 means padding
        phoneme_mask = (
            torch.nn.utils.rnn.pad_sequence(
                phoneme_mask, batch_first=True, padding_value=0
            )
            .to(device)
            .bool()
        )
        return phoneme_ids, phoneme_mask


def compute_sma_loss_many_to_one(alpha, imv, mel_mask, text_mask):
    """
    Args:
        alpha: [B, T_frames, T_phonemes] - Attenzione many-to-one (softmax sui fonemi)
        imv: [B, T_frames] - Traiettoria monotona (indice fonema atteso per frame)
        mel_mask: [B, T_frames] (1 valid, 0 padding)
        text_mask: [B, T_phonemes] (1 valid, 0 padding)
    """
    B, T_f, T_p = alpha.shape
    device = alpha.device

    # 1. Normalizziamo gli indici dei fonemi tra 0 e 1
    # Usiamo text_mask per sapere quanti fonemi reali ci sono
    text_lens = text_mask.sum(dim=-1).clamp(min=2.0)
    # p_idx: [B, 1, T_p]
    p_idx = torch.arange(T_p, device=device).view(1, 1, T_p).float()
    p_idx_norm = p_idx / (text_lens.view(B, 1, 1) - 1)

    # 2. Normalizziamo l'IMV tra 0 e 1
    # imv_norm: [B, T_f, 1]
    imv_norm = imv.unsqueeze(-1) / (text_lens.view(B, 1, 1) - 1)

    # 3. Calcolo della distanza quadratica
    # Per ogni frame (T_f), quanto ogni fonema (T_p) è lontano dall'IMV?
    # dist: [B, T_f, T_p]
    dist = (p_idx_norm - imv_norm) ** 2

    # 4. Pesatura con l'attenzione
    # Se alpha[t, p] è alto ma dist[t, p] è grande, la loss sale.
    sma_map = alpha * dist

    # 5. Mascheramento
    # Ignoriamo i frame di padding e i fonemi di padding
    combined_mask = mel_mask.unsqueeze(-1) * text_mask.unsqueeze(1)
    sma_map = sma_map * combined_mask.float()

    # 6. Riduzione
    # Sommiamo l'errore per ogni frame e facciamo la media sui frame reali
    loss_per_frame = sma_map.sum(dim=-1)  # Somma sui fonemi
    total_loss = loss_per_frame.sum() / mel_mask.sum().clamp(min=1.0)

    return total_loss


class Aligner(nn.Module):
    def __init__(
        self,
        attn_dim: int,
        text_dim: int,
        audio_dim: int,
        num_embeddings: int,
        delta: float = 1.0,
        vocab_path: str = "data/vocab.json",
        parsing_mode: str = "phoneme",
    ):
        super().__init__()
        self.attention = EfficientAlphaAttention(audio_dim, text_dim, attn_dim)
        self.imv = IMV(20.0)
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.phonemes_embeddings = PhonemesEmbeddings(num_embeddings, text_dim)
        self.phoneme_vocab = PhonemeVocab(vocab_path, parsing_mode=parsing_mode)
        self.layernorm_mel = nn.LayerNorm(audio_dim)
        self.final_layer = nn.Linear(text_dim + audio_dim, audio_dim)

    def generate_pos_encoding(
        self, max_len: int, d_model: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        position = torch.arange(max_len).unsqueeze(1).to(device)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        ).to(device)

        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.to(device=device, dtype=dtype)

    def forward(
        self,
        mels: torch.FloatTensor,
        phonemes: List[str],
        mels_mask: torch.BoolTensor,
    ):
        # get phonemes
        phonemes_ids, phonemes_mask = self.phoneme_vocab(phonemes, device=mels.device)
        phonemes_embeddings = self.phonemes_embeddings(phonemes_ids)
        # add position encoding
        phonemes_attention = phonemes_embeddings + self.generate_pos_encoding(
            max_len=phonemes_ids.size(1),
            d_model=self.text_dim,
            device=mels.device,
            dtype=mels.dtype,
        )
        mels = self.layernorm_mel(mels)
        mels_attention = mels + self.generate_pos_encoding(
            max_len=mels.size(1),
            d_model=self.audio_dim,
            device=mels.device,
            dtype=mels.dtype,
        )
        # get scores
        alpha = self.attention(
            frames=mels_attention,
            phonemes_embeddings=phonemes_attention,
            audio_mask=mels_mask,
            text_mask=phonemes_mask,
        )
        # get aligned mels and durations
        aligned_mels, durations, imv = self.imv(
            alpha=alpha, mels=mels, mel_mask=mels_mask, text_mask=phonemes_mask
        )
        print(durations[0, :20])
        min_mass = 1.0  # in "frame", soft
        cov_loss = torch.relu(min_mass - durations) * phonemes_mask.to(durations.dtype)
        cov_loss = cov_loss.sum() / phonemes_mask.sum().clamp(min=1)

        # compute guided attention loss
        align_loss = None
        align_loss = (
            compute_sma_loss_many_to_one(
                alpha=alpha,
                imv=imv,
                mel_mask=mels_mask,
                text_mask=phonemes_mask,
            )
            * 0.5
        ) + cov_loss * 0.5

        final = torch.cat([phonemes_attention, aligned_mels], dim=-1)
        final = self.final_layer(final)
        return final, durations, ~phonemes_mask, align_loss, imv
