import json
import math
import torch
import torch.nn as nn
from typing import List
import matplotlib.pyplot as plt
import torch.nn.functional as F


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
        self.query_proj = nn.Linear(audio_dim, attn_dim)
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
            audio_mask: [Batch, T_frames] - 1 per i frame reali, 0 per padding
            text_mask: [Batch, T_phonemes] - 1 per i fonemi reali, 0 per padding

        Returns:
            alpha: [Batch, T_phonemes, T_frames]
        """
        B, T_frames, _ = frames.shape
        T_P = phonemes_embeddings.shape[1]

        # 1. Proiezioni
        Q = self.query_proj(frames)  # [B, T_frames, attn_dim]
        K = self.key_proj(phonemes_embeddings)  # [B, T_P, attn_dim]

        # 2. Score [B, T_frames, T_P]
        scores = torch.matmul(Q, K.transpose(1, 2)) * self.scale

        # 3. Applicazione della Text Mask (prima della Softmax)
        # Impedisce a ogni frame audio di assegnare peso ai fonemi di padding
        if text_mask is not None:
            # Espandiamo la maschera per il broadcasting: [B, 1, T_P]
            t_mask = text_mask.unsqueeze(1)
            scores = scores.masked_fill(t_mask == 0, -1e9)

        # Softmax sulla dimensione dei fonemi
        alpha_prime = F.softmax(scores, dim=-1)  # [B, T_frames, T_P]

        # 4. Applicazione della Audio Mask (dopo la Softmax)
        # Impedisce ai frame audio di padding di avere pesi validi
        if audio_mask is not None:
            # Espandiamo la maschera per il broadcasting: [B, T_frames, 1]
            a_mask = audio_mask.unsqueeze(2)
            alpha_prime = alpha_prime * a_mask

        # alpha shape: [Batch, T_phonemes, T_frames]
        alpha = alpha_prime.transpose(1, 2)

        return alpha


class IMV(torch.nn.Module):
    def __init__(
        self,
        sigma: float = 0.5,
        delta: float = 0.1,
    ):
        super().__init__()
        self.sigma = sigma
        self.delta = delta

    def generate_index_vector(self, text_mask, dtype=torch.float32):
        """Create index vector of text sequence.
        Args:
            text_mask: text_mask: mask of text-sequence. [B,T1]
        returns:
            index vector of text sequence. [B,T1]
        """
        p = (
            torch.arange(0, text_mask.size(-1))
            .repeat(text_mask.size(0), 1)
            .to(device=text_mask.device, dtype=dtype)
        )
        return p * text_mask.to(dtype)

    def imv_generator(self, alpha, p, mel_mask, text_length):
        """Compute imv from alignment matrix alpha. Implementation of HMA
        Args:
            alpha: scaled dot attention [B,T1,T2]
            p: index vector, output of generate_index_vector. [B,T1]
            mel_mask: mask of mel-spectrogram [B, T2]
            text_length: lengths of input text-sequence [B]
        returns:
            Index mapping vector (IMV) [B,T2]
        """
        imv_dummy = torch.bmm(alpha.transpose(1, 2), p.unsqueeze(-1)).squeeze(
            -1
        )  # [B,T2]
        delta_imv = torch.relu(imv_dummy[:, 1:] - imv_dummy[:, :-1])  # [B, T2-1]
        delta_imv = torch.cat(
            (torch.zeros(alpha.size(0), 1).type_as(alpha), delta_imv), -1
        )  # [B, T2-1] -> [B, T2]
        imv = torch.cumsum(delta_imv, -1) * mel_mask.to(dtype=alpha.dtype)
        last_imv = torch.max(imv, dim=-1)[0]  # get last element of imv
        last_imv = torch.clamp(last_imv, min=1e-8).unsqueeze(-1)  # avoid zeros
        imv = (
            imv / last_imv * (text_length.to(dtype=alpha.dtype).unsqueeze(-1))
        )  # multiply imv by a positive scalar to enforce 0<imv<T1-1
        return imv

    def get_aligned_positions(self, imv, p, mel_mask, text_mask):
        """compute aligned positions from imv
        Args:
            imv: index mapping vector #[B,T2]
            p: index vector, output of generate_index_vector. [B,T1]
            sigma: a scalar, default 0.5
            mel_mask: mask of mel-spectrogram [B, T2]
        returns:
            Aligned positions [B,T1]
        """
        energies = -1 * ((imv.unsqueeze(1) - p.unsqueeze(-1)) ** 2) * self.sigma
        energies = energies.masked_fill(
            ~(mel_mask.unsqueeze(1).repeat(1, energies.size(1), 1)), -1e9
        )
        beta = torch.softmax(energies, dim=-1)
        q = (
            torch.arange(0, mel_mask.size(-1))
            .unsqueeze(0)
            .repeat(imv.size(0), 1)
            .to(device=imv.device, dtype=imv.dtype)
        )
        q = q * mel_mask.to(dtype=imv.dtype)  # generate index vector of target sequence.
        return torch.bmm(beta, q.unsqueeze(-1)) * text_mask.unsqueeze(-1).to(dtype=imv.dtype)

    def reconstruct_align_from_aligned_positions(
        self, e, mel_mask=None, text_mask=None
    ):
        """reconstruct alignment matrix from aligned positions
        Args:
            e: aligned positions [B,T1]
            delta: a scalar, default 0.1
            mel_mask: mask of mel-spectrogram [B, T2], None if inference or B==1
            text_mask: mask of text-sequence, None if B==1
        returns:
            alignment matrix [B,T1,T2]
        """
        if mel_mask is None:  # inference phase:
            max_length = torch.round(e[:, -1] + (e[:, -1] - e[:, -2]))
        else:
            max_length = mel_mask.size(-1)
        q = (
            torch.arange(0, max_length)
            .unsqueeze(0)
            .repeat(e.size(0), 1)
            .to(device=e.device, dtype=e.dtype)
        )
        if mel_mask is not None:
            q = q * mel_mask.to(dtype=e.dtype)
        energies = -1 * self.delta * (q.unsqueeze(1) - e.unsqueeze(-1)) ** 2
        if text_mask is not None:
            energies = energies.masked_fill(
                ~(text_mask.unsqueeze(-1).repeat(1, 1, max_length)), -1e9
            )
        return torch.softmax(energies, dim=-1)

    def algin_frames(self, energies, mels):
        """Align frames to text
        Args:
            energies: energies [B,T1,T2]
            mels: mels [B,T2,F]
        Returns:
            aligned_mels: aligned mels [B,T1,F]
        """
        aligned_mels = torch.bmm(energies, mels)
        return aligned_mels

    def extract_durations(self, alignement):
        """Align frames to text
        Args:
            alignement: alignement [B,T1]
        Returns:
            durations: durations [B,T1]
        """
        alignement = torch.cat(
            (torch.zeros(alignement.size(0), 1).type_as(alignement), alignement), -1
        )
        delta = alignement.diff(dim=-1)
        delta = torch.relu(delta)
        return (
            delta.cumsum(dim=-1)
            .round()
            .diff(dim=-1, prepend=torch.zeros(delta.size(0), 1).type_as(delta))
        ).long()

    def forward(
        self,
        mels: torch.FloatTensor,
        alpha: torch.FloatTensor,
        mel_mask: torch.FloatTensor,
        text_mask: torch.FloatTensor,
    ):
        p = self.generate_index_vector(text_mask=text_mask, dtype=alpha.dtype)
        text_length = text_mask.sum(dim=1)
        imv = self.imv_generator(
            alpha=alpha, p=p, mel_mask=mel_mask, text_length=text_length
        )
        alignement = self.get_aligned_positions(
            imv=imv, p=p, mel_mask=mel_mask, text_mask=text_mask
        ).squeeze(-1)
        energies = self.reconstruct_align_from_aligned_positions(
            e=alignement, mel_mask=mel_mask, text_mask=text_mask
        )
        aligned_mels = self.algin_frames(energies=energies, mels=mels)
        durations = self.extract_durations(alignement=alignement)

        return aligned_mels, durations


class PhonemeVocab:
    def __init__(self, path_to_vocab: str, parsing_mode: str = "phoneme"):
        self.vocab = self._load_vocab(path_to_vocab)
        self.parsing_mode = parsing_mode
        assert "<pad>" in self.vocab
        assert "<sil>" in self.vocab
        assert "<unk>" in self.vocab

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


class Aligner(nn.Module):
    def __init__(
        self,
        attn_dim: int,
        text_dim: int,
        audio_dim: int,
        embedding_dim: int,
        num_embeddings: int,
        sigma: float = 0.5,
        delta: float = 0.1,
        vocab_path: str = "data/vocab.json",
        parsing_mode: str = "phoneme",
    ):
        super().__init__()
        self.attention = EfficientAlphaAttention(audio_dim, text_dim, attn_dim)
        self.imv = IMV(sigma, delta)
        self.audio_dim = audio_dim
        self.embedding_dim = embedding_dim
        self.phonemes_embeddings = PhonemesEmbeddings(num_embeddings, embedding_dim)
        self.phoneme_vocab = PhonemeVocab(vocab_path, parsing_mode=parsing_mode)

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
        phonemes_embeddings += self.generate_pos_encoding(
            max_len=phonemes_ids.size(1),
            d_model=self.embedding_dim,
            device=mels.device,
            dtype=mels.dtype,
        )
        mels += self.generate_pos_encoding(
            max_len=mels.size(1),
            d_model=self.audio_dim,
            device=mels.device,
            dtype=mels.dtype,
        )
        # get scores
        alpha = self.attention(
            frames=mels,
            phonemes_embeddings=phonemes_embeddings,
            audio_mask=mels_mask,
            text_mask=phonemes_mask,
        )
        # get aligned mels and durations
        aligned_mels, durations = self.imv(
            alpha=alpha, mels=mels, mel_mask=mels_mask, text_mask=phonemes_mask
        )
        return aligned_mels, durations, ~phonemes_mask
