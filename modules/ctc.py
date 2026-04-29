import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from phonemizer import phonemize
from phonemizer.separator import Separator


class PhonemeAligner(nn.Module):
    """
    Attention-based phoneme alignment module.
    Uses cross-attention between phoneme embeddings and audio features to find
    frame-level phoneme boundaries.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 4,
        max_phonemes: int = 250,
        phoneme_embed_dim: int = 128,
    ):
        super(PhonemeAligner, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_phonemes = max_phonemes
        self.phoneme_embed_dim = phoneme_embed_dim

        # Audio feature encoder
        self.audio_encoder = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
        )

        # Phoneme embedding layer
        self.phoneme_embedding = nn.Embedding(max_phonemes, phoneme_embed_dim)

        # Phoneme encoder
        self.phoneme_encoder = nn.Sequential(
            nn.Linear(phoneme_embed_dim, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size), nn.Dropout(0.1)
        )

        # Cross-attention: phonemes (query) attend to audio frames (key, value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=False,  # Expects [seq_len, batch, embed_dim]
        )

        # Positional encoding for audio frames
        self.register_buffer("pos_encoding", self._generate_pos_encoding(5000, hidden_size))

        # Dynamic vocabulary for phoneme-to-index mapping (blank=0, unk=1)
        self.vocab = {"<blank>": 0, "<unk>": 1}
        self._unk_warnings = set()

    def _generate_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def _update_vocab(self, phoneme: str) -> int:
        """Add a new phoneme to vocabulary and return its index."""
        if phoneme not in self.vocab:
            new_idx = len(self.vocab)
            if new_idx >= self.max_phonemes:
                if phoneme not in self._unk_warnings:
                    print(
                        f"Warning: Vocabulary full (size={self.max_phonemes}). "
                        f"Mapping phoneme '{phoneme}' to <unk> token."
                    )
                    self._unk_warnings.add(phoneme)
                return self.vocab["<unk>"]
            self.vocab[phoneme] = new_idx
        return self.vocab[phoneme]

    def get_vocab_size(self) -> int:
        """Return current vocabulary size."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Return copy of current vocabulary."""
        return self.vocab.copy()

    def set_vocab(self, vocab: Dict[str, int]):
        """Set vocabulary (useful for loading pretrained models)."""
        self.vocab = vocab.copy()
        if len(self.vocab) > self.max_phonemes:
            raise ValueError(
                f"Provided vocabulary size ({len(self.vocab)}) exceeds max_phonemes ({self.max_phonemes})"
            )
        if "<unk>" not in self.vocab:
            print("Warning: Loaded vocabulary missing <unk> token. Adding it at index 1.")
            self.vocab["<unk>"] = 1
        self._unk_warnings = set()

    def get_phonemes(self, transcript: List[str], language: str = "en-us") -> List[List[str]]:
        """
        Convert transcripts to phoneme tokens.
        Returns: List[B] of List[phoneme]
        """
        phonemes_batch = []
        sep = Separator(phone=" ", word=" | ", syllable=None)

        for text in transcript:
            if not text or text.strip() == "":
                phonemes_batch.append([])
                continue

            try:
                phoneme_str = phonemize(
                    text,
                    language=language,
                    backend="espeak",
                    separator=sep,
                    strip=True,
                    preserve_punctuation=False,
                    njobs=1,
                )
            except RuntimeError as e:
                if "failed to find espeak" in str(e).lower() or "espeak" in str(e).lower():
                    msg = (
                        "Phonemizer non trova la libreria espeak. "
                        "Soluzione: export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
                    )
                    raise RuntimeError(msg) from e
                raise

            tokens = phoneme_str.split()
            tokens = [t for t in tokens if t != "|"]
            phonemes_batch.append(tokens)

        return phonemes_batch

    def phonemes_to_indices(self, phonemes: List[List[str]]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Convert phoneme tokens to indices."""
        max_len = max(len(p) for p in phonemes) if phonemes else 0
        batch_size = len(phonemes)

        indices_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, phoneme_list in enumerate(phonemes):
            for j, phoneme in enumerate(phoneme_list):
                idx = self._update_vocab(phoneme)
                indices_batch[i, j] = idx
            lengths[i] = len(phoneme_list)

        return indices_batch, lengths

    def extract_boundaries_from_attention(
        self,
        attention_weights: torch.Tensor,
        phoneme_lengths: torch.Tensor,
        audio_lengths: torch.Tensor,
        phonemes: List[List[str]],
    ) -> Tuple[List[List[Tuple[int, int, str]]], List[List[int]]]:
        """
        Extract frame boundaries from attention weights using peak detection.

        Args:
            attention_weights: [B, num_phonemes, T] attention weights
            phoneme_lengths: [B] actual phoneme sequence lengths
            audio_lengths: [B] actual audio frame lengths
            phonemes: List[B] of phoneme lists

        Returns:
            boundaries_batch: [(start_frame, end_frame, phoneme), ...]
            durations_batch: [duration1, duration2, ...]
        """
        batch_size = attention_weights.size(0)
        boundaries_batch = []
        durations_batch = []

        for b in range(batch_size):
            num_phonemes = phoneme_lengths[b].item()
            num_frames = audio_lengths[b].item()

            if num_phonemes == 0:
                boundaries_batch.append([])
                durations_batch.append([])
                continue

            # Get attention weights for this sample [num_phonemes, T]
            attn = attention_weights[b, :num_phonemes, :num_frames]

            boundaries = []
            durations = []

            # For each phoneme, find the frame range where attention is highest
            for p in range(num_phonemes):
                phoneme_attn = attn[p]  # [T]

                # Find weighted center of attention
                frame_indices = torch.arange(num_frames, device=phoneme_attn.device, dtype=phoneme_attn.dtype)
                center = (phoneme_attn * frame_indices).sum() / (phoneme_attn.sum() + 1e-8)

                # Find frames with significant attention (> threshold)
                threshold = phoneme_attn.max() * 0.1
                significant_frames = (phoneme_attn > threshold).nonzero(as_tuple=True)[0]

                if len(significant_frames) > 0:
                    start_frame = significant_frames[0].item()
                    end_frame = significant_frames[-1].item() + 1  # +1 for exclusive end
                else:
                    # Fallback: use center ± 1
                    start_frame = max(0, int(center.item()) - 1)
                    end_frame = min(num_frames, int(center.item()) + 2)

                duration = end_frame - start_frame
                phoneme_text = phonemes[b][p]

                boundaries.append((start_frame, end_frame, phoneme_text))
                durations.append(duration)

            boundaries_batch.append(boundaries)
            durations_batch.append(durations)

        return boundaries_batch, durations_batch

    def compute_alignment_loss(
        self, attention_weights: torch.Tensor, phoneme_lengths: torch.Tensor, audio_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alignment loss to encourage monotonic and complete alignments.

        Args:
            attention_weights: [B, max_phonemes, T]
            phoneme_lengths: [B]
            audio_lengths: [B]
        """
        batch_size = attention_weights.size(0)

        # Monotonicity loss: encourage attention to move forward
        monotonic_loss = 0.0

        # Coverage loss: ensure all frames are attended to
        coverage_loss = 0.0

        for b in range(batch_size):
            num_phonemes = phoneme_lengths[b].item()
            num_frames = audio_lengths[b].item()

            if num_phonemes == 0:
                continue

            attn = attention_weights[b, :num_phonemes, :num_frames]  # [P, T]

            # Monotonicity: penalize backward attention
            # Compute center of mass for each phoneme
            frame_indices = torch.arange(num_frames, device=attn.device, dtype=attn.dtype)
            centers = (attn * frame_indices.unsqueeze(0)).sum(dim=1) / (attn.sum(dim=1) + 1e-8)

            # Penalize if center moves backward
            if num_phonemes > 1:
                center_diffs = centers[1:] - centers[:-1]
                monotonic_loss += F.relu(-center_diffs).sum()

            # Coverage: sum of attention over phonemes should be ~1 for each frame
            coverage = attn.sum(dim=0)  # [T]
            coverage_loss += ((coverage - 1.0) ** 2).mean()

        monotonic_loss = monotonic_loss / batch_size
        coverage_loss = coverage_loss / batch_size

        return monotonic_loss + coverage_loss

    def forward(
        self,
        z: torch.FloatTensor,
        transcript: List[str],
        input_lengths: torch.LongTensor = None,
    ):
        """
        Args:
            z: [B, T, D] audio features (mel spectrogram latents)
            transcript: [B] list of transcriptions
            input_lengths: [B] actual lengths of sequences (excluding padding)

        Returns:
            loss: alignment loss
            boundaries: List of frame boundaries per batch item
            durations: List of durations per batch item
            attention_weights: [B, max_phonemes, T] attention weights
        """
        batch_size, seq_len, _ = z.shape

        # If input_lengths not provided, assume no padding
        if input_lengths is None:
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=z.device)

        # Encode audio features
        audio_features = self.audio_encoder(z)  # [B, T, H]

        # Add positional encoding to audio features
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).to(z.device)  # [1, T, H]
        audio_features = audio_features + pos_enc

        # Get phonemes and convert to indices
        phonemes = self.get_phonemes(transcript)
        phoneme_indices, phoneme_lengths = self.phonemes_to_indices(phonemes)
        phoneme_indices = phoneme_indices.to(z.device)
        phoneme_lengths = phoneme_lengths.to(z.device)

        # Embed and encode phonemes
        phoneme_embeds = self.phoneme_embedding(phoneme_indices)  # [B, P, E]
        phoneme_features = self.phoneme_encoder(phoneme_embeds)  # [B, P, H]

        # Prepare for cross-attention: [seq_len, batch, embed_dim]
        # Query: phonemes, Key/Value: audio
        query = phoneme_features.permute(1, 0, 2)  # [P, B, H]
        key = value = audio_features.permute(1, 0, 2)  # [T, B, H]

        # Create attention masks
        # Key padding mask: [B, T] - True where padding
        key_padding_mask = torch.arange(seq_len, device=z.device).unsqueeze(0) >= input_lengths.unsqueeze(1)

        # Query padding mask: [B, P] - True where padding
        max_phoneme_len = phoneme_indices.size(1)
        query_padding_mask = torch.arange(max_phoneme_len, device=z.device).unsqueeze(0) >= phoneme_lengths.unsqueeze(
            1
        )

        # Cross-attention
        attn_output, attention_weights = self.cross_attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # Get attention per head
        )

        # Average attention over heads: [P, B, T] -> [B, P, T]
        attention_weights = attention_weights.mean(dim=1).permute(1, 0, 2)

        # Compute alignment loss
        loss = self.compute_alignment_loss(attention_weights, phoneme_lengths, input_lengths)

        # Extract boundaries and durations
        with torch.no_grad():
            boundaries, durations = self.extract_boundaries_from_attention(
                attention_weights, phoneme_lengths, input_lengths, phonemes
            )

        return loss, boundaries, durations, attention_weights
