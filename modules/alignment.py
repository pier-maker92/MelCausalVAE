"""
AlignmentMatrixBuilder — Converts phoneme-level timestamp alignments
(start/end in seconds) into binary mel-spectrogram alignment matrices.

Uses MelSpectrogramConfig for the exact time-to-frame mapping:
    frames_per_second = sampling_rate / hop_length   (e.g. 24000/256 = 93.75 Hz)

Alignment strategy:
  - Stage 1: map each phoneme interval to frames with ±1 frame tolerance.
  - Stage 2: any remaining unassigned frames (discontinuity gaps) are labelled
             with the special GAP_TOKEN so they appear as their own segment
             in the alignment matrix.

Includes a dynamic phoneme vocabulary and an nn.Embedding layer so that
segment labels can be converted to dense vectors.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from modules.melspecEncoder import MelSpectrogramConfig

# Special token used for frames that fall in alignment gaps.
GAP_TOKEN = "<GAP>"
# Padding token for label-ID tensors.
PAD_TOKEN = "<PAD>"


# ------------------------------------------------------------------
# Output dataclass
# ------------------------------------------------------------------

@dataclass
class AlignmentOutput:
    """Structured output of :py:meth:`AlignmentMatrixBuilder.build`."""

    alignments: torch.Tensor
    """``(B, max_T, max_N)`` binary alignment matrices (float, 0/1)."""

    phoneme_mask: torch.BoolTensor
    """``(B, max_N)`` — ``False`` = valid, ``True`` = padding."""

    durations: torch.Tensor
    """``(B, max_N)`` per-segment frame counts (long)."""

    segment_labels: List[List[str]]
    """Per-utterance list of segment phoneme labels (including GAP_TOKEN)."""

    label_ids: torch.Tensor
    """``(B, max_N)`` integer IDs from the dynamic vocabulary (long).
    Padding positions are filled with the PAD_TOKEN id."""

    embeddings: torch.Tensor
    """``(B, max_N, embedding_dim)`` dense phoneme embeddings."""


# ------------------------------------------------------------------
# Dynamic phoneme vocabulary
# ------------------------------------------------------------------

class PhonemeVocabulary:
    """
    Auto-growing label → integer mapping.

    Pre-reserves slot 0 for ``PAD_TOKEN`` and slot 1 for ``GAP_TOKEN``.
    New labels are assigned the next available id on first encounter.
    """

    def __init__(self, max_size: int = 120):
        self.max_size = max_size
        self._label2id: Dict[str, int] = {PAD_TOKEN: 0, GAP_TOKEN: 1}
        self._id2label: Dict[int, str] = {0: PAD_TOKEN, 1: GAP_TOKEN}

    # --- public ---

    @property
    def pad_id(self) -> int:
        return 0

    def __len__(self) -> int:
        return len(self._label2id)

    def encode(self, label: str) -> int:
        """Return the id for *label*, registering it if unseen."""
        if label in self._label2id:
            return self._label2id[label]
        new_id = len(self._label2id)
        if new_id >= self.max_size:
            raise RuntimeError(
                f"PhonemeVocabulary is full ({self.max_size}). "
                f"Cannot register new label '{label}'."
            )
        self._label2id[label] = new_id
        self._id2label[new_id] = label
        return new_id

    def encode_batch(self, labels: List[str]) -> List[int]:
        return [self.encode(l) for l in labels]

    def decode(self, idx: int) -> str:
        return self._id2label[idx]

    def label2id(self) -> Dict[str, int]:
        return dict(self._label2id)


# ------------------------------------------------------------------
# Main builder (nn.Module because it owns an Embedding)
# ------------------------------------------------------------------

class AlignmentMatrixBuilder(nn.Module):
    """
    Build binary monotonic alignment matrices from phoneme timestamps,
    convert segment labels to integer IDs via a dynamic vocabulary,
    and produce dense phoneme embeddings.

    Args:
        mel_config: MelSpectrogramConfig used to derive the exact frame rate.
        num_embeddings: vocabulary capacity (default 120).
        embedding_dim: size of each embedding vector (default 64).
    """

    _GAP_SENTINEL = None  # internal sentinel, never leaks to outputs

    def __init__(
        self,
        mel_config: Optional[MelSpectrogramConfig] = None,
        num_embeddings: int = 120,
        embedding_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        if mel_config is None:
            mel_config = MelSpectrogramConfig()
        self.mel_config = mel_config

        # Derived constants
        self.frames_per_second = mel_config.sampling_rate / mel_config.hop_length
        self.ms_per_frame = 1000.0 / self.frames_per_second

        # Vocabulary & embedding
        self.vocab = PhonemeVocabulary(max_size=num_embeddings)
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=self.vocab.pad_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ceil(x: float) -> int:
        return math.ceil(x)

    # ------------------------------------------------------------------
    # Label → IDs → Embeddings
    # ------------------------------------------------------------------

    def labels_to_ids(
        self,
        segment_labels: List[List[str]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert per-utterance segment labels to a padded ``(B, max_N)``
        integer tensor using the dynamic vocabulary.
        """
        B = len(segment_labels)
        max_N = max((len(sl) for sl in segment_labels), default=0)
        pad_id = self.vocab.pad_id

        ids = torch.full((B, max_N), pad_id, dtype=torch.long, device=device)
        for i, sl in enumerate(segment_labels):
            encoded = self.vocab.encode_batch(sl)
            ids[i, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        return ids

    def get_embeddings(
        self,
        label_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Look up dense embeddings for a ``(B, N)`` tensor of label IDs.

        Returns:
            ``(B, N, embedding_dim)``
        """
        return self.embedding(label_ids)

    # ------------------------------------------------------------------
    # Stage 1: frame-wise label assignment
    # ------------------------------------------------------------------

    def _fill_framewise(
        self,
        phoneme_alignment: List[dict],
        total_frames: int,
    ) -> list:
        """
        Assign a phoneme label to each mel frame.

        Returns:
            framewise_label: list[str] of length *total_frames*.
                             Unassigned frames are marked with ``GAP_TOKEN``.
        """
        gap = self._GAP_SENTINEL
        ms_per_frame = self.ms_per_frame

        # Convert seconds → ms and sort by onset
        ts_items = sorted(
            [
                {
                    "phoneme": p["phoneme"],
                    "start_ms": p["start"] * 1000.0,
                    "end_ms": p["end"] * 1000.0,
                }
                for p in phoneme_alignment
            ],
            key=lambda x: x["start_ms"],
        )

        framewise_label: list = [gap] * total_frames

        # ---- Stage 1: exact boundaries with ±1 frame tolerance ----
        for ts in ts_items:
            start_idx = max(int(ts["start_ms"] / ms_per_frame), 0)
            end_idx = min(self._ceil(ts["end_ms"] / ms_per_frame), total_frames)

            if end_idx - start_idx > total_frames:
                continue  # degenerate entry — skip
            end_idx = min(end_idx, total_frames)

            # Fill the exact range, expanding ±1 only into gaps
            for frame_i in range(max(start_idx - 1, 0), min(end_idx + 1, total_frames)):
                if framewise_label[frame_i] is gap:
                    framewise_label[frame_i] = ts["phoneme"]

        # ---- Stage 2: assign remaining gaps to GAP_TOKEN ----
        for i in range(total_frames):
            if framewise_label[i] is gap:
                framewise_label[i] = GAP_TOKEN

        return framewise_label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        all_phoneme_alignments: List[List[dict]],
        total_frames_list: List[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> AlignmentOutput:
        """
        Build binary alignment matrices for a batch of utterances,
        padded to uniform size.

        Returns an :class:`AlignmentOutput` containing alignments,
        phoneme_mask, durations, segment_labels, label_ids, and embeddings.
        """
        per_utt: List[torch.Tensor] = []
        all_segment_labels: List[List[str]] = []

        for phoneme_alignment, T in zip(all_phoneme_alignments, total_frames_list):
            T = int(T)
            if T == 0 or len(phoneme_alignment) == 0:
                per_utt.append(torch.empty((0, 0), dtype=torch.float32))
                all_segment_labels.append([])
                continue

            # Stage 1 + gap assignment
            framewise_label = self._fill_framewise(phoneme_alignment, T)

            # Build segments from contiguous runs of the same label
            segments: list = []  # (label, start_frame, end_frame)
            current = framewise_label[0]
            start = 0
            for i in range(1, T):
                if framewise_label[i] != current:
                    segments.append((current, start, i))
                    current = framewise_label[i]
                    start = i
            segments.append((current, start, T))

            # Construct the binary alignment matrix
            N = len(segments)
            alignment = torch.zeros((T, N), dtype=torch.float32)
            for col, (_label, s, e) in enumerate(segments):
                alignment[s:e, col] = 1.0

            per_utt.append(alignment)
            all_segment_labels.append([label for label, _s, _e in segments])

        # ---- Pad to (B, max_T, max_N) ----
        B = len(per_utt)
        max_T = max((a.shape[0] for a in per_utt), default=0)
        max_N = max((a.shape[1] for a in per_utt), default=0)

        padded = torch.zeros(B, max_T, max_N, dtype=torch.float32)
        phoneme_lengths = torch.zeros(B, dtype=torch.long)

        for i, a in enumerate(per_utt):
            if a.numel() > 0:
                padded[i, : a.shape[0], : a.shape[1]] = a
                phoneme_lengths[i] = a.shape[1]

        # Phoneme mask: False = valid, True = padding
        phoneme_mask = torch.ones(B, max_N, dtype=torch.bool)
        for i, length in enumerate(phoneme_lengths):
            phoneme_mask[i, :length] = False

        # Durations: per-segment frame counts
        durations = padded.sum(dim=1)

        # Label IDs & embeddings
        label_ids = self.labels_to_ids(all_segment_labels, device=device)
        embeddings = self.get_embeddings(label_ids)

        return AlignmentOutput(
            alignments=padded.to(device=device, dtype=dtype),
            phoneme_mask=phoneme_mask.to(device=device, dtype=torch.bool),
            durations=durations.to(device=device, dtype=torch.long),
            segment_labels=all_segment_labels,
            label_ids=label_ids,
            embeddings=embeddings,
        )

    def build_single(
        self,
        phoneme_alignment: List[dict],
        total_frames: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> AlignmentOutput:
        """Convenience wrapper for a single utterance (unbatched)."""
        return self.build([phoneme_alignment], [total_frames], device=device, dtype=dtype)
