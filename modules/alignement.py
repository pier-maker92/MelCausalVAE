"""
AlignmentMatrixBuilder — Converts phoneme-level timestamp alignments
(start/end in seconds) into binary mel-spectrogram alignment matrices.

Uses MelSpectrogramConfig for the exact time-to-frame mapping:
    frames_per_second = sampling_rate / hop_length   (e.g. 24000/256 = 93.75 Hz)

Optional temporal compression (compress_factor):
    When the mel spectrogram is passed through an encoder that downsamples
    the time axis (e.g. a CausalVAE with stride 2 or 4), the effective frame
    rate seen by downstream modules is:

        effective_fps = frames_per_second / compress_factor

    Setting ``compress_factor=2`` (2X) or ``compress_factor=4`` (4X) makes the
    builder map phoneme timestamps directly into the *compressed* frame space,
    so ``total_frames`` passed to ``build()`` should already be the compressed
    length (T // compress_factor).

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
        mel_config: MelSpectrogramConfig used to derive the base frame rate.
        compress_factor: Temporal downsampling factor applied by the encoder
            *after* the mel spectrogram is computed (default ``1`` = no
            compression).  Supported values: any positive integer, typically
            ``1``, ``2``, or ``4`` (1X / 2X / 4X).  The effective frame rate
            used when mapping phoneme timestamps to frame indices is::

                effective_fps = (sampling_rate / hop_length) / compress_factor

            ``total_frames`` passed to :py:meth:`build` / :py:meth:`build_single`
            must already be the *compressed* frame count
            (i.e. ``mel_T // compress_factor``).
        num_embeddings: vocabulary capacity (default 120).
        embedding_dim: size of each embedding vector (default 64).
    """

    _GAP_SENTINEL = None  # internal sentinel, never leaks to outputs

    def __init__(
        self,
        mel_config: Optional[MelSpectrogramConfig] = None,
        compress_factor: int = 1,
        num_embeddings: int = 120,
        embedding_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        if mel_config is None:
            mel_config = MelSpectrogramConfig()
        if compress_factor < 1 or not isinstance(compress_factor, int):
            raise ValueError(
                f"compress_factor must be a positive integer, got {compress_factor!r}."
            )
        self.mel_config = mel_config
        self.compress_factor = compress_factor

        # Base frame rate derived from the mel config
        _base_fps = mel_config.sampling_rate / mel_config.hop_length

        # Effective frame rate after temporal compression
        # e.g. compress_factor=2 → frames_per_second is halved
        self.frames_per_second: float = _base_fps / compress_factor
        self.ms_per_frame: float = 1000.0 / self.frames_per_second

        # Vocabulary & embedding
        self.vocab = PhonemeVocabulary(max_size=num_embeddings)
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=self.vocab.pad_id,
        )

    # ------------------------------------------------------------------
    # Frame-rate introspection
    # ------------------------------------------------------------------

    @property
    def effective_frame_rate_info(self) -> dict:
        """
        Return a dict summarising the frame-rate parameters in use.

        Example (sampling_rate=24000, hop_length=256, compress_factor=4)::

            {
                'base_fps':        93.75,   # mel frames per second
                'compress_factor': 4,
                'effective_fps':   23.4375, # compressed frames per second
                'ms_per_frame':    42.667,  # ms per compressed frame
            }
        """
        base_fps = self.mel_config.sampling_rate / self.mel_config.hop_length
        return {
            "base_fps": base_fps,
            "compress_factor": self.compress_factor,
            "effective_fps": self.frames_per_second,
            "ms_per_frame": self.ms_per_frame,
        }

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
    ) -> List[int]:
        """
        Assign an index (pointing to phoneme_alignment) to each mel frame.

        Returns:
            framewise_idx: list[int] of length *total_frames*.
                           Values are indices into *phoneme_alignment*.
                           Gaps are marked with -1.
        """
        ms_per_frame = self.ms_per_frame

        # Convert seconds → ms
        ts_items = [
            {
                "index": i,
                "start_ms": p["start"] * 1000.0,
                "end_ms": p["end"] * 1000.0,
                "center_ms": (p["start"] + p["end"]) * 500.0,
            }
            for i, p in enumerate(phoneme_alignment)
        ]
        # Sort by onset to handle overlapping timestamps gracefully
        ts_items.sort(key=lambda x: x["start_ms"])

        # -1 represents a gap
        framewise_idx: List[int] = [-1] * total_frames

        # ---- Pass 1: standard assignment with ±1 frame tolerance ----
        for ts in ts_items:
            start_idx = max(int(ts["start_ms"] / ms_per_frame), 0)
            end_idx = min(self._ceil(ts["end_ms"] / ms_per_frame), total_frames)

            # Stage 1: fill range into gaps (±1 frame tolerance)
            for frame_i in range(max(start_idx - 1, 0), min(end_idx + 1, total_frames)):
                if framewise_idx[frame_i] == -1:
                    framewise_idx[frame_i] = ts["index"]

        # ---- Pass 2: 1-frame minimum guarantee ----
        # If any phoneme index is missing, force it into its closest frame
        assigned_indices = set(framewise_idx)
        for ts in ts_items:
            if ts["index"] not in assigned_indices:
                # Target the frame closest to the phoneme's temporal center
                center_frame = int(ts["center_ms"] / ms_per_frame)
                center_frame = max(0, min(center_frame, total_frames - 1))
                framewise_idx[center_frame] = ts["index"]

        return framewise_idx

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
        Build binary alignment matrices for a batch of utterances.
        Every input phoneme is guaranteed to have at least one frame.
        """
        per_utt: List[torch.Tensor] = []
        all_segment_labels: List[List[str]] = []

        for phoneme_alignment, T in zip(all_phoneme_alignments, total_frames_list):
            T = int(T)
            if T == 0 or len(phoneme_alignment) == 0:
                per_utt.append(torch.empty((0, 0), dtype=torch.float32))
                all_segment_labels.append([])
                continue

            # Get frame-to-index mapping (index into phoneme_alignment or -1)
            framewise_idx = self._fill_framewise(phoneme_alignment, T)

            # Build segments from contiguous runs of the same index.
            # This prevents merging of identical adjacent phonemes because
            # they have different indices.
            segments: List[Tuple[str, int, int]] = []
            curr_idx = framewise_idx[0]
            start = 0
            for i in range(1, T):
                if framewise_idx[i] != curr_idx:
                    label = (
                        phoneme_alignment[curr_idx]["phoneme"]
                        if curr_idx != -1
                        else GAP_TOKEN
                    )
                    segments.append((label, start, i))
                    curr_idx = framewise_idx[i]
                    start = i
            # Final segment
            label = (
                phoneme_alignment[curr_idx]["phoneme"]
                if curr_idx != -1
                else GAP_TOKEN
            )
            segments.append((label, start, T))

            # Construct the binary alignment matrix (T, N)
            N = len(segments)
            alignment = torch.zeros((T, N), dtype=torch.float32)
            segment_labels = []
            for col, (label, s, e) in enumerate(segments):
                alignment[s:e, col] = 1.0
                segment_labels.append(label)

            per_utt.append(alignment)
            all_segment_labels.append(segment_labels)

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
        return self.build(
            [phoneme_alignment], [total_frames], device=device, dtype=dtype
        )


# ------------------------------------------------------------------
# Word-level alignment utilities
# ------------------------------------------------------------------


def build_word_frame_alignment(
    word_alignment: List[dict],
    total_frames: int,
    frames_per_second: float,
) -> torch.LongTensor:
    """
    Map word-level timestamps to per-frame word IDs.

    Args:
        word_alignment: List of dicts with ``start`` (seconds), ``end`` (seconds),
            and ``word`` keys, in temporal order.
        total_frames: Number of (possibly compressed) frames.
        frames_per_second: Effective frame rate (base FPS / compress_factor).

    Returns:
        ``(T,)`` LongTensor of word indices.  Words are numbered ``0 .. W-1``
        in order; inter-word gaps receive unique IDs that continue the count
        so that each gap is its own "word" and never merges with neighbours.
    """
    if total_frames == 0:
        return torch.zeros(0, dtype=torch.long)

    ms_per_frame = 1000.0 / frames_per_second

    # -1 = unassigned / gap
    frame_word: List[int] = [-1] * total_frames

    for word_idx, w in enumerate(word_alignment):
        start_ms = w["start"] * 1000.0
        end_ms = w["end"] * 1000.0

        start_frame = max(int(start_ms / ms_per_frame), 0)
        end_frame = min(math.ceil(end_ms / ms_per_frame), total_frames)

        # Fill with ±1 tolerance so every word gets at least 1 frame
        for f in range(max(start_frame - 1, 0), min(end_frame + 1, total_frames)):
            if frame_word[f] == -1:
                frame_word[f] = word_idx

    # Guarantee: every word index is assigned at least one frame
    assigned = set(frame_word)
    for word_idx, w in enumerate(word_alignment):
        if word_idx not in assigned:
            center_ms = (w["start"] + w["end"]) * 500.0
            center_frame = int(center_ms / ms_per_frame)
            center_frame = max(0, min(center_frame, total_frames - 1))
            frame_word[center_frame] = word_idx

    # Convert to monotonic word IDs.
    # Gaps (-1) get unique IDs so they never merge with adjacent segments.
    num_words = len(word_alignment)
    gap_counter = num_words  # IDs for gap frames start after real words
    word_ids: List[int] = []
    prev_raw = None
    for raw in frame_word:
        if raw == -1:
            # Each contiguous gap run gets one unique ID
            if prev_raw != -1 or len(word_ids) == 0:
                word_ids.append(gap_counter)
                gap_counter += 1
            else:
                word_ids.append(word_ids[-1])
        else:
            word_ids.append(raw)
        prev_raw = raw

    return torch.tensor(word_ids, dtype=torch.long)


def build_word_causal_attention_mask(
    word_ids: torch.LongTensor,
) -> torch.BoolTensor:
    """
    Build a ``(T, T)`` attention mask from per-frame word IDs.

    Semantics of ``mask[i, j] = True``: frame *i* **can** attend to frame *j*.

    - Frames within the same word attend **bidirectionally**.
    - Frames can attend to all frames of **past** words.
    - Frames **cannot** attend to frames of **future** words.

    This reduces to ``word_ids[j] <= word_ids[i]``.
    """
    # word_ids: (T,)
    # Compare every pair: result[i, j] = (word_ids[j] <= word_ids[i])
    return word_ids.unsqueeze(0) <= word_ids.unsqueeze(1)  # (T, T)


def build_word_causal_attention_mask_batch(
    word_ids_list: List[torch.LongTensor],
    padding_mask: torch.BoolTensor,
    device: torch.device,
) -> torch.BoolTensor:
    """
    Batched version of :func:`build_word_causal_attention_mask`.

    Args:
        word_ids_list: Per-utterance word-ID tensors (variable length).
        padding_mask: ``(B, T_max)`` with ``True`` = **valid**, ``False`` = padding.
        device: Target device.

    Returns:
        ``(B, T_max, T_max)`` bool mask.  Padded positions are ``False``.
    """
    B = len(word_ids_list)
    T_max = padding_mask.shape[1]

    masks = torch.zeros(B, T_max, T_max, dtype=torch.bool, device=device)
    for i, wids in enumerate(word_ids_list):
        T = wids.shape[0]
        m = build_word_causal_attention_mask(wids.to(device))  # (T, T)
        masks[i, :T, :T] = m

    # Zero out padding rows/cols
    masks = masks & padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)
    return masks
