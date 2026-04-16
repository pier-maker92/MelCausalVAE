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
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
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
        if max_N == 0:
            return torch.zeros(B, 0, dtype=torch.long, device=device)
        pad_id = self.vocab.pad_id
        ids = np.full((B, max_N), pad_id, dtype=np.int64)
        for i, sl in enumerate(segment_labels):
            if sl:
                ids[i, : len(sl)] = self.vocab.encode_batch(sl)
        return torch.from_numpy(ids).to(device=device)

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
    ) -> np.ndarray:
        """
        Fast CPU assignment using numpy slice ops.

        Loops only over P phonemes (small, ~50-200); all frame-level work
        is numpy C-level slicing — no Python loop over T frames.

        Returns:
            ``(total_frames,)`` int64 ndarray.  Gaps are ``-1``.
        """
        P = len(phoneme_alignment)
        T = total_frames
        ms = self.ms_per_frame

        starts_ms = [p["start"] * 1000.0 for p in phoneme_alignment]
        ends_ms = [p["end"] * 1000.0 for p in phoneme_alignment]
        centers_ms = [(p["start"] + p["end"]) * 500.0 for p in phoneme_alignment]
        order = sorted(range(P), key=lambda i: starts_ms[i])

        fw = np.full(T, -1, dtype=np.int64)

        # Pass 1: first-writer-wins with ±1 frame tolerance
        for i in order:
            s = max(int(starts_ms[i] / ms) - 1, 0)
            e = min(math.ceil(ends_ms[i] / ms) + 1, T)
            if s >= e:
                continue
            sl = fw[s:e]
            gap = sl == -1
            if gap.any():
                sl[gap] = i

        # Pass 2: guarantee ≥ 1 frame per phoneme
        present = np.bincount(fw[fw >= 0], minlength=P) > 0
        for i in order:
            if present[i]:
                continue
            cf = max(0, min(int(centers_ms[i] / ms), T - 1))
            fw[cf] = i

        return fw

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

        Loops only over B (batch, small) and P (phonemes, small).
        All frame-level work uses numpy C-level ops.
        """
        B = len(all_phoneme_alignments)
        per_utt_np: List[Optional[np.ndarray]] = []
        N_list: List[int] = []
        all_segment_labels: List[List[str]] = []

        for phoneme_alignment, T_raw in zip(all_phoneme_alignments, total_frames_list):
            T = int(T_raw)
            if T == 0 or not phoneme_alignment:
                per_utt_np.append(None)
                N_list.append(0)
                all_segment_labels.append([])
                continue

            fw = self._fill_framewise(phoneme_alignment, T)

            # Segment detection (numpy, no loop over T)
            change = np.empty(T, dtype=np.bool_)
            change[0] = True
            np.not_equal(fw[1:], fw[:-1], out=change[1:])

            seg_ids = np.cumsum(change, dtype=np.int64) - 1
            N = int(seg_ids[-1]) + 1
            N_list.append(N)

            # Segment labels: gather phoneme index at each segment start
            starts_idx = np.flatnonzero(change)
            phon_at_start = fw[starts_idx]
            labels = [
                phoneme_alignment[i]["phoneme"] if i >= 0 else GAP_TOKEN
                for i in phon_at_start.tolist()
            ]
            all_segment_labels.append(labels)

            # Alignment matrix (numpy advanced indexing, no loop)
            alignment = np.zeros((T, N), dtype=np.float32)
            alignment[np.arange(T, dtype=np.int64), seg_ids] = 1.0
            per_utt_np.append(alignment)

        # --- batch padding ---
        max_T = max(total_frames_list) if total_frames_list else 0
        max_N = max(N_list) if N_list else 0

        padded_np = np.zeros((B, max_T, max_N), dtype=np.float32)
        phoneme_lengths = np.array(N_list, dtype=np.int64)

        for b, a in enumerate(per_utt_np):
            if a is not None:
                padded_np[b, : a.shape[0], : a.shape[1]] = a

        padded = torch.from_numpy(padded_np)

        # Phoneme mask (numpy broadcast, no loop)
        if max_N == 0:
            phoneme_mask = torch.ones(B, 0, dtype=torch.bool)
        else:
            phoneme_mask = torch.from_numpy(
                np.arange(max_N)[None, :] >= phoneme_lengths[:, None]
            )

        durations = padded.sum(dim=1)

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

    Loops only over W words (small); frame-level work is numpy slicing.

    Returns:
        ``(T,)`` LongTensor.  Words ``0..W-1``; inter-word gaps get
        unique IDs ``≥ W``.
    """
    if total_frames == 0:
        return torch.zeros(0, dtype=torch.long)

    T = total_frames
    W = len(word_alignment)
    ms = 1000.0 / frames_per_second

    fw = np.full(T, -1, dtype=np.int64)

    # Pass 1: first-writer-wins with ±1 tolerance (loop over W, not T)
    for i, w in enumerate(word_alignment):
        s = max(int(w["start"] * 1000.0 / ms) - 1, 0)
        e = min(math.ceil(w["end"] * 1000.0 / ms) + 1, T)
        if s >= e:
            continue
        sl = fw[s:e]
        gap = sl == -1
        if gap.any():
            sl[gap] = i

    # Pass 2: guarantee ≥ 1 frame per word
    present = np.bincount(fw[fw >= 0], minlength=W) > 0
    for i, w in enumerate(word_alignment):
        if present[i]:
            continue
        cf = max(0, min(int((w["start"] + w["end"]) * 500.0 / ms), T - 1))
        fw[cf] = i

    # Gap IDs: each contiguous gap run gets a unique ID (numpy, no loop)
    is_gap = fw < 0
    if not is_gap.any():
        return torch.from_numpy(fw)

    gap_start = is_gap.copy()
    gap_start[1:] &= ~is_gap[:-1]
    gap_run_id = np.cumsum(gap_start)
    fw[is_gap] = W + gap_run_id[is_gap] - 1

    return torch.from_numpy(fw)


def build_word_causal_attention_mask(
    word_ids: torch.LongTensor,
) -> torch.BoolTensor:
    """``(T, T)`` mask: ``mask[i, j] = word_ids[j] <= word_ids[i]``."""
    return word_ids.unsqueeze(0) <= word_ids.unsqueeze(1)


def build_word_causal_attention_mask_batch(
    word_ids_list: List[torch.LongTensor],
    padding_mask: torch.BoolTensor,
    device: torch.device,
) -> torch.BoolTensor:
    """
    Batched causal mask via pad + broadcast.

    Args:
        word_ids_list: per-utterance word-ID tensors (variable length).
        padding_mask: ``(B, T_max)`` — ``True`` = valid, ``False`` = padding.
        device: target device.

    Returns:
        ``(B, T_max, T_max)`` bool mask.
    """
    from torch.nn.utils.rnn import pad_sequence

    T_max = padding_mask.shape[1]
    _INF = torch.iinfo(torch.long).max

    padded = pad_sequence(
        [w.to(device) for w in word_ids_list],
        batch_first=True,
        padding_value=_INF,
    )
    if padded.shape[1] < T_max:
        padded = torch.nn.functional.pad(
            padded, (0, T_max - padded.shape[1]), value=_INF
        )
    else:
        padded = padded[:, :T_max]

    masks = padded.unsqueeze(1) <= padded.unsqueeze(2)
    masks = masks & padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)
    return masks