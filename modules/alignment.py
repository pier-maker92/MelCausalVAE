"""
AlignmentMatrixBuilder — Converts phoneme-level timestamp alignments
(start/end in seconds) into binary mel-spectrogram alignment matrices.

Uses MelSpectrogramConfig for the exact time-to-frame mapping:
    frames_per_second = sampling_rate / hop_length   (e.g. 24000/256 = 93.75 Hz)

Gap-filling follows a contraction strategy (adapted from framewise_assortment):
  - Stage 1: map each phoneme interval to frames with ±1 frame tolerance.
  - Stage 2: contract remaining gaps from both sides using neighbouring labels,
             controlled by `gap_contraction` (in frames).
  - Stage 3: any still-uncovered frames are flood-filled from the nearest
             assigned neighbour so that every row of the matrix has exactly one 1.
"""

import math
from typing import List, Optional, Tuple

import torch

from modules.melspecEncoder import MelSpectrogramConfig


class AlignmentMatrixBuilder:
    """
    Build binary monotonic alignment matrices from phoneme timestamps.

    Args:
        mel_config: MelSpectrogramConfig used to derive the exact frame rate.
        gap_contraction: number of frames to absorb from each side of a gap
                         during Stage-2 contraction (default 5).
    """

    # Sentinel that must not collide with any valid phoneme label.
    _GAP_SENTINEL = None

    def __init__(
        self,
        mel_config: Optional[MelSpectrogramConfig] = None,
        gap_contraction: int = 5,
    ):
        if mel_config is None:
            mel_config = MelSpectrogramConfig()
        self.mel_config = mel_config
        self.gap_contraction = gap_contraction

        # Derived constants
        self.frames_per_second = mel_config.sampling_rate / mel_config.hop_length
        self.ms_per_frame = 1000.0 / self.frames_per_second

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ceil(x: float) -> int:
        return math.ceil(x)

    # ------------------------------------------------------------------
    # Stage 1 + 2: frame-wise label assignment with gap contraction
    # ------------------------------------------------------------------

    def _fill_framewise(
        self,
        phoneme_alignment: List[dict],
        total_frames: int,
    ) -> list:
        """
        Assign a phoneme label to each mel frame.

        Args:
            phoneme_alignment: list of dicts, each containing:
                - 'start': float — onset time in **seconds**
                - 'end':   float — offset time in **seconds**
                - 'phoneme': str  — phoneme label
            total_frames: number of mel frames for the utterance.

        Returns:
            framewise_label: list[str | None] of length *total_frames*.
                             None marks frames still unassigned after contraction.
        """
        gap = self._GAP_SENTINEL
        ms_per_frame = self.ms_per_frame
        gap_contraction = self.gap_contraction

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

        # ---- Stage 2: contract silent gaps ----
        i = 0
        gaps: list = []
        while i < total_frames:
            if framewise_label[i] is gap:
                gap_start = i
                while i < total_frames and framewise_label[i] is gap:
                    i += 1
                gaps.append((gap_start, i))
            else:
                i += 1

        for gap_start, gap_end in gaps:
            left_value = framewise_label[gap_start - 1] if gap_start > 0 else gap
            right_value = framewise_label[gap_end] if gap_end < total_frames else gap
            gap_size = gap_end - gap_start

            if left_value is gap and right_value is gap:
                continue  # both sides unknown — cannot contract

            # If only one side is known, treat the other as that side
            if left_value is gap:
                left_value = right_value
            if right_value is gap:
                right_value = left_value

            if gap_size <= gap_contraction:
                # Small gap: fill entirely with left neighbour
                for j in range(gap_start, gap_end):
                    framewise_label[j] = left_value
            elif gap_size <= gap_contraction * 2:
                # Medium gap: split at midpoint
                midpoint = gap_start + gap_size // 2
                for j in range(gap_start, gap_end):
                    framewise_label[j] = left_value if j < midpoint else right_value
            else:
                # Large gap: absorb gap_contraction frames from each side
                for j in range(gap_start, min(gap_start + gap_contraction, gap_end)):
                    framewise_label[j] = left_value
                for j in range(
                    gap_end - 1,
                    max(gap_end - gap_contraction - 1, gap_start - 1),
                    -1,
                ):
                    framewise_label[j] = right_value

        return framewise_label

    # ------------------------------------------------------------------
    # Stage 3: ensure full coverage (flood-fill remaining gaps)
    # ------------------------------------------------------------------

    @staticmethod
    def _flood_fill_gaps(framewise_label: list) -> list:
        """
        Fill any remaining gap sentinels by propagating the nearest
        assigned label (forward then backward), guaranteeing every
        frame is covered.
        """
        gap = AlignmentMatrixBuilder._GAP_SENTINEL
        T = len(framewise_label)
        labels = list(framewise_label)  # copy

        # Forward fill
        for i in range(1, T):
            if labels[i] is gap and labels[i - 1] is not gap:
                labels[i] = labels[i - 1]
        # Backward fill (handles leading gaps)
        for i in range(T - 2, -1, -1):
            if labels[i] is gap and labels[i + 1] is not gap:
                labels[i] = labels[i + 1]

        return labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        all_phoneme_alignments: List[List[dict]],
        total_frames_list: List[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.Tensor]:
        """
        Build binary alignment matrices for a batch of utterances,
        padded to uniform size.

        Args:
            all_phoneme_alignments: batch of phoneme alignments.
                Each element is a list of dicts:
                    {'start': float (s), 'end': float (s), 'phoneme': str}
            total_frames_list: number of mel frames per utterance.

        Returns:
            alignments: ``torch.Tensor``  ``(B, max_T, max_N)``, float32, binary.
                        Zero-padded to the maximum T and N in the batch.
            phoneme_mask: ``torch.BoolTensor`` ``(B, max_N)``.
                          ``False`` = valid phoneme column,
                          ``True``  = padding column.
            durations: ``torch.Tensor`` ``(B, max_N)``, float32.
                       Per-segment frame counts (sum of each alignment column).
        """
        per_utt: List[torch.Tensor] = []

        for phoneme_alignment, T in zip(all_phoneme_alignments, total_frames_list):
            T = int(T)
            if T == 0 or len(phoneme_alignment) == 0:
                per_utt.append(torch.empty((0, 0), dtype=torch.float32))
                continue

            # Stages 1+2: timestamp → framewise labels with gap contraction
            framewise_label = self._fill_framewise(phoneme_alignment, T)

            # Stage 3: guarantee full coverage
            framewise_label = self._flood_fill_gaps(framewise_label)

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

        # Durations: per-segment frame counts  (B, max_N)
        durations = padded.sum(dim=1)

        return (
            padded.to(device=device, dtype=dtype),
            phoneme_mask.to(device=device, dtype=torch.bool),
            durations.to(device=device, dtype=torch.long),
        )

    def build_single(
        self,
        phoneme_alignment: List[dict],
        total_frames: int,
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.Tensor]:
        """Convenience wrapper for a single utterance (unbatched)."""
        return self.build([phoneme_alignment], [total_frames])
