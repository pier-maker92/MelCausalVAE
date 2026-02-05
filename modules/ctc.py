import os
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from phonemizer import phonemize
from phonemizer.separator import Separator

# Se espeak non viene trovato in un passo successivo (es. evaluation), di solito è perché
# il processo figlio non eredita LD_LIBRARY_PATH. Imposta le variabili d'ambiente prima
# di avviare Python, es.: export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH


class CTC(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 250):
        super(CTC, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size),
        )
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        # Dynamic vocabulary for phoneme-to-index mapping (blank=0, unk=1)
        self.vocab = {"<blank>": 0, "<unk>": 1}
        self.output_size = output_size
        self._unk_warnings = set()  # Track which phonemes have been warned about

    def _update_vocab(self, phoneme: str) -> int:
        """Add a new phoneme to vocabulary and return its index. Returns <unk> if vocab is full."""
        if phoneme not in self.vocab:
            new_idx = len(self.vocab)
            if new_idx >= self.output_size:
                # Vocabulary is full, map to <unk> token
                if phoneme not in self._unk_warnings:
                    print(
                        f"Warning: Vocabulary full (size={self.output_size}). "
                        f"Mapping phoneme '{phoneme}' to <unk> token. "
                        f"Consider increasing output_size in CTC initialization."
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
        if len(self.vocab) > self.output_size:
            raise ValueError(f"Provided vocabulary size ({len(self.vocab)}) exceeds output_size ({self.output_size})")
        # Ensure <unk> token exists
        if "<unk>" not in self.vocab:
            print("Warning: Loaded vocabulary missing <unk> token. Adding it at index 1.")
            self.vocab["<unk>"] = 1
        self._unk_warnings = set()  # Reset warnings when loading vocab

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
                        "Phonemizer non trova la libreria espeak. Succede spesso quando "
                        "l'evaluation o un processo figlio non eredita LD_LIBRARY_PATH. "
                        "Soluzione: avvia il training con la lib in path, es.\n"
                        "  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH\n"
                        "  python train.py ...\n"
                        "Oppure installa: apt install espeak-ng libespeak-ng-dev"
                    )
                    raise RuntimeError(msg) from e
                raise

            tokens = phoneme_str.split()
            tokens = [t for t in tokens if t != "|"]

            phonemes_batch.append(tokens)

        return phonemes_batch

    def phonemes_to_indices(self, phonemes: List[List[str]]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Convert phoneme tokens to CTC targets.
        """
        targets = []
        target_lengths = []

        for phoneme_list in phonemes:
            indices = []

            for phoneme in phoneme_list:
                idx = self._update_vocab(phoneme)
                indices.append(idx)

            targets.extend(indices)
            target_lengths.append(len(indices))

        return torch.LongTensor(targets), torch.LongTensor(target_lengths)

    def get_frame_boundaries(
        self, log_probs: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[List[List[Tuple[int, int, str]]], List[List[int]]]:
        """
        Decode CTC output and extract frame boundaries for each phoneme.

        Args:
            log_probs: [T, B, C] log probabilities from CTC
            input_lengths: [B] actual lengths of each sequence

        Returns:
            boundaries_batch: List of boundaries per batch item: [(start_frame, end_frame, phoneme), ...]
            durations_batch: List of durations per batch item: [duration1, duration2, ...]
        """
        batch_size = log_probs.size(1)
        boundaries_batch = []
        durations_batch = []

        # Convert vocab to index->phoneme mapping
        idx_to_phoneme = {v: k for k, v in self.vocab.items()}

        for b in range(batch_size):
            # Get predictions for this batch item
            probs = log_probs[: input_lengths[b], b, :]  # [T, C]
            predictions = torch.argmax(probs, dim=-1)  # [T]

            boundaries = []
            durations = []
            prev_token = None
            start_frame = 0

            for t, token_idx in enumerate(predictions):
                token_idx = token_idx.item()

                # Skip blank tokens (0)
                if token_idx == 0:
                    if prev_token is not None:
                        # End of a phoneme segment
                        phoneme = idx_to_phoneme.get(prev_token, "<unk>")
                        if phoneme != "<blank>":
                            duration = t - start_frame
                            boundaries.append((start_frame, t, phoneme))
                            durations.append(duration)
                        prev_token = None
                    continue

                # New phoneme or continuation
                if token_idx != prev_token:
                    if prev_token is not None:
                        # Save previous phoneme boundary
                        phoneme = idx_to_phoneme.get(prev_token, "<unk>")
                        if phoneme != "<blank>":
                            duration = t - start_frame
                            boundaries.append((start_frame, t, phoneme))
                            durations.append(duration)
                    start_frame = t
                    prev_token = token_idx

            # Handle last phoneme if sequence doesn't end with blank
            if prev_token is not None:
                phoneme = idx_to_phoneme.get(prev_token, "<unk>")
                if phoneme != "<blank>":
                    end_frame = input_lengths[b].item()
                    duration = end_frame - start_frame
                    boundaries.append((start_frame, end_frame, phoneme))
                    durations.append(duration)

            boundaries_batch.append(boundaries)
            durations_batch.append(durations)

        return boundaries_batch, durations_batch

    def forward(
        self,
        z: torch.FloatTensor,
        transcript: List[str],
        input_lengths: torch.LongTensor = None,
    ):
        """
        Args:
            z:
                [B, T, D] latent features, B is the batch size, T is the length of the sequence, D is the dimension of the latent features.
                consider that z contains also padding values
            transcript:
                [B] list of transcriptions, B is the batch size
            input_lengths:
                [B] actual lengths of sequences (excluding padding)

        Returns:
            loss: CTC loss value
            boundaries: List of frame boundaries per batch item: [(start, end, phoneme), ...]
            durations: List of durations per batch item: [duration1, duration2, ...]
            log_probs: [T, B, C] log probabilities for potential further use
        """
        batch_size, seq_len, _ = z.shape

        # If input_lengths not provided, assume no padding
        if input_lengths is None:
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=z.device)

        # Forward pass through layers
        x = self.net(z)

        # CTC expects [T, B, C] format
        log_probs = torch.log_softmax(x, dim=-1).permute(1, 0, 2)  # [T, B, C]

        # Get phonemes and convert to indices
        phonemes = self.get_phonemes(transcript)
        targets, target_lengths = self.phonemes_to_indices(phonemes)
        targets = targets.to(z.device)
        target_lengths = target_lengths.to(z.device)

        # Compute CTC loss (PyTorch CTC does not support bfloat16 on CUDA, use float32)
        log_probs_f32 = log_probs.float()
        loss = self.ctc_loss(log_probs_f32, targets, input_lengths, target_lengths)
        loss = loss.to(log_probs.dtype)

        # Extract frame boundaries and durations
        with torch.no_grad():
            boundaries, durations = self.get_frame_boundaries(log_probs, input_lengths)

        return loss, boundaries, durations, log_probs
