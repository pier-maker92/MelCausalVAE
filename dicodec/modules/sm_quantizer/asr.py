"""BiLSTM + CTC from quantized features, matching the audiocodecs ASR architecture.

Reference: downstream/train_asr.py and downstream/hparams/tasks/asr.yaml in
https://github.com/lucadellalib/audiocodecs (use_quantized_feats path).
Unlike the frozen-codec downstream recipe, gradients flow into the tokenizer here.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .configs import ASRConfig
from .output_dataclasses import ASROutput


class TextTokenizer:
    def __init__(self, config: ASRConfig):
        self.config = config
        self.symbols = {char: index for index, char in enumerate(config.characters)}
        self.sentencepiece = None
        if config.tokenizer == "sentencepiece":
            import sentencepiece as spm

            self.sentencepiece = spm.SentencePieceProcessor(model_file=config.tokenizer_model)
            if self.sentencepiece.get_piece_size() != config.vocab_size:
                raise ValueError("ASR vocab_size does not match the SentencePiece model.")

    def encode(self, text: str) -> torch.Tensor:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("ASR requires a nonempty transcript for every utterance.")
        if self.sentencepiece is not None:
            ids = self.sentencepiece.encode(text, out_type=int)
        else:
            text = " ".join(text.lower().split())
            unknown = set(text) - self.symbols.keys()
            if unknown:
                raise ValueError(f"Characters absent from ASR alphabet: {sorted(unknown)}")
            ids = [self.symbols[char] for char in text]
        if not ids:
            raise ValueError("ASR transcript produced no tokens.")
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: list[int]) -> str:
        if self.sentencepiece is not None:
            return self.sentencepiece.decode(ids)
        return "".join(self.config.characters[index] for index in ids)


class ASRHead(nn.Module):
    def __init__(self, code_dim: int, config: ASRConfig):
        super().__init__()
        self.config = config
        self.blank_id = config.num_tokens
        self.input_projection = nn.Linear(code_dim, config.embedding_dim)
        self.encoder = nn.LSTM(config.embedding_dim, config.hidden_size, config.layers,
                               batch_first=True, bidirectional=True,
                               dropout=config.dropout if config.layers > 1 else 0)
        self.head = nn.Linear(2 * config.hidden_size, config.num_tokens + 1)
        self._wer_tokenizer = None

    def validate_targets(self, targets, target_lengths, input_lengths):
        if targets is None or target_lengths is None:
            raise ValueError("ASR is enabled: text_targets and text_lengths are required.")
        if targets.dtype != torch.long or target_lengths.dtype != torch.long:
            raise ValueError("CTC targets and lengths must be int64.")
        if targets.ndim != 2 or target_lengths.shape != input_lengths.shape or targets.shape[0] != len(input_lengths):
            raise ValueError("Expected ASR targets [B, L] and lengths [B].")
        if (target_lengths < 1).any() or (target_lengths > targets.shape[1]).any():
            raise ValueError("Invalid ASR target lengths.")
        for row, length, frames in zip(targets, target_lengths.tolist(), input_lengths.tolist()):
            text = row[:length]
            if ((text < 0) | (text >= self.blank_id)).any():
                raise ValueError("ASR targets contain out-of-vocabulary IDs or CTC blank.")
            minimum = length + int((text[1:] == text[:-1]).sum())
            if frames < minimum:
                raise ValueError("CTC alignment impossible: need more frames (including repeated labels). "
                                 "Use shorter BPE targets or increase asr.upsample_factor.")

    def forward(self, codes, valid, targets, target_lengths) -> ASROutput:
        input_lengths = valid.sum(1) * self.config.upsample_factor
        self.validate_targets(targets, target_lengths, input_lengths)
        hidden = self.input_projection(codes).repeat_interleave(self.config.upsample_factor, dim=1)
        packed = pack_padded_sequence(hidden, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(packed, batch_first=True, total_length=hidden.shape[1])
        logits = self.head(hidden)
        loss = F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), targets,
                          input_lengths.cpu(), target_lengths.cpu(), blank=self.blank_id,
                          reduction="mean", zero_infinity=False)
        return ASROutput(loss, logits, input_lengths)

    @torch.no_grad()
    def decode(self, logits, lengths) -> list[list[int]]:
        return [[token for token in row[:length].argmax(-1).unique_consecutive().tolist()
                 if token != self.blank_id] for row, length in zip(logits, lengths.tolist())]

    @torch.no_grad()
    def word_error_counts(self, logits, input_lengths, targets, target_lengths) -> tuple[int, int]:
        from .wer import word_error_counts

        if self._wer_tokenizer is None:
            self._wer_tokenizer = TextTokenizer(self.config)
        hypotheses = self.decode(logits.detach(), input_lengths)
        references = [row[:length] for row, length in zip(targets.tolist(), target_lengths.tolist())]
        errors = words = 0
        for reference, hypothesis in zip(references, hypotheses):
            count, length = word_error_counts(self._wer_tokenizer.decode(reference),
                                              self._wer_tokenizer.decode(hypothesis))
            errors += count
            words += length
        return errors, words
