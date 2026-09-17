"""Masked audio Transformer and autoregressive text decoder (no CTC upsampling)."""

import torch
from torch import nn
from torch.nn import functional as F

from .asr import TextTokenizer
from .configs import ASRConfig
from .output_dataclasses import ASROutput
from .wer import word_error_counts


def causal_mask(length, device):
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


class AudioTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: ASRConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(input_dim, config.transformer_dim)
        self.positions = nn.Embedding(config.max_audio_length, config.transformer_dim)
        layer = nn.TransformerEncoderLayer(
            config.transformer_dim, config.transformer_heads, config.transformer_ff_dim,
            config.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, config.encoder_layers,
            norm=nn.LayerNorm(config.transformer_dim), enable_nested_tensor=False)
        self.output_projection = nn.Linear(config.transformer_dim, output_dim)

    def forward(self, frames, valid):
        length = frames.shape[1]
        if length > self.config.max_audio_length:
            raise ValueError("Audio exceeds asr.max_audio_length.")
        hidden = self.input_projection(frames.masked_fill(~valid.unsqueeze(-1), 0))
        hidden = hidden + self.positions(torch.arange(length, device=frames.device))
        mask = causal_mask(length, frames.device) if self.config.encoder_causal else None
        hidden = self.transformer(hidden, mask=mask, src_key_padding_mask=~valid)
        return self.output_projection(hidden).masked_fill(~valid.unsqueeze(-1), 0)


class Seq2SeqASRHead(nn.Module):
    def __init__(self, code_dim: int, config: ASRConfig):
        super().__init__()
        self.config = config
        self.pad_id, self.bos_id, self.eos_id = range(config.num_tokens, config.num_tokens + 3)
        self.audio_encoder = (AudioTransformer(code_dim, config.transformer_dim, config)
            if config.quantizer_position == "before_transformer" else None)
        self.memory_projection = (nn.Linear(code_dim, config.transformer_dim)
            if self.audio_encoder is None else None)
        self.audio_positions = (nn.Embedding(config.max_audio_length, config.transformer_dim)
            if self.audio_encoder is None else None)
        self.embedding = nn.Embedding(config.num_tokens + 3, config.transformer_dim, padding_idx=self.pad_id)
        self.positions = nn.Embedding(config.max_text_length, config.transformer_dim)
        layer = nn.TransformerDecoderLayer(
            config.transformer_dim, config.transformer_heads, config.transformer_ff_dim,
            config.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, config.decoder_layers,
                                             norm=nn.LayerNorm(config.transformer_dim))
        self.output = nn.Linear(config.transformer_dim, config.num_tokens + 3)
        self.tokenizer = TextTokenizer(config)

    def validate_targets(self, targets, lengths, input_lengths):
        if targets is None or lengths is None:
            raise ValueError("ASR requires text_targets and text_lengths.")
        if targets.dtype != torch.long or lengths.dtype != torch.long:
            raise ValueError("ASR targets and lengths must be int64.")
        if targets.ndim != 2 or lengths.shape != input_lengths.shape or len(targets) != len(lengths):
            raise ValueError("Expected targets [B, L] and lengths [B].")
        if (lengths < 1).any() or (lengths > targets.shape[1]).any():
            raise ValueError("Invalid text lengths.")
        if (lengths + 1 > self.config.max_text_length).any():
            raise ValueError("Text including EOS exceeds asr.max_text_length.")
        valid = torch.arange(targets.shape[1], device=targets.device)[None] < lengths[:, None]
        if ((targets[valid] < 0) | (targets[valid] >= self.config.num_tokens)).any():
            raise ValueError("Text contains out-of-vocabulary IDs.")

    def teacher_forcing_tokens(self, targets, lengths):
        width = int(lengths.max()) + 1
        labels = targets.new_full((len(targets), width), self.pad_id)
        valid = torch.arange(width - 1, device=targets.device)[None] < lengths[:, None]
        labels[:, :-1] = targets[:, :width - 1].masked_fill(~valid, self.pad_id)
        labels.scatter_(1, lengths[:, None], self.eos_id)
        inputs = labels.new_full(labels.shape, self.pad_id)
        inputs[:, 0] = self.bos_id
        inputs[:, 1:] = labels[:, :-1]
        return inputs, labels

    def encode_memory(self, codes, valid):
        if codes.shape[1] > self.config.max_audio_length:
            raise ValueError("Audio exceeds asr.max_audio_length.")
        if self.audio_encoder is not None:
            return self.audio_encoder(codes, valid)
        hidden = self.memory_projection(codes.masked_fill(~valid.unsqueeze(-1), 0))
        return hidden + self.audio_positions(torch.arange(codes.shape[1], device=codes.device))

    def decode_logits(self, inputs, memory, valid):
        hidden = self.embedding(inputs) + self.positions(torch.arange(inputs.shape[1], device=inputs.device))
        hidden = self.decoder(hidden, memory, tgt_mask=causal_mask(inputs.shape[1], inputs.device),
            tgt_key_padding_mask=inputs.eq(self.pad_id), memory_key_padding_mask=~valid)
        return self.output(hidden)

    def forward(self, codes, valid, targets, lengths):
        self.validate_targets(targets, lengths, valid.sum(1))
        inputs, labels = self.teacher_forcing_tokens(targets, lengths)
        logits = self.decode_logits(inputs, self.encode_memory(codes, valid), valid)
        loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(),
            ignore_index=self.pad_id, label_smoothing=self.config.label_smoothing)
        return ASROutput(loss, logits, lengths + 1)

    @torch.no_grad()
    def generate(self, codes, valid, max_length=None):
        limit = self.config.max_decode_length if max_length is None else max_length
        if not 1 <= limit <= self.config.max_text_length:
            raise ValueError("Invalid ASR generation length.")
        modes = {module: module.training for module in self.modules()}
        self.eval()
        try:
            memory = self.encode_memory(codes, valid)
            tokens = torch.full((len(codes), 1), self.bos_id, device=codes.device, dtype=torch.long)
            finished = torch.zeros(len(codes), device=codes.device, dtype=torch.bool)
            for _ in range(limit):
                logits = self.decode_logits(tokens, memory, valid)[:, -1]
                logits[:, [self.pad_id, self.bos_id]] = -torch.inf
                next_token = logits.argmax(-1).masked_fill(finished, self.eos_id)
                tokens = torch.cat((tokens, next_token[:, None]), dim=1)
                finished |= next_token.eq(self.eos_id)
                if finished.all():
                    break
            results = []
            for row in tokens[:, 1:].tolist():
                results.append(row[:row.index(self.eos_id)] if self.eos_id in row else row)
            return results
        finally:
            for module, mode in modes.items():
                module.training = mode

    @torch.no_grad()
    def word_error_counts(self, codes, valid, targets, lengths):
        hypotheses = self.generate(codes.detach(), valid)
        errors = words = 0
        for hypothesis, row, length in zip(hypotheses, targets.tolist(), lengths.tolist()):
            count, total = word_error_counts(self.tokenizer.decode(row[:length]), self.tokenizer.decode(hypothesis))
            errors += count
            words += total
        return errors, words
