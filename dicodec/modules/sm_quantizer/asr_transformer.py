"""Masked audio Transformer and autoregressive text decoder (no CTC upsampling)."""

import torch
from torch import nn

from .asr import TextTokenizer
from .configs import ASRConfig
from .asr_seq2seq import AutoregressiveASRHead


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


class Seq2SeqASRHead(AutoregressiveASRHead):
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
