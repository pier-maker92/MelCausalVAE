"""BiLSTM + location-aware GRU ASR, implemented entirely in PyTorch."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .asr import TextTokenizer
from .asr_seq2seq import AutoregressiveASRHead
from .configs import ASRConfig


@dataclass
class GRUDecoderState:
    hidden: torch.Tensor
    context: torch.Tensor
    attention: torch.Tensor


class LocationAttention(nn.Module):
    """Additive attention conditioned on the previous frame alignment."""

    def __init__(self, memory_dim: int, config: ASRConfig):
        super().__init__()
        self.keys = nn.Linear(memory_dim, config.attention_dim)
        self.query = nn.Linear(config.decoder_hidden_size, config.attention_dim)
        radius = config.attention_kernel_size
        self.location = nn.Conv1d(1, config.attention_channels, 2 * radius + 1,
                                  padding=radius, bias=False)
        self.location_projection = nn.Linear(config.attention_channels, config.attention_dim)
        self.energy = nn.Linear(config.attention_dim, 1, bias=False)
        self.context_projection = nn.Linear(memory_dim, config.decoder_hidden_size)
        self.scaling = config.attention_scaling

    def forward(self, query, memory, keys, previous, valid):
        location = self.location_projection(self.location(previous.unsqueeze(1)).transpose(1, 2))
        scores = self.energy(torch.tanh(keys + self.query(query).unsqueeze(1) + location)).squeeze(-1)
        weights = (scores.float() * self.scaling).masked_fill(~valid, -torch.inf).softmax(-1)
        context = torch.bmm(weights.to(memory.dtype).unsqueeze(1), memory).squeeze(1)
        return self.context_projection(context), weights


class GRUASRHead(AutoregressiveASRHead):
    def __init__(self, code_dim: int, config: ASRConfig):
        super().__init__()
        self.config = config
        self.pad_id, self.bos_id, self.eos_id = range(config.num_tokens, config.num_tokens + 3)
        self.tokenizer = TextTokenizer(config)
        self.input_projection = nn.Linear(code_dim, config.embedding_dim)
        self.audio_encoder = nn.LSTM(config.embedding_dim, config.encoder_hidden_size,
            config.encoder_layers, batch_first=True, bidirectional=True,
            dropout=config.dropout if config.encoder_layers > 1 else 0.0)
        self.embedding = nn.Embedding(config.num_tokens + 3, config.decoder_embedding_dim,
                                      padding_idx=self.pad_id)
        hidden = config.decoder_hidden_size
        self.decoder = nn.GRU(config.decoder_embedding_dim + hidden, hidden,
            config.decoder_layers, batch_first=True,
            dropout=config.dropout if config.decoder_layers > 1 else 0.0)
        self.attention = LocationAttention(2 * config.encoder_hidden_size, config)
        self.fusion = nn.Linear(2 * hidden, hidden)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(hidden, config.num_tokens + 3)
        self.initialize_recurrent_weights()

    def initialize_recurrent_weights(self):
        # Gate-wise orthogonal initialization, corresponding to re_init in the guide.
        for recurrent, gates in ((self.audio_encoder, 4), (self.decoder, 3)):
            for name, parameter in recurrent.named_parameters():
                if name.startswith("weight"):
                    for gate in parameter.chunk(gates, dim=0):
                        nn.init.orthogonal_(gate)
                else:
                    nn.init.zeros_(parameter)

    def encode_memory(self, codes, valid):
        if codes.shape[1] > self.config.max_audio_length:
            raise ValueError("Audio exceeds asr.max_audio_length.")
        hidden = self.input_projection(codes.masked_fill(~valid.unsqueeze(-1), 0))
        packed = pack_padded_sequence(hidden, valid.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.audio_encoder(packed)
        memory, _ = pad_packed_sequence(encoded, batch_first=True, total_length=codes.shape[1])
        return memory

    def initial_state(self, memory, valid):
        hidden = self.config.decoder_hidden_size
        return GRUDecoderState(
            memory.new_zeros(self.config.decoder_layers, len(memory), hidden),
            memory.new_zeros(len(memory), hidden),
            valid.to(memory.dtype) / valid.sum(1, keepdim=True),
        )

    def decode_step(self, token, memory, keys, valid, state):
        embedded = self.dropout(self.embedding(token))
        inputs = torch.cat((embedded, state.context), dim=-1).unsqueeze(1)
        output, hidden = self.decoder(inputs, state.hidden)
        query = output[:, 0]
        context, attention = self.attention(query, memory, keys, state.attention, valid)
        features = self.dropout(self.fusion(torch.cat((query, context), dim=-1)))
        # Carry the previous alignment as a location feature, as in the guide.
        return self.output(features), GRUDecoderState(hidden, context, attention.detach())

    def decode_logits(self, inputs, memory, valid):
        keys = self.attention.keys(memory)
        state = self.initial_state(memory, valid)
        logits = []
        for token in inputs.unbind(1):
            step_logits, state = self.decode_step(token, memory, keys, valid, state)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(self, codes, valid, max_length=None):
        limit = self.config.max_decode_length if max_length is None else max_length
        if not 1 <= limit <= self.config.max_text_length:
            raise ValueError("Invalid ASR generation length.")
        modes = {module: module.training for module in self.modules()}
        self.eval()
        try:
            memory = self.encode_memory(codes, valid)
            keys = self.attention.keys(memory)
            state = self.initial_state(memory, valid)
            token = torch.full((len(codes),), self.bos_id, device=codes.device, dtype=torch.long)
            finished = torch.zeros_like(token, dtype=torch.bool)
            generated = []
            for _ in range(limit):
                logits, state = self.decode_step(token, memory, keys, valid, state)
                logits[:, [self.pad_id, self.bos_id]] = -torch.inf
                token = logits.argmax(-1).masked_fill(finished, self.eos_id)
                generated.append(token)
                finished |= token.eq(self.eos_id)
                if finished.all():
                    break
            results = []
            for row in torch.stack(generated, dim=1).tolist():
                results.append(row[:row.index(self.eos_id)] if self.eos_id in row else row)
            return results
        finally:
            for module, mode in modes.items():
                module.training = mode
