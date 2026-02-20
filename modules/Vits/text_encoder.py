import json
import math
import torch
import torch.nn as nn
from typing import List


class PhonemeVocab:
    def __init__(self, path_to_vocab: str, parsing_mode: str = "phoneme"):
        self.vocab = self._load_vocab(path_to_vocab)
        self.parsing_mode = parsing_mode
        assert "<pad>" in self.vocab
        assert "<sil>" in self.vocab  # for blank token
        assert "<unk>" in self.vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _load_vocab(self, path_to_vocab: str):
        try:
            with open(path_to_vocab, "r") as f:
                vocab = json.load(f)
        except FileNotFoundError:
            vocab = {"<pad>": 0, "<sil>": 1, "<unk>": 2}
        return vocab

    def token2id(self, token: str):
        return self.vocab.get(token, self.vocab["<unk>"])

    def _get_phonemes(self, phonemes: List[str]):
        phonemes_batch = []
        for phoneme_str in phonemes:
            phoneme_str = f"<sil> {phoneme_str} <sil>"

            if self.parsing_mode == "phoneme":
                tokens = phoneme_str.split()
            elif self.parsing_mode == "char":
                tokens = []
                for p in phoneme_str.split():
                    if p == "<sil>":
                        tokens.append(p)
                    else:
                        tokens.extend(list(p))
            else:
                raise ValueError(f"Unknown parsing mode: {self.parsing_mode}")

            phonemes_batch.append(
                torch.tensor([self.token2id(t) for t in tokens]).long()
            )
        return phonemes_batch

    def __call__(self, phonemes: List[str], device: torch.device):
        phoneme_ids = self._get_phonemes(phonemes)
        phoneme_mask = [torch.ones(len(ph)) for ph in phoneme_ids]

        phoneme_ids = (
            torch.nn.utils.rnn.pad_sequence(
                phoneme_ids, batch_first=True, padding_value=self.vocab["<pad>"]
            )
            .to(device)
            .long()
        )

        phoneme_mask = (
            torch.nn.utils.rnn.pad_sequence(
                phoneme_mask, batch_first=True, padding_value=0
            )
            .to(device)
            .bool()
        )
        return phoneme_ids, phoneme_mask.unsqueeze(1)


# Utility per il Positional Encoding standard
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1), :].to(x.device)
        return self.dropout(x)


class TextEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        n_heads,
        n_layers,
        kernel_size,  # Può essere ignorato/rimosso se si usa il Transformer di PyTorch
        p_dropout,
        output_dim: int,
        vocab_path: str,
        parsing_mode: str = "phoneme",
    ):
        super().__init__()
        self.out_channels = output_dim
        self.hidden_channels = hidden_channels
        self.vocab = PhonemeVocab(vocab_path, parsing_mode)

        # 1. Embedding
        self.emb = nn.Embedding(len(self.vocab.vocab), hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        # 2. Encoder Transformer
        self.pos_encoder = PositionalEncoding(hidden_channels, p_dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=n_heads,
            dim_feedforward=int(hidden_channels * 4),
            dropout=p_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        # 3. Proiezione Finale (Projection Layer)
        # ---> MODIFICA: Proietta direttamente in out_channels (niente più * 2 per logs_p)
        self.proj = nn.Conv1d(hidden_channels, self.out_channels, 1)

    def forward(self, phonemes: List[str]):
        # ---> MODIFICA: Prende la device dinamicamente dai pesi dell'embedding
        device = self.emb.weight.device
        
        # x: [B, T_text], x_mask: [B, 1, T_text]
        x, x_mask = self.vocab(phonemes, device)

        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T, H]
        x = self.pos_encoder(x)
        
        # TIP: Per far sì che l'attention ignori davvero il padding, dovresti 
        # passare src_key_padding_mask all'encoder. Il formato atteso è True per il padding.
        # padding_mask = (~x_mask.squeeze(1).bool())
        # x = self.encoder(x, src_key_padding_mask=padding_mask)
        
        x = self.encoder(x)

        # Trasponi per convoluzione: [B, T, H] -> [B, H, T]
        x = x.transpose(1, 2)

        # ---> MODIFICA: Proiezione finale per l'allineamento. È già l'output definitivo.
        m = self.proj(x) * x_mask

        return m, x_mask