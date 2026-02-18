import math
import torch
import numpy as np
import torch.nn as nn
import monotonic_align  # Fallback per esecuzione diretta


def search_path(z_p, m_p, logs_p, x_mask, y_mask):
    with torch.no_grad():
        b, c, t_y = z_p.shape
        t_x = m_p.shape[2]

        # z_p: [B, C, T_audio] (Variabili latenti audio)
        # m_p: [B, C, T_text] (Media predetta dal testo)
        # logs_p: [B, C, T_text] (Log-varianza predetta dal testo)

        # 1. Calcoliamo l'inverso della varianza (precisione)
        # 1 / sigma^2 = exp(-2 * logs)
        z_p_sq = z_p**2
        o_p = torch.exp(-2 * logs_p)  # [B, C, T_x]

        # Dobbiamo calcolare: sum_channels( (z - m)^2 / sigma^2 )
        # Espandendo il quadrato: (z^2)/sigma^2 - (2zm)/sigma^2 + (m^2)/sigma^2

        # Termine A: sum(z^2 * (1/sigma^2))
        # Usiamo Matmul: [B, T_x, C] * [B, C, T_y] -> [B, T_x, T_y]
        term_a = torch.matmul(o_p.transpose(1, 2), z_p_sq)

        # Termine B: -2 * sum(z * m * (1/sigma^2))
        # Weighted mean: m_p * o_p
        term_b = -2 * torch.matmul((m_p * o_p).transpose(1, 2), z_p)

        # Termine C: sum(m^2 * (1/sigma^2))
        # Questo è costante per ogni frame audio, dipende solo dal testo
        term_c = torch.sum(m_p**2 * o_p, dim=1, keepdim=True).transpose(
            1, 2
        )  # [B, T_x, 1]

        # Termine D: Log Determinante (Parte della normalizzazione della Gaussiana)
        # sum(log(2*pi) + log(sigma^2)) -> ignoriamo costanti, teniamo 2 * logs_p
        term_d = torch.sum(2 * logs_p, dim=1, keepdim=True).transpose(
            1, 2
        )  # [B, T_x, 1]

        # Somma totale per la Log Likelihood Negativa (pesata)
        # log_p = -0.5 * ( (z-m)^2/var + log_var )
        log_p = -0.5 * (term_a + term_b + term_c + term_d)  # [B, T_x, T_y]

        # 2. Preparazione per Cython
        # Cython implementation expects [T_y, T_x] but we have [T_x, T_y]
        # We need to transpose inputs and outputs [B, T_y, T_x]
        log_p_transposed = log_p.transpose(1, 2).contiguous()

        # Path buffer [B, T_y, T_x]
        path = torch.zeros(b, t_y, t_x, dtype=torch.int32).to(device=z_p.device)

        # Ensure contiguous memory layout for Cython
        log_p_cpu = log_p_transposed.data.cpu().numpy().astype(np.float32)
        path_cpu = path.data.cpu().numpy().astype(np.int32)
        t_x_len = x_mask.sum([1, 2]).data.cpu().numpy().astype(np.int32)
        t_y_len = y_mask.sum([1, 2]).data.cpu().numpy().astype(np.int32)

        # Chiamata Cython
        monotonic_align.maximum_path_c(path_cpu, log_p_cpu, t_y_len, t_x_len)

        # Ritorna il path come tensore float (0.0 o 1.0)
        # Converti e transponi back a [B, T_x, T_y]
        path = torch.from_numpy(path_cpu).to(device=z_p.device, dtype=torch.float32)
        path = path.transpose(1, 2)
        return path


# Riutilizziamo le tue classi base
class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # 1. Embedding (dal tuo codice)
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        # 2. Encoder Transformer (Semplificato con implementazione PyTorch standard)
        # In VITS reale si usa Relative Positional Encoding, qui uso Absolute per brevità
        self.pos_encoder = PositionalEncoding(hidden_channels, p_dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=n_heads,
            dim_feedforward=filter_channels,
            dropout=p_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        # 3. Proiezione Finale (Projection Layer)
        # Proietta l'hidden state in Media e Log-Varianza
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        """
        x: [B, T_text] (Phoneme IDs)
        x_lengths: [B]
        """
        # Creazione maschera per il padding
        # Nota: Transformer di PyTorch vuole maschera (True = ignora)
        # Qui costruiamo una maschera semplice per la convoluzione finale
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(1)), 1).to(
            x.device
        )

        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T, H]
        x = self.pos_encoder(x)
        x = self.encoder(x)

        # Trasponi per convoluzione [B, T, H] -> [B, H, T]
        x = x.transpose(1, 2)

        # Proiezione finale
        stats = self.proj(x) * x_mask

        # Split in Media e Log-Varianza
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return m, logs, x_mask


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


# Utility maschera (spesso chiamata commons.sequence_mask nelle repo VITS)
class commons:
    @staticmethod
    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
    ):
        super().__init__()
        self.half_channels = channels // 2

        # Rete non causale (WaveNet-like) che trasforma metà dei canali
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = nn.ModuleList()
        self.dilations = [
            1
        ] * n_layers  # Semplificazione: dilatations fisse o crescenti

        for i in range(n_layers):
            self.enc.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        dilation=self.dilations[i],
                        padding=(kernel_size - 1) * self.dilations[i] // 2,
                    ),
                    nn.GELU(),
                    nn.Dropout(p_dropout),
                )
            )

        # Proietta indietro per ottenere shift (m) e scale (logs)
        # Nota: l'output è doppio perché predice m e logs per l'altra metà
        self.post = nn.Conv1d(hidden_channels, self.half_channels * 2, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, reverse=False):
        # Split canali: [x0, x1]
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        # x0 passa attraverso la rete per trasformare x1
        h = self.pre(x0) * x_mask
        for layer in self.enc:
            h = layer(h) + h  # Residual connection interna
            h = h * x_mask

        stats = self.post(h) * x_mask
        m, logs = torch.split(stats, [self.half_channels] * 2, 1)

        if not reverse:
            # Forward: Training (Normalizing: Audio complesso -> Gaussiana semplice)
            # x1 = (x1 - m) * exp(-logs)
            x1 = (
                (m - x1) * torch.exp(logs) * x_mask
            )  # Implementazione VITS standard varia leggermente nel segno
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            # Reverse: Inference (Generating: Gaussiana semplice -> Audio complesso)
            # x1 = x1 * exp(-logs) + m (inversione della formula sopra)
            # Nota: L'algebra esatta dipende dalla convenzione affine.
            # Qui usiamo: x1_new = m - x1_old * exp(-logs)
            x1 = m - x1 * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class VitsKLProcessor(nn.Module):
    def __init__(self, text_encoder, flow_model):
        super().__init__()
        self.text_encoder = text_encoder
        self.flow = flow_model

    def forward(self, z_spec, y_mask, phonemes, phoneme_lengths):
        """
        z_spec: [B, C, T_audio] (Output del Posterior Encoder - campionato con reparam trick)
        y_mask: [B, 1, T_audio] (Mask dell'audio)
        phonemes: [B, T_text]
        phoneme_lengths: [B]
        """

        # 1. Text Encoding (Prior)
        # m_p, logs_p: statistiche previste dal testo [B, C, T_text]
        m_p, logs_p, x_mask = self.text_encoder(phonemes, phoneme_lengths)

        # 2. Flow Model (Trasformazione)
        # Trasforma z_spec (complesso) verso z_flow (semplice)
        z_flow, logdet_flow = self.flow(z_spec, y_mask, reverse=False)

        # 3. Monotonic Alignment Search (MAS)
        # Trova la mappa ottimale tra z_flow (audio trasformato) e m_p (testo)
        # Nota: MAS usa z_flow, non z_spec, perché z_flow è comparabile al testo
        attn_mask = search_path(z_flow, m_p, logs_p, x_mask, y_mask)
        # attn_mask shape: [B, T_text, T_audio]

        # 4. Espansione delle statistiche del testo
        # Moltiplica Prior per Allineamento: [B, C, T_text] x [B, T_text, T_audio] = [B, C, T_audio]
        m_p_expanded = torch.matmul(m_p, attn_mask)
        logs_p_expanded = torch.matmul(logs_p, attn_mask)

        # 5. Calcolo KL Loss
        # La formula è: -LogLikelihood(z_flow | N(m_p_exp, logs_p_exp)) - log_det_jacobian

        # Log Normal PDF
        # Costante -0.5 * log(2pi) omessa perché costante durante l'ottimizzazione
        kl_loss_elements = logs_p_expanded + 0.5 * torch.exp(-2 * logs_p_expanded) * (
            (z_flow - m_p_expanded) ** 2
        )

        # Somma su canali e tempo, media sul batch
        kl_loss_raw = torch.sum(kl_loss_elements * y_mask) - torch.sum(logdet_flow)

        # Normalizzazione per batch e time (o per numero di elementi totali mascherati)
        # In VITS solitamente si normalizza per numero totale di frame nel batch
        total_frames = torch.sum(y_mask)
        loss = kl_loss_raw / total_frames

        return loss, attn_mask, z_flow


# Esempio di utilizzo fittizio
if __name__ == "__main__":
    # Config
    vocab_size = 100
    channels = 192
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init Moduli
    txt_enc = TextEncoder(
        vocab_size,
        out_channels=channels,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
    ).to(device)

    # Flow semplice (solo 1 layer per demo)
    flow = ResidualCouplingLayer(channels, 192, 5, 1, 4).to(device)

    # Processore
    vits_loss_module = VitsKLProcessor(txt_enc, flow).to(device)

    # Input Finti
    B = 24
    T_audio = 1500
    T_text = 340

    # z dal posterior encoder (già campionato)
    z_from_posterior = torch.randn(B, channels, T_audio).to(device)
    y_mask = torch.ones(B, 1, T_audio).to(device)

    # Fonemi
    phonemes = torch.randint(0, vocab_size, (B, T_text)).to(device)
    phoneme_lens = torch.tensor([T_text] * B).to(device)

    # Calcolo
    loss, alignment, z_transformed = vits_loss_module(
        z_from_posterior, y_mask, phonemes, phoneme_lens
    )

    print(f"KL Loss: {loss.item()}")
    print(f"Alignment Shape: {alignment.shape}")  # Dovrebbe essere [2, 10, 50]

    # save a figure with  the alignement
    import matplotlib.pyplot as plt

    plt.imshow(alignment[0].cpu().detach().numpy(), aspect="auto")
    plt.savefig("alignment.png")
