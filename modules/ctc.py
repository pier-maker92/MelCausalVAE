import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


# -----------------------------
# Phoneme tokenizer (dynamic vocab) + optional <sil> handling
# -----------------------------
class PhonemeTokenizer:
    """
    Dynamic vocab:
      <blank>=0  (CTC blank)
      <unk>=1
    Targets for CTC must NOT include <blank>.
    """

    def __init__(
        self,
        output_size: int = 250,
        add_edge_sil: bool = True,
        sil_token: str = "<sil>",
    ):
        self.vocab: Dict[str, int] = {"<blank>": 0, "<unk>": 1}
        self.output_size = output_size
        self._unk_warnings = set()
        self.add_edge_sil = add_edge_sil
        self.sil_token = sil_token

        # ensure sil exists early if desired
        if self.add_edge_sil:
            self._update_vocab(self.sil_token)

    def _update_vocab(self, phoneme: str) -> int:
        if phoneme not in self.vocab:
            new_idx = len(self.vocab)
            if new_idx >= self.output_size:
                if phoneme not in self._unk_warnings:
                    print(
                        f"Warning: Vocabulary full (size={self.output_size}). "
                        f"Mapping phoneme '{phoneme}' to <unk>. Increase output_size."
                    )
                    self._unk_warnings.add(phoneme)
                return self.vocab["<unk>"]
            self.vocab[phoneme] = new_idx
        return self.vocab[phoneme]

    def parse_phoneme_strings(self, phoneme_strings: List[str]) -> List[List[str]]:
        """
        Input: List[str], each string is like: "p h o n e m e s"
               you can include word separators; we just split by spaces.
        Adds edge silences if enabled.
        """
        out: List[List[str]] = []
        for s in phoneme_strings:
            s = s.strip()
            tokens = s.split() if len(s) else []
            if self.add_edge_sil:
                tokens = [self.sil_token] + tokens + [self.sil_token]
            out.append(tokens)
        return out

    def phonemes_to_indices(
        self, phonemes: List[List[str]]
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        targets = []
        target_lengths = []
        for phoneme_list in phonemes:
            indices = [self._update_vocab(p) for p in phoneme_list]
            targets.extend(indices)
            target_lengths.append(len(indices))
        return torch.LongTensor(targets), torch.LongTensor(target_lengths)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()

    def set_vocab(self, vocab: Dict[str, int]):
        self.vocab = vocab.copy()
        if "<unk>" not in self.vocab:
            self.vocab["<unk>"] = 1


# -----------------------------
# CTC head on frame-level h (BEFORE CIF)
# -----------------------------
class CTCHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 250,
        blank: int = 0,
        add_edge_sil: bool = True,
        sil_token: str = "<sil>",
    ):
        super().__init__()
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
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)
        self.blank = blank
        self.output_size = output_size
        self.tokenizer = PhonemeTokenizer(
            output_size=output_size, add_edge_sil=add_edge_sil, sil_token=sil_token
        )

    def forward(
        self,
        h: torch.FloatTensor,  # [B,T,D]
        phoneme_strings: List[str],  # List[str] e.g. "p h o n ..."
        input_lengths: Optional[torch.LongTensor] = None,  # [B]
    ):
        B, T, _ = h.shape
        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=h.device)

        logits = self.net(h)  # [B,T,C]
        log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)  # [T,B,C]

        phonemes = self.tokenizer.parse_phoneme_strings(
            phoneme_strings
        )  # List[List[str]]
        targets, target_lengths = self.tokenizer.phonemes_to_indices(
            phonemes
        )  # flat targets + lengths
        targets = targets.to(h.device)
        target_lengths = target_lengths.to(h.device)

        # CTCLoss prefers float32 on CUDA
        loss = self.ctc_loss(
            log_probs.float(), targets, input_lengths, target_lengths
        ).to(log_probs.dtype)

        return {
            "ctc_loss": loss,
            "log_probs": log_probs,  # [T,B,C]
            "phonemes": phonemes,  # List[List[str]]
            "target_lengths": target_lengths,  # [B]  (N per sample)
            "vocab": self.tokenizer.get_vocab(),
        }


# -----------------------------
# Fully-differentiable CIF with exact K=N via alpha normalization
# -----------------------------
class CIFFullyDiffAligner(nn.Module):
    """
    CIF via overlap in mass-space (vectorized, differentiable).

    Steps:
      - raw_alpha = softplus(alpha_net(h))  [B,T] > 0
      - mask padding, then scale raw_alpha so that sum_t alpha_t = N exactly (per sample)
      - define cumulative mass c_t = sum_{<=t} alpha
        frame interval in mass-space is [c_{t-1}, c_t]
        token k corresponds to mass interval [k-1, k]
      - overlap mass assigned from frame t to token k is:
          w_{t,k} = clamp(min(c_t, k) - max(c_{t-1}, k-1), 0)
      - token embedding: token_k = sum_t w_{t,k} * v_t
      - CIF mass per token: sum_t w_{t,k}  (≈1)
      - frame-duration per token: sum_t (w_{t,k} / alpha_t)  (fractions of frames) -> sums to L
    """

    def __init__(
        self, input_size: int, value_size: Optional[int] = None, alpha_hidden: int = 128
    ):
        super().__init__()
        self.value_size = value_size if value_size is not None else input_size

        self.alpha_net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, alpha_hidden),
            nn.SiLU(),
            nn.Linear(alpha_hidden, 1),
        )
        self.value_proj = nn.Linear(input_size, self.value_size)

    @staticmethod
    def _length_mask(T: int, lengths: torch.Tensor) -> torch.Tensor:
        # lengths: [B]
        ar = torch.arange(T, device=lengths.device).unsqueeze(0)  # [1,T]
        return ar < lengths.unsqueeze(1)  # [B,T] bool

    @staticmethod
    def _normalize_alpha_to_N(
        raw_alpha: torch.Tensor,  # [B,T] >0
        target_lengths: torch.LongTensor,  # [B] = N
        input_lengths: torch.LongTensor,  # [B] = L
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Mask padding, then scale so sum(alpha[b,:L]) == N[b] exactly (up to numerical tolerance).
        """
        B, T = raw_alpha.shape
        mask = CIFFullyDiffAligner._length_mask(T, input_lengths).to(
            raw_alpha.dtype
        )  # [B,T]
        alpha = raw_alpha * mask

        sums = alpha.sum(dim=1)  # [B]
        N = target_lengths.to(raw_alpha.dtype)  # [B]

        # scale factor; if sum is tiny, fall back to uniform distribution over valid frames
        scale = N / (sums + eps)  # [B]
        alpha_scaled = alpha * scale.unsqueeze(1)  # [B,T]

        # uniform fallback when sums ~ 0
        # uniform alpha on valid frames: N/L
        L = input_lengths.to(raw_alpha.dtype).clamp_min(1.0)
        alpha_uniform = mask * (N / L).unsqueeze(1)

        use_uniform = (sums < 10 * eps).to(raw_alpha.dtype).unsqueeze(1)  # [B,1]
        alpha_final = alpha_scaled * (1.0 - use_uniform) + alpha_uniform * use_uniform

        return alpha_final  # [B,T], sum over valid frames == N

    def forward(
        self,
        h: torch.FloatTensor,  # [B,T,D]
        target_lengths: torch.LongTensor,  # [B] = N
        input_lengths: torch.LongTensor,  # [B] = L
    ):
        B, T, _ = h.shape
        V = self.value_size
        eps = 1e-8

        raw_alpha = F.softplus(self.alpha_net(h)).squeeze(-1)  # [B,T] >0
        alpha = self._normalize_alpha_to_N(
            raw_alpha, target_lengths, input_lengths, eps=eps
        )  # [B,T]

        # values to accumulate
        v = self.value_proj(h)  # [B,T,V]

        # Nmax for padding to a fixed tensor
        Nmax = int(target_lengths.max().item()) if target_lengths.numel() else 0
        if Nmax == 0:
            tokens = h.new_zeros((B, 0, V))
            frame_durations = h.new_zeros((B, 0))
            cif_mass_per_token = h.new_zeros((B, 0))
            token_mask = torch.zeros((B, 0), device=h.device, dtype=torch.bool)
            return {
                "tokens": tokens,
                "frame_durations": frame_durations,
                "cif_mass_per_token": cif_mass_per_token,
                "token_mask": token_mask,
                "alpha": alpha,
                "raw_alpha": raw_alpha,
            }

        # cumulative mass
        c = torch.cumsum(alpha, dim=1)  # [B,T]
        c_prev = c - alpha  # [B,T]

        # token indices in mass-space:
        # token k (1..Nmax) corresponds to interval [k-1, k]
        k = torch.arange(1, Nmax + 1, device=h.device, dtype=h.dtype).view(
            1, 1, Nmax
        )  # [1,1,N]
        left = torch.maximum(c_prev.unsqueeze(-1), (k - 1.0))  # [B,T,N]
        right = torch.minimum(c.unsqueeze(-1), k)  # [B,T,N]
        w = torch.clamp(right - left, min=0.0)  # [B,T,N]
        # w is "mass assigned from frame t to token k" (CIF overlap)

        # token mask: only first N tokens are valid per sample
        token_mask = torch.arange(Nmax, device=h.device).unsqueeze(
            0
        ) < target_lengths.unsqueeze(
            1
        )  # [B,N] bool

        # Token embeddings: sum_t w[t,k] * v[t]
        # w: [B,T,N], v: [B,T,V] -> tokens: [B,N,V]
        tokens = torch.einsum("btn,btv->bnv", w, v)

        # CIF mass per token (should be ~1 for valid tokens)
        cif_mass_per_token = w.sum(dim=1)  # [B,N]

        # Frame durations (soft): fraction of each frame assigned to token
        # fraction = w / alpha_t
        denom = (alpha + eps).unsqueeze(-1)  # [B,T,1]
        frac = w / denom  # [B,T,N]
        frame_durations = frac.sum(dim=1)  # [B,N]

        # Zero out padding tokens explicitly
        tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)
        cif_mass_per_token = cif_mass_per_token * token_mask.to(
            cif_mass_per_token.dtype
        )
        frame_durations = frame_durations * token_mask.to(frame_durations.dtype)

        return {
            "tokens": tokens,  # [B,Nmax,V]
            "frame_durations": frame_durations,  # [B,Nmax] sum ~= input_lengths
            "cif_mass_per_token": cif_mass_per_token,  # [B,Nmax] each ~1
            "token_mask": token_mask,  # [B,Nmax]
            "alpha": alpha,  # [B,T] sum==N (valid frames)
            "raw_alpha": raw_alpha,  # [B,T]
        }


# -----------------------------
# Wrapper: CTC (frame) + CIF (fully diff)
# -----------------------------
class CTCThenCIF(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        ctc_hidden: int,
        ctc_output: int = 250,
        cif_value_dim: Optional[int] = None,
        add_edge_sil: bool = True,
        sil_token: str = "<sil>",
    ):
        super().__init__()
        self.ctc = CTCHead(
            input_size=encoder_dim,
            hidden_size=ctc_hidden,
            output_size=ctc_output,
            add_edge_sil=add_edge_sil,
            sil_token=sil_token,
        )
        self.cif = CIFFullyDiffAligner(input_size=encoder_dim, value_size=cif_value_dim)

    def forward(
        self,
        h: torch.FloatTensor,  # [B,T,D] encoder output (causale)
        phoneme_strings: List[str],  # List[str] "p h o n ..."
        input_lengths: Optional[torch.LongTensor] = None,
        return_checks: bool = True,
    ):
        B, T, _ = h.shape
        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=h.device)

        # 1) CTC
        ctc_out = self.ctc(
            h=h, phoneme_strings=phoneme_strings, input_lengths=input_lengths
        )
        target_lengths = ctc_out["target_lengths"]  # [B] = N

        # 2) CIF (K=N exact via alpha normalization)
        cif_out = self.cif(
            h=h, target_lengths=target_lengths, input_lengths=input_lengths
        )

        out = {
            "ctc_loss": ctc_out["ctc_loss"],
            "log_probs": ctc_out["log_probs"],  # [T,B,C]
            "phonemes": ctc_out["phonemes"],  # List[List[str]]
            "target_lengths": target_lengths,  # [B]
            "tokens": cif_out["tokens"],  # [B,N,V]
            "frame_durations": cif_out["frame_durations"],  # [B,N]
            "cif_mass_per_token": cif_out["cif_mass_per_token"],  # [B,N]
            "token_mask": cif_out["token_mask"],  # [B,N]
            "alpha": cif_out["alpha"],  # [B,T]
            "raw_alpha": cif_out["raw_alpha"],  # [B,T]
            "vocab": ctc_out["vocab"],
        }

        if return_checks:
            # Useful diagnostics (no breakpoints)
            with torch.no_grad():
                # alpha sum should match N (per sample, valid frames)
                alpha_sum = out["alpha"].sum(dim=1)  # [B]
                # frame durations sum should match L (per sample)
                dur_sum = out["frame_durations"].sum(dim=1)  # [B]
                out["check_alpha_sum_minus_N"] = alpha_sum - target_lengths.to(
                    alpha_sum.dtype
                )
                out["check_dur_sum_minus_L"] = dur_sum - input_lengths.to(dur_sum.dtype)
                # token mass should be ~1 for valid tokens
                # (mean over valid tokens)
                m = out["cif_mass_per_token"]
                mask = out["token_mask"]
                if mask.any():
                    out["check_token_mass_mean"] = (m[mask]).mean()
                    out["check_token_mass_std"] = (m[mask]).std(unbiased=False)
                else:
                    out["check_token_mass_mean"] = torch.tensor(0.0, device=h.device)
                    out["check_token_mass_std"] = torch.tensor(0.0, device=h.device)

        return out
