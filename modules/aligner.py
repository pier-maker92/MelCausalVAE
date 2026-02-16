# aligner_dp_debug.py
import json
import math
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_durations_on_mel(
    mels,
    durations,
    mel_mask,
    text_length,
    batch_idx=0,
    step=0,
    labels=None,
    device_id=0,
):
    # mel_mask is True for padding. We want valid length.
    valid_len = (~mel_mask[batch_idx]).long().sum().item()

    mel = mels[batch_idx, :valid_len].detach().float().cpu().numpy().T
    dur = durations[batch_idx, : text_length[batch_idx]].detach().float().cpu().numpy()

    positions = dur.cumsum()
    fig, (ax_mel, ax_dur) = plt.subplots(2, 1, figsize=(16, 6))

    ax_mel.imshow(mel, origin="lower", aspect="auto")
    ax_mel.set_xlim(0, mel.shape[1])
    for pos in positions:
        ax_mel.axvline(pos, color="white", linestyle="--", linewidth=0.8, alpha=0.7)

    ax_mel.set_ylabel("Mel bin")
    ax_mel.set_title(f"Sample {batch_idx} - Step {step} - Device {device_id}")
    ax_mel.set_xticks([])

    norm_dur = (dur - dur.min()) / (dur.max() - dur.min() + 1e-8) * 0.7 + 0.3
    ax_dur.bar(
        range(len(dur)),
        dur,
        color=plt.cm.Blues(norm_dur),
        edgecolor="black",
        linewidth=0.5,
    )
    ax_dur.set_xlabel("z position")
    ax_dur.set_ylabel("Duration (frames)")
    ax_dur.set_xlim(-0.5, len(dur) - 0.5)
    ax_dur.set_xticks(range(len(dur)))
    ax_dur.set_xticklabels(
        labels[: len(dur)] if labels else range(len(dur)), rotation=0, ha="right"
    )

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def neg_inf(dtype: torch.dtype) -> float:
    """
    A large negative number that is representable in the given dtype.
    - fp16: -1e4 is safe-ish (avoid -inf)
    - bf16/fp32: -1e9 is fine
    """
    return -1e4 if dtype == torch.float16 else -1e9


def safe_isfinite(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


def debug_print(enabled: bool, *args):
    if enabled:
        print(*args)


def _masked_log_softmax(
    logits: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    logits: [..., K]
    mask:   same shape broadcastable to logits, True=valid
    """
    neg = neg_inf(logits.dtype)
    logits = logits.masked_fill(~mask, neg)

    # If an entire row is masked, logsumexp becomes neg (or -inf).
    # We still compute it, but downstream we must detect impossible cases (logZ=-inf).
    lse = torch.logsumexp(logits, dim=dim, keepdim=True)
    return logits - lse


# -----------------------------------------------------------------------------
# Vocab + Embeddings (unchanged)
# -----------------------------------------------------------------------------
class PhonemeVocab:
    def __init__(self, path_to_vocab: str, parsing_mode: str = "phoneme"):
        self.vocab = self._load_vocab(path_to_vocab)
        self.parsing_mode = parsing_mode
        assert "<pad>" in self.vocab
        assert "<sil>" in self.vocab
        assert "<unk>" in self.vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _load_vocab(self, path_to_vocab: str):
        with open(path_to_vocab, "r") as f:
            vocab = json.load(f)
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
        return phoneme_ids, phoneme_mask


class PhonemesEmbeddings(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, phonemes: torch.Tensor) -> torch.Tensor:
        return self.embedding(phonemes)


# -----------------------------------------------------------------------------
# Monotonic DP Aligner with FULL DEBUG
# -----------------------------------------------------------------------------
class MonotonicDPAligner(nn.Module):
    """
    Monotonic alignment with forward-sum / backward-sum (stay or advance by 1).
    Returns:
      gamma:    [B, T2, T1] posterior (frame->token)
      pooled:   [B, T1, D]  frames pooled to tokens
      context:  [B, T2, D]  tokens expanded back to frames via gamma
      durations:[B, T1]
      align_loss: scalar (normalized per-frame)
      debug: dict (optional)
    """

    def __init__(
        self, audio_dim: int, text_dim: int, attn_dim: int = 256, eps: float = 1e-8
    ):
        super().__init__()
        self.q = nn.Linear(audio_dim, attn_dim, bias=False)
        self.k = nn.Linear(text_dim, attn_dim, bias=False)
        self.scale = 1.0 / math.sqrt(attn_dim)
        self.eps = eps

    # ---------------------------
    # Debug helpers
    # ---------------------------
    def _debug_masks(
        self, audio_mask: torch.Tensor, text_mask: torch.Tensor, debug: bool
    ):
        with torch.no_grad():
            debug_print(
                debug,
                f"[DPAligner] audio_mask True(valid) ratio: {audio_mask.float().mean().item():.4f}",
            )
            debug_print(
                debug,
                f"[DPAligner] text_mask  True(valid) ratio: {text_mask.float().mean().item():.4f}",
            )
            debug_print(debug, f"[DPAligner] audio_len: {audio_mask.sum(1).tolist()}")
            debug_print(debug, f"[DPAligner] text_len : {text_mask.sum(1).tolist()}")

    def compute_logp(
        self,
        frames: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        Q = self.q(frames)  # [B,T2,A]
        K = self.k(text_emb)  # [B,T1,A]
        logits = torch.matmul(Q, K.transpose(1, 2)) * self.scale  # [B,T2,T1]
        logp = _masked_log_softmax(logits, text_mask.unsqueeze(1), dim=-1)

        if debug:
            with torch.no_grad():
                ok = safe_isfinite(logp)
                debug_print(debug, f"[DPAligner] logp finite: {ok}")
                if not ok:
                    finite = torch.isfinite(logp)
                    debug_print(
                        debug,
                        f"[DPAligner] logp finite ratio: {finite.float().mean().item():.4f}",
                    )
                    debug_print(
                        debug,
                        f"[DPAligner] logits dtype={logits.dtype} logp dtype={logp.dtype}",
                    )
                    debug_print(
                        debug,
                        f"[DPAligner] logits min/max: {logits.min().item():.3f} {logits.max().item():.3f}",
                    )
                    # Crash hard so you see the real source
                    raise RuntimeError("Non-finite logp detected")
        return logp

    def forward_sum(
        self,
        logp: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
        debug: bool = False,
        fail_fast: bool = True,
    ):
        """
        Forward DP in log-space.
        """
        B, T2, T1 = logp.shape
        device = logp.device
        dtype = logp.dtype
        neg = neg_inf(dtype)

        text_len = text_mask.sum(dim=1).long().clamp(min=1)  # [B]
        last_p = (text_len - 1).clamp(min=0)

        # init alpha with NEG
        log_alpha = torch.full((B, T1), neg, device=device, dtype=dtype)

        # Base case:
        # if first frame is invalid, this model can't start correctly.
        # We'll still set alpha[:,0] from t=0, but we'll detect impossibility via logZ.
        log_alpha[:, 0] = logp[:, 0, 0]
        log_alpha = log_alpha.masked_fill(~text_mask, neg)

        log_alpha_all = torch.full((B, T2, T1), neg, device=device, dtype=dtype)
        log_alpha_all[:, 0, :] = log_alpha

        for t in range(1, T2):
            prev = log_alpha

            stay = prev
            adv = torch.cat(
                [torch.full((B, 1), neg, device=device, dtype=dtype), prev[:, :-1]],
                dim=1,
            )

            log_alpha_new = logp[:, t, :] + torch.logsumexp(
                torch.stack([stay, adv], dim=0), dim=0
            )
            log_alpha_new = log_alpha_new.masked_fill(~text_mask, neg)

            # If frame t is padding => do not update
            m = audio_mask[:, t].unsqueeze(1)  # [B,1] True=valid
            log_alpha = torch.where(m, log_alpha_new, prev)
            log_alpha_all[:, t, :] = log_alpha

        audio_len = audio_mask.sum(dim=1).long().clamp(min=1)
        last_t = (audio_len - 1).clamp(min=0)

        logZ = log_alpha_all[torch.arange(B, device=device), last_t, last_p]

        if debug:
            with torch.no_grad():
                finiteZ = torch.isfinite(logZ)
                debug_print(
                    debug,
                    f"[DPAligner] logZ finite ratio: {finiteZ.float().mean().item():.4f}",
                )
                if (~finiteZ).any():
                    bad = (~finiteZ).nonzero().squeeze(-1)
                    debug_print(debug, f"[DPAligner] BAD logZ idx: {bad.tolist()}")
                    debug_print(
                        debug, f"[DPAligner] BAD audio_len: {audio_len[bad].tolist()}"
                    )
                    debug_print(
                        debug, f"[DPAligner] BAD text_len : {text_len[bad].tolist()}"
                    )
                    debug_print(
                        debug,
                        f"[DPAligner] BAD last_t/last_p: {last_t[bad].tolist()} / {last_p[bad].tolist()}",
                    )

                    # This condition is REQUIRED for stay/advance DP:
                    # to consume T1 tokens, need at least T1 frames.
                    impossible = audio_len < text_len
                    if impossible.any():
                        ib = impossible.nonzero().squeeze(-1)
                        debug_print(
                            debug,
                            f"[DPAligner] IMPOSSIBLE (audio_len < text_len) idx: {ib.tolist()}",
                        )
                        debug_print(
                            debug, f"[DPAligner] audio_len: {audio_len[ib].tolist()}"
                        )
                        debug_print(
                            debug, f"[DPAligner] text_len : {text_len[ib].tolist()}"
                        )

                    if fail_fast:
                        raise RuntimeError(
                            "logZ is not finite -> no valid monotonic path (mask or lengths issue)."
                        )

        return log_alpha_all, logZ, last_t, last_p, text_len, audio_len

    def backward_sum(
        self,
        logp: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
        last_t: torch.Tensor,
        last_p: torch.Tensor,
        debug: bool = False,
    ):
        """
        Backward DP in log-space.
        IMPORTANT: initialize beta at each sample's last valid frame (last_t), not at T2-1.
        """
        B, T2, T1 = logp.shape
        device = logp.device
        dtype = logp.dtype
        neg = neg_inf(dtype)

        log_beta = torch.full((B, T1), neg, device=device, dtype=dtype)
        log_beta.scatter_(
            1, last_p.unsqueeze(1), torch.zeros((B, 1), device=device, dtype=dtype)
        )

        log_beta_all = torch.full((B, T2, T1), neg, device=device, dtype=dtype)

        # Place the terminal condition at last_t for each sample
        log_beta_all[torch.arange(B, device=device), last_t, :] = log_beta

        for t in range(T2 - 2, -1, -1):
            nxt = log_beta  # beta_{t+1} in "packed" form

            # stay: p -> p
            stay = logp[:, t + 1, :] + nxt
            # advance: p -> p+1 (so beta_t[p] depends on beta_{t+1}[p+1])
            adv = logp[:, t + 1, 1:] + nxt[:, 1:]
            adv = torch.cat(
                [adv, torch.full((B, 1), neg, device=device, dtype=dtype)], dim=1
            )

            log_beta_new = torch.logsumexp(torch.stack([stay, adv], dim=0), dim=0)
            log_beta_new = log_beta_new.masked_fill(~text_mask, neg)

            # Only propagate if frame t+1 is valid
            m = audio_mask[:, t + 1].unsqueeze(1)
            log_beta = torch.where(m, log_beta_new, nxt)

            log_beta_all[:, t, :] = log_beta

        if debug:
            with torch.no_grad():
                ok = safe_isfinite(log_beta_all)
                debug_print(debug, f"[DPAligner] log_beta_all finite: {ok}")
                if not ok:
                    raise RuntimeError("Non-finite log_beta_all")

        return log_beta_all

    def posterior(
        self, log_alpha_all, log_beta_all, logZ, audio_mask, text_mask, debug=False
    ):
        # NOTE: logZ qui non serve per gamma; lo lasciamo per align_loss.
        dtype = log_alpha_all.dtype
        neg = neg_inf(dtype)

        # valid cells
        valid = audio_mask.unsqueeze(-1) & text_mask.unsqueeze(1)  # [B,T2,T1]

        # unnormalized log posterior
        log_post = log_alpha_all + log_beta_all
        log_post = log_post.masked_fill(~valid, neg)

        if debug:
            with torch.no_grad():
                if torch.isnan(log_post).any():
                    print("[DPAligner] log_post has NaN")
                    # Debug mirato: capire se nasce in alpha o beta
                    print("  alpha NaN:", torch.isnan(log_alpha_all).any().item())
                    print("  beta  NaN:", torch.isnan(log_beta_all).any().item())
                    # Trova un indice esempio
                    idx = torch.isnan(log_post).nonzero()[0].tolist()
                    b, t, p = idx
                    print("  example idx b,t,p:", idx)
                    print(
                        "  alpha:",
                        log_alpha_all[b, t, p].item(),
                        "beta:",
                        log_beta_all[b, t, p].item(),
                    )
                    raise RuntimeError("NaN in log_post")

        # normalized posterior per-frame (softmax stabile)
        # softmax gestisce bene valori molto negativi e -inf
        gamma = torch.softmax(log_post, dim=-1)

        # (opzionale) ripulisci padding: su frame padding gamma deve essere 0
        gamma = gamma * audio_mask.unsqueeze(-1).to(gamma.dtype)

        if debug:
            with torch.no_grad():
                if torch.isnan(gamma).any():
                    print("[DPAligner] gamma has NaN AFTER softmax (unexpected)")
                    raise RuntimeError("NaN in gamma")

        return gamma

    def forward(
        self,
        frames: torch.Tensor,
        text_emb: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
        debug: bool = False,
        fail_fast: bool = True,
        return_debug: bool = False,
    ):
        """
        frames:    [B,T2,D]
        text_emb:  [B,T1,Dtext]
        audio_mask:[B,T2] True=valid
        text_mask: [B,T1] True=valid
        """
        # Basic mask sanity
        if debug:
            self._debug_masks(audio_mask, text_mask, debug=True)

        # Critical feasibility check for stay/advance DP
        # Need at least as many valid frames as valid tokens.
        # If not, logZ will be -inf. We'll surface it clearly.
        with torch.no_grad():
            audio_len = audio_mask.sum(1)
            text_len = text_mask.sum(1)
            impossible = audio_len < text_len
            if debug and impossible.any():
                idx = impossible.nonzero().squeeze(-1).tolist()
                debug_print(
                    True,
                    f"[DPAligner] IMPOSSIBLE samples (audio_len < text_len): {idx}",
                )
                debug_print(
                    True, f"[DPAligner] audio_len: {audio_len[impossible].tolist()}"
                )
                debug_print(
                    True, f"[DPAligner] text_len : {text_len[impossible].tolist()}"
                )

        # 1) emissions
        logp = self.compute_logp(frames, text_emb, text_mask, debug=debug)

        # 2) forward DP
        log_alpha_all, logZ, last_t, last_p, text_len_i, audio_len_i = self.forward_sum(
            logp, audio_mask, text_mask, debug=debug, fail_fast=fail_fast
        )

        # 3) backward DP
        log_beta_all = self.backward_sum(
            logp, audio_mask, text_mask, last_t=last_t, last_p=last_p, debug=debug
        )

        # 4) posterior
        gamma = self.posterior(
            log_alpha_all,
            log_beta_all,
            logZ,
            audio_mask,
            text_mask,
            debug=debug,
        )  # [B,T2,T1]

        # 5) durations
        durations = gamma.sum(dim=1) * text_mask.to(gamma.dtype)  # [B,T1]

        # 6) pooling (token->frame weights)
        w = gamma.transpose(1, 2)  # [B,T1,T2]
        w = w * audio_mask.unsqueeze(1).to(w.dtype)
        denom_w = w.sum(dim=-1, keepdim=True)
        w = torch.where(denom_w > 0, w / (denom_w + self.eps), torch.zeros_like(w))
        pooled = torch.bmm(w, frames)  # [B,T1,D]

        # 7) unpool context for decoder
        context = torch.bmm(gamma.to(pooled.dtype), pooled)  # [B,T2,D]

        # 8) align loss normalized per frame
        audio_len_f = audio_mask.sum(dim=1).clamp(min=1).to(logZ.dtype)  # [B]
        align_loss = (-(logZ) / audio_len_f).mean()

        if debug:
            with torch.no_grad():
                debug_print(
                    debug, f"[DPAligner] align_loss(per-frame)={align_loss.item():.4f}"
                )
                debug_print(
                    debug,
                    f"[DPAligner] durations min/max={durations.min().item():.4f}/{durations.max().item():.4f}",
                )
                # Check for NaNs in outputs
                for name, t in [
                    ("gamma", gamma),
                    ("pooled", pooled),
                    ("context", context),
                    ("durations", durations),
                ]:
                    if torch.isnan(t).any():
                        debug_print(True, f"[DPAligner] NaN in {name}")
                        if fail_fast:
                            raise RuntimeError(f"NaN in {name}")

        dbg: Optional[Dict[str, Any]] = None
        if return_debug:
            dbg = dict(
                logZ=logZ.detach().float().cpu(),
                audio_len=audio_len_i.detach().cpu(),
                text_len=text_len_i.detach().cpu(),
                last_t=last_t.detach().cpu(),
                last_p=last_p.detach().cpu(),
                impossible=(audio_len_i < text_len_i).detach().cpu(),
            )
        return gamma, pooled, context, durations, align_loss, dbg


# -----------------------------------------------------------------------------
# High-level Aligner module (adds text pos enc + final projection)
# -----------------------------------------------------------------------------
class Aligner(nn.Module):
    def __init__(
        self,
        attn_dim: int,
        text_dim: int,
        audio_dim: int,
        num_embeddings: int,
        vocab_path: str = "data/vocab.json",
        parsing_mode: str = "phoneme",
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim

        self.phonemes_embeddings = PhonemesEmbeddings(num_embeddings, text_dim)
        self.phoneme_vocab = PhonemeVocab(vocab_path, parsing_mode=parsing_mode)

        self.dp_aligner = MonotonicDPAligner(
            audio_dim=audio_dim, text_dim=text_dim, attn_dim=attn_dim
        )

        self.layernorm_mel = nn.LayerNorm(audio_dim)
        self.final_layer = nn.Linear(text_dim + audio_dim, audio_dim)

    def generate_pos_encoding(
        self, max_len: int, d_model: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        ).to(device=device, dtype=torch.float32)

        pe = torch.zeros(max_len, d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe.to(device=device, dtype=dtype)

    def forward(
        self,
        mels: torch.FloatTensor,  # [B,T2,D]
        phonemes: List[str],
        mels_mask: torch.BoolTensor,  # MUST be True=valid frames for DP
        debug: bool = True,
        fail_fast: bool = True,
        return_debug: bool = False,
    ):
        # Normalize audio tokens (optional but often helps)
        mels = self.layernorm_mel(mels)

        # Text ids + mask
        phonemes_ids, phonemes_mask = self.phoneme_vocab(
            phonemes, device=mels.device
        )  # mask True=valid
        phonemes_emb = self.phonemes_embeddings(phonemes_ids)

        # Add text positional encoding
        phonemes_emb = phonemes_emb + self.generate_pos_encoding(
            max_len=phonemes_ids.size(1),
            d_model=self.text_dim,
            device=mels.device,
            dtype=mels.dtype,
        )

        # DP align
        gamma, pooled, context, durations, align_loss, dbg = self.dp_aligner(
            frames=mels,
            text_emb=phonemes_emb,
            audio_mask=mels_mask,
            text_mask=phonemes_mask,
            debug=debug,
            fail_fast=fail_fast,
            return_debug=return_debug,
        )
        print(durations[0, :20])

        # Produce token-level output (phoneme-length) if you want compressed sequence
        final = torch.cat([phonemes_emb, pooled], dim=-1)
        final = self.final_layer(final)

        # Return also ~phonemes_mask if you expect padding_mask True=padding elsewhere
        return final, durations, gamma, align_loss, ~phonemes_mask


"""
HOW TO USE DEBUG (examples)

1) During training/inference, call:
   final, durations, gamma, align_loss, pad_mask_tok, context, dbg = aligner(
       mels=z, phonemes=phonemes, mels_mask=audio_valid_mask, debug=True, fail_fast=True, return_debug=True
   )

2) If it crashes, you will see:
   - whether audio_mask/text_mask are inverted
   - which samples have logZ=-inf
   - whether audio_len < text_len (impossible for stay/advance DP)

COMMON ROOT CAUSES:
- mels_mask is True for padding (invert it!)
- T2_valid < T1_valid (too many phonemes for too few audio tokens at that resolution)
- some sample has audio_len==0 or text_len==0 due to preprocessing/padding bugs
"""
