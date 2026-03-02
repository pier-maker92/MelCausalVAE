import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
import torchaudio
import torch.nn as nn
from einops import rearrange
from dataclasses import dataclass
from typing import Optional, List
from bournemouth_aligner import PhonemeTimestampAligner

from modules.flash_attn_encoder import FlashTransformerEncoder
from modules.resnet import LinearResNet as ResNet
from modules.downsampler import DownSampler

# from modules.similarity import Aligner, SimilarityUpsamplerBatch

# from modules.Vits.aligner import Aligner as VitsAligner
# from modules.Vits.aligner import AlignerConfig


@dataclass
class ConvformerOutput:
    z: torch.FloatTensor
    kl_loss: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.BoolTensor] = None
    mu: Optional[torch.FloatTensor] = None
    align_loss: Optional[torch.FloatTensor] = None
    durations: Optional[torch.LongTensor] = None
    z_upsampled: Optional[torch.FloatTensor] = None
    upsampled_padding_mask: Optional[torch.BoolTensor] = None


@dataclass
class SigmaVAEencoderConfig:
    logvar_layer: bool = True
    kl_loss_weight: float = 1e-3
    target_std: Optional[float] = None
    use_sofplus: Optional[bool] = None
    kl_loss_warmup_steps: int = 1000
    semantic_regulation: bool = True
    freeze_encoder: bool = False
    semantic_kl_loss_weight: Optional[float] = None


@dataclass
class ConvformerEncoderConfig(SigmaVAEencoderConfig):
    compress_factor_C: int = 8
    tf_heads: int = 8
    tf_layers: int = 4
    drop_p: float = 0.1
    latent_dim: int = 64
    n_residual_blocks: int = 3
    num_embeddings: int = 100
    embedding_dim: int = 80
    phoneme_parsing_mode: str = "phoneme"
    vocab_path: str = "data/vocab.json"
    use_aligner: bool = False
    threshold: float = 0.95
    force_downsample: bool = False
    split_route: bool = False
    semantic_dim: Optional[int] = None


class SigmaVAEEncoder(nn.Module):
    def __init__(self, config: SigmaVAEencoderConfig):
        super().__init__()
        self.config = config
        self.std_activation = (
            nn.Softplus() if self.config.use_sofplus else nn.Identity()
        )
        self.kl_loss_weight = float(config.kl_loss_weight)
        if self.config.semantic_kl_loss_weight is not None:
            self.semantic_kl_loss_weight = float(config.semantic_kl_loss_weight)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.semantic_regulator = InterpolateRegulator(
        #     depth=2, in_channels=1024, channels=256, out_channels=config.latent_dim
        # )

    def forward(self, **kwargs):
        pass

    def reparameterize(
        self, mu: torch.FloatTensor, logvar: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        eps = torch.randn_like(mu)
        if logvar is None:
            std = self.sample_scalar_std(mu)
            while std.dim() < mu.dim():
                std = std.unsqueeze(-1)
        else:
            std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def kl_divergence(
        self,
        mu: torch.FloatTensor,
        logvar: Optional[torch.FloatTensor],
        padding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        if logvar is None:
            # Compute in fp32 for numerical stability
            mu_valid = mu[~padding_mask].float()
            return torch.nn.functional.mse_loss(
                mu_valid, torch.zeros_like(mu_valid)
            ).to(mu.dtype)
        # Compute KL divergence in fp32 for numerical stability with fp16
        mu_valid = mu[~padding_mask].float()
        logvar_valid = logvar[~padding_mask].float()
        kl = -0.5 * torch.sum(1 + logvar_valid - mu_valid.pow(2) - logvar_valid.exp())
        return kl.to(mu.dtype)

    def sample_scalar_std(self, mu: torch.FloatTensor) -> torch.FloatTensor:
        return self.std_activation(
            torch.randn(mu.shape[0], mu.shape[1], dtype=mu.dtype, device=mu.device)
            * self.config.target_std
        )

    def _resize_padding_mask(
        self, padding_mask: torch.BoolTensor, target_length: int
    ) -> torch.BoolTensor:
        padding_mask = (
            torch.nn.functional.interpolate(
                padding_mask.unsqueeze(1).float(),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            > 0.5  # Use threshold instead of .bool()
        )
        return padding_mask

    def get_kl_cosine_schedule(self, step, kl_loss_weight):
        """
        Returns the scaled KL loss weight following a cosine schedule
        ranging from 0 to self.kl_loss_weight over total_steps.
        Once step surpasses total_steps, stays at self.kl_loss_weight.
        """
        if self.config.kl_loss_warmup_steps == 0:
            return kl_loss_weight
        if step >= self.config.kl_loss_warmup_steps:
            return kl_loss_weight
        # Cosine schedule: start at 0, increase to kl_loss_weight in total_steps
        cosine = 0.5 * (1 - math.cos(math.pi * step / self.config.kl_loss_warmup_steps))
        return kl_loss_weight * cosine


class CausalTransformerTail(nn.Module):
    def __init__(self, d_model=512, nheads=8, nlayers=4, drop_p=0.1):
        super().__init__()
        self.enc = FlashTransformerEncoder(
            d_model=d_model, nhead=nheads, nlayers=nlayers, drop_p=drop_p
        )

    def forward(self, tokens):  # [B, T_tok, d_model]
        return self.enc(tokens, causal=True)


# TransformerEncoder with causal masking via is_causal for left-only attention [web:84][web:92]


class ConvformerEncoder(SigmaVAEEncoder):
    def __init__(self, config: ConvformerEncoderConfig):
        super().__init__(config)

        compress_factor_C = config.compress_factor_C
        tf_heads = config.tf_heads
        tf_layers = config.tf_layers
        drop_p = config.drop_p
        latent_dim = config.latent_dim
        n_residual_blocks = config.n_residual_blocks

        if config.force_downsample:
            self.downsampler = DownSampler(
                d_in=100,
                d_hidden=512,
                d_out=512,
                compress_factor=compress_factor_C,
                causal=True,
            )
        else:
            self.downsampler = ResNet(
                in_dim=100,
                hidden_dim=512,
                output_dim=512,
                num_blocks=n_residual_blocks,
            )

        # Causal Transformer tail operating on tokens of size 512
        self.transformer = CausalTransformerTail(
            d_model=512, nheads=tf_heads, nlayers=tf_layers, drop_p=drop_p
        )

        self.mu = nn.Linear(512, latent_dim)
        if config.logvar_layer:
            self.logvar = nn.Linear(512, latent_dim)

        if config.split_route:
            self.semantic_mu = nn.Linear(512, config.semantic_dim)
            self.semantic_logvar = nn.Linear(512, config.semantic_dim)
            self.final_projection = nn.Linear(
                latent_dim + config.semantic_dim, latent_dim
            )

        if config.use_aligner:
            aligner_config = AlignerConfig(z_dim=config.latent_dim, hidden_dim=512)
            self.aligner = VitsAligner(aligner_config)
            self.upsampler = SimilarityUpsamplerBatch()

        if config.freeze_encoder:
            for name, param in self.named_parameters():
                if name.split(".")[0] in ["mu", "logvar", "aligner", "upsampler"]:
                    continue
                param.requires_grad = False
        self.extractor = PhonemeTimestampAligner(
            preset="en-us",
            duration_max=30,
            device="cuda",
            enforce_all_targets=True,
            silence_anchors=0,
            boundary_softness=2,
        )
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=24000, new_freq=16000
        ).to(self.mu.weight.device)

    def build_alignment_matrix(self, all_frames_assorted):
        """
        Build binary monotonic alignment matrices for a batch of utterances.

        Each matrix's columns are derived from the contiguous runs in the
        corresponding *frames_assorted*, including runs of -1 (gaps).
        Every row has exactly one 1.

        Args:
            all_frames_assorted: list of lists — each inner list contains
                                phoneme IDs per frame (length T_i).
                                -1 represents an unaligned gap / silence.

        Returns:
            results: list of (alignment_matrix, segment_ids) tuples.
                    alignment_matrix: torch.Tensor (T_i, N_i), float32, binary.
                    segment_ids:      list[int] of length N_i.
        """
        results = []
        for frames_assorted in all_frames_assorted:
            T = len(frames_assorted)
            if T == 0:
                results.append((torch.empty((0, 0), dtype=torch.float32), []))
                continue

            # Identify contiguous runs (same logic as compress_frames)
            segments = []  # (phoneme_id, start_frame, end_frame)
            current_id = frames_assorted[0]
            start = 0
            for i in range(1, T):
                if frames_assorted[i] != current_id:
                    segments.append((current_id, start, i))
                    current_id = frames_assorted[i]
                    start = i
            segments.append((current_id, start, T))

            N = len(segments)
            alignment = torch.zeros((T, N), dtype=torch.float32)
            segment_ids = []

            for col, (ph_id, s, e) in enumerate(segments):
                alignment[s:e, col] = 1.0
                segment_ids.append(ph_id)

            # results.append((alignment, segment_ids))
            results.append(alignment)

        return results

    def get_ctc_alignement(self, transcriptions, audios, frames_durations):
        batch_results = self.extractor.process_sentences_batch(transcriptions, audios)
        all_frames_assorted = []
        for result, total_frames in zip(batch_results, frames_durations):
            seg = result["segments"][0]
            # Mel spectrogram for this segment
            s = int(seg["start"] * self.extractor.resampler_sample_rate)
            e = int(seg["end"] * self.extractor.resampler_sample_rate)

            # Framewise assortment
            duration = seg["end"] - seg["start"]
            fps = total_frames / duration
            frames = self.extractor.framewise_assortment(
                aligned_ts=seg["phoneme_ts"],
                total_frames=total_frames,
                frames_per_second=fps,
                gap_contraction=75,
                select_key="phoneme_id",
            )
            all_frames_assorted.append(frames)
        alignements = self.build_alignment_matrix(all_frames_assorted)

        # Pad to (max_T, max_N) and stack into a batched tensor [B, max_T, max_N]
        max_T = max(a.shape[0] for a in alignements)
        max_N = max(a.shape[1] for a in alignements)
        padded = torch.zeros(len(alignements), max_T, max_N, dtype=torch.float32)
        phonemes_lengths = torch.tensor(
            [a.shape[1] for a in alignements], dtype=torch.long
        )
        for i, a in enumerate(alignements):
            padded[i, : a.shape[0], : a.shape[1]] = a

        phoneme_mask = torch.ones((len(alignements), max_N), dtype=torch.bool)
        for i, length in enumerate(phonemes_lengths):
            phoneme_mask[i, :length] = False

        return padded, phoneme_mask, padded.sum(1)

    def forward(
        self,
        x: torch.FloatTensor,
        phonemes: List[str],
        padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):  # x: [B, T, 100]

        target_T = (~padding_mask).sum(dim=1)
        original_padding_mask = padding_mask.clone()
        x, padding_mask = self.downsampler(x, padding_mask.bool())
        x = self.transformer(x)  # [B, T/C, 512]

        # resample to 16KHz and run CTC alignment
        # Wrap in autocast to handle dtype mismatch (model is bf16 but audio is fp32)
        # During training autocast is already active (no-op); during eval this provides it
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            audios = [
                self.resampler(audio[0]).unsqueeze(0) for audio in kwargs["audios_srs"]
            ]
            alignements, padding_mask, durations = self.get_ctc_alignement(
                kwargs["transcription"], audios, target_T
            )
        alignements = alignements.to(dtype=x.dtype, device=x.device)
        durations = durations.to(dtype=x.dtype, device=x.device)
        x = torch.bmm(alignements.permute(0, 2, 1), x)
        x = x / (durations.unsqueeze(-1) + 1e-8)
        phoneme_lengths = (~padding_mask).sum(dim=1)
        # durations = [d[: phoneme_lengths[i]].long() for i, d in enumerate(durations)]

        if self.config.split_route:
            raise NotImplementedError("split route not implemented yet")

        mu = self.mu(x)
        logvar = None
        if hasattr(self, "logvar"):
            logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        assert not torch.isnan(z).any(), "z contains nan after reparameterization"

        # get alignement
        align_loss = None

        kl_loss = None
        if kwargs.get("step", None) is not None:
            kl_loss = self.kl_divergence(
                mu, logvar, padding_mask
            ) * self.get_kl_cosine_schedule(
                kwargs["step"], kl_loss_weight=self.kl_loss_weight
            )

        z_upsampled, upsampled_padding_mask = (
            torch.bmm(alignements, z),
            original_padding_mask,
        )

        return ConvformerOutput(
            z=z,
            kl_loss=kl_loss,
            padding_mask=padding_mask,
            mu=mu,
            durations=durations,
            z_upsampled=z_upsampled,
            align_loss=align_loss,
            upsampled_padding_mask=upsampled_padding_mask,
        )


@torch.no_grad()
def forward_latent(model, x):
    model.eval()
    return model(x)  # returns mu: [B, T/C, latent_dim]


def test_time_causality_invariance(model):
    torch.manual_seed(0)
    B, T, F = 1, 64, 100

    x = torch.randn(B, T, F)
    y_ref = forward_latent(model, x)  # [B, T/C, latent_dim]

    for j in range(1, y_ref.shape[1]):
        t_max = j * model.config.compress_factor_C
        x_pert = x.clone()
        if t_max < T:
            x_pert[:, t_max:, :] = torch.randn_like(x_pert[:, t_max:, :]) * 5.0
        y_pert = forward_latent(model, x_pert)

        ok = torch.allclose(y_pert[:, j - 1, :], y_ref[:, j - 1, :], atol=1e-7, rtol=0)
        assert ok, f"Causality violated at output index {j}"


if __name__ == "__main__":
    config = ConvformerEncoderConfig(
        compress_factor_C=8,
        tf_heads=8,
        tf_layers=4,
        drop_p=0.1,
        latent_dim=64,
        n_residual_blocks=3,
    )
    model = ConvformerEncoder(config=config)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    test_time_causality_invariance(model)
    breakpoint()
    # print number of parameters
