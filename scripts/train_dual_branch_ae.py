from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import safetensors.torch
from tqdm.auto import tqdm

# Add root to sys.path to access dicodec modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dicodec.modules.builder import load_pretrained_model


# --- PROVIDED CLASSES ---

@dataclass
class QuantizerOutput:
    """Normalized shape expected from the external quantizer."""
    z_q: torch.Tensor
    loss: torch.Tensor
    indices: torch.Tensor | None = None


@dataclass
class DualBranchAEOutput:
    z_rec: torch.Tensor
    z_sem_q: torch.Tensor
    z_pros_enc: torch.Tensor
    quantizer_loss: torch.Tensor
    consistency_loss: torch.Tensor
    indices: torch.Tensor | None


class ResNetBlock1D(nn.Module):
    """Pre-activation residual block over [B, D, T]."""
    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.act = nn.SiLU()

    @staticmethod
    def _normalize(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.conv1(self.act(self._normalize(x, self.norm1)))
        if valid_mask is not None:
            x = x * valid_mask
        x = self.conv2(self.act(self._normalize(x, self.norm2)))
        x = residual + x
        if valid_mask is not None:
            x = x * valid_mask
        return x


class ResNetStack1D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
    ):
        super().__init__()
        dilations = dilations or [1] * num_blocks
        assert len(dilations) == num_blocks
        self.blocks = nn.ModuleList(
            ResNetBlock1D(dim, kernel_size=kernel_size, dilation=d) for d in dilations
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, valid_mask=valid_mask)
        return x


class SemanticEncoder(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, quant_dim: int, num_blocks: int = 2, kernel_size: int = 3,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.resnet = ResNetStack1D(hidden_dim, num_blocks, kernel_size=kernel_size)
        self.out_proj = nn.Conv1d(hidden_dim, quant_dim, kernel_size=1)

    def forward(self, z_sem: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = z_sem.transpose(1, 2)
        channel_mask = None
        if valid_mask is not None:
            channel_mask = valid_mask.transpose(1, 2)
            x = x * channel_mask
        x = self.in_proj(x)
        if channel_mask is not None:
            x = x * channel_mask
        x = self.out_proj(self.resnet(x, valid_mask=channel_mask))
        if valid_mask is not None:
            x = x * channel_mask
        return x.transpose(1, 2)



class Decoder(nn.Module):
    def __init__(
        self, sem_dim: int, hidden_dim: int, out_dim: int, num_blocks: int = 2, kernel_size: int = 3,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(sem_dim, hidden_dim, kernel_size=1)
        self.resnet = ResNetStack1D(hidden_dim, num_blocks, kernel_size=kernel_size)
        self.out_proj = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, z_sem_q: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = z_sem_q.transpose(1, 2)
        channel_mask = None
        if valid_mask is not None:
            channel_mask = valid_mask.transpose(1, 2)
            x = x * channel_mask
        x = self.in_proj(x)
        if channel_mask is not None:
            x = x * channel_mask
        x = self.out_proj(self.resnet(x, valid_mask=channel_mask))
        if valid_mask is not None:
            x = x * channel_mask
        return x.transpose(1, 2)


class DualBranchQuantizedAE(nn.Module):
    def __init__(
        self,
        dim: int,
        quantizer: nn.Module,
        quant_dim: int | None = None,
        hidden_dim: int = 256,
        num_sem_blocks: int = 2,
        num_dec_blocks: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        quant_dim = quant_dim or dim

        self.sem_encoder = SemanticEncoder(
            in_dim=dim, hidden_dim=hidden_dim, quant_dim=quant_dim,
            num_blocks=num_sem_blocks, kernel_size=kernel_size,
        )
        self.quantizer = quantizer
        self.decoder = Decoder(
            sem_dim=quant_dim, hidden_dim=hidden_dim, out_dim=dim,
            num_blocks=num_dec_blocks, kernel_size=kernel_size,
        )

    def _run_quantizer(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> QuantizerOutput:
        # Pass valid_mask to the wrapper so losses can be masked
        out = self.quantizer(x, valid_mask)
        if isinstance(out, QuantizerOutput):
            return out
        if isinstance(out, tuple):
            z_q, loss = out[0], out[1]
            indices = out[2] if len(out) > 2 else None
            return QuantizerOutput(z_q=z_q, loss=loss, indices=indices)
        
        # specific handler for VectorQuantizer that returns (toks, codes)
        if isinstance(out, tuple) and len(out) == 2:
            toks, codes = out
            return QuantizerOutput(z_q=codes, loss=torch.tensor(0.0, device=x.device), indices=toks)

        raise TypeError(
            f"Unrecognized quantizer output type {type(out)}."
        )

    def forward(
        self,
        z: torch.Tensor,
        z_sem_target: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> DualBranchAEOutput:
        if valid_mask is not None and valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(-1)

        sem_enc = self.sem_encoder(z, valid_mask=valid_mask)
        consistency_loss = z.new_zeros(())

        if z_sem_target is not None:
            with torch.no_grad():
                target_enc = self.sem_encoder(z_sem_target, valid_mask=valid_mask)

            joint_enc = torch.cat([sem_enc, target_enc], dim=0)
            joint_mask = (
                torch.cat([valid_mask, valid_mask], dim=0)
                if valid_mask is not None
                else None
            )
            quant_out = self._run_quantizer(joint_enc, valid_mask=joint_mask)
            z_sem_q, target_q = quant_out.z_q.chunk(2, dim=0)
            indices = (
                quant_out.indices.chunk(2, dim=0)[0]
                if quant_out.indices is not None
                else None
            )

            if valid_mask is None:
                consistency_loss = F.mse_loss(z_sem_q, target_q.detach())
            else:
                valid_elements = (
                    valid_mask.sum() * z_sem_q.shape[-1]
                ).clamp_min(1.0)
                consistency_loss = F.mse_loss(
                    z_sem_q * valid_mask,
                    target_q.detach() * valid_mask,
                    reduction="sum",
                ) / valid_elements
        else:
            quant_out = self._run_quantizer(sem_enc, valid_mask=valid_mask)
            z_sem_q = quant_out.z_q
            indices = quant_out.indices

        z_rec = self.decoder(z_sem_q, valid_mask=valid_mask)

        return DualBranchAEOutput(
            z_rec=z_rec,
            z_sem_q=z_sem_q,
            z_pros_enc=torch.zeros_like(z_sem_q), # dummy to avoid breaking typing

            quantizer_loss=quant_out.loss,
            consistency_loss=consistency_loss,
            indices=indices,
        )


# --- UTILS TO LOAD PRETRAINED ENCODER (REMOVED: using Dicodec natively) ---

# --- DATASET & TRAIN LOGIC ---

def get_dataloader(data_dir: Path, target_sr: int, batch_size: int = 8, max_samples: int = 48000):
    from datasets import load_dataset, Audio
    
    parquet_files = sorted(str(path) for path in data_dir.rglob("*.parquet") if not path.name.startswith("._"))
    dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    if "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sr))
    
    def collate_fn(batch):
        wavs = []
        for item in batch:
            wav = torch.as_tensor(item["audio"]["array"], dtype=torch.float32)
            if wav.dim() > 1:
                wav = wav.mean(dim=0)
            # Clip to max_samples for memory bounds
            if wav.shape[0] > max_samples:
                wav = wav[:max_samples]
            wav = wav / (wav.abs().max() + 1e-8)
            wavs.append(wav)
        return wavs
        
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

# extract_features (REMOVED: using Dicodec natively)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/workspace/MelCausalVAE/checkpoints/18-denc128-novq"))
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/datasets/librispeech-aligned/train_clean_100"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--quantizer", type=str, choices=["vq_ema", "bsq", "std_vq", "fsq"], default="vq_ema", help="Quantizer to use for z_sem")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-batches", type=int, default=None, help="Limit number of batches for testing")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum total number of training steps")
    
    # Model Architecture
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension for the dual branch AE")
    parser.add_argument("--codebook-size", type=int, default=1024, help="Number of codes in the codebook")
    parser.add_argument("--num-sem-blocks", type=int, default=4, help="Number of residual blocks in SemanticEncoder")
    parser.add_argument("--num-dec-blocks", type=int, default=4, help="Number of residual blocks in Decoder")
    
    # Regularizations
    parser.add_argument("--commit-weight", type=float, default=0.25)
    parser.add_argument("--ema-decay", type=float, default=0.85)
    parser.add_argument("--ema-eps", type=float, default=1e-5)
    parser.add_argument("--reset-dead-codes", action="store_true")
    parser.add_argument("--reset-every-forward", type=int, default=10)
    parser.add_argument("--entropy-loss-weight", type=float, default=0.01)
    parser.add_argument("--entropy-temperature", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading base Dicodec from {args.checkpoint_dir}...")
    dicodec = load_pretrained_model(str(args.checkpoint_dir))
    dicodec.to(device)
    dicodec.eval()
    
    print(f"Initializing DualBranchQuantizedAE with {args.quantizer} quantizer...")
    latent_dim = dicodec.encoder.config.latent_dim
    quant_dim = latent_dim # usually same as latent_dim
    
    class QuantizerWrapper(nn.Module):
        def __init__(self, quantizer_module, quantizer_type, args):
            super().__init__()
            self.quantizer_module = quantizer_module
            self.quantizer_type = quantizer_type
            self.args = args
            
        def forward(self, x, valid_mask=None):
            if valid_mask is None:
                valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            else:
                valid = valid_mask.squeeze(-1).bool()

            x_valid = x[valid]
            if x_valid.numel() == 0:
                return (
                    torch.zeros_like(x),
                    x.new_zeros(()),
                    torch.zeros(x.shape[:2], dtype=torch.long, device=x.device),
                )

            toks_valid, codes_valid = self.quantizer_module(x_valid)

            if self.quantizer_type == "vq_ema":
                commit_loss = F.mse_loss(codes_valid.detach(), x_valid)
                quantizer_loss = self.args.commit_weight * commit_loss

            elif self.quantizer_type == "bsq":
                probs = torch.sigmoid(x_valid / self.args.entropy_temperature)
                avg_probs = probs.mean(dim=0)
                negative_entropy = torch.mean(avg_probs * torch.log(avg_probs + 1e-5) + (1 - avg_probs) * torch.log(1 - avg_probs + 1e-5))
                centered = probs - avg_probs
                covariance = centered.transpose(0, 1) @ centered / max(probs.shape[0], 1)
                off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
                entropy_loss = negative_entropy + off_diagonal.square().mean()
                quantizer_loss = self.args.entropy_loss_weight * entropy_loss

            elif self.quantizer_type == "std_vq":
                commit_loss = F.mse_loss(codes_valid.detach(), x_valid)
                codebook_loss = F.mse_loss(codes_valid, x_valid.detach())
                quantizer_loss = codebook_loss + self.args.commit_weight * commit_loss

            elif self.quantizer_type == "fsq":
                quantizer_loss = x.new_zeros(())

            if self.quantizer_type == "fsq":
                codes_valid_ste = codes_valid
            else:
                codes_valid_ste = x_valid + (codes_valid - x_valid).detach()
            codes_ste = torch.zeros_like(x)
            codes_ste[valid] = codes_valid_ste
            toks = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
            toks[valid] = toks_valid.long()
            return codes_ste, quantizer_loss, toks

    if args.quantizer == "vq_ema":
        from dicodec.modules.quantizer.vq_ema import EMAVectorQuantizer
        base_quantizer = EMAVectorQuantizer(
            dim=quant_dim, 
            codebook_size=args.codebook_size, 
            decay=args.ema_decay,
            eps=args.ema_eps,
            reset_dead_codes=args.reset_dead_codes,
            reset_every_forward=args.reset_every_forward
        )
    elif args.quantizer == "bsq":
        from dicodec.modules.quantizer.bsq import BinarySphericalQuantizer
        base_quantizer = BinarySphericalQuantizer(codebook_size=args.codebook_size)
        quant_dim = base_quantizer.dim
    elif args.quantizer == "std_vq":
        from dicodec.modules.quantizer.std_vq import StandardVectorQuantizer
        base_quantizer = StandardVectorQuantizer(dim=quant_dim, codebook_size=args.codebook_size)
    elif args.quantizer == "fsq":
        from dicodec.modules.quantizer.fsq import FiniteScalarQuantizer
        base_quantizer = FiniteScalarQuantizer(codebook_size=args.codebook_size)
        quant_dim = base_quantizer.dim
        
    quantizer_wrapper = QuantizerWrapper(base_quantizer, args.quantizer, args).to(device)

    model = DualBranchQuantizedAE(
        dim=latent_dim,
        quantizer=quantizer_wrapper,
        quant_dim=quant_dim,
        hidden_dim=args.hidden_dim,
        num_sem_blocks=args.num_sem_blocks,
        num_dec_blocks=args.num_dec_blocks,
    ).to(device)
    
    # Enable training mode for AE
    model.train()
    
    sem_enc_params = sum(p.numel() for p in model.sem_encoder.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"SemanticEncoder Trainable Parameters: {sem_enc_params:,}")
    print(f"Decoder Trainable Parameters: {dec_params:,}")
    print(f"Total Model Trainable Parameters: {sem_enc_params + dec_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    dataloader = get_dataloader(args.data_dir, target_sr=dicodec.config.sample_rate, batch_size=args.batch_size)
    
    global_step = 0
    cumulative_code_counts = torch.zeros(args.codebook_size, dtype=torch.long)
    print("Starting training loop...")
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        
        for batch_idx, wavs in enumerate(pbar):
            if args.limit_batches and batch_idx >= args.limit_batches:
                break
            if args.max_steps and global_step >= args.max_steps:
                break
                
            wavs = [w.to(device) for w in wavs]
            audios_srs = [(w, dicodec.config.sample_rate) for w in wavs]
            
            with torch.no_grad():
                enc_features, enc_padding_mask, _, _ = dicodec.extract_features(audios_srs)
                encoder_output = dicodec.encode(enc_features, enc_padding_mask)
                z = encoder_output.z
                padding_mask = encoder_output.padding_mask
                
                attrs = dicodec.encode_attributes(z, padding_mask=padding_mask)
                z_sem, z_pros, z_mean = attrs.z_sem, attrs.z_pros, attrs.z_mean

            optimizer.zero_grad()
            ae_out = model(
                z,
                z_sem_target=z_sem,
                valid_mask=~padding_mask if padding_mask is not None else None,
            )
            
            # Mask out padding for reconstruction loss
            if padding_mask is not None:
                valid_mask = (~padding_mask).unsqueeze(-1)
                valid_elems = (valid_mask.sum() * z_sem.shape[-1]).clamp_min(1.0)
                rec_loss = F.mse_loss(ae_out.z_rec * valid_mask, z_sem * valid_mask, reduction='sum') / valid_elems
            else:
                rec_loss = F.mse_loss(ae_out.z_rec, z_sem)
            
            loss = (
                rec_loss
                + ae_out.quantizer_loss
                + args.consistency_weight * ae_out.consistency_loss
            )
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Compute codebook utilization and perplexity
            postfix_dict = {
                "loss": f"{loss.item():.4f}",
                "rec_loss": f"{rec_loss.item():.4f}",
                "cons": f"{ae_out.consistency_loss.item():.4f}",
            }
            if ae_out.indices is not None:
                codebook_size = args.codebook_size
                if padding_mask is not None:
                    # Flatten only valid indices
                    valid_idx = (~padding_mask).view(-1)
                    toks_flat = ae_out.indices.view(-1)[valid_idx]
                else:
                    toks_flat = ae_out.indices.view(-1)
                    
                if toks_flat.numel() > 0:
                    batch_counts = torch.bincount(
                        toks_flat.detach().cpu(), minlength=codebook_size
                    )
                    cumulative_code_counts += batch_counts
                    batch_utilization = (
                        (batch_counts > 0).sum().item() / codebook_size * 100.0
                    )
                    utilization = (
                        (cumulative_code_counts > 0).sum().item()
                        / codebook_size
                        * 100.0
                    )

                    probs = cumulative_code_counts.float() / cumulative_code_counts.sum()
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    perplexity = torch.exp(entropy)

                    postfix_dict["batch_util%"] = f"{batch_utilization:.1f}"
                    postfix_dict["util_all%"] = f"{utilization:.1f}"
                    postfix_dict["ppl_all"] = f"{perplexity.item():.1f}"
                    
            pbar.set_postfix(postfix_dict)
        
        # Save model checkpoint at the end of each epoch
        save_dir = Path("checkpoints/dual_branch_ae")
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"model_epoch_{epoch+1}_{args.quantizer}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        if args.max_steps and global_step >= args.max_steps:
            print(f"Reached max steps ({args.max_steps}). Stopping training.")
            break
            
if __name__ == "__main__":
    main()
