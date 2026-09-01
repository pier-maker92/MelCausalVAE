from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dicodec.modules.semantic_quantizer_ae import SemanticQuantizerAE


@dataclass
class TrainQuantizerConfig:
    model_type: str
    checkpoint_dir: str
    data_dir: str
    input_source: str
    target_source: str
    quantizer_type: str
    num_codebooks: int
    num_embeddings: int
    codebook_size: int
    latent_dim: int
    quant_dim: int
    vq_dim: int
    hidden_dim: int
    num_sem_blocks: int
    num_dec_blocks: int
    kernel_size: int
    batch_size: int
    epochs: int
    lr: float
    max_steps: int | None
    commit_weight: float
    ema_decay: float
    ema_eps: float
    reset_dead_codes: bool
    reset_every_forward: int
    entropy_loss_weight: float
    entropy_temperature: float
    wandb_mode: str
    wandb_project: str
    wandb_run_name: str | None
    wandb_id: str | None


class QuantizerWrapper(nn.Module):
    def __init__(self, quantizer_module: nn.Module, quantizer_type: str, args):
        super().__init__()
        self.quantizer_module = quantizer_module
        self.quantizer_type = quantizer_type
        self.args = args

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None):
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
            negative_entropy = torch.mean(
                avg_probs * torch.log(avg_probs + 1e-5)
                + (1 - avg_probs) * torch.log(1 - avg_probs + 1e-5)
            )
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
        else:
            raise ValueError(f"Unsupported quantizer type: {self.quantizer_type}")

        if self.quantizer_type == "fsq":
            codes_valid_ste = codes_valid
        else:
            codes_valid_ste = x_valid + (codes_valid - x_valid).detach()

        codes_ste = torch.zeros_like(x)
        codes_ste[valid] = codes_valid_ste
        toks = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        toks[valid] = toks_valid.long()
        return codes_ste, quantizer_loss, toks


def get_dataloader(
    data_dir: Path,
    target_sr: int,
    batch_size: int = 8,
    max_samples: int = 48000,
):
    from datasets import Audio, load_dataset

    parquet_files = sorted(
        str(path)
        for path in data_dir.rglob("*.parquet")
        if not path.name.startswith("._")
    )
    dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    if "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sr))

    def collate_fn(batch):
        wavs = []
        for item in batch:
            wav = torch.as_tensor(item["audio"]["array"], dtype=torch.float32)
            if wav.dim() > 1:
                wav = wav.mean(dim=0)
            if wav.shape[0] > max_samples:
                wav = wav[:max_samples]
            wav = wav / (wav.abs().max() + 1e-8)
            wavs.append(wav)
        return wavs

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )


def build_quantizer(args, quant_dim: int):
    if args.quantizer == "vq_ema":
        from dicodec.modules.quantizer.vq_ema import EMAVectorQuantizer

        return (
            EMAVectorQuantizer(
                dim=quant_dim,
                codebook_size=args.codebook_size,
                decay=args.ema_decay,
                eps=args.ema_eps,
                reset_dead_codes=True,
                reset_every_forward=args.reset_every_forward,
            ),
            quant_dim,
        )
    if args.quantizer == "bsq":
        from dicodec.modules.quantizer.bsq import BinarySphericalQuantizer

        quantizer = BinarySphericalQuantizer(codebook_size=args.codebook_size)
        return quantizer, quantizer.dim
    if args.quantizer == "std_vq":
        from dicodec.modules.quantizer.std_vq import StandardVectorQuantizer

        return (
            StandardVectorQuantizer(dim=quant_dim, codebook_size=args.codebook_size),
            quant_dim,
        )
    if args.quantizer == "fsq":
        from dicodec.modules.quantizer.fsq import FiniteScalarQuantizer

        quantizer = FiniteScalarQuantizer(codebook_size=args.codebook_size)
        return quantizer, quantizer.dim
    raise ValueError(f"Unsupported quantizer: {args.quantizer}")


def save_checkpoint(
    model: SemanticQuantizerAE,
    output_dir: Path,
    config: TrainQuantizerConfig,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)


def make_config(args, latent_dim: int, quant_dim: int) -> TrainQuantizerConfig:
    return TrainQuantizerConfig(
        model_type="semantic_quantizer_ae",
        checkpoint_dir=str(args.checkpoint_dir),
        data_dir=str(args.data_dir),
        input_source=args.input_source,
        target_source="z_sem",
        quantizer_type=args.quantizer,
        num_codebooks=args.codebook_size,
        num_embeddings=args.codebook_size,
        codebook_size=args.codebook_size,
        latent_dim=latent_dim,
        quant_dim=quant_dim,
        vq_dim=quant_dim,
        hidden_dim=args.hidden_dim,
        num_sem_blocks=args.num_sem_blocks,
        num_dec_blocks=args.num_dec_blocks,
        kernel_size=args.kernel_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_steps=args.max_steps,
        commit_weight=args.commit_weight,
        ema_decay=args.ema_decay,
        ema_eps=args.ema_eps,
        reset_dead_codes=True,
        reset_every_forward=args.reset_every_forward,
        entropy_loss_weight=args.entropy_loss_weight,
        entropy_temperature=args.entropy_temperature,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_id=args.wandb_id,
    )


def maybe_init_wandb(config: TrainQuantizerConfig):
    if config.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Use --wandb-mode disabled or install the training extra."
        ) from exc

    return wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        id=config.wandb_id,
        resume="allow" if config.wandb_id else None,
        mode=config.wandb_mode,
        config=asdict(config),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train an external semantic quantizer for Dicodec."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "/scratch/piermel/MelCausalVAE/checkpoints/paper/baseline/18-denc128-novq"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/scratch/piermel/datasets/libritts-r/train"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-source", choices=["z", "z_sem"], default="z_sem")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--quantizer",
        choices=["vq_ema", "bsq", "std_vq", "fsq"],
        default="vq_ema",
        help="Quantizer type for the semantic latent.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every-steps", type=int, default=1000)

    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--num-sem-blocks", type=int, default=4)
    parser.add_argument("--num-dec-blocks", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)

    parser.add_argument("--commit-weight", type=float, default=0.25)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--ema-eps", type=float, default=1e-5)
    parser.add_argument("--reset-every-forward", type=int, default=10)
    parser.add_argument("--entropy-loss-weight", type=float, default=0.1)
    parser.add_argument("--entropy-temperature", type=float, default=1.0)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb-project", type=str, default="dicodec-quantizer")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)

    args = parser.parse_args()

    from dicodec.modules.builder import load_pretrained_model

    device = torch.device(args.device)
    print(f"Loading base Dicodec from {args.checkpoint_dir}...")
    dicodec = load_pretrained_model(str(args.checkpoint_dir))
    dicodec.to(device)
    dicodec.eval()

    latent_dim = dicodec.encoder.config.latent_dim
    base_quantizer, quant_dim = build_quantizer(args, latent_dim)
    quantizer_wrapper = QuantizerWrapper(base_quantizer, args.quantizer, args).to(
        device
    )

    model = SemanticQuantizerAE(
        dim=latent_dim,
        quantizer=quantizer_wrapper,
        quant_dim=quant_dim,
        hidden_dim=args.hidden_dim,
        num_sem_blocks=args.num_sem_blocks,
        num_dec_blocks=args.num_dec_blocks,
        kernel_size=args.kernel_size,
    ).to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    dataloader = get_dataloader(
        args.data_dir,
        target_sr=dicodec.config.sample_rate,
        batch_size=args.batch_size,
    )

    config = make_config(args, latent_dim=latent_dim, quant_dim=quant_dim)
    wandb_run = maybe_init_wandb(config)
    root_output_dir = args.output_dir
    if root_output_dir is None:
        root_output_dir = args.checkpoint_dir / "quantized"

    global_step = 0
    cumulative_code_counts = torch.zeros(args.codebook_size, dtype=torch.long)
    print(
        "Starting training loop "
        f"(input_source={args.input_source}, target_source=z_sem, quantizer={args.quantizer})..."
    )

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, wavs in enumerate(pbar):
            if args.limit_batches and batch_idx >= args.limit_batches:
                break
            if args.max_steps and global_step >= args.max_steps:
                break

            wavs = [w.to(device) for w in wavs]
            audios_srs = [(w, dicodec.config.sample_rate) for w in wavs]

            with torch.no_grad():
                enc_features, enc_padding_mask, _, _ = dicodec.extract_features(
                    audios_srs
                )
                encoder_output = dicodec.encode(enc_features, enc_padding_mask)
                z = encoder_output.z
                padding_mask = encoder_output.padding_mask
                attrs = dicodec.encode_attributes(z, padding_mask=padding_mask)
                z_sem = attrs.z_sem

            valid_mask = ~padding_mask if padding_mask is not None else None
            model_input = z if args.input_source == "z" else z_sem

            optimizer.zero_grad()
            ae_out = model(model_input, valid_mask=valid_mask)

            if padding_mask is not None:
                valid_3d = (~padding_mask).unsqueeze(-1)
                valid_elems = (valid_3d.sum() * z_sem.shape[-1]).clamp_min(1.0)
                rec_loss = (
                    F.mse_loss(
                        ae_out.z_rec * valid_3d,
                        z_sem * valid_3d,
                        reduction="sum",
                    )
                    / valid_elems
                )
            else:
                rec_loss = F.mse_loss(ae_out.z_rec, z_sem)

            loss = rec_loss + ae_out.quantizer_loss
            loss.backward()
            optimizer.step()
            global_step += 1

            postfix = {
                "loss": f"{loss.item():.4f}",
                "rec_loss": f"{rec_loss.item():.4f}",
            }
            log_metrics = {
                "train/loss": loss.item(),
                "train/rec_loss": rec_loss.item(),
                "train/quantizer_loss": ae_out.quantizer_loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch + 1,
                "train/batch_idx": batch_idx,
            }
            if ae_out.indices is not None:
                if padding_mask is not None:
                    toks_flat = ae_out.indices.view(-1)[(~padding_mask).view(-1)]
                else:
                    toks_flat = ae_out.indices.view(-1)

                if toks_flat.numel() > 0:
                    batch_counts = torch.bincount(
                        toks_flat.detach().cpu(), minlength=args.codebook_size
                    )
                    cumulative_code_counts += batch_counts
                    probs = (
                        cumulative_code_counts.float() / cumulative_code_counts.sum()
                    )
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    batch_util = (
                        (batch_counts > 0).sum().item()
                        / args.codebook_size
                        * 100.0
                    )
                    util_all = (
                        (cumulative_code_counts > 0).sum().item()
                        / args.codebook_size
                        * 100.0
                    )
                    ppl_all = torch.exp(entropy).item()
                    postfix["batch_util%"] = f"{batch_util:.1f}"
                    postfix["util_all%"] = f"{util_all:.1f}"
                    postfix["ppl_all"] = f"{ppl_all:.1f}"
                    log_metrics.update(
                        {
                            "codebook/batch_util_pct": batch_util,
                            "codebook/util_all_pct": util_all,
                            "codebook/ppl_all": ppl_all,
                            "codebook/used_all": int(
                                (cumulative_code_counts > 0).sum().item()
                            ),
                        }
                    )

            pbar.set_postfix(postfix)
            if wandb_run is not None:
                wandb_run.log(log_metrics, step=global_step)

            if args.save_every_steps and global_step % args.save_every_steps == 0:
                step_dir = (
                    root_output_dir
                    / f"{global_step // 1000}kstep"
                    / (f"{args.quantizer}_cb{args.codebook_size}")
                )
                config.max_steps = args.max_steps
                save_checkpoint(model, step_dir, config)
                print(f"\nSaved checkpoint to {step_dir}")
                if wandb_run is not None:
                    wandb_run.summary["latest_checkpoint"] = str(step_dir)

        epoch_dir = (
            root_output_dir
            / f"epoch_{epoch + 1}"
            / (f"{args.quantizer}_cb{args.codebook_size}")
        )
        save_checkpoint(model, epoch_dir, config)
        print(f"Saved checkpoint to {epoch_dir}")
        if wandb_run is not None:
            wandb_run.summary["latest_checkpoint"] = str(epoch_dir)

        if args.max_steps and global_step >= args.max_steps:
            print(f"Reached max steps ({args.max_steps}). Stopping training.")
            break

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
