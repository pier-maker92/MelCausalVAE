import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dicodec.modules.configs import VQConfig
from dicodec.modules.quantizer.vq import VectorQuantizer


INPUT_DIR = Path("/Volumes/Crucial X6/Research/dicodec-attributes")
OUTPUT_DIR = INPUT_DIR / "quantizers"
ATTRIBUTE = "z_sem"
SEED = 1234
VQ_TYPE_ALIASES = {"std_vq": "vq"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_attribute_files(input_dir: Path, attribute: str) -> List[Path]:
    filename = f"{attribute}.npy"
    return sorted(
        path
        for path in input_dir.glob(f"*/{filename}")
        if path.parent.name != "kmeans"
    )


class ZSemChunkDataset(Dataset):
    def __init__(
        self,
        paths: List[Path],
        seq_len: int,
        steps: int,
        frames_per_sample: int,
        seed: int,
    ) -> None:
        self.paths = list(paths)
        self.seq_len = seq_len
        self.steps = steps
        self.frames_per_sample = frames_per_sample
        self.seed = seed
        self.lengths = []

        for path in self.paths:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2:
                raise ValueError(f"{path} must have shape [T, D], found {array.shape}.")
            self.lengths.append(int(array.shape[0]))

        eligible = [n for n in self.lengths if n > 0]
        if not eligible:
            raise RuntimeError("No non-empty z_sem files found.")

    def __len__(self) -> int:
        return self.steps

    def _sample_path_index(self, rng: random.Random) -> int:
        for _ in range(32):
            idx = rng.randrange(len(self.paths))
            if self.lengths[idx] > 0:
                return idx
        for idx, length in enumerate(self.lengths):
            if length > 0:
                return idx
        raise RuntimeError("No valid z_sem files available.")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + index)
        chunks = []
        masks = []
        remaining = self.frames_per_sample

        while remaining > 0:
            path_idx = self._sample_path_index(rng)
            path = self.paths[path_idx]
            array = np.load(path, mmap_mode="r")
            length = int(array.shape[0])
            take = min(self.seq_len, remaining)

            if length >= take:
                start = 0 if length == take else rng.randint(0, length - take)
                chunk = np.asarray(array[start : start + take], dtype=np.float32)
                mask = np.zeros((take,), dtype=bool)
            else:
                start = 0
                chunk = np.zeros((take, array.shape[1]), dtype=np.float32)
                chunk[:length] = np.asarray(array[:length], dtype=np.float32)
                mask = np.ones((take,), dtype=bool)
                mask[:length] = False

            chunks.append(chunk)
            masks.append(mask)
            remaining -= take

        sample = np.concatenate(chunks, axis=0)
        padding_mask = np.concatenate(masks, axis=0)
        return {
            "z": torch.from_numpy(sample),
            "padding_mask": torch.from_numpy(padding_mask),
        }


class ZSemFullSequenceDataset(Dataset):
    def __init__(self, paths: List[Path]) -> None:
        self.paths = []
        self.lengths = []
        for path in paths:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2:
                raise ValueError(f"{path} must have shape [T, D], found {array.shape}.")
            if array.shape[0] == 0:
                continue
            self.paths.append(path)
            self.lengths.append(int(array.shape[0]))
        if not self.paths:
            raise RuntimeError("No non-empty z_sem files found.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        array = np.load(self.paths[index], mmap_mode="r")
        z = np.asarray(array, dtype=np.float32)
        padding_mask = np.zeros((z.shape[0],), dtype=bool)
        return {
            "z": torch.from_numpy(z),
            "padding_mask": torch.from_numpy(padding_mask),
        }


def collate_with_padding(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(item["z"].shape[0] for item in batch)
    dim = batch[0]["z"].shape[1]
    batch_size = len(batch)

    z = torch.zeros((batch_size, max_len, dim), dtype=batch[0]["z"].dtype)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        length = item["z"].shape[0]
        z[i, :length] = item["z"]
        padding_mask[i, :length] = item["padding_mask"]

    return {"z": z, "padding_mask": padding_mask}


def infer_feature_dim(paths: List[Path]) -> int:
    for path in paths:
        array = np.load(path, mmap_mode="r")
        if array.ndim == 2 and array.shape[0] > 0:
            return int(array.shape[1])
    raise RuntimeError("Could not infer feature dimension from z_sem files.")


def validate_args(args: argparse.Namespace) -> None:
    if args.seq_len is not None and args.seq_len <= 0:
        raise ValueError("--seq-len must be > 0 when provided.")
    vq_type = VQ_TYPE_ALIASES.get(args.vq_type, args.vq_type)
    if vq_type == "bsq":
        if args.codebook_size < 2 or args.codebook_size & (args.codebook_size - 1):
            raise ValueError("--codebook-size for bsq must be a power of two >= 2.")


def build_model(args: argparse.Namespace, dim: int, device: str) -> Tuple[VectorQuantizer, VQConfig]:
    vq_type = VQ_TYPE_ALIASES.get(args.vq_type, args.vq_type)

    config = VQConfig(
        num_embeddings=args.codebook_size,
        add_residual=False,
        add_residual_p=0.0,
        drop_acoustic_p=0.0,
        vq_type=vq_type,
        vq_dim=args.vq_dim,
        commitment_weight=args.commitment_weight,
        ema_decay=args.ema_decay,
        ema_eps=args.ema_eps,
        reset_dead_codes=vq_type == "vq_ema",
        reset_every_forward=10,
        entropy_loss_weight=args.entropy_loss_weight,
        entropy_temperature=args.entropy_temperature,
        recon_weight=args.recon_weight,
    )
    model = VectorQuantizer(config=config, dim=dim).to(device)
    return model, config


def save_checkpoint(
    model: VectorQuantizer,
    config: VQConfig,
    output_dir: Path,
    step: int,
    avg_loss: float,
    dim: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "step": step,
            "avg_loss": avg_loss,
            "feature_dim": dim,
        },
        checkpoint_path,
    )
    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                "feature_dim": dim,
                "vq_config": asdict(config),
                "step": step,
                "avg_loss": avg_loss,
            },
            f,
            indent=2,
        )

    quantizer = model.quantizer
    if hasattr(quantizer, "embedding"):
        embedding = quantizer.embedding
        if isinstance(embedding, torch.nn.Embedding):
            embedding = embedding.weight
        torch.save(
            {
                "embedding": embedding.detach().cpu(),
                "cluster_size": getattr(quantizer, "cluster_size", None),
                "embed_avg": getattr(quantizer, "embed_avg", None),
                "step": step,
                "avg_loss": avg_loss,
            },
            output_dir / "codebook.pt",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a quantizer on saved z_sem attributes only."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--attribute", default=ATTRIBUTE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--vq-type",
        choices=["vq_ema", "vq", "std_vq", "bsq", "fsq"],
        default="vq_ema",
    )
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--vq-dim", type=int, default=64)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--ema-eps", type=float, default=1e-5)
    parser.add_argument("--entropy-loss-weight", type=float, default=0.0)
    parser.add_argument("--entropy-temperature", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)

    args = parser.parse_args()

    validate_args(args)
    seed_everything(args.seed)
    device = get_device(args.device)

    paths = find_attribute_files(args.input_dir, args.attribute)
    if not paths:
        raise RuntimeError(f"No {args.attribute}.npy files found under {args.input_dir}.")

    dim = infer_feature_dim(paths)
    print(f"Found {len(paths):,} {args.attribute}.npy files.")
    print(f"Feature dim: {dim}")
    print(f"Device: {device}")

    if args.seq_len is None:
        dataset = ZSemFullSequenceDataset(paths=paths)
        shuffle = True
        collate_fn = collate_with_padding
    else:
        dataset = ZSemChunkDataset(
            paths=paths,
            seq_len=args.seq_len,
            steps=args.steps * args.batch_size,
            frames_per_sample=args.seq_len,
            seed=args.seed,
        )
        shuffle = False
        collate_fn = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )

    model, config = build_model(args, dim=dim, device=device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    step = 0
    running_loss = 0.0
    cumulative_code_counts = torch.zeros(args.codebook_size, dtype=torch.long)
    progress = tqdm(
        total=args.steps,
        desc=f"Training {args.vq_type}",
        dynamic_ncols=True,
    )

    for epoch in range(args.epochs):
        del epoch
        for batch in dataloader:
            if step >= args.steps:
                break

            z = batch["z"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            output = model(z, padding_mask=padding_mask)
            loss = output.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            loss_value = float(loss.detach().item())
            running_loss += loss_value
            progress.update(1)

            valid_indices = output.indices[~padding_mask].detach().cpu().long()
            if valid_indices.numel() > 0:
                cumulative_code_counts += torch.bincount(
                    valid_indices, minlength=args.codebook_size
                )

            if step % args.log_every == 0 or step == 1:
                avg = running_loss / step
                total_codes = cumulative_code_counts.sum().clamp_min(1)
                probs = cumulative_code_counts.float() / total_codes
                perplexity = float(torch.exp(-(probs * torch.log(probs + 1e-10)).sum()))
                used = float(
                    (cumulative_code_counts > 0).sum().item()
                    / args.codebook_size
                )
                progress.set_postfix(
                    loss=f"{avg:.4f}",
                    ppl_all=f"{perplexity:.1f}",
                    used_all=f"{used:.3f}",
                )

            if step % args.save_every == 0:
                avg = running_loss / step
                save_dir = args.output_dir / f"{args.attribute}_{args.vq_type}_{args.codebook_size}"
                save_checkpoint(model, config, save_dir, step, avg, dim)

        if step >= args.steps:
            break

    progress.close()

    save_dir = args.output_dir / f"{args.attribute}_{args.vq_type}_{args.codebook_size}"
    avg = running_loss / max(step, 1)
    save_checkpoint(model, config, save_dir, step, avg, dim)
    print(f"Saved checkpoint to {save_dir}")
    print(f"Final step: {step}")
    print(f"Average loss: {avg:.6f}")


if __name__ == "__main__":
    main()
