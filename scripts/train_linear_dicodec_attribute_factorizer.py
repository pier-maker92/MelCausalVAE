#!/usr/bin/env python3
"""Train a conservative linear prosody/residual factorizer from saved attributes.

The saved attributes are expected at:
    <attributes-dir>/<sample_id>/z_sem.npy
    <attributes-dir>/<sample_id>/z_pros.npy
    <attributes-dir>/<sample_id>/z_mean.npy

This script reconstructs z = z_sem + z_pros + z_mean and uses
x = z - z_mean as the centered latent. It first estimates A0 with ridge
least squares so x @ A0 approximates the current z_pros target, then trains
A = A0 + delta to reduce cross-correlation while staying close to the anchor.

It can also train a shared-shift factorizer. In that mode it starts from the
existing z_sem/z_pros split and learns a shared component predicted from z_pros:
shared = z_pros @ B, z_sem_new = z_sem + shared, z_pros_new = z_pros - shared.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

sys.dont_write_bytecode = True


DEFAULT_ATTRIBUTES_DIR = (
    "/Volumes/Crucial X6/Research/Datasets/libritts-r-dicodec-attributes"
)
DEFAULT_OUTPUT_DIR = (
    "/Volumes/Crucial X6/Research/Datasets/libritts-r-dicodec-factorizer"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear factorizer from saved dicodec attributes."
    )
    parser.add_argument("--attributes-dir", default=DEFAULT_ATTRIBUTES_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--factorizer-type",
        default="shared_shift",
        choices=("projection", "shared_shift"),
        help=(
            "projection learns P=xA,R=x(I-A). shared_shift learns "
            "shared=z_pros B, z_sem+=shared, z_pros-=shared."
        ),
    )

    parser.add_argument("--a0-max-samples", type=int, default=None)
    parser.add_argument("--a0-max-frames-per-sample", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--shared-init-scale", type=float, default=1.0)

    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--samples-per-step", type=int, default=64)
    parser.add_argument("--frames-per-sample", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument("--lambda-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-corr", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=1e-3)
    parser.add_argument("--lambda-proj", type=float, default=1e-2)
    parser.add_argument("--lambda-sym", type=float, default=1e-2)
    parser.add_argument("--lambda-shared", type=float, default=1e-3)
    parser.add_argument("--corr-eps", type=float, default=1e-5)
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _sample_dirs(attributes_dir: Path) -> list[Path]:
    dirs = [
        item
        for item in attributes_dir.iterdir()
        if item.is_dir()
        and (item / "z_sem.npy").is_file()
        and (item / "z_pros.npy").is_file()
        and (item / "z_mean.npy").is_file()
    ]
    dirs.sort()
    if not dirs:
        raise FileNotFoundError(f"No attribute triplets found in {attributes_dir}")
    return dirs


def _load_triplet(sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_sem = np.load(sample_dir / "z_sem.npy", mmap_mode="r")
    z_pros = np.load(sample_dir / "z_pros.npy", mmap_mode="r")
    z_mean = np.load(sample_dir / "z_mean.npy", mmap_mode="r")
    if z_sem.shape != z_pros.shape:
        raise ValueError(f"Shape mismatch in {sample_dir}: {z_sem.shape} vs {z_pros.shape}")
    if z_mean.ndim == 1:
        z_mean = z_mean[None, :]
    if z_mean.shape[-1] != z_sem.shape[-1]:
        raise ValueError(
            f"z_mean dim mismatch in {sample_dir}: {z_mean.shape} vs {z_sem.shape}"
        )
    return z_sem, z_pros, z_mean


def _choose_frames(num_frames: int, max_frames: int, rng: random.Random) -> np.ndarray:
    if max_frames <= 0 or num_frames <= max_frames:
        return np.arange(num_frames)
    return np.array(rng.sample(range(num_frames), max_frames), dtype=np.int64)


def _centered_and_targets(
    sample_dir: Path,
    frame_indices: np.ndarray | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_sem_np, z_pros_np, z_mean_np = _load_triplet(sample_dir)
    if frame_indices is None:
        z_sem_np = np.array(z_sem_np, dtype=np.float32, copy=True)
        z_pros_np = np.array(z_pros_np, dtype=np.float32, copy=True)
    else:
        z_sem_np = np.array(z_sem_np[frame_indices], dtype=np.float32, copy=True)
        z_pros_np = np.array(z_pros_np[frame_indices], dtype=np.float32, copy=True)
    z_mean_np = np.array(z_mean_np, dtype=np.float32, copy=True)

    z_sem = torch.as_tensor(z_sem_np, dtype=torch.float32, device=device)
    z_pros = torch.as_tensor(z_pros_np, dtype=torch.float32, device=device)
    z_mean = torch.as_tensor(z_mean_np, dtype=torch.float32, device=device)

    z = z_sem + z_pros + z_mean
    x = z - z_mean
    return x, z_pros, z_sem


@torch.no_grad()
def _estimate_a0(
    sample_dirs: list[Path],
    max_samples: int | None,
    max_frames_per_sample: int,
    ridge: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    rng = random.Random(seed)
    selected = list(sample_dirs)
    rng.shuffle(selected)
    if max_samples is not None:
        selected = selected[:max_samples]

    accum_dtype = torch.float32 if device.type == "mps" else torch.float64
    dim = None
    xtx = None
    xtp = None
    total_frames = 0

    for sample_dir in tqdm(selected, desc="Estimating A0", unit="sample"):
        z_sem_np, _, _ = _load_triplet(sample_dir)
        frame_indices = _choose_frames(
            int(z_sem_np.shape[0]),
            max_frames=max_frames_per_sample,
            rng=rng,
        )
        x, p0, _ = _centered_and_targets(sample_dir, frame_indices, device=device)
        if dim is None:
            dim = int(x.shape[-1])
            xtx = torch.zeros(dim, dim, dtype=accum_dtype, device=device)
            xtp = torch.zeros(dim, dim, dtype=accum_dtype, device=device)

        x_accum = x.to(accum_dtype)
        p_accum = p0.to(accum_dtype)
        xtx += x_accum.T @ x_accum
        xtp += x_accum.T @ p_accum
        total_frames += int(x.shape[0])

    if dim is None or xtx is None or xtp is None or total_frames == 0:
        raise RuntimeError("No frames collected for A0 estimation.")

    eye = torch.eye(dim, dtype=accum_dtype, device=device)
    system = xtx + ridge * eye
    try:
        a0 = torch.linalg.solve(system, xtp).to(torch.float32)
    except RuntimeError as exc:
        if device.type != "mps":
            raise
        print(f"MPS solve failed ({exc}); solving 64x64 system on CPU.")
        a0 = torch.linalg.solve(system.cpu(), xtp.cpu()).to(torch.float32).to(device)
    stats = {
        "samples": len(selected),
        "frames": total_frames,
        "dim": dim,
        "ridge": ridge,
    }
    return a0, stats


class LinearDeltaFactorizer(nn.Module):
    def __init__(self, a0: torch.Tensor):
        super().__init__()
        self.register_buffer("A0", a0.detach().clone())
        self.delta = nn.Parameter(torch.zeros_like(a0))

    @property
    def A(self) -> torch.Tensor:
        return self.A0 + self.delta

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pros = x @ self.A
        res = x - pros
        return pros, res


@torch.no_grad()
def _estimate_shared_b0(
    sample_dirs: list[Path],
    max_samples: int | None,
    max_frames_per_sample: int,
    ridge: float,
    seed: int,
    init_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    rng = random.Random(seed)
    selected = list(sample_dirs)
    rng.shuffle(selected)
    if max_samples is not None:
        selected = selected[:max_samples]

    accum_dtype = torch.float32 if device.type == "mps" else torch.float64
    dim = None
    ptp = None
    ptr = None
    total_frames = 0

    for sample_dir in tqdm(selected, desc="Estimating shared B0", unit="sample"):
        z_sem_np, _, _ = _load_triplet(sample_dir)
        frame_indices = _choose_frames(
            int(z_sem_np.shape[0]),
            max_frames=max_frames_per_sample,
            rng=rng,
        )
        _, p0, r0 = _centered_and_targets(sample_dir, frame_indices, device=device)
        if dim is None:
            dim = int(p0.shape[-1])
            ptp = torch.zeros(dim, dim, dtype=accum_dtype, device=device)
            ptr = torch.zeros(dim, dim, dtype=accum_dtype, device=device)

        p_accum = p0.to(accum_dtype)
        r_accum = r0.to(accum_dtype)
        ptp += p_accum.T @ p_accum
        ptr += p_accum.T @ r_accum
        total_frames += int(p0.shape[0])

    if dim is None or ptp is None or ptr is None or total_frames == 0:
        raise RuntimeError("No frames collected for shared B0 estimation.")

    eye = torch.eye(dim, dtype=accum_dtype, device=device)
    system = ptp + ridge * eye
    try:
        b0 = torch.linalg.solve(system, ptr).to(torch.float32)
    except RuntimeError as exc:
        if device.type != "mps":
            raise
        print(f"MPS solve failed ({exc}); solving 64x64 system on CPU.")
        b0 = torch.linalg.solve(system.cpu(), ptr.cpu()).to(torch.float32).to(device)
    b0 = b0 * float(init_scale)
    stats = {
        "samples": len(selected),
        "frames": total_frames,
        "dim": dim,
        "ridge": ridge,
        "shared_init_scale": init_scale,
        "regression": "z_pros -> z_sem",
    }
    return b0, stats


class SharedShiftFactorizer(nn.Module):
    def __init__(self, b0: torch.Tensor):
        super().__init__()
        self.register_buffer("B0", b0.detach().clone())
        self.delta = nn.Parameter(torch.zeros_like(b0))

    @property
    def B(self) -> torch.Tensor:
        return self.B0 + self.delta

    def forward(
        self,
        z_pros: torch.Tensor,
        z_sem: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = z_pros @ self.B
        pros = z_pros - shared
        sem = z_sem + shared
        return pros, sem, shared


def cross_correlation_loss(
    p: torch.Tensor,
    r: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    p = p.reshape(-1, p.shape[-1])
    r = r.reshape(-1, r.shape[-1])

    p = p - p.mean(dim=0, keepdim=True)
    r = r - r.mean(dim=0, keepdim=True)

    p = p / (p.std(dim=0, keepdim=True, unbiased=False) + eps)
    r = r / (r.std(dim=0, keepdim=True, unbiased=False) + eps)

    c = p.T @ r / p.shape[0]
    return c.square().mean()


def _sample_frame_batch(
    sample_dirs: list[Path],
    samples_per_step: int,
    frames_per_sample: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen = rng.choices(sample_dirs, k=samples_per_step)
    xs = []
    p0s = []
    r0s = []
    for sample_dir in chosen:
        z_sem_np, _, _ = _load_triplet(sample_dir)
        frame_indices = _choose_frames(
            int(z_sem_np.shape[0]),
            max_frames=frames_per_sample,
            rng=rng,
        )
        x, p0, r0 = _centered_and_targets(sample_dir, frame_indices, device=device)
        xs.append(x)
        p0s.append(p0)
        r0s.append(r0)
    return torch.cat(xs, dim=0), torch.cat(p0s, dim=0), torch.cat(r0s, dim=0)


def _projection_loss(a: torch.Tensor) -> torch.Tensor:
    return (a @ a - a).square().mean()


def _symmetric_loss(a: torch.Tensor) -> torch.Tensor:
    return (a - a.T).square().mean()


def _train(
    factorizer: LinearDeltaFactorizer,
    sample_dirs: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    rng = random.Random(args.seed + 1)
    optimizer = torch.optim.AdamW(
        [factorizer.delta],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history = []
    progress = tqdm(range(1, args.train_steps + 1), desc="Training factorizer", unit="step")
    for step in progress:
        x, p0, r0 = _sample_frame_batch(
            sample_dirs=sample_dirs,
            samples_per_step=args.samples_per_step,
            frames_per_sample=args.frames_per_sample,
            rng=rng,
            device=device,
        )
        pros, res = factorizer(x)
        loss_anchor = F.mse_loss(pros, p0) + F.mse_loss(res, r0)
        loss_corr = cross_correlation_loss(pros, res, eps=args.corr_eps)
        loss_delta = factorizer.delta.square().mean()
        loss_proj = _projection_loss(factorizer.A)
        loss_sym = _symmetric_loss(factorizer.A)

        loss = (
            args.lambda_anchor * loss_anchor
            + args.lambda_corr * loss_corr
            + args.lambda_delta * loss_delta
            + args.lambda_proj * loss_proj
            + args.lambda_sym * loss_sym
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_([factorizer.delta], args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            with torch.no_grad():
                delta_ratio = (
                    factorizer.delta.norm() / factorizer.A0.norm().clamp_min(1e-12)
                )
            item = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "anchor": float(loss_anchor.detach().cpu()),
                "corr": float(loss_corr.detach().cpu()),
                "delta": float(loss_delta.detach().cpu()),
                "proj": float(loss_proj.detach().cpu()),
                "sym": float(loss_sym.detach().cpu()),
                "delta_over_a0": float(delta_ratio.detach().cpu()),
            }
            history.append(item)
            progress.set_postfix(
                loss=f"{item['loss']:.4g}",
                corr=f"{item['corr']:.4g}",
                dA=f"{item['delta_over_a0']:.4g}",
            )
    return history


def _train_shared_shift(
    factorizer: SharedShiftFactorizer,
    sample_dirs: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    rng = random.Random(args.seed + 1)
    optimizer = torch.optim.AdamW(
        [factorizer.delta],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history = []
    progress = tqdm(
        range(1, args.train_steps + 1),
        desc="Training shared-shift factorizer",
        unit="step",
    )
    for step in progress:
        _, p0, r0 = _sample_frame_batch(
            sample_dirs=sample_dirs,
            samples_per_step=args.samples_per_step,
            frames_per_sample=args.frames_per_sample,
            rng=rng,
            device=device,
        )
        pros, sem, shared = factorizer(p0, r0)
        loss_corr = cross_correlation_loss(pros, sem, eps=args.corr_eps)
        loss_delta = factorizer.delta.square().mean()
        loss_shared = shared.square().mean()
        loss_anchor = F.mse_loss(pros, p0) + F.mse_loss(sem, r0)
        loss_sym = _symmetric_loss(factorizer.B)

        loss = (
            args.lambda_corr * loss_corr
            + args.lambda_delta * loss_delta
            + args.lambda_shared * loss_shared
            + args.lambda_anchor * loss_anchor
            + args.lambda_sym * loss_sym
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_([factorizer.delta], args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            with torch.no_grad():
                delta_ratio = (
                    factorizer.delta.norm() / factorizer.B0.norm().clamp_min(1e-12)
                )
                shared_ratio = shared.norm() / p0.norm().clamp_min(1e-12)
            item = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "anchor": float(loss_anchor.detach().cpu()),
                "corr": float(loss_corr.detach().cpu()),
                "delta": float(loss_delta.detach().cpu()),
                "shared": float(loss_shared.detach().cpu()),
                "sym": float(loss_sym.detach().cpu()),
                "delta_over_b0": float(delta_ratio.detach().cpu()),
                "shared_over_pros": float(shared_ratio.detach().cpu()),
            }
            history.append(item)
            progress.set_postfix(
                loss=f"{item['loss']:.4g}",
                corr=f"{item['corr']:.4g}",
                dB=f"{item['delta_over_b0']:.4g}",
                sh=f"{item['shared_over_pros']:.4g}",
            )
    return history


@torch.no_grad()
def _evaluate(
    factorizer: LinearDeltaFactorizer,
    sample_dirs: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    rng = random.Random(args.seed + 2)
    x, p0, r0 = _sample_frame_batch(
        sample_dirs=sample_dirs,
        samples_per_step=min(args.samples_per_step, len(sample_dirs)),
        frames_per_sample=args.frames_per_sample,
        rng=rng,
        device=device,
    )
    pros, res = factorizer(x)
    a = factorizer.A
    return {
        "anchor": float((F.mse_loss(pros, p0) + F.mse_loss(res, r0)).cpu()),
        "corr": float(cross_correlation_loss(pros, res, eps=args.corr_eps).cpu()),
        "delta_over_a0": float((factorizer.delta.norm() / factorizer.A0.norm()).cpu()),
        "projection": float(_projection_loss(a).cpu()),
        "symmetric": float(_symmetric_loss(a).cpu()),
    }


@torch.no_grad()
def _evaluate_shared_shift(
    factorizer: SharedShiftFactorizer,
    sample_dirs: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    rng = random.Random(args.seed + 2)
    _, p0, r0 = _sample_frame_batch(
        sample_dirs=sample_dirs,
        samples_per_step=min(args.samples_per_step, len(sample_dirs)),
        frames_per_sample=args.frames_per_sample,
        rng=rng,
        device=device,
    )
    pros, sem, shared = factorizer(p0, r0)
    return {
        "anchor": float((F.mse_loss(pros, p0) + F.mse_loss(sem, r0)).cpu()),
        "corr": float(cross_correlation_loss(pros, sem, eps=args.corr_eps).cpu()),
        "delta_over_b0": float(
            (factorizer.delta.norm() / factorizer.B0.norm().clamp_min(1e-12)).cpu()
        ),
        "shared": float(shared.square().mean().cpu()),
        "shared_over_pros": float((shared.norm() / p0.norm().clamp_min(1e-12)).cpu()),
        "symmetric": float(_symmetric_loss(factorizer.B).cpu()),
    }


def _save_projection_factorizer(
    factorizer: LinearDeltaFactorizer,
    output_dir: Path,
    args: argparse.Namespace,
    a0_stats: dict[str, Any],
    history: list[dict[str, float]],
    eval_stats: dict[str, float],
) -> None:
    a0_cpu = factorizer.A0.detach().cpu()
    delta_cpu = factorizer.delta.detach().cpu()
    a_cpu = factorizer.A.detach().cpu()

    torch.save(
        {
            "factorizer_type": "projection",
            "A0": a0_cpu,
            "delta": delta_cpu,
            "A": a_cpu,
            "args": vars(args),
            "a0_stats": a0_stats,
            "history": history,
            "eval": eval_stats,
        },
        output_dir / "linear_delta_factorizer.pt",
    )
    np.save(output_dir / "A0.npy", a0_cpu.numpy())
    np.save(output_dir / "delta.npy", delta_cpu.numpy())
    np.save(output_dir / "A.npy", a_cpu.numpy())


def _save_shared_shift_factorizer(
    factorizer: SharedShiftFactorizer,
    output_dir: Path,
    args: argparse.Namespace,
    b0_stats: dict[str, Any],
    history: list[dict[str, float]],
    eval_stats: dict[str, float],
) -> None:
    b0_cpu = factorizer.B0.detach().cpu()
    delta_cpu = factorizer.delta.detach().cpu()
    b_cpu = factorizer.B.detach().cpu()

    torch.save(
        {
            "factorizer_type": "shared_shift",
            "B0": b0_cpu,
            "delta": delta_cpu,
            "B": b_cpu,
            "args": vars(args),
            "b0_stats": b0_stats,
            "history": history,
            "eval": eval_stats,
        },
        output_dir / "shared_shift_factorizer.pt",
    )
    np.save(output_dir / "B0.npy", b0_cpu.numpy())
    np.save(output_dir / "delta.npy", delta_cpu.numpy())
    np.save(output_dir / "B.npy", b_cpu.numpy())


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _resolve_device(args.device)
    attributes_dir = Path(args.attributes_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = _sample_dirs(attributes_dir)
    print(f"Found {len(sample_dirs)} attribute samples in {attributes_dir}")
    print(f"Using device={device}")

    init_stats_key = "a0_stats"
    init_stats = {}
    if args.factorizer_type == "projection":
        a0, init_stats = _estimate_a0(
            sample_dirs=sample_dirs,
            max_samples=args.a0_max_samples,
            max_frames_per_sample=args.a0_max_frames_per_sample,
            ridge=args.ridge,
            seed=args.seed,
            device=device,
        )
        factorizer = LinearDeltaFactorizer(a0).to(device)
        history = _train(factorizer, sample_dirs, args, device=device)
        eval_stats = _evaluate(factorizer, sample_dirs, args, device=device)
        _save_projection_factorizer(
            factorizer=factorizer,
            output_dir=output_dir,
            args=args,
            a0_stats=init_stats,
            history=history,
            eval_stats=eval_stats,
        )
    else:
        init_stats_key = "b0_stats"
        b0, init_stats = _estimate_shared_b0(
            sample_dirs=sample_dirs,
            max_samples=args.a0_max_samples,
            max_frames_per_sample=args.a0_max_frames_per_sample,
            ridge=args.ridge,
            seed=args.seed,
            init_scale=args.shared_init_scale,
            device=device,
        )
        factorizer = SharedShiftFactorizer(b0).to(device)
        history = _train_shared_shift(factorizer, sample_dirs, args, device=device)
        eval_stats = _evaluate_shared_shift(factorizer, sample_dirs, args, device=device)
        _save_shared_shift_factorizer(
            factorizer=factorizer,
            output_dir=output_dir,
            args=args,
            b0_stats=init_stats,
            history=history,
            eval_stats=eval_stats,
        )

    summary = {
        "attributes_dir": str(attributes_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "factorizer_type": args.factorizer_type,
        "num_attribute_samples": len(sample_dirs),
        init_stats_key: init_stats,
        "final": history[-1] if history else {},
        "eval": eval_stats,
    }
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
