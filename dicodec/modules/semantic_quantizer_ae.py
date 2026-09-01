from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class QuantizerOutput:
    z_q: torch.Tensor
    loss: torch.Tensor
    indices: torch.Tensor | None = None


@dataclass
class SemanticQuantizerAEOutput:
    z_rec: torch.Tensor
    z_sem_q: torch.Tensor
    z_pros_enc: torch.Tensor
    quantizer_loss: torch.Tensor
    indices: torch.Tensor | None


class ResNetBlock1D(nn.Module):
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
        if len(dilations) != num_blocks:
            raise ValueError("dilations length must match num_blocks.")
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
        self,
        in_dim: int,
        hidden_dim: int,
        quant_dim: int,
        num_blocks: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.resnet = ResNetStack1D(hidden_dim, num_blocks, kernel_size=kernel_size)
        self.out_proj = nn.Conv1d(hidden_dim, quant_dim, kernel_size=1)

    def forward(
        self,
        z_sem: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        self,
        sem_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_blocks: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(sem_dim, hidden_dim, kernel_size=1)
        self.resnet = ResNetStack1D(hidden_dim, num_blocks, kernel_size=kernel_size)
        self.out_proj = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)

    def forward(
        self,
        z_sem_q: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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


class SemanticQuantizerAE(nn.Module):
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
            in_dim=dim,
            hidden_dim=hidden_dim,
            quant_dim=quant_dim,
            num_blocks=num_sem_blocks,
            kernel_size=kernel_size,
        )
        self.quantizer = quantizer
        self.decoder = Decoder(
            sem_dim=quant_dim,
            hidden_dim=hidden_dim,
            out_dim=dim,
            num_blocks=num_dec_blocks,
            kernel_size=kernel_size,
        )

    def _run_quantizer(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> QuantizerOutput:
        out = self.quantizer(x, valid_mask)
        if isinstance(out, QuantizerOutput):
            return out
        if isinstance(out, tuple):
            if len(out) == 2:
                toks, codes = out
                return QuantizerOutput(
                    z_q=codes,
                    loss=torch.tensor(0.0, device=x.device),
                    indices=toks,
                )
            z_q, loss = out[0], out[1]
            indices = out[2] if len(out) > 2 else None
            return QuantizerOutput(z_q=z_q, loss=loss, indices=indices)
        raise TypeError(f"Unrecognized quantizer output type {type(out)}.")

    def forward(
        self,
        z: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> SemanticQuantizerAEOutput:
        if valid_mask is not None and valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(-1)

        sem_enc = self.sem_encoder(z, valid_mask=valid_mask)
        quant_out = self._run_quantizer(sem_enc, valid_mask=valid_mask)
        z_sem_q = quant_out.z_q
        indices = quant_out.indices
        z_rec = self.decoder(z_sem_q, valid_mask=valid_mask)

        return SemanticQuantizerAEOutput(
            z_rec=z_rec,
            z_sem_q=z_sem_q,
            z_pros_enc=torch.zeros_like(z_sem_q),
            quantizer_loss=quant_out.loss,
            indices=indices,
        )


class InferenceQuantizerWrapper(nn.Module):
    def __init__(self, quantizer_module: nn.Module):
        super().__init__()
        self.quantizer_module = quantizer_module

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> QuantizerOutput:
        if valid_mask is None:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            valid = valid_mask.squeeze(-1).bool()

        x_valid = x[valid]
        z_q = torch.zeros_like(x)
        toks = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        if x_valid.numel() == 0:
            return QuantizerOutput(z_q=z_q, loss=x.new_zeros(()), indices=toks)

        toks_valid, codes_valid = self.quantizer_module(x_valid)
        z_q[valid] = codes_valid
        toks[valid] = toks_valid.long()
        return QuantizerOutput(z_q=z_q, loss=x.new_zeros(()), indices=toks)


def _count_blocks(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    indices = {
        int(key.split(".")[3])
        for key in state_dict
        if key.startswith(prefix) and len(key.split(".")) > 3
    }
    return len(indices)


def infer_semantic_quantizer_hparams(
    state_dict: dict[str, torch.Tensor],
    fallback_dim: int,
    quantizer_type: str,
    codebook_size: int | None = None,
) -> dict[str, int]:
    hidden_dim = int(state_dict["sem_encoder.in_proj.weight"].shape[0])
    latent_dim = int(state_dict["sem_encoder.in_proj.weight"].shape[1])
    quant_dim = int(state_dict["sem_encoder.out_proj.weight"].shape[0])
    num_sem_blocks = _count_blocks(state_dict, "sem_encoder.resnet.blocks.")
    num_dec_blocks = _count_blocks(state_dict, "decoder.resnet.blocks.")

    inferred_codebook_size = codebook_size
    embedding_key = "quantizer.quantizer_module.embedding.weight"
    if embedding_key in state_dict:
        inferred_codebook_size = int(state_dict[embedding_key].shape[0])
    elif quantizer_type == "vq_ema":
        ema_key = "quantizer.quantizer_module.embedding"
        if ema_key in state_dict:
            inferred_codebook_size = int(state_dict[ema_key].shape[0])

    return {
        "latent_dim": latent_dim or fallback_dim,
        "hidden_dim": hidden_dim,
        "quant_dim": quant_dim,
        "num_sem_blocks": num_sem_blocks or 2,
        "num_dec_blocks": num_dec_blocks or 2,
        "codebook_size": inferred_codebook_size or 1024,
    }


def build_base_quantizer(
    quantizer_type: str,
    quant_dim: int,
    codebook_size: int,
    device: torch.device | str,
) -> Tuple[nn.Module, int]:
    if quantizer_type == "vq_ema":
        from .quantizer.vq_ema import EMAVectorQuantizer

        return EMAVectorQuantizer(dim=quant_dim, codebook_size=codebook_size), quant_dim
    if quantizer_type == "bsq":
        from .quantizer.bsq import BinarySphericalQuantizer

        quantizer = BinarySphericalQuantizer(codebook_size=codebook_size)
        return quantizer, quantizer.dim
    if quantizer_type == "std_vq":
        from .quantizer.std_vq import StandardVectorQuantizer

        return StandardVectorQuantizer(dim=quant_dim, codebook_size=codebook_size), quant_dim
    if quantizer_type == "fsq":
        from .quantizer.fsq import FiniteScalarQuantizer

        quantizer = FiniteScalarQuantizer(codebook_size=codebook_size)
        return quantizer, quantizer.dim
    raise ValueError(f"Unsupported external semantic quantizer type: {quantizer_type}")


def load_semantic_quantizer_ae(
    checkpoint_path: str | Path,
    latent_dim: int,
    quantizer_type: str = "std_vq",
    codebook_size: int | None = None,
    device: torch.device | str = "cpu",
) -> SemanticQuantizerAE:
    checkpoint_path = Path(checkpoint_path)
    config = {}
    if checkpoint_path.is_dir():
        config_path = checkpoint_path / "config.json"
        model_path = checkpoint_path / "model.pt"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing quantizer config: {config_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing quantizer checkpoint: {model_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        checkpoint_path = model_path

    quantizer_type = config.get("quantizer_type", quantizer_type)
    codebook_size = config.get(
        "codebook_size",
        config.get("num_embeddings", config.get("num_codebooks", codebook_size)),
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    hparams = infer_semantic_quantizer_hparams(
        state_dict=state_dict,
        fallback_dim=latent_dim,
        quantizer_type=quantizer_type,
        codebook_size=codebook_size,
    )
    base_quantizer, quant_dim = build_base_quantizer(
        quantizer_type=quantizer_type,
        quant_dim=hparams["quant_dim"],
        codebook_size=hparams["codebook_size"],
        device=device,
    )
    wrapper = InferenceQuantizerWrapper(base_quantizer)
    model = SemanticQuantizerAE(
        dim=hparams["latent_dim"],
        quantizer=wrapper,
        quant_dim=quant_dim,
        hidden_dim=hparams["hidden_dim"],
        num_sem_blocks=hparams["num_sem_blocks"],
        num_dec_blocks=hparams["num_dec_blocks"],
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def read_semantic_quantizer_config(checkpoint_path: str | Path) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_dir():
        return {}
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)
