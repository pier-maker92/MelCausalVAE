#!/usr/bin/env python3
"""Evaluate reconstructions on local librispeech-aligned test_clean.

Metrics:
  - dWER: Whisper transcription WER between hypothesis and reference audio
  - UTMOS: predicted MOS for reference and hypothesis audio
  - SIM: WavLM speaker embedding cosine similarity
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_HF_HOME = "/Volumes/Crucial X6/HF_HOME"
DEFAULT_DATASET_DIR = "/Users/software/Research/datasets/librispeech-aligned/test_clean"
DEFAULT_CHECKPOINT_DIR = (
    "/Users/software/Research/MelCausalVAE/checkpoints/paper/baseline/"
    "18-denc128-novq"
)
DEFAULT_OUTPUT_DIR = "/Users/software/Research/MelCausalVAE/evaluation/librispeech_test_clean"
DEFAULT_FACTORIZER_PATH = (
    "/Volumes/Crucial X6/Research/Datasets/libritts-r-dicodec-factorizer/"
    "shared_shift_factorizer.pt"
)


def _set_external_caches(hf_home: str) -> Path:
    hf_home_path = Path(hf_home).expanduser()
    for child in (
        "datasets",
        "hub",
        "assets",
        "transformers",
        "xdg",
        "torch",
        "tmp",
        "matplotlib",
    ):
        (hf_home_path / child).mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home_path)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home_path / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home_path / "hub")
    os.environ["HF_ASSETS_CACHE"] = str(hf_home_path / "assets")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home_path / "transformers")
    os.environ["XDG_CACHE_HOME"] = str(hf_home_path / "xdg")
    os.environ["TORCH_HOME"] = str(hf_home_path / "torch")
    os.environ["TMPDIR"] = str(hf_home_path / "tmp")
    os.environ["TEMP"] = str(hf_home_path / "tmp")
    os.environ["TMP"] = str(hf_home_path / "tmp")
    os.environ["MPLCONFIGDIR"] = str(hf_home_path / "matplotlib")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return hf_home_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dWER/UTMOS/SIM on local librispeech-aligned test_clean."
    )
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--metrics-device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-audio-len", type=float, default=20.0)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--guidance-scale", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--whisper-size", default="small")
    parser.add_argument("--sim-model", default="microsoft/wavlm-base-sv")
    parser.add_argument(
        "--mode",
        default="native",
        choices=("native", "kmeans"),
        help="native uses z from encode_decode; kmeans quantizes z_sem/z_sem_new.",
    )
    parser.add_argument("--kmeans-path", default=None)
    parser.add_argument("--factorizer-path", default=DEFAULT_FACTORIZER_PATH)
    parser.add_argument("--kmeans-chunk-size", type=int, default=16384)
    parser.add_argument("--save-audio", action="store_true")
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_device(requested: str) -> "torch.device":
    import torch

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _load_local_dataset(dataset_dir: Path, max_audio_len: float | None):
    from datasets import load_dataset

    parquet_files = sorted(
        str(path)
        for path in dataset_dir.glob("*.parquet")
        if not path.name.startswith("._")
    )
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")

    dataset = load_dataset(
        "parquet",
        data_files={"test": parquet_files},
        split="test",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )
    if "subset" in dataset.column_names:
        before = len(dataset)
        dataset = dataset.filter(
            lambda subset: subset == "test-clean",
            input_columns=["subset"],
            desc="Filtering test-clean",
        )
        print(f"Filtered test-clean subset: {before} -> {len(dataset)} samples")

    if max_audio_len is not None:
        before = len(dataset)
        duration_column = None
        if "duration" in dataset.column_names:
            duration_column = "duration"
        elif "duration_sec" in dataset.column_names:
            duration_column = "duration_sec"

        if duration_column is None:
            print(
                "Skipping max_audio_len filter because no duration metadata column "
                "is available; avoiding audio decode during filtering."
            )
        else:
            dataset = dataset.filter(
                lambda duration: duration is not None and float(duration) <= max_audio_len,
                input_columns=[duration_column],
                desc="Filtering length",
            )
            print(
                f"Filtered max_audio_len={max_audio_len}: "
                f"{before} -> {len(dataset)} samples"
            )

    return dataset


class LocalTestCleanDataset:
    def __init__(self, dataset):
        from data.audio_dataset import SimpleAudioDataset

        class _Processor(SimpleAudioDataset):
            pass

        self.dataset = dataset
        self.processor = _Processor()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        out: dict[str, Any] = {}
        self.processor._process_audio_output(out, item["audio"], target_sr=24000)
        self.processor._process_audio_output(
            out,
            item["audio"],
            key_name="16k_audio",
            target_sr=16000,
        )
        out["ids"] = item.get("id") or str(index)
        out["language"] = item.get("language", "en")
        out["transcription"] = (
            item.get("text_normalized")
            or item.get("transcript")
            or item.get("text")
            or ""
        )
        return out


def _make_dataloader(dataset, batch_size: int, num_workers: int, num_samples: int | None):
    import torch
    from data.audio_dataset import EvalDataCollator

    wrapped = LocalTestCleanDataset(dataset)
    if num_samples is not None:
        wrapped = torch.utils.data.Subset(wrapped, range(min(num_samples, len(wrapped))))
    return torch.utils.data.DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=EvalDataCollator(),
    )


def _load_model(checkpoint_dir: Path, device: "torch.device"):
    import torch
    from modules.builder import build_model
    from vocos import Vocos

    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")
    with config_path.open() as config_file:
        cfg_dict = json.load(config_file)

    model = build_model(cfg_dict)
    model.from_pretrained(str(checkpoint_dir))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    return model, vocoder, cfg_dict.get("model_name", checkpoint_dir.name)


def _load_kmeans(path: Path) -> tuple[dict, dict | None]:
    import torch

    if path.is_dir():
        summary_path = path / "summary.json"
        codebook_path = path / "encoder_kmeans.pt"
    else:
        codebook_path = path
        summary_path = path.parent / "summary.json"
    if not codebook_path.is_file():
        raise FileNotFoundError(f"KMeans codebook not found: {codebook_path}")
    codebook = torch.load(codebook_path, map_location="cpu")
    if "centroids" not in codebook:
        raise ValueError(f"KMeans codebook has no centroids: {codebook_path}")
    summary = None
    if summary_path.is_file():
        with summary_path.open() as summary_file:
            summary = json.load(summary_file)
    codebook["path"] = str(codebook_path)
    codebook["summary"] = summary
    return codebook, summary


def _kmeans_source(codebook: dict) -> str:
    selection = codebook.get("latent_selection") or {}
    return str(selection.get("source") or codebook.get("source") or "z_sem")


def _load_shared_shift_matrix(path: Path, device: "torch.device"):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    factorizer_type = payload.get("factorizer_type")
    if factorizer_type != "shared_shift" and "B" not in payload and "B0" not in payload:
        raise ValueError(f"Expected shared_shift factorizer, got {path}")
    if "B" in payload:
        matrix = payload["B"]
    elif "B0" in payload and "delta" in payload:
        matrix = payload["B0"] + payload["delta"]
    else:
        raise KeyError(f"Shared-shift payload has no B or B0+delta keys: {path}")
    return matrix.to(device=device, dtype=torch.float32)


def _save_audio(path: Path, audio: "torch.Tensor", sample_rate: int) -> None:
    import torchaudio

    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().cpu()
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak.clamp_min(1e-8)
    torchaudio.save(str(path), audio, sample_rate)


def _decode_native(model, vocoder, audios_srs, audio_16khz, args, model_device):
    import torch

    generator = torch.Generator(device=model_device).manual_seed(args.seed)
    out = model.encode_decode(
        audios_srs=audios_srs,
        num_steps=args.num_steps,
        temperature=args.temperature,
        guidance_scale=args.guidance_scale,
        generator=generator,
        audio_16khz=audio_16khz,
    )
    reconstructed_mel = out["decoder_output"].audio_features
    padding_mask = out["decoder_output"].padding_mask
    return [
        vocoder.decode(mel[~mask].unsqueeze(0).permute(0, 2, 1)).squeeze(0)
        for mel, mask in zip(reconstructed_mel, padding_mask)
        if not mask.all()
    ]


def _decode_kmeans(
    model,
    vocoder,
    audios_srs,
    audio_16khz,
    args,
    model_device,
    codebook,
    factorizer_matrix,
):
    import torch

    enc_features, enc_padding_mask, _, _, _ = model.extract_features(
        audios_srs,
        audio_16khz=audio_16khz,
    )
    encoded = model.encode(enc_features, enc_padding_mask)
    attrs = model.encode_attributes(encoded.z, padding_mask=encoded.padding_mask)

    source = _kmeans_source(codebook)
    z_pros = attrs.z_pros
    z_sem = attrs.z_sem
    if source == "z_sem_new":
        if factorizer_matrix is None:
            raise ValueError("KMeans source is z_sem_new but no factorizer was loaded.")
        shared = attrs.z_pros @ factorizer_matrix
        z_sem = attrs.z_sem + shared
        z_pros = attrs.z_pros - shared
    elif source != "z_sem":
        raise ValueError(f"Unsupported KMeans latent source: {source}")

    kmeans_encoded = model._kmeans_encode(
        z_sem,
        padding_mask=encoded.padding_mask,
        kmeans_codebook=codebook,
        chunk_size=args.kmeans_chunk_size,
    )
    z = kmeans_encoded["quantized"] + z_pros + attrs.z_mean
    speaker_embedding = model.extract_speaker_embedding(audios_srs)

    generator = torch.Generator(device=model_device).manual_seed(args.seed)
    mel, mel_mask = model.sample(
        num_steps=args.num_steps,
        temperature=args.temperature,
        guidance_scale=args.guidance_scale,
        z=z,
        generator=generator,
        padding_mask=encoded.padding_mask,
        speaker_embedding=speaker_embedding,
    )
    return [
        vocoder.decode(item[~mask].unsqueeze(0).permute(0, 2, 1)).squeeze(0)
        for item, mask in zip(mel, mel_mask)
        if not mask.all()
    ]


def main() -> None:
    args = _parse_args()
    hf_home = _set_external_caches(args.hf_home)
    sys.path.insert(0, str(_repo_root()))

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import set_seed
    from evaluation.scripts.dwer import DWER
    from evaluation.scripts.utmos import UTMOS
    from evaluation.scripts.speaker_similarity import SpkSimWavLM

    set_seed(args.seed)
    model_device = _resolve_device(args.model_device)
    metrics_device = _resolve_device(args.metrics_device)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_local_dataset(
        Path(args.dataset_dir).expanduser(),
        max_audio_len=args.max_audio_len if args.max_audio_len > 0 else None,
    )
    dataloader = _make_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
    )
    model, vocoder, model_name = _load_model(Path(args.checkpoint_dir), model_device)

    codebook = None
    kmeans_summary = None
    factorizer_matrix = None
    if args.mode == "kmeans":
        if args.kmeans_path is None:
            raise ValueError("--mode kmeans requires --kmeans-path")
        codebook, kmeans_summary = _load_kmeans(Path(args.kmeans_path).expanduser())
        if _kmeans_source(codebook) == "z_sem_new":
            factorizer_matrix = _load_shared_shift_matrix(
                Path(args.factorizer_path).expanduser(),
                model_device,
            )

    dwer = DWER(args.whisper_size, device=metrics_device)
    utmos_ref = UTMOS(sample_rate=16000, device=metrics_device)
    utmos_hyp = UTMOS(sample_rate=24000, device=metrics_device)
    sim = SpkSimWavLM(args.sim_model, device=metrics_device)

    rows = []
    audio_dir = output_dir / "audio"
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating test_clean", unit="batch"):
            references = [audio.to(metrics_device) for audio in batch["16k_audio"]]
            audios_srs = [
                (audio.to(model_device), sr) for audio, sr in batch["audio_input_srs"]
            ]
            audio_16khz = [
                audio.to(model_device) for audio in batch["16k_audio"]
            ]

            if args.mode == "native":
                hypotheses = _decode_native(
                    model,
                    vocoder,
                    audios_srs,
                    audio_16khz,
                    args,
                    model_device,
                )
            else:
                hypotheses = _decode_kmeans(
                    model,
                    vocoder,
                    audios_srs,
                    audio_16khz,
                    args,
                    model_device,
                    codebook,
                    factorizer_matrix,
                )
            hypotheses = [audio.to(metrics_device) for audio in hypotheses]

            ids = [str(item) for item in batch["ids"]]
            utmos_ref.append(ids, references)
            utmos_hyp.append(ids, hypotheses)
            dwer.append(
                hyp_sr=24000,
                ref_sr=16000,
                ids=ids,
                hyp_sig=hypotheses,
                ref_sig=references,
            )
            sim.append(
                hyp_sr=24000,
                ref_sr=16000,
                ids=ids,
                hyp_sig=hypotheses,
                ref_sig=references,
            )

            for id_, ref, hyp in zip(ids, references, hypotheses):
                row = {"id": id_}
                if args.save_audio:
                    ref_path = audio_dir / f"{id_}_ref.wav"
                    hyp_path = audio_dir / f"{id_}_hyp.wav"
                    _save_audio(ref_path, ref.unsqueeze(0), 16000)
                    _save_audio(hyp_path, hyp.unsqueeze(0), 24000)
                    row.update({"reference_wav": str(ref_path), "hypothesis_wav": str(hyp_path)})
                rows.append(row)

    dwer_summary = dwer.summarize()
    results = {
        "model_name": model_name,
        "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser()),
        "dataset_dir": str(Path(args.dataset_dir).expanduser()),
        "hf_home": str(hf_home),
        "mode": args.mode,
        "num_samples": len(rows),
        "num_steps": args.num_steps,
        "temperature": args.temperature,
        "guidance_scale": args.guidance_scale,
        "model_device": str(model_device),
        "metrics_device": str(metrics_device),
        "dwer": float(dwer_summary["error_rate"]),
        "dcer": float(dwer_summary["error_rate_char"]),
        "utmos_ref": float(utmos_ref.summarize("average")),
        "utmos_hyp": float(utmos_hyp.summarize("average")),
        "sim": float(sim.summarize("average")),
        "kmeans_path": args.kmeans_path,
        "kmeans_source": _kmeans_source(codebook) if codebook is not None else None,
        "kmeans_summary": kmeans_summary,
        "factorizer_path": args.factorizer_path if factorizer_matrix is not None else None,
    }

    result_path = output_dir / "summary.json"
    with result_path.open("w") as result_file:
        json.dump(results, result_file, indent=2, sort_keys=True)

    rows_path = output_dir / "samples.csv"
    with rows_path.open("w", newline="") as rows_file:
        fieldnames = sorted({key for row in rows for key in row.keys()}) or ["id"]
        writer = csv.DictWriter(rows_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Saved summary to {result_path}")
    print(f"Saved samples to {rows_path}")


if __name__ == "__main__":
    main()
