#!/usr/bin/env python3
"""
Carica gli output VQ in ``audio_outputs/`` e pubblica una ``wandb.Table`` con anteprime audio.

Uso (dalla root del repo, con ``wandb login`` già fatto):

  python wandb_log_vq_audio_table.py --project MelCausalVAE --run-name vq-audio-abl

Opzionale: ``--entity``, ``--offline``, ``--table-key`` per il nome della chiave nel log.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent

# La cartella di log ``MelCausalVAE/wandb/`` ha lo stesso nome del pacchetto: ``import wandb``
# risolverebbe alla directory di log se repo root o cwd è sulla path. Togliamo tali voci
# solo per caricare l'SDK.
_path_backup = sys.path.copy()
_cwd = Path.cwd().resolve()


def _path_is_repo_root(p: str) -> bool:
    if p in ("", "."):
        return _cwd == REPO_ROOT.resolve()
    try:
        return Path(p).resolve() == REPO_ROOT.resolve()
    except OSError:
        return False


sys.path = [p for p in sys.path if not _path_is_repo_root(p)]
try:
    wandb = importlib.import_module("wandb")
except ImportError as e:
    sys.path[:] = _path_backup
    raise SystemExit("Installa wandb: pip install wandb") from e
sys.path[:] = _path_backup

import wave

import numpy as np

try:
    import soundfile as sf

    _HAS_SF = True
except ImportError:
    sf = None  # type: ignore[assignment]
    _HAS_SF = False


def _read_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    """Mono float32 [-1,1], sample rate Hz."""
    if _HAS_SF:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data.squeeze(-1)
        return np.ascontiguousarray(data), int(sr)
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sw} ({path})")
    x = x.reshape(-1, nch)
    if nch > 1:
        x = x.mean(axis=1)
    else:
        x = x.squeeze(-1)
    return np.ascontiguousarray(x), int(sr)


INPUT_MALE = REPO_ROOT / "ablations" / "male.wav"
TARGET_FEMALE = REPO_ROOT / "ablations" / "female.wav"
OUT_DIR = REPO_ROOT / "audio_outputs"

# Righe: output relativo a REPO_ROOT, input/target opzionali, nota breve.
ROWS: List[Dict[str, Any]] = [
    {
        "name": "only_quantizer",
        "output": OUT_DIR / "only_quantizer.wav",
        "input": INPUT_MALE,
        "target": None,
        "notes": "Solo parte quantizzata (VQ head); input: male.wav",
    },
    {
        "name": "only_residual",
        "output": OUT_DIR / "only_residual.wav",
        "input": INPUT_MALE,
        "target": None,
        "notes": "Solo residuo sulla testa VQ; input: male.wav",
    },
    {
        "name": "only_tail",
        "output": OUT_DIR / "only_tail.wav",
        "input": INPUT_MALE,
        "target": None,
        "notes": "Solo tail non quantizzato; input: male.wav",
    },
    {
        "name": "quantizer_and_residual_and_tail",
        "output": OUT_DIR / "quantizer_and_residual_and_tail.wav",
        "input": INPUT_MALE,
        "target": None,
        "notes": "Quantizzato + residuo + tail (both completo); input: male.wav",
    },
    {
        "name": "quantizer_and_residual",
        "output": OUT_DIR / "quantizer_and_residual.wav",
        "input": INPUT_MALE,
        "target": None,
        "notes": "Quantizzato + residuo (--mode both, tail azzerato); input: male.wav",
    },
    {
        "name": "quantizer_and_tail",
        "output": OUT_DIR / "qunaitzer_and_tail.wav",
        "input": INPUT_MALE,
        "target": None,
        "notes": "Quantizzato + tail (file con typo nel nome); input: male.wav",
    },
    {
        "name": "residual_and_tail",
        "output": OUT_DIR / "residual_and_tail.wav",
        "input": INPUT_MALE,
        "target": None,
        "notes": "Residuo + tail; input: male.wav",
    },
    {
        "name": "swap1_3",
        "output": OUT_DIR / "swap1_3.wav",
        "input": INPUT_MALE,
        "target": TARGET_FEMALE,
        "notes": "Swap / blend: input male.wav, target female.wav",
    },
]


def _wav_to_wandb_audio(path: Path) -> "wandb.Audio":
    data, sr = _read_wav_mono_float32(path)
    return wandb.Audio(data, sample_rate=sr, caption=path.name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Log VQ ablation WAVs to a wandb Table.")
    p.add_argument("--project", type=str, required=True)
    p.add_argument("--run-name", type=str, default="vq-audio-outputs")
    p.add_argument("--entity", type=str, default=None, help="wandb entity (team o user).")
    p.add_argument("--table-key", type=str, default="vq_component_ablations")
    p.add_argument("--offline", action="store_true", help="wandb offline mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    missing: List[str] = []
    for row in ROWS:
        for key in ("output", "input"):
            p = row[key]
            if p is not None and not Path(p).is_file():
                missing.append(str(p))
        t = row.get("target")
        if t is not None and not Path(t).is_file():
            missing.append(str(t))
    if missing:
        print("File mancanti:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        raise SystemExit(1)

    columns = [
        "name",
        "notes",
        "output_audio",
        "input_audio",
        "target_audio",
        "output_path",
        "input_path",
        "target_path",
    ]
    table = wandb.Table(columns=columns)

    for row in ROWS:
        out_p = Path(row["output"])
        in_p = Path(row["input"])
        tgt_p: Optional[Path] = Path(row["target"]) if row.get("target") else None

        out_a = _wav_to_wandb_audio(out_p)
        in_a = _wav_to_wandb_audio(in_p)
        if tgt_p is not None:
            tgt_a: Optional[wandb.Audio] = _wav_to_wandb_audio(tgt_p)
            tgt_path_str = str(tgt_p.relative_to(REPO_ROOT))
        else:
            tgt_a = None
            tgt_path_str = ""

        table.add_data(
            row["name"],
            row["notes"],
            out_a,
            in_a,
            tgt_a,
            str(out_p.relative_to(REPO_ROOT)),
            str(in_p.relative_to(REPO_ROOT)),
            tgt_path_str,
        )

    wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        dir=str(REPO_ROOT / "wandb"),
        mode="offline" if args.offline else None,
    )
    wandb.log({args.table_key: table})
    wandb.finish()
    print(f"Logged wandb.Table '{args.table_key}' to project {args.project!r} run {args.run_name!r}.")


if __name__ == "__main__":
    main()
