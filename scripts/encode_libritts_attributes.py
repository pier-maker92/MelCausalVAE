import argparse
import json
import os
import random
import re
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

HF_HOME = Path("/Volumes/Crucial X6/HF_HOME")
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import safetensors.torch
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dicodec.modules.configs import (
    DropoutConfig,
    EncoderConfig,
    KLChunkRegularizer,
    LowPassFilterConfig,
    MelSpectrogramConfig,
    NoiseConfig,
    WavLMConfig,
)
from dicodec.modules.encoder.encoder import Encoder
from dicodec.modules.feature_extractor import FeatureExtractor, WavLMFeatureExtractor
from dicodec.modules.lp_filter import LowPassFilter


INPUT_DIR = Path("/Volumes/Crucial X6/Research/Datasets/libritts-r/train-clean-100")
OUTPUT_DIR = Path("/Volumes/Crucial X6/Research/dicodec-attributes")
CHECKPOINT_DIR = Path(
    "/Users/software/Research/MelCausalVAE/checkpoints/paper/baseline/18-denc128-novq"
)
TARGET_FRAMES = 1_000_000
SHUFFLE = True
SEED = 1234
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def filter_dataclass_kwargs(config_cls, values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {field.name for field in fields(config_cls)}
    return {key: value for key, value in values.items() if key in allowed}


def build_encoder_config(cfg_dict: Dict[str, Any]) -> EncoderConfig:
    encoder_cfg = cfg_dict.get("encoder_config", cfg_dict.get("encoder", {})).copy()
    encoder_cfg.setdefault("use_reparameterization_trick", False)
    encoder_cfg.setdefault("use_std_sweep", False)

    dropout_dict = encoder_cfg.pop("dropout_regularizer_config", None)
    dropout_config = (
        DropoutConfig(**filter_dataclass_kwargs(DropoutConfig, dropout_dict))
        if dropout_dict
        else None
    )

    kl_dict = encoder_cfg.pop("kl_chunk_regularizer_config", None)
    kl_config = (
        KLChunkRegularizer(**filter_dataclass_kwargs(KLChunkRegularizer, kl_dict))
        if kl_dict
        else None
    )

    noise_dict = encoder_cfg.pop("noise_regularizer_config", None)
    noise_config = (
        NoiseConfig(**filter_dataclass_kwargs(NoiseConfig, noise_dict))
        if noise_dict
        else None
    )

    return EncoderConfig(
        dropout_regularizer_config=dropout_config,
        kl_chunk_regularizer_config=kl_config,
        noise_regularizer_config=noise_config,
        **filter_dataclass_kwargs(EncoderConfig, encoder_cfg),
    )


def build_lowpass_filter_config(cfg_dict: Dict[str, Any]) -> LowPassFilterConfig:
    lowpass_filter_dict = cfg_dict.get(
        "lowpass_filter_config", cfg_dict.get("lowpass_filter", None)
    )
    if lowpass_filter_dict and "kernel_size" in lowpass_filter_dict:
        lowpass_filter_dict = lowpass_filter_dict.copy()
        lowpass_filter_dict.setdefault("order", lowpass_filter_dict["kernel_size"] - 1)
        lowpass_filter_dict.pop("kernel_size", None)
    return (
        LowPassFilterConfig(
            **filter_dataclass_kwargs(LowPassFilterConfig, lowpass_filter_dict)
        )
        if lowpass_filter_dict
        else LowPassFilterConfig()
    )


def load_prefixed_state(
    module: torch.nn.Module, state_dict: Dict[str, torch.Tensor], prefix: str
) -> None:
    prefix_with_dot = f"{prefix}."
    module_state = {
        key[len(prefix_with_dot) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix_with_dot)
    }
    missing, unexpected = module.load_state_dict(module_state, strict=False)
    print(
        f"Loaded {prefix}: {len(module_state)} tensors "
        f"({len(missing)} missing, {len(unexpected)} unexpected)."
    )


def build_attribute_modules(checkpoint_dir: Path, device: torch.device):
    config_path = checkpoint_dir / "config.json"
    checkpoint_path = checkpoint_dir / "model.safetensors"
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    mel_config = MelSpectrogramConfig(
        **filter_dataclass_kwargs(
            MelSpectrogramConfig, cfg_dict.get("mel_spectrogram_config", {})
        )
    )
    mel_config.n_mels = cfg_dict["mel_dim"]
    mel_config.sampling_rate = cfg_dict["sample_rate"]
    encoder_config = build_encoder_config(cfg_dict)
    lowpass_config = build_lowpass_filter_config(cfg_dict)
    if lowpass_config.sample_rate is None:
        lowpass_config.sample_rate = (
            cfg_dict["sample_rate"]
            / mel_config.hop_length
            / cfg_dict["compress_factor"]
        )
        lowpass_config.__post_init__()

    feature_extractor = FeatureExtractor(mel_config)
    encoder = Encoder(encoder_config)
    lowpass_filter = LowPassFilter(
        cutoff_hz=lowpass_config.cutoff_hz,
        sample_rate=lowpass_config.sample_rate,
        order=lowpass_config.order,
    )

    wavlm = None
    wavlm_extractor = None
    wavlm_module_config = cfg_dict.get("wavlm_module_config", None)
    if wavlm_module_config and wavlm_module_config.get("feature_extractor_config"):
        encoder_config.mel_dim = 1024
        from transformers import WavLMModel

        wavlm = WavLMModel.from_pretrained(
            wavlm_module_config["pretrained_model_name"],
            use_safetensors=False,
        )
        wavlm.eval()
        for parameter in wavlm.parameters():
            parameter.requires_grad_(False)
        wavlm_config = WavLMConfig(
            **filter_dataclass_kwargs(
                WavLMConfig, wavlm_module_config["feature_extractor_config"]
            )
        )
        wavlm_extractor = WavLMFeatureExtractor(wavlm_config, wavlm=wavlm)
    else:
        encoder_config.mel_dim = cfg_dict["mel_dim"]

    encoder_config.latent_dim = cfg_dict["latent_dim"]
    encoder_config.compress_factor_C = cfg_dict["compress_factor"]

    state_dict = safetensors.torch.load_file(str(checkpoint_path), device="cpu")
    load_prefixed_state(feature_extractor, state_dict, "feature_extractor")
    load_prefixed_state(encoder, state_dict, "encoder")

    feature_extractor.to(device).eval()
    encoder.to(device).eval()
    lowpass_filter.to(device).eval()
    if wavlm is not None:
        wavlm.to(device).eval()
    if wavlm_extractor is not None:
        wavlm_extractor.to(device).eval()

    return {
        "sample_rate": cfg_dict["sample_rate"],
        "dtype": encoder.dtype,
        "device": device,
        "feature_extractor": feature_extractor,
        "wavlm_extractor": wavlm_extractor,
        "encoder": encoder,
        "lowpass_filter": lowpass_filter,
    }


def sanitize_id(raw_id: str) -> str:
    clean = raw_id.strip().replace(os.sep, "__")
    clean = re.sub(r"[^A-Za-z0-9_.=-]+", "_", clean)
    return clean.strip("_") or "sample"


def get_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_audio_file(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    wav = wav.squeeze(0).to(torch.float32)
    return wav / (wav.abs().max() + 1e-8)


def load_audio_array(audio: Dict, target_sr: int) -> torch.Tensor:
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        wav = samples.data
        sr = int(samples.sample_rate)
    else:
        wav = torch.as_tensor(audio["array"], dtype=torch.float32)
        sr = int(audio["sampling_rate"])
    if wav.dim() > 1:
        wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    return wav / (wav.abs().max() + 1e-8)


def iter_audio_files(input_dir: Path, target_sr: int) -> Iterator[Dict]:
    paths = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if SHUFFLE:
        random.Random(SEED).shuffle(paths)

    for path in paths:
        rel = path.relative_to(input_dir).with_suffix("")
        yield {
            "id": sanitize_id(str(rel)),
            "wav": load_audio_file(path, target_sr),
        }


def iter_hf_dataset(input_dir: Path, target_sr: int) -> Optional[Iterator[Dict]]:
    parquet_files = sorted(str(path) for path in input_dir.rglob("*.parquet"))
    arrow_files = sorted(
        str(path)
        for path in input_dir.rglob("*.arrow")
        if not path.name.startswith("._")
    )
    is_saved_dataset = (input_dir / "state.json").exists() and bool(arrow_files)
    if not parquet_files and not is_saved_dataset:
        return None

    try:
        from datasets import Audio, Dataset, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Found a Hugging Face dataset, but the 'datasets' package is not installed."
        ) from exc

    if is_saved_dataset:
        shard_paths = arrow_files.copy()
        if SHUFFLE:
            random.Random(SEED).shuffle(shard_paths)
        print(f"Found {len(shard_paths):,} Arrow shards.")

        def _iterator() -> Iterator[Dict]:
            order_idx = 0
            for shard_idx, path in enumerate(shard_paths):
                dataset = Dataset.from_file(path)
                if "audio" in dataset.column_names:
                    dataset = dataset.cast_column(
                        "audio", Audio(sampling_rate=target_sr)
                    )
                row_indices = list(range(len(dataset)))
                if SHUFFLE:
                    random.Random(SEED + shard_idx).shuffle(row_indices)
                print(
                    f"Reading shard {shard_idx + 1:,}/{len(shard_paths):,}: "
                    f"{Path(path).name} ({len(dataset):,} examples)"
                )
                for idx in row_indices:
                    row = dataset[idx]
                    raw_id = (
                        row.get("id")
                        or row.get("speaker_id")
                        and row.get("chapter_id")
                        and row.get("utterance_id")
                        and f"{row['speaker_id']}_{row['chapter_id']}_{row['utterance_id']}"
                        or f"sample_{order_idx:08d}"
                    )
                    order_idx += 1
                    yield {
                        "id": sanitize_id(str(raw_id)),
                        "wav": load_audio_array(row["audio"], target_sr),
                    }

        return _iterator()
    else:
        dataset = load_dataset("parquet", data_files=parquet_files, split="train")

    if "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sr))

    indices = list(range(len(dataset)))
    if SHUFFLE:
        random.Random(SEED).shuffle(indices)
    print(f"Loaded Hugging Face dataset with {len(dataset):,} examples.")

    def _iterator() -> Iterator[Dict]:
        for order_idx, idx in enumerate(indices):
            row = dataset[idx]
            raw_id = (
                row.get("id")
                or row.get("speaker_id")
                and row.get("chapter_id")
                and row.get("utterance_id")
                and f"{row['speaker_id']}_{row['chapter_id']}_{row['utterance_id']}"
                or f"sample_{order_idx:08d}"
            )
            yield {
                "id": sanitize_id(str(raw_id)),
                "wav": load_audio_array(row["audio"], target_sr),
            }

    return _iterator()


def iter_examples(input_dir: Path, target_sr: int) -> Iterator[Dict]:
    hf_iterator = iter_hf_dataset(input_dir, target_sr)
    if hf_iterator is not None:
        yield from hf_iterator
        return
    yield from iter_audio_files(input_dir, target_sr)


def save_npy(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().float().cpu().numpy()
    np.save(path, array)


def existing_frame_count(output_dir: Path) -> int:
    total = 0
    for path in output_dir.glob("*/z.npy"):
        try:
            total += int(np.load(path, mmap_mode="r").shape[0])
        except Exception:
            pass
    return total


def extract_features(modules: Dict[str, Any], audios_srs, audio_16khz=None):
    feature_extractor = modules["feature_extractor"]
    wavlm_extractor = modules["wavlm_extractor"]
    target_output = feature_extractor(audios_srs)
    target_length = target_output.audio_features.shape[1]
    if wavlm_extractor is None:
        return (
            target_output.audio_features.to(modules["dtype"]),
            target_output.padding_mask,
        )

    wavlm_output = wavlm_extractor(audios_srs, audio_16khz=audio_16khz)
    wavlm_features = wavlm_output.audio_features.to(modules["dtype"])
    wavlm_features = wavlm_features.repeat_interleave(2, dim=1)
    wavlm_features = (
        F.interpolate(
            wavlm_features.float().transpose(1, 2),
            size=target_length,
            mode="linear",
            align_corners=False,
        )
        .transpose(1, 2)
        .to(wavlm_features.dtype)
    )
    wavlm_padding_mask = (
        F.interpolate(
            wavlm_output.padding_mask.float().unsqueeze(1),
            size=target_length,
            mode="nearest",
        )
        .squeeze(1)
        .bool()
    )
    return wavlm_features, wavlm_padding_mask


def encode_attributes(
    lowpass_filter: LowPassFilter,
    z: torch.Tensor,
    padding_mask: Optional[torch.BoolTensor] = None,
):
    if padding_mask is not None:
        valid_mask = (~padding_mask).to(device=z.device, dtype=z.dtype).unsqueeze(-1)
        valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        z_mean = (z * valid_mask).sum(dim=1, keepdim=True) / valid_count
    else:
        valid_mask = None
        z_mean = z.mean(dim=1, keepdim=True)

    z_centered = z - z_mean
    if valid_mask is not None:
        z_centered = z_centered * valid_mask

    z_lp = lowpass_filter(z_centered, valid_mask=valid_mask)
    dot_product = torch.sum(z_centered * z_lp, dim=1, keepdim=True)
    norm_sq = torch.sum(z_lp.square(), dim=1, keepdim=True)
    z_pros = (dot_product / (norm_sq + 1e-8)) * z_lp
    z_sem = z_centered - z_pros
    if valid_mask is not None:
        z_pros = z_pros * valid_mask
        z_sem = z_sem * valid_mask
    return z_sem, z_pros, z_mean


@torch.inference_mode()
def encode_one(
    modules: Dict[str, Any],
    wav: torch.Tensor,
    sample_id: str,
    output_dir: Path,
    remaining: int,
):
    device = modules["device"]
    wav = wav.to(device)
    audios_srs = [(wav, modules["sample_rate"])]
    audio_16khz = None
    if modules["wavlm_extractor"] is not None:
        wavlm_sr = modules["wavlm_extractor"].sampling_rate
        audio_16khz = [
            T.Resample(modules["sample_rate"], wavlm_sr)(wav.cpu()).to(device)
        ]

    enc_features, enc_padding_mask = extract_features(
        modules, audios_srs, audio_16khz=audio_16khz
    )
    encoder_output = modules["encoder"](enc_features, enc_padding_mask)
    z = encoder_output.z
    padding_mask = encoder_output.padding_mask
    z_sem, z_pros, z_mean = encode_attributes(
        modules["lowpass_filter"], z, padding_mask=padding_mask
    )

    valid = ~padding_mask[0]
    frame_count = int(valid.sum().item())
    if frame_count == 0:
        return 0

    keep = min(frame_count, remaining)
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    save_npy(sample_dir / "z.npy", z[0, valid][:keep])
    save_npy(sample_dir / "z_sem.npy", z_sem[0, valid][:keep])
    save_npy(sample_dir / "z_pros.npy", z_pros[0, valid][:keep])
    save_npy(sample_dir / "z_mean.npy", z_mean[0].expand(keep, -1))
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode LibriTTS-R attributes and save z/z_sem/z_pros/z_mean npy files."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--target-frames", type=int, default=TARGET_FRAMES)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print(f"SHUFFLE = {SHUFFLE}")
    print(f"Loading attribute modules from {args.checkpoint_dir} on {device}...")
    modules = build_attribute_modules(args.checkpoint_dir, device)
    print("Attribute modules ready.")

    saved_frames = existing_frame_count(args.output_dir) if args.resume else 0
    print(f"Starting from {saved_frames:,}/{args.target_frames:,} frames.")

    with tqdm(
        initial=saved_frames,
        total=args.target_frames,
        unit="frames",
        dynamic_ncols=True,
    ) as progress:
        for example in iter_examples(args.input_dir, modules["sample_rate"]):
            if saved_frames >= args.target_frames:
                break

            sample_id = example["id"]
            sample_dir = args.output_dir / sample_id
            if args.resume and (sample_dir / "z.npy").exists():
                continue

            remaining = args.target_frames - saved_frames
            try:
                written = encode_one(
                    modules=modules,
                    wav=example["wav"],
                    sample_id=sample_id,
                    output_dir=args.output_dir,
                    remaining=remaining,
                )
            except Exception as exc:
                progress.write(f"Skipping {sample_id}: {exc}")
                continue

            saved_frames += written
            progress.update(written)
            progress.set_postfix_str(sample_id[:40])

    print(f"Done. Saved {saved_frames:,} frames under {args.output_dir}")


if __name__ == "__main__":
    main()
