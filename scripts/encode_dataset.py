"""Export selected audio partitions as Parquet containing only z and attributes.

The run_job experiment is the source of truth for both dataset_partitions (copy)
and encoding (checkpoint/output). No training dataset merging or filtering occurs.
"""
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile

import hydra
from omegaconf import DictConfig, OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logger = logging.getLogger(__name__)

MATRIX = pa.list_(pa.list_(pa.float32()))
SCHEMA = pa.schema([
    ("z", MATRIX),
    ("attributes", pa.struct([
        ("z_sem", MATRIX), ("z_pros", MATRIX), ("z_mean", MATRIX),
    ])),
])
PARTITIONS = {
    "train_clean_100", "train_clean_360", "train_other_500",
    "dev_clean", "dev_other", "test_clean", "test_other",
}


def read_experiment(path):
    experiment = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if experiment.get("dataset_name") != "librispeech-aligned":
        raise ValueError("This export requires dataset_name: librispeech-aligned.")
    if experiment.get("num_gpus") != 1:
        raise ValueError("Encoding requires num_gpus: 1 (one writer per partition).")
    if experiment.get("dataset_extension") != "parquet":
        raise ValueError("dataset_extension must be parquet.")
    partitions = experiment.get("dataset_partitions")
    if not isinstance(partitions, list) or not partitions:
        raise ValueError("Set a non-empty dataset_partitions list in the experiment.")
    if any(part not in PARTITIONS for part in partitions):
        raise ValueError(f"Unknown partition; choose from {sorted(PARTITIONS)}.")
    if len(set(partitions)) != len(partitions):
        raise ValueError("dataset_partitions must not contain duplicates.")
    if experiment.get("dataset_filter"):
        raise ValueError("Use dataset_partitions, not dataset_filter.")
    encoding = experiment["encoding"]
    label = encoding["model_version"]
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise ValueError("encoding.model_version must be a safe directory suffix.")
    if not isinstance(encoding["shard_size_mb"], int) or encoding["shard_size_mb"] <= 0:
        raise ValueError("encoding.shard_size_mb must be a positive integer (MiB).")
    checkpoint = Path(encoding["checkpoint"])
    for filename in ("config.json", "model.safetensors"):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(checkpoint / filename)
    if encoding.get("input_root"):
        source = Path(encoding["input_root"])
    else:
        tmpdir = os.environ.get("SLURM_TMPDIR")
        if not tmpdir:
            raise ValueError("SLURM_TMPDIR is unset; set encoding.input_root for local runs.")
        source = Path(tmpdir) / "datasets" / experiment.get("dataset_path", experiment["dataset_name"])
    destination = Path(encoding["output_root"]) / f"librispeech-dicodec-{label}"
    files = {}
    for part in partitions:
        files[part] = sorted((source / part).glob("*.parquet"))
        if not files[part]:
            raise FileNotFoundError(f"No input Parquet files for selected partition: {source / part}")
        if (destination / part).exists():
            raise FileExistsError(f"Output partition already exists: {destination / part}")
    return experiment, files, destination


class ShardWriter:
    """Bounded Arrow buffer, actual on-disk size check, and no empty shards.

    Use uncompressed Parquet without dictionaries so buffered Arrow size tracks
    the desired file size closely. If Parquet overhead exceeds the hard limit,
    find the largest fitting row prefix and carry the remaining rows forward.
    """
    def __init__(self, directory, size_limit, schema=SCHEMA):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.size_limit = size_limit
        self.schema = schema
        self.tables = []
        self.buffered_bytes = 0
        self.index = 0
        self.rows_written = 0

    def append(self, row):
        table = pa.Table.from_pylist([row], schema=self.schema)
        self.tables.append(table)
        self.buffered_bytes += table.nbytes
        if self.buffered_bytes >= self.size_limit:
            self.flush()

    def flush(self):
        if not self.tables:
            return
        remaining = pa.concat_tables(self.tables)
        self.tables = []
        self.buffered_bytes = 0
        while remaining.num_rows:
            temporary = self.directory / ".shard.partial"

            def write_prefix(count):
                pq.write_table(
                    remaining.slice(0, count), temporary,
                    compression=None, use_dictionary=False,
                )
                return temporary.stat().st_size

            count = remaining.num_rows
            if write_prefix(count) > self.size_limit:
                low, high = 0, count - 1
                while low < high:
                    middle = (low + high + 1) // 2
                    if write_prefix(middle) <= self.size_limit:
                        low = middle
                    else:
                        high = middle - 1
                count = low
                if count == 0:
                    raise ValueError("A single encoded sample exceeds shard_size_mb.")
                write_prefix(count)
            final = self.directory / f"shard_{self.index:05d}.parquet"
            temporary.rename(final)
            self.index += 1
            self.rows_written += count
            remaining = remaining.slice(count)
            if remaining.num_rows and remaining.nbytes < self.size_limit:
                self.tables = [remaining]
                self.buffered_bytes = remaining.nbytes
                return

    def close(self):
        while self.tables:
            self.flush()


def iter_audio_rows(files):
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=16, columns=["audio"]):
            for row in batch.to_pylist():
                yield row["audio"], path.parent


def prepare_audio(audio, source_directory, device, sample_rate):
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    if audio.get("bytes") is not None:
        stream = io.BytesIO(audio["bytes"])
    else:
        stream = Path(audio["path"])
        if not stream.is_absolute():
            stream = source_directory / stream
    array, sr = sf.read(stream, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(array).mean(dim=1)
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError("Input contains empty or non-finite audio.")

    def resample(target_rate):
        result = AF.resample(waveform, sr, target_rate) if sr != target_rate else waveform
        result = result / (result.abs().max() + 1e-8)
        return result.to(device=device)

    return [(resample(sample_rate), sample_rate)], [resample(16000)]


def encoded_row(output):
    import torch

    z = output.z[0]
    valid = ~output.padding_mask[0] if output.padding_mask is not None else torch.ones(
        z.shape[0], dtype=torch.bool, device=z.device
    )
    if not valid.any():
        raise ValueError("Encoder returned no valid frames.")
    tensors = {"z": z[valid], "z_sem": output.attributes.z_sem[0][valid],
               "z_pros": output.attributes.z_pros[0][valid],
               "z_mean": output.attributes.z_mean[0]}
    if any(not torch.isfinite(tensor).all() for tensor in tensors.values()):
        raise ValueError("Encoder returned non-finite latents or attributes.")
    values = {key: tensor.detach().float().cpu().tolist() for key, tensor in tensors.items()}
    return {"z": values.pop("z"), "attributes": values}


def encode_partitions(model, files, destination, size_limit):
    import torch
    from tqdm import tqdm

    destination.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for part, paths in files.items():
            count = sum(pq.read_metadata(path).num_rows for path in paths)
            if count == 0:
                raise ValueError(f"Selected partition {part} has no rows.")
            final = destination / part
            if final.exists():
                raise FileExistsError(final)
            # Publish the partition only after every row has been written successfully.
            with tempfile.TemporaryDirectory(prefix=f".{part}-", dir=destination) as staging:
                writer = ShardWriter(staging, size_limit)
                for audio, parent in tqdm(iter_audio_rows(paths), total=count, desc=part):
                    audios, audio16 = prepare_audio(audio, parent, model.device, model.config.sample_rate)
                    features, mask, _, _ = model.extract_features(audios, audio_16khz=audio16)
                    output = model.encode(features, mask, compute_attributes=True)
                    writer.append(encoded_row(output))
                writer.close()
                if writer.rows_written != count:
                    raise RuntimeError(f"Row count mismatch for {part}: {writer.rows_written} != {count}")
                Path(staging).rename(final)
            logger.info("%s: %d rows, %d shards -> %s", part, count, writer.index, final)


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    import torch
    from dicodec.modules.builder import build_model
    import safetensors.torch

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Launch encoding with one process/GPU to avoid duplicate writers.")
    experiment, files, destination = read_experiment(cfg.encoding.experiment_config)
    encoding = experiment["encoding"]
    checkpoint = Path(encoding["checkpoint"])
    with (checkpoint / "config.json").open() as stream:
        model_config = json.load(stream)
    # The checkpoint config, including compression factor, owns the architecture.
    model = build_model(model_config)
    state = safetensors.torch.load_file(str(checkpoint / "model.safetensors"))
    incompatible = model.load_state_dict(state, strict=False)
    # Learned encoder weights and normalization statistics must match the checkpoint.
    critical = ("encoder.", "wavlm.", "feature_extractor.", "wavlm_extractor.")
    mismatched = [key for key in incompatible.missing_keys + incompatible.unexpected_keys
                  if key.startswith(critical)]
    if mismatched:
        raise ValueError(f"Checkpoint does not match the encoder: {mismatched}")
    del state
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    # Encode one utterance at a time: normalization must not depend on batch padding.
    logger.info("Checkpoint: %s; partitions: %s; destination: %s", checkpoint, list(files), destination)
    encode_partitions(model, files, destination, encoding["shard_size_mb"] * 1024 * 1024)


if __name__ == "__main__":
    main()
