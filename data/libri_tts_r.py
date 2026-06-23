import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from datasets import load_dataset, concatenate_datasets
from data.audio_dataset import SimpleAudioDataset

SLURM_TMPDIR = os.getenv("SLURM_TMPDIR")
parquet_dir = f"{SLURM_TMPDIR}/datasets/libritts-r"


class LibriTTSR(SimpleAudioDataset):
    def __init__(self):
        super().__init__()
        dataset = load_dataset("parquet", data_dir=parquet_dir)

        partitions_per_destination = defaultdict(list)
        for partition in dataset:
            dest = self._partition_to_destination(partition)
            print(f"partition: {partition}, destination: {dest}")
            partitions_per_destination[dest].append(dataset[partition])

        for destination, parts in partitions_per_destination.items():
            setattr(self, f"{destination}_dataset", concatenate_datasets(parts))

    def _partition_to_destination(self, partition_name):
        if partition_name == "train":
            return "train"
        return "test"
