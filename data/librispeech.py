import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import concatenate_datasets, load_dataset

from data.audio_dataset import SimpleAudioDataset

SLURM_TMPDIR = os.getenv("SLURM_TMPDIR")
parquet_dir = f"{SLURM_TMPDIR}/datasets/librispeech-aligned"


class LibriSpeechDataset(SimpleAudioDataset):
    def __init__(self, debug: bool = False):
        super().__init__()
        dataset = load_dataset("parquet", data_dir=parquet_dir)
        self.dataset = dataset

        partitions_per_destination = defaultdict(list)
        for partition in dataset:
            destination = self._partition_to_destination(partition)
            print(f"partition: {partition}, destination: {destination}")
            partitions_per_destination[destination].append(dataset[partition])

        for destination, parts in partitions_per_destination.items():
            setattr(self, f"{destination}_dataset", concatenate_datasets(parts))

    def _partition_to_destination(self, partition_name):
        if partition_name == "train":
            return "train"
        return "test"
