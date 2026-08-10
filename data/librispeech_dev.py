import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from data.audio_dataset import SimpleAudioDataset
from datasets import load_dataset, concatenate_datasets

# Specify custom cache directory
SLURM_TMPDIR = os.getenv("SLURM_TMPDIR")
parquet_dir = f"{SLURM_TMPDIR}/datasets/librispeech-aligned"


class LibriSpeechAlignDatasetDev(SimpleAudioDataset):
    def __init__(self):
        super().__init__()
        dataset = load_dataset(
            "parquet",
            data_files={
                "dev": f"{parquet_dir}/dev_clean/*.parquet",
            },
        )
        partitions_per_destination = defaultdict(list)
        for partition in dataset:
            print(
                f"partition: {partition}, destination: {self._partition_to_destination(partition)}"
            )
            partitions_per_destination[
                self._partition_to_destination(partition)
            ].append(dataset[partition])

        for destination in partitions_per_destination:
            setattr(
                self,
                f"{destination}_dataset",
                concatenate_datasets(partitions_per_destination[destination]),
            )

    def _partition_to_destination(self, partition_name):
        if partition_name in ["train"]:
            return "train"
        else:
            return "test"
