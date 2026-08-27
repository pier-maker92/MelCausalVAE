import os
import sys
from collections import defaultdict
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import concatenate_datasets, load_dataset

from dicodec.data.audio_dataset import SimpleAudioDataset

SLURM_TMPDIR = os.getenv("SLURM_TMPDIR")
cache_dir = os.getenv("HF_DATASETS_CACHE")
parquet_dir = f"{SLURM_TMPDIR}/datasets/mls"


class MLSDataset(SimpleAudioDataset):
    def __init__(self, languages: Optional[list[str]] = None):
        super().__init__()
        self.language_id_map = {
            "french": "fr",
            "german": "de",
            "spanish": "es",
            "english": "en",
        }
        if languages is None:
            languages = ["french", "german", "spanish", "english"]

        datasets = []
        for language in languages:
            lang_id = self.language_id_map[language]
            datasets.append(
                load_dataset(
                    f"{parquet_dir}/{lang_id}",
                    cache_dir=cache_dir,
                    num_proc=min(os.cpu_count() or 1, 16),
                )
            )

        partitions_per_destination = defaultdict(list)
        for dataset in datasets:
            for partition in dataset:
                destination = self._partition_to_destination(partition)
                print(f"partition: {partition}, destination: {destination}")
                partitions_per_destination[destination].append(dataset[partition])

        for destination, parts in partitions_per_destination.items():
            setattr(self, f"{destination}_dataset", concatenate_datasets(parts))

    def _partition_to_destination(self, partition_name):
        split = partition_name.split(".")[0]
        if split == "train":
            return "train"
        return "test"
