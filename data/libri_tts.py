import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from modules.feature_extractor import FeatureExtractor, MelSpectrogramConfig
from data.audio_dataset import SimpleAudioDataset, DataCollator, TrainDatasetWrapper

<<<<<<< HEAD
# Specify custom cache directory
parquet_dir = "/home/ec2-user/data"
=======
# set the HF_HOME environment variable
# os.environ["HF_HOME"] = "/Volumes/Crucial X6/HF_HOME"
cache_dir = "/Users/pierfrancescomelucci/Research/Playground/data_cache"


>>>>>>> origin/main
# import mel spec encoder
mel_spec_encoder = FeatureExtractor(config=MelSpectrogramConfig())


def simple_collate_fn(batch):
    return batch


class LibriTTS(SimpleAudioDataset):
    def __init__(self):
        super().__init__()
        # Load the two datasets
        dataset = load_dataset(
            "parquet",
            data_dir=f"{parquet_dir}/libritts",
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
        if partition_name in ["train", "validation"]:
            return "train"
        elif partition_name in ["test"]:
            return "test"


# parser = argparse.ArgumentParser()
# parser.add_argument("-b", "--batch_size", type=int, default=1)
# parser.add_argument("-s", "--stats", action="store_true", default=False)
# parser.add_argument("-n", "--num_samples", type=int, default=10000)
# args = parser.parse_args()
# if __name__ == "__main__":
#     # data collator
#     data_collator = DataCollator()
<<<<<<< HEAD
#     dataset = DatasetWrapper(LibriTTS(), "train")
=======
#     dataset = TrainDatasetWrapper(LibriTTS(), "train")
>>>>>>> origin/main
#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         collate_fn=data_collator,
#         num_workers=min(os.cpu_count(), 16),
#         shuffle=True,
#     )
#     means = []
#     stds = []
#     counter = 0
#     if args.stats:
#         pbar = tqdm(total=min(args.num_samples, len(dataloader)))
#         for batch in dataloader:
#             audio_srs = batch["output_audios_srs"]
#             mel_spec = mel_spec_encoder(audio_srs)
#             featues, padding_mask = mel_spec.audio_features, mel_spec.padding_mask
#             for feature, mask in zip(featues, padding_mask):
#                 means.append(feature[~mask].mean())
#                 stds.append(feature[~mask].std())
#             counter += args.batch_size
#             if counter >= args.num_samples or counter >= len(dataloader):
#                 break
#             pbar.update(args.batch_size)
#         pbar.close()
#         print(f"Mean: {np.mean(means)}")
#         print(f"Std: {np.mean(stds)}")
#     else:
#         print(dataset[0])
