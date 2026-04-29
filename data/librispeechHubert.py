import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from modules.melspecEncoder import MelSpectrogramEncoder, MelSpectrogramConfig
from data.audio_dataset import SimpleAudioDataset, DataCollator, TrainDatasetWrapper

# set the HF_HOME environment variable
#os.environ["HF_HOME"] = "/Volumes/Crucial X6/HF_HOME"
cache_dir = "/Users/pierfrancescomelucci/Research/Playground/data_cache"


# import mel spec encoder
mel_spec_encoder = MelSpectrogramEncoder(config=MelSpectrogramConfig())


def simple_collate_fn(batch):
    return batch


class LibriSpeech100h(SimpleAudioDataset):
    def __init__(self):
        super().__init__()
        # Load the two datasets
        datasets = []
        for subset in ["train"]:
            ds = load_dataset(
                "cmu-mlsp/hubert_layer9-librispeech-asr100h",
                cache_dir=cache_dir,
                num_proc=min(
                    os.cpu_count(),
                    16,
                ),
            )
            datasets.append(ds)

        partitions_per_destination = defaultdict(list)
        for dataset in datasets:
            for partition in dataset:
                partitions_per_destination[self._partition_to_destination(partition)].append(dataset[partition])

        for destination in partitions_per_destination:
            setattr(
                self,
                f"{destination}_dataset",
                concatenate_datasets(partitions_per_destination[destination]),
            )
        # select only the "audio_codes" column
        self.train_dataset = self.train_dataset.select_columns(["audio_codes"])

    def _partition_to_destination(self, partition_name):
        if "train" in partition_name:
            return "train"
        elif "dev" in partition_name or "test" in partition_name:
            return "test"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-s", "--stats", action="store_true", default=False)
    parser.add_argument("-n", "--num_samples", type=int, default=10000)
    args = parser.parse_args()
    # data collator
    data_collator = DataCollator()
    dataset = TrainDatasetWrapper(LibriSpeech100h(), "train")
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
