import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from audio_dataset import SimpleAudioDataset

# Specify custom cache directory
cache_dir = "/home/ec2-user/dataset_cache"
# import mel spec encoder
mel_spec_encoder = MelSpectrogramEncoder(config=MelSpectrogramConfig())


def simple_collate_fn(batch):
    return batch


class LibriTTS(SimpleAudioDataset):
    def __init__(self):
        super().__init__()
        # Load the two datasets
        datasets = []
        for subset in ["clean", "other"]:
            ds = load_dataset(
                "parler-tts/libritts_r_filtered",
                subset,
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

    def _partition_to_destination(self, partition_name):
        if partition_name.split(".")[0] in ["train"]:
            return "train"
        elif partition_name.split(".")[0] in ["dev", "test"]:
            return "test"

    def __getitem__(self, idx):
        data_dict = {}
        data = self.train_dataset[idx]
        self._process_audio_output(data_dict, data)
        return data_dict


# parser = argparse.ArgumentParser()
# parser.add_argument("-b", "--batch_size", type=int, default=1)
# parser.add_argument("-s", "--stats", action="store_true", default=False)
# parser.add_argument("-n", "--num_samples", type=int, default=100000)
# args = parser.parse_args()
# if __name__ == "__main__":
#     # tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
#     tokenizer.pad_token = tokenizer.eos_token
#     # data collator
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     dataset = LJSpeech(tokenizer=tokenizer, conversation_version="llama_3_1")
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
#             if counter >= args.num_samples:
#                 break
#             pbar.update(args.batch_size)
#         pbar.close()
#         print(f"Mean: {np.mean(means)}")
#         print(f"Std: {np.mean(stds)}")
#     else:
#         print(dataset[0])
