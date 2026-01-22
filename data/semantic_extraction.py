import os
import sys

sys.path.append("/leonardo_scratch/large/userexternal/pmelucci/textlesslib")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torchaudio
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from textless.data.speech_encoder import SpeechEncoder
from data.librispeechHubert import LibriSpeech100h
from torch.utils.data import DataLoader
from data.audio_dataset import DataCollator, HubertDatasetWrapper

# Available models
EXPRESSO_MODELS = [
    ("hubert-base-ls960-layer-9", "kmeans", 500),
    ("hubert-base-ls960-layer-9", "kmeans-expresso", 2000),
    ("mhubert-base-vp_mls_cv_8lang", "kmeans", 2000),
    ("mhubert-base-vp_mls_cv_8lang", "kmeans-expresso", 2000),
    ("hubert-base-ls960", "kmeans", 200),
]

# Try one model
dense_model, quantizer_model, vocab = EXPRESSO_MODELS[1]
semantic_encoder = SpeechEncoder.by_name(
    dense_model_name=dense_model,
    quantizer_model_name=quantizer_model,
    vocab_size=vocab,
    deduplicate=False,
)


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=1)
parser.add_argument("-n", "--num_samples", type=int, default=10000)
parser.add_argument(
    "-o", "--output_dir", type=str, default="/Volumes/Crucial X6/librispeech100h_semantic"
)
parser.add_argument("-s", "--shard_size_mb", type=int, default=512, help="Target shard size in MB")
args = parser.parse_args()


def get_shard_path(output_dir: str, shard_idx: int) -> str:
    """Generate shard filename with zero-padded index."""
    return os.path.join(output_dir, f"semantic_features_{shard_idx:04d}.parquet")


if __name__ == "__main__":
    # Create output directory for parquet files
    semantic_parquet_dir = args.output_dir
    os.makedirs(semantic_parquet_dir, exist_ok=True)

    # Target shard size in bytes
    target_shard_size = args.shard_size_mb * 1024 * 1024

    # Define schema for parquet file
    schema = pa.schema(
        [
            ("audio_id", pa.string()),
            ("units", pa.list_(pa.int64())),
            ("centroids", pa.list_(pa.list_(pa.float32()))),
            ("durations", pa.list_(pa.int64())),
        ]
    )

    # data collator
    data_collator = DataCollator()
    dataset = HubertDatasetWrapper(LibriSpeech100h(), split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=0,
        shuffle=False,  # Set to False for reproducibility when saving
    )

    for batch in tqdm(dataloader, desc="Processing batches"):
        audio_codes = batch["audio_codes"]
        breakpoint()
        


    #     # Create a PyArrow table for this batch and write it
    #     table = pa.table(
    #         {
    #             "audio_id": audio_ids,
    #             "units": batch_units,
    #             "centroids": batch_centroids,
    #             "durations": batch_durations,
    #         },
    #         schema=schema,
    #     )
    #     writer.write_table(table)

    #     # Check if current shard exceeds target size, start a new shard
    #     current_size = os.path.getsize(current_shard_path)
    #     if current_size >= target_shard_size:
    #         writer.close()
    #         print(f"Completed shard {shard_idx}: {current_shard_path} ({current_size / 1024 / 1024:.1f} MB)")
    #         shard_idx += 1
    #         current_shard_path = get_shard_path(semantic_parquet_dir, shard_idx)
    #         writer = pq.ParquetWriter(current_shard_path, schema)

    # # Close the final shard
    # writer.close()
    # print(f"Saved {shard_idx + 1} shards to {semantic_parquet_dir}")
