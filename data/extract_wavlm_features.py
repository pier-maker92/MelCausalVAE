import os
import sys

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import importlib
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

def collate_fn(batch):
    return {key: [item[key] for item in batch] for key in batch[0].keys()}

from modules.feature_extractor import WavLMFeatureExtractor
from modules.configs import WavLMConfig
from data.audio_dataset import TrainDatasetWrapper, TestDatasetWrapper


def main(args):
    is_custom_dataset = True

    print(f"Loading dataset {args.dataset_name}...")
    if args.dataset_name == "mls":
        from data.mls import MLSDataset
        base_dataset = MLSDataset()
    elif args.dataset_name == "libritts":
        from data.libri_tts import LibriTTS
        base_dataset = LibriTTS()
    elif args.dataset_name in ["librispeech_aligned", "librispeech-aligned"]:
        from data.librispeech_align import LibriSpeechAlignDataset
        base_dataset = LibriSpeechAlignDataset(debug=args.debug)
    else:
        is_custom_dataset = False
        print(f"Fallback to HF dataset {args.dataset_name}...")
        hf_dataset = load_dataset(args.dataset_name)
        splits = list(hf_dataset.keys())

    if is_custom_dataset:
        splits = ["train", "test"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    config = WavLMConfig()
    # Initialize the extractor and ensure it's in eval mode
    extractor = WavLMFeatureExtractor(config=config).to(device)
    extractor.eval()

    schema = pa.schema(
        [
            ("id", pa.string()),
            ("audio", pa.list_(pa.float32())),
            ("wav_lm", pa.list_(pa.list_(pa.float32()))),
        ]
    )

    import torchaudio.transforms as T
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Starting extraction to {args.output_dir}...")

    for split in splits:
        print(f"Processing split: {split}")
        if is_custom_dataset:
            if split == "train":
                dataset = TrainDatasetWrapper(base_dataset, "train")
            else:
                dataset = TestDatasetWrapper(base_dataset, "test")
        else:
            dataset = hf_dataset[split]

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        shard_idx = 0
        def get_parquet_file(idx):
            return os.path.join(split_dir, f"data_{idx:05d}.parquet")
            
        parquet_file = get_parquet_file(shard_idx)
        writer = None

        for batch in tqdm(dataloader, desc=f"Extracting {split}"):
            extract_inputs = []
            batch_ids = []
            batch_audio_arrays = []
            
            batch_size_actual = len(batch.get("ids", batch.get("id", batch.get("file", []))))
            if batch_size_actual == 0:
                continue
                
            for b_idx in range(batch_size_actual):
                if is_custom_dataset:
                    audio_tensor = batch["audio_output"][b_idx][0].squeeze()
                    sr = batch["audio_output_sr"][b_idx][0]
                    item_id = str(batch["ids"][b_idx])
                else:
                    audio_data = batch["audio"][b_idx]
                    audio_array = audio_data["array"]
                    sr = audio_data["sampling_rate"]
                    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
                    item_id = str(batch.get("id", batch.get("file", batch.get("ids", [])))[b_idx])
                    
                audio_tensor = audio_tensor.to(device)

                if sr != config.sampling_rate:
                    resampler = T.Resample(sr, config.sampling_rate).to(device)
                    audio_tensor = resampler(audio_tensor)
                    sr = config.sampling_rate
                
                audio_array = audio_tensor.cpu().numpy()
                
                batch_ids.append(item_id)
                batch_audio_arrays.append(audio_array)
                extract_inputs.append((audio_tensor, sr))
                
            if not extract_inputs:
                continue

            with torch.no_grad():
                out = extractor(extract_inputs)
            
            features_batch = out.audio_features.cpu()
            padding_mask_batch = out.padding_mask.cpu()
            
            rows = []
            for b_idx, (b_item_id, b_audio_array) in enumerate(zip(batch_ids, batch_audio_arrays)):
                valid_len = (~padding_mask_batch[b_idx]).sum().item()
                feat = features_batch[b_idx, :valid_len].numpy().tolist()
                rows.append({
                    "id": b_item_id,
                    "audio": b_audio_array.tolist(),
                    "wav_lm": feat
                })
            
            table = pa.Table.from_pylist(rows, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(parquet_file, table.schema)
            writer.write_table(table)
            
            # Check shard size
            if os.path.exists(parquet_file) and os.path.getsize(parquet_file) >= args.shard_size_mb * 1024 * 1024:
                writer.close()
                writer = None
                shard_idx += 1
                parquet_file = get_parquet_file(shard_idx)

        if writer is not None:
            writer.close()

    print(f"Finished writing extracted features to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract WavLM features and save to a Parquet file."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset name or local dataset key (e.g. mls, libritts, librispeech_aligned)",
    )
    default_dir = "data/wavlm_features"
    if "SLURM_TMPDIR" in os.environ:
        default_dir = os.path.join(os.environ["SLURM_TMPDIR"], "wavlm_features")

    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_dir,
        help="Path to the output directory",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader workers")
    parser.add_argument("--shard_size_mb", type=int, default=512, help="Max size in MB for each parquet shard")
    args = parser.parse_args()
    main(args)
