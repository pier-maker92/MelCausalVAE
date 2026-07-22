import os

if "SCRATCH" in os.environ:
    os.environ["HF_HOME"] = os.path.join(os.environ["SCRATCH"], ".cache/huggingface")
    os.environ["TORCH_HOME"] = os.path.join(os.environ["SCRATCH"], ".cache/torch")

import json
import torch
import logging
import argparse
import torchaudio
from tqdm import tqdm
from vocos import Vocos
from transformers import set_seed
import torchaudio.transforms as T
from modules.builder import build_model

from data.audio_dataset import EvalDataCollator
from data.audio_dataset import TestDatasetWrapper
from data.librispeech_align import LibriSpeechAlignDataset

from evaluation.scripts.dwer import DWER
from evaluation.scripts.utmos import UTMOS
from evaluation.scripts.speaker_similarity import SpkSimWavLM


def load_model(checkpoint_dir, device):
    print(f"Loading model from {checkpoint_dir}...")
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    model = build_model(cfg_dict)

    checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
    model.from_pretrained(checkpoint_path)
    model.eval()
    model.to(device)
    assert not model.training, "Model must be in eval mode"
    assert not model.encoder.training, (
        "Encoder must be in eval mode: reparameterization trick and "
        "dropout regularizer are only disabled when training=False"
    )
    print("Initializing Vocoder...")
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    return model, vocoder


def load_test_dataset(num_workers: int, batch_size: int):
    dataset = LibriSpeechAlignDataset()
    test_dataset = TestDatasetWrapper(dataset, "test")
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=EvalDataCollator(),
    )
    return dataloader


def get_hypothesis(model, vocoder, audios_srs, args):
    params = {
        "audios_srs": audios_srs,
        "num_steps": args.num_steps,
        "temperature": args.temperature,
        "guidance_scale": args.guidance_scale,
    }
    if args.quantized or args.residual or args.tail:
        params["quantized"] = False
        params["residual"] = False
        params["tail"] = False

    if args.quantized:
        params["quantized"] = True
    if args.residual:
        params["residual"] = True
    if args.tail:
        params["tail"] = True

    if getattr(args, "chunk_size", None) is not None:
        params["chunk_size"] = args.chunk_size
    if getattr(args, "chunk", None) is not None:
        params["chunk"] = args.chunk
    if getattr(args, "exclude_chunk", None) is not None:
        params["exclude_chunk"] = args.exclude_chunk
    if getattr(args, "exclude_start_chunk", None) is not None:
        params["exclude_start_chunk"] = args.exclude_start_chunk
    if getattr(args, "exclude_end_chunk", None) is not None:
        params["exclude_end_chunk"] = args.exclude_end_chunk

    out = model.encode_decode(**params)

    reconstructed_mel = out["decoder_output"].audio_features
    padding_mask = out["decoder_output"].padding_mask
    audio = vocoder.decode(reconstructed_mel.permute(0, 2, 1))
    return audio


def main(args):
    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError(
            "No CUDA device is available. CPU inference is strongly discouraged."
        )

    if args.batch_size > 1:
        raise NotImplementedError("Batch size > 1 is not supported for evaluation")

    model, vocoder = load_model(args.checkpoint_dir, device)

    # get models
    UTMOS_reference = UTMOS(sample_rate=16000)
    UTMOS_hypothesis = UTMOS(sample_rate=24000)
    DWER_computer = DWER("small", device=device)  # FIXME
    SpkSim_computer = SpkSimWavLM("microsoft/wavlm-base-sv", device=device)

    # load Librispeech test dataset
    dataloader = load_test_dataset(args.num_workers, args.batch_size)

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            # get references
            references = batch["16k_audios"]
            # get hypotheses
            audios_srs = [
                (audio.to(device), sr) for audio, sr in batch["output_audios_srs"]
            ]
            hypotheses = get_hypothesis(model, vocoder, audios_srs, args)

            UTMOS_reference.append(batch["ids"], references)
            UTMOS_hypothesis.append(batch["ids"], hypotheses)
            DWER_computer.append(
                hyp_sr=24000,
                ref_sr=16000,
                ids=batch["ids"],
                hyp_sig=hypotheses,
                ref_sig=references,
            )
            SpkSim_computer.append(
                hyp_sr=24000,
                ref_sr=16000,
                ids=batch["ids"],
                hyp_sig=hypotheses,
                ref_sig=references,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("-b", "--batch_size", type=int, default=1)

    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        type=str,
        help="Path to the model checkpoint directory",
    )

    parser.add_argument("--num_steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--guidance_scale", type=float, default=1.3)

    parser.add_argument("-q", "--quantized", action="store_true")
    parser.add_argument("-r", "--residual", action="store_true")
    parser.add_argument("-t", "--tail", action="store_true")

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Chunk size for the bottleneck tail",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Number of chunks to use, starting from the bottom of the bottleneck",
    )
    parser.add_argument(
        "--exclude_chunk",
        type=int,
        default=None,
        help="Number of chunks to exclude, starting from the bottom of the bottleneck",
    )
    args = parser.parse_args()
    main(args)
