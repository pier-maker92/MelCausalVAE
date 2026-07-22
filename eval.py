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


def load_model(checkpoint, device):
    print(f"Loading model from {checkpoint}...")
    config_path = os.path.join(checkpoint, "config.json")
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    model_name = cfg_dict.get("model_name")

    model = build_model(cfg_dict)

    checkpoint_path = os.path.join(checkpoint, "model.safetensors")
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
    return model, vocoder, model_name


def load_test_dataset(
    num_workers: int, batch_size: int, num_samples=None, max_audio_len=None
):
    dataset = LibriSpeechAlignDataset()
    test_dataset = TestDatasetWrapper(dataset, "test", max_audio_len=max_audio_len)
    if num_samples is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(num_samples))
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=EvalDataCollator(),
    )
    return dataloader


def get_hypothesis(model, vocoder, audios_srs, args, device):
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

    generator = torch.Generator(device=device).manual_seed(args.seed)
    params["generator"] = generator

    out = model.encode_decode(**params)
    reconstructed_mel = out["decoder_output"].audio_features
    padding_mask = out["decoder_output"].padding_mask
    audios = [
        vocoder.decode(mel[~mask].unsqueeze(0).permute(0, 2, 1)).squeeze(0)
        for mel, mask in zip(reconstructed_mel, padding_mask)
        if not mask.all()
    ]
    return audios


def get_eval_id(args):
    eval_id = f"eval_{args.num_samples}"
    if args.num_steps is not None:
        eval_id += f"_nsteps{args.num_steps}"
    if args.temperature is not None:
        eval_id += f"_temp{args.temperature}"
    if args.guidance_scale is not None:
        eval_id += f"_guidance{args.guidance_scale}"
    if args.quantized:
        eval_id += "_quantized"
    if args.residual:
        eval_id += "_residual"
    if args.tail:
        eval_id += "_tail"
    return eval_id


def main(args):
    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError(
            "No CUDA device is available. CPU inference is strongly discouraged."
        )
    model, vocoder, model_name = load_model(args.checkpoint, device)

    # get models
    DWER_computer = DWER("small", device=device)  # FIXME
    UTMOS_reference = UTMOS(sample_rate=16000, device=device)
    UTMOS_hypothesis = UTMOS(sample_rate=24000, device=device)
    SpkSim_computer = SpkSimWavLM("microsoft/wavlm-base-sv", device=device)

    # load Librispeech test dataset
    dataloader = load_test_dataset(
        args.num_workers,
        args.batch_size,
        num_samples=args.num_samples,
        max_audio_len=args.max_audio_len,
    )

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            # get references
            references = [ref.to(device) for ref in batch["16k_audios"]]
            # get hypotheses
            audios_srs = [
                (audio.to(device), sr) for audio, sr in batch["output_audios_srs"]
            ]
            hypotheses = get_hypothesis(
                model=model,
                vocoder=vocoder,
                audios_srs=audios_srs,
                args=args,
                device=device,
            )

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

        utmos_ref = UTMOS_reference.summarize("average")
        utmos_hyp = UTMOS_hypothesis.summarize("average")
        dwer = DWER_computer.summarize("error_rate")
        spksim = SpkSim_computer.summarize("average")

        output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)

        eval_id = get_eval_id(args)

        with open(os.path.join(output_dir, f"{eval_id}.json"), "w") as f:
            json.dump(
                {
                    "utmos_ref": utmos_ref,
                    "utmos_hyp": utmos_hyp,
                    "dwer": dwer,
                    "spksim": spksim,
                    "checkpoint": args.checkpoint,
                    "hparams": {
                        "num_samples": args.num_samples,
                        "num_steps": args.num_steps,
                        "temperature": args.temperature,
                        "guidance_scale": args.guidance_scale,
                        "quantized": args.quantized,
                        "residual": args.residual,
                        "tail": args.tail,
                        "chunk_size": args.chunk_size,
                        "chunk": args.chunk,
                        "exclude_chunk": args.exclude_chunk,
                    },
                },
                f,
            )

        print(f"UTMOS (reference): {utmos_ref:.3f}")
        print(f"UTMOS (hypothesis): {utmos_hyp:.3f}")
        print(f"DWER: {dwer:.3f}%")
        print(f"Speaker Similarity: {spksim:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("-b", "--batch_size", type=int, default=1)

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help="Path to the model checkpoint directory",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation",
        help="Directory to save the evaluation results",
    )

    parser.add_argument(
        "--max_audio_len",
        type=float,
        default=20.0,
        help="Maximum audio length in seconds to filter the dataset",
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
