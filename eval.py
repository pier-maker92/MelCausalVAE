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
from transformers import set_seed
import torchaudio.transforms as T
from dicodec.modules.builder import build_model

from dicodec.data.audio_dataset import EvalDataCollator
from dicodec.data.audio_dataset import TestDatasetWrapper
from dicodec.data.librispeech import LibriSpeechDataset

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
    return model, model_name


def load_test_dataset(
    num_workers: int, batch_size: int, num_samples=None, max_audio_len=None
):
    dataset = LibriSpeechDataset()
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


def get_hypothesis(model, audios_srs, args, device, audio_16khz=None):
    params = {
        "audios_srs": audios_srs,
        "num_steps": args.num_steps,
        "temperature": args.temperature,
        "guidance_scale": args.guidance_scale,
    }
    if audio_16khz is not None:
        params["audio_16khz"] = audio_16khz

    generator = torch.Generator(device=device).manual_seed(args.seed)
    params["generator"] = generator

    out = model.encode_decode(**params)
    padding_mask = out["decoder_output"].padding_mask
    audio_waveform = out["audio_waveform"]

    audios = []
    for audio, mask in zip(audio_waveform, padding_mask):
        if not mask.all():
            valid_frames = (~mask).sum().item()
            hop_length = audio.shape[-1] // mask.shape[-1]
            valid_audio_len = valid_frames * hop_length
            audios.append(audio[..., :valid_audio_len].squeeze())

    return audios


def run_evaluation(
    model: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    dataset_name: str,
    num_samples: int = 100,
    run_id: str = "default_run",
    quantized: bool = False,
    residual: bool = False,
    tail: bool = False,
    chunk: int | None = None,
    chunk_size: int | None = None,
) -> dict[str, float]:
    """
    Perform evaluation during training using eval.py metric classes.
    """
    model.eval()
    
    DWER_computer = DWER("small", device=device)
    UTMOS_reference = UTMOS(sample_rate=16000, device=device)
    UTMOS_hypothesis = UTMOS(sample_rate=24000, device=device)

    processed_count = 0
    print(f"Starting evaluation on {num_samples} samples...")
    
    for batch in eval_dataloader:
        if processed_count >= num_samples:
            break
            
        references = [ref.to(device) for ref in batch["16k_audio"]]
        audios_srs = [
            (audio.to(device), sr) for audio, sr in batch["audio_input_srs"]
        ]
        
        params = {
            "audios_srs": audios_srs,
            "num_steps": 16,
            "temperature": 0.2,
            "guidance_scale": 1.3,
            "audio_16khz": references,
        }
        has_vq = getattr(model.config.encoder_config, "vq_config", None) is not None
        if tail:
            if not has_vq:
                raise ValueError("The -t flag was passed, but the model's config does not have a vq_config.")
            params["quantized"] = False
            params["residual"] = False
            params["tail"] = True
        else:
            if has_vq:
                params["quantized"] = True
                params["residual"] = False
                params["tail"] = True
            else:
                params["quantized"] = False
                params["residual"] = False
                params["tail"] = False
        if chunk is not None:
            params["chunk"] = chunk
        if chunk_size is not None:
            params["chunk_size"] = chunk_size
            
        out = model.encode_decode(**params)
        padding_mask = out["decoder_output"].padding_mask
        audio_waveform = out["audio_waveform"]
        
        hypotheses = []
        for audio, mask in zip(audio_waveform, padding_mask):
            if not mask.all():
                valid_frames = (~mask).sum().item()
                hop_length = audio.shape[-1] // mask.shape[-1]
                valid_audio_len = valid_frames * hop_length
                hypotheses.append(audio[..., :valid_audio_len].squeeze())
            else:
                hypotheses.append(audio.squeeze())

        remaining = num_samples - processed_count
        references = references[:remaining]
        hypotheses = hypotheses[:remaining]
        ids = batch.get("ids", [str(i) for i in range(len(references))])[:remaining]
        
        UTMOS_reference.append(ids, references)
        UTMOS_hypothesis.append(ids, hypotheses)
        DWER_computer.append(
            hyp_sr=24000,
            ref_sr=16000,
            ids=ids,
            hyp_sig=hypotheses,
            ref_sig=references,
        )
        processed_count += len(references)

    utmos_ref = UTMOS_reference.summarize("average")
    utmos_hyp = UTMOS_hypothesis.summarize("average")
    dwer = DWER_computer.summarize("error_rate")
    cer = DWER_computer.summarize("CER")
    
    DWER_computer.clear()
    
    summary_metrics = {
        "avg_UTMOS": utmos_hyp,
        "avg_UTMOS_ref": utmos_ref,
        "avg_dWER": dwer,
        "avg_dCER": cer,
        "avg_dUTMOS": utmos_hyp - utmos_ref
    }
    print(f"Evaluation complete. Metrics: {summary_metrics}")
    return summary_metrics


def get_eval_id(args):
    eval_id = f"eval_{args.num_samples}"
    if args.num_steps is not None:
        eval_id += f"_nsteps{args.num_steps}"
    if args.temperature is not None:
        eval_id += f"_temp{args.temperature}"
    if args.guidance_scale is not None:
        eval_id += f"_guidance{args.guidance_scale}"
    return eval_id


def main(args):
    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError(
            "No CUDA device is available. CPU inference is strongly discouraged."
        )
    model, model_name = load_model(args.checkpoint, device)

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
            references = [ref.to(device) for ref in batch["16k_audio"]]
            # get hypotheses
            audios_srs = [
                (audio.to(device), sr) for audio, sr in batch["audio_input_srs"]
            ]
            hypotheses = get_hypothesis(
                model=model,
                audios_srs=audios_srs,
                args=args,
                device=device,
                audio_16khz=[audio.to(device) for audio in batch["16k_audio"]],
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

    args = parser.parse_args()
    main(args)
