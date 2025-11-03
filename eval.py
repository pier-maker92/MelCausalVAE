import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import json
import yaml
import math
import time
import shutil
import random
import logging
import argparse
import tempfile
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader
from vocos import Vocos
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MelCausalVAE.modules.cfm import DiTConfig
from MelCausalVAE.modules.VAE import VAE, VAEConfig
from MelCausalVAE.modules.Encoder import ConvformerEncoderConfig
from MelCausalVAE.modules.melspecEncoder import MelSpectrogramConfig
from MelCausalVAE.data.libri_tts import LibriTTS
from MelCausalVAE.data.mls import MLSDataset
from MelCausalVAE.data.audio_dataset import DataCollator, TestDatasetWrapper


# optional: we will directly construct MLS test datasets for OOD
from datasets import load_dataset, concatenate_datasets

# For metrics (CER, UTMOS) reuse EvalTTS from CST
import sys as _sys

_sys.path.append(str(Path("/home/ec2-user/CST")))
from llava.eval.eval_tts_class import EvalTTS  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model_from_config(
    config_dict: Dict[str, Any], checkpoint_path: Optional[str], device: torch.device
) -> Tuple[VAE, Vocos]:

    encoder_cfg = ConvformerEncoderConfig(**config_dict["convformer"])  # type: ignore
    decoder_cfg = DiTConfig(**config_dict["cfm"])  # type: ignore
    decoder_cfg.expansion_factor = encoder_cfg.compress_factor_C
    mel_cfg = MelSpectrogramConfig()  # type: ignore
    vae_cfg = VAEConfig(encoder_config=encoder_cfg, decoder_config=decoder_cfg, mel_spec_config=mel_cfg)

    model = VAE(vae_cfg).to(device)
    model.from_pretrained(checkpoint_path)
    model.set_device(device)
    model.set_dtype(torch.bfloat16)
    model.eval()

    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocos.to(device)
    return model, vocos, vae_cfg


def mel_to_audio(vocoder: Vocos, mel: torch.Tensor, device: torch.device) -> torch.Tensor:
    features = mel.permute(0, 2, 1).to(device)
    vocoder.to(device)
    waveform = vocoder.decode(features)  # [1, samples]
    waveform = waveform.float().squeeze(0).detach().cpu()
    # normalize to prevent clipping
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform.view(-1)


def save_wav(path: Path, audio: torch.Tensor, sr: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio.unsqueeze(0).cpu(), sample_rate=sr)


def get_available_gpus() -> List[int]:
    """Detect available GPUs without initializing CUDA contexts (safe before forking)."""
    try:
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def generate_hyperparam_combinations(hyperparam_config: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not hyperparam_config:
        return [{}]
    keys = list(hyperparam_config.keys())
    values = list(hyperparam_config.values())
    combos: List[Dict[str, Any]] = []

    # cartesian product
    def _prod(arrs: List[List[Any]], idx: int, cur: Dict[str, Any]):
        if idx == len(arrs):
            combos.append(dict(cur))
            return
        k = keys[idx]
        for v in arrs[idx]:
            cur[k] = v
            _prod(arrs, idx + 1, cur)
            cur.pop(k, None)

    _prod(values, 0, {})
    return combos


def evaluate_dataset(
    setting: str,
    languages: Optional[List[str]],
    num_samples: int,
    batch_size: int,
    model: VAE,
    vocoder: Vocos,
    n_steps: int,
    temperature: float,
    guidance_scale: float,
    device: torch.device,
    work_dir: Path,
    evaluator: EvalTTS,
    log_images: bool,
    max_images: int,
    run,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"samples": [], "aggregates": {}}

    if setting == "ID":
        ds = TestDatasetWrapper(LibriTTS(), "test")
    else:
        ds = TestDatasetWrapper(MLSDataset(), "train")

    test_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=DataCollator())

    # Aggregation structures
    per_language = {}

    tmp_audio_dir = work_dir / "tmp_wavs"
    tmp_audio_dir.mkdir(parents=True, exist_ok=True)
    img_dir = work_dir / "images"
    if log_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    image_count = 0
    total_samples = min(num_samples, len(test_dataloader) * batch_size)
    pbar = tqdm(total=total_samples, desc="Evaluating dataset")
    dtype = torch.bfloat16
    images = []
    reconstructed_audios = []
    original_audios = []
    for batch in test_dataloader:
        audios_srs = batch["output_audios_srs"]
        sr = audios_srs[0][1]
        lang = batch["language"]
        gt_text = batch["transcription"]
        audios_srs = [(audio.to(device, dtype=dtype), sr) for audio, sr in audios_srs]

        with torch.no_grad():
            out = model.encode_and_sample(
                audios_srs=audios_srs,
                num_steps=n_steps,
                temperature=temperature,
                guidance_scale=guidance_scale,
                generator=None,
            )
        original_mel = out["original_mel"].detach().cpu()
        reconstructed_mel = out["reconstructed_mel"].detach().cpu()
        padding_mask = out["padding_mask"].detach().cpu()

        # save wavs and a prompt text file
        for idx in range(len(audios_srs)):
            min_length = min(original_mel[idx].size(0), reconstructed_mel[idx].size(0))
            cur_original_mel = original_mel[idx][:min_length]
            cur_reconstructed_mel = reconstructed_mel[idx][:min_length]
            cur_padding_mask = padding_mask[idx][:min_length]
            cur_original_mel = cur_original_mel[~cur_padding_mask]
            cur_reconstructed_mel = cur_reconstructed_mel[~cur_padding_mask]

            original_audio = mel_to_audio(vocoder, cur_original_mel.unsqueeze(0), device)
            reconstructed_audio = mel_to_audio(vocoder, cur_reconstructed_mel.unsqueeze(0), device)

            orig_wav = tmp_audio_dir / f"sample_{idx}_orig.wav"
            recon_wav = tmp_audio_dir / f"sample_{idx}_recon.wav"
            prompt_txt = tmp_audio_dir / f"sample_{idx}_prompt.txt"
            save_wav(orig_wav, original_audio, sr)
            save_wav(recon_wav, reconstructed_audio, sr)
            prompt_txt.write_text(gt_text[idx])

            # compute metrics using EvalTTS
            ref_meta, _ = evaluator.evaluate(
                target_audio_path=str(orig_wav),
                prompt_text_path=str(prompt_txt),
                reference_audio_path=None,
                language="en" if lang[idx] in ["en", "english"] else lang[idx],
                debug=False,
            )
            recon_meta, _ = evaluator.evaluate(
                target_audio_path=str(recon_wav),
                prompt_text_path=str(prompt_txt),
                reference_audio_path=None,
                language="en" if lang[idx] in ["en", "english"] else lang[idx],
                debug=False,
            )

            ref_scores = {k: v for k, v in ref_meta.items() if k in ["CER", "UTMOS"]}
            recon_scores = {k: v for k, v in recon_meta.items() if k in ["CER", "UTMOS"]}
            discrepancy_pct = {}
            for m in ["CER", "UTMOS"]:
                ref_v = float(ref_scores.get(m, float("nan")))
                rec_v = float(recon_scores.get(m, float("nan")))
                if math.isnan(ref_v) or abs(ref_v) < 1e-8:
                    discrepancy_pct[m] = None
                else:
                    discrepancy_pct[m] = float((rec_v - ref_v) / (ref_v + 1e-8) * 100.0)

            results["samples"].append(
                {
                    "index": idx,
                    "language": lang[idx],
                    "ref": ref_scores,
                    "reconstructed": recon_scores,
                    "discrepancy_pct": discrepancy_pct,
                }
            )

            # optionally save spectrogram images
            if log_images and image_count < max_images:

                def _to_numpy(m: torch.Tensor):
                    m = m.squeeze(0)
                    return m.detach().cpu().float().numpy().T

                fig, axes = plt.subplots(2, 1, figsize=(12, 4))
                im0 = axes[0].imshow(_to_numpy(cur_original_mel), aspect="auto", origin="lower")
                axes[0].set_title(f"Original Mel #{idx}")
                im1 = axes[1].imshow(_to_numpy(cur_reconstructed_mel), aspect="auto", origin="lower")
                axes[1].set_title(f"Reconstructed Mel #{idx}")
                for ax in axes:
                    ax.set_xlabel("Frames")
                    ax.set_ylabel("Bins")
                out_img_path = img_dir / f"sample_{idx}.png"
                fig.tight_layout()
                fig.savefig(out_img_path)
                plt.close(fig)
                image_count += 1
                # also log audio to wandb as audio files
                images.append(wandb.Image(fig))
                original_audios.append(wandb.Audio(original_audio.numpy(), sample_rate=sr))
                reconstructed_audios.append(wandb.Audio(reconstructed_audio.numpy(), sample_rate=sr))

            # aggregate per language
            if setting == "OOD":
                key = lang[idx]
            else:
                key = "all"
            if key not in per_language:
                per_language[key] = {"CER": [], "UTMOS": []}
            if "CER" in ref_scores:
                per_language[key].setdefault("ref_CER", [])
                per_language[key].setdefault("recon_CER", [])
                per_language[key]["ref_CER"].append(float(ref_scores["CER"]))
                per_language[key]["recon_CER"].append(float(recon_scores.get("CER", float("nan"))))
            if "UTMOS" in ref_scores:
                per_language[key].setdefault("ref_UTMOS", [])
                per_language[key].setdefault("recon_UTMOS", [])
                per_language[key]["ref_UTMOS"].append(float(ref_scores["UTMOS"]))
                per_language[key]["recon_UTMOS"].append(float(recon_scores.get("UTMOS", float("nan"))))

        processed += batch_size
        if processed >= num_samples:
            break
        pbar.update(batch_size)
    pbar.close()
    run.log(
        {
            "images": images,
            "original_audios": original_audios,
            "reconstructed_audios": reconstructed_audios,
        },
    )

    # finalize aggregates
    for key, agg in per_language.items():

        def _mean(arr: List[float]) -> Optional[float]:
            arr = [x for x in arr if not math.isnan(x)]
            return float(sum(arr) / len(arr)) if arr else None

        ref_cer = _mean(agg.get("ref_CER", []))
        recon_cer = _mean(agg.get("recon_CER", []))
        ref_utmos = _mean(agg.get("ref_UTMOS", []))
        recon_utmos = _mean(agg.get("recon_UTMOS", []))

        results["aggregates"][key] = {
            "ref": {"CER": ref_cer, "UTMOS": ref_utmos},
            "reconstructed": {"CER": recon_cer, "UTMOS": recon_utmos},
            "discrepancy_pct": {
                "CER": (
                    None
                    if (ref_cer is None or abs(ref_cer) < 1e-8)
                    else float((recon_cer - ref_cer) / (ref_cer + 1e-8) * 100.0)
                ),
                "UTMOS": (
                    None
                    if (ref_utmos is None or abs(ref_utmos) < 1e-8)
                    else float((recon_utmos - ref_utmos) / (ref_utmos + 1e-8) * 100.0)
                ),
            },
        }

    # cleanup temp wavs
    try:
        shutil.rmtree(tmp_audio_dir)
    except Exception:
        pass

    return results


def save_json(output: Dict[str, Any], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved metrics to {out_path}")


def run_single_eval(
    checkpoint: str,
    gpu_index: int,
    base_args: Dict[str, Any],
    hyperparams: Dict[str, Any],
    timeout: int,
    run,
) -> Dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(base_args.get("seed", 42))

    # build model
    model, vocoder, vae_cfg = build_model_from_config(base_args["config_dict"], checkpoint, device)

    evaluator = EvalTTS()

    # resolve args
    setting = base_args["setting"]
    languages = base_args.get("languages")
    num_samples = base_args["num_samples"]
    batch_size = base_args["batch_size"]
    n_steps = int(hyperparams.get("n_steps", base_args["n_steps"]))
    temperature = float(hyperparams.get("temperature", base_args["temperature"]))
    guidance_scale = float(hyperparams.get("guidance_scale", base_args["guidance_scale"]))
    log_images = bool(base_args.get("wandb_log_images", False))
    wandb_max_images = int(base_args.get("wandb_max_images", 10))

    work_dir = (
        Path(base_args["output_dir"])
        / base_args["exp_name"]
        / Path(checkpoint).name
        / f"gpu{gpu_index}_n{n_steps}_t{temperature}_g{guidance_scale}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    try:
        eval_output = evaluate_dataset(
            setting=setting,
            languages=languages,
            num_samples=num_samples,
            batch_size=batch_size,
            model=model,
            vocoder=vocoder,
            n_steps=n_steps,
            temperature=temperature,
            guidance_scale=guidance_scale,
            device=device,
            work_dir=work_dir,
            evaluator=evaluator,
            log_images=log_images,
            max_images=wandb_max_images,
            run=run,
        )
        status = "success"
        error = None
    except Exception as e:
        logger.exception("Evaluation failed")
        eval_output = {}
        status = "failed"
        error = str(e)
    duration = time.time() - start

    # save json for this run
    out_path = work_dir / "metrics.json"
    if eval_output:
        save_json(eval_output, out_path)

    return {
        "checkpoint": checkpoint,
        "gpu_index": gpu_index,
        "hyperparams": {"n_steps": n_steps, "temperature": temperature, "guidance_scale": guidance_scale},
        "status": status,
        "duration": duration,
        "metrics_path": str(out_path) if eval_output else None,
        "work_dir": str(work_dir),
        "error": error,
    }


def orchestrate(
    checkpoints: List[str],
    base_args: Dict[str, Any],
    hyperparam_combinations: Optional[List[Dict[str, Any]]],
    max_workers: Optional[int],
    timeout: int,
    multi_gpu: bool,
) -> List[Dict[str, Any]]:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # generate task list
    if hyperparam_combinations is None:
        hyperparam_combinations = [{}]
    tasks: List[Tuple[str, Dict[str, Any]]] = []
    for ckpt in checkpoints:
        for combo in hyperparam_combinations:
            tasks.append((ckpt, combo))

    gpus = get_available_gpus()
    run = wandb.init(
        project="MelCausalVAE-eval",
        name=base_args["exp_name"],
        config={
            "setting": base_args["setting"],
            "languages": base_args["languages"],
        },
    )

    # Single-process path (default): run sequentially on first GPU (or CPU if none)
    if not multi_gpu:
        if gpus:
            gpu_index = gpus[0]
        else:
            gpu_index = 0  # will run on CPU inside run_single_eval if CUDA not available
        logger.info(
            f"Running in single-process mode on gpu_index={gpu_index} (available_gpus={gpus}) with {len(tasks)} task(s)"
        )
        results: List[Dict[str, Any]] = []
        for i, (ckpt, combo) in enumerate(tasks):
            logger.info(f"Running task {i+1}/{len(tasks)}: ckpt={ckpt}, combo={combo}")
            res = run_single_eval(ckpt, gpu_index, base_args, combo, timeout, run)
            logger.info(
                f"Task completed: {res.get('checkpoint')} on GPU {res.get('gpu_index')} status={res.get('status')}"
            )
            results.append(res)
        return results

    # Multi-GPU parallel path
    if not gpus:
        raise RuntimeError("No available GPUs found for --multi-gpu mode")
    max_workers = max_workers or len(gpus)

    # Use 'spawn' to avoid CUDA fork-related deadlocks
    ctx = mp.get_context("spawn")

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = {}
        for i, (ckpt, combo) in enumerate(tasks):
            gpu_index = gpus[i % len(gpus)]
            logger.info(f"Submitting task: ckpt={ckpt}, gpu={gpu_index}, combo={combo}")
            futures[ex.submit(run_single_eval, ckpt, gpu_index, base_args, combo, timeout, run)] = (
                ckpt,
                combo,
                gpu_index,
            )
        for fut in as_completed(futures):
            res = fut.result()
            logger.info(
                f"Task completed: {res.get('checkpoint')} on GPU {res.get('gpu_index')} status={res.get('status')}"
            )
            results.append(res)
    run.finish()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VAE reconstructions with CER and UTMOS, with GPU orchestration"
    )
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--report-to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--guidance-scale", type=float, default=1.3)
    parser.add_argument("--setting", type=str, default="ID", choices=["ID", "OOD"])
    parser.add_argument("--languages", nargs="+", help="Languages for OOD setting (e.g., english german spanish)")
    parser.add_argument("--wandb-log-images", action="store_true", help="If set, also log spectrogram images to W&B")
    parser.add_argument("--wandb-max-images", type=int, default=10, help="Maximum number of images to log (>1)")

    # orchestration inputs
    parser.add_argument("--checkpoints", nargs="+", help="List of checkpoints to evaluate")
    parser.add_argument("--checkpoints-file", type=str, help="File with one checkpoint path per line")
    parser.add_argument(
        "--hyperparam-sweep",
        type=str,
        help="JSON path for hyperparam sweep with keys: n_steps, temperature, guidance_scale",
    )
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="default_exp")
    parser.add_argument("--output-dir", type=str, default="/home/ec2-user/MelCausalVAE/evaluation")
    parser.add_argument("--multi-gpu", action="store_true", help="Enable multi-GPU parallel inference")

    args = parser.parse_args()

    # set default dtype to bfloat16
    torch.set_default_dtype(torch.bfloat16)

    # resolve checkpoints
    checkpoints: List[str] = []
    if args.checkpoints_file:
        with open(args.checkpoints_file, "r") as f:
            checkpoints = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if args.checkpoints:
        checkpoints.extend(args.checkpoints)
    if not checkpoints:
        raise ValueError("No checkpoints provided. Use --checkpoints or --checkpoints-file")

    # load config dict once and pass down
    config_dict = load_config(args.config_path)

    # hyperparam combinations
    if args.hyperparam_sweep:
        with open(args.hyperparam_sweep, "r") as f:
            sweep_cfg = json.load(f)
        hyperparam_combinations = generate_hyperparam_combinations(sweep_cfg)
    else:
        hyperparam_combinations = generate_hyperparam_combinations(
            {
                "n_steps": [args.n_steps],
                "temperature": [args.temperature],
                "guidance_scale": [args.guidance_scale],
            }
        )
        # collapse if singular defaults
        if len(hyperparam_combinations) == 1:
            hyperparam_combinations = None

    # validate wandb-max-images
    if args.wandb_max_images is not None and args.wandb_max_images <= 1:
        raise ValueError("--wandb-max-images must be greater than 1")

    base_args: Dict[str, Any] = {
        "config_dict": config_dict,
        "report_to": args.report_to,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "temperature": args.temperature,
        "guidance_scale": args.guidance_scale,
        "setting": args.setting,
        "languages": args.languages,
        "seed": args.seed,
        "exp_name": args.exp_name,
        "output_dir": args.output_dir,
        "wandb_log_images": args.wandb_log_images,
        "wandb_max_images": args.wandb_max_images,
    }

    # orchestrate one job per GPU
    results = orchestrate(
        checkpoints=checkpoints,
        base_args=base_args,
        hyperparam_combinations=hyperparam_combinations,
        max_workers=args.max_workers,
        timeout=args.timeout,
        multi_gpu=args.multi_gpu,
    )

    # summary file
    summary_path = Path(args.output_dir) / args.exp_name / "summary_results.json"
    save_json(results, summary_path)

    # # wandb: upload json metrics, and optionally images
    # if args.report_to == "wandb":
    #     try:
    #         import wandb

    #         wandb.save(str(summary_path))
    #         if args.wandb_log_images:
    #             # collect image files across runs up to the cap
    #             image_files: List[str] = []
    #             for res in results:
    #                 work_dir = res.get("work_dir")
    #                 if not work_dir:
    #                     continue
    #                 img_dir = Path(work_dir) / "images"
    #                 if img_dir.exists():
    #                     for p in sorted(img_dir.glob("*.png")):
    #                         image_files.append(str(p))
    #                         if len(image_files) >= args.wandb_max_images:
    #                             break
    #                 if len(image_files) >= args.wandb_max_images:
    #                     break
    #             if image_files:
    #                 wandb.log({"mel_spectrograms": [wandb.Image(p) for p in image_files]})
    #         run.finish()
    #     except Exception:
    #         logger.warning("W&B logging failed; continuing without upload")


if __name__ == "__main__":
    main()

# python eval.py \
#   --config-path /home/ec2-user/MelCausalVAE/configs/settings/setting1.yaml \
#   --checkpoints /home/ec2-user/checkpoints/setting1/checkpoint-44000/model.safetensors \
#   --setting ID \
#   --num-samples 24 --batch-size 4 \
#   --n-steps 4 --temperature 0.2 --guidance-scale 1.3 \
#   --exp-name ID_eval \
#   --output-dir /home/ec2-user/MelCausalVAE/evaluation \
#   --report-to wandb \
#   --wandb-log-images \
#   --wandb-max-images 24
