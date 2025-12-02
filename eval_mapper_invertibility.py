"""
Evaluate mapper invertibility by comparing:
  1. Standard reconstruction: z -> decoder -> audio
  2. Mapper reconstruction: z -> y -> z_rec -> decoder -> audio

If the mapper is perfectly invertible, both outputs should be identical
(when using the same generator seed).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import yaml
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchaudio
from torch.utils.data import DataLoader
from vocos import Vocos

from MelCausalVAE.modules.cfm import DiTConfig
from MelCausalVAE.modules.VAE import VAE, VAEConfig
from MelCausalVAE.modules.Encoder import ConvformerEncoderConfig
from MelCausalVAE.modules.melspecEncoder import MelSpectrogramConfig
from MelCausalVAE.modules.semantic_mapper import Z2YMapper, SemanticMapperConfig
from MelCausalVAE.data.libri_tts import LibriTTS
from MelCausalVAE.data.audio_dataset import DataCollator, TrainDatasetWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(
    config_dict: Dict[str, Any],
    vae_checkpoint: str,
    mapper_checkpoint: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
):
    """Build VAE, mapper, and vocoder."""
    encoder_cfg = ConvformerEncoderConfig(**config_dict["convformer"])
    decoder_cfg = DiTConfig(**config_dict["cfm"])
    decoder_cfg.expansion_factor = encoder_cfg.compress_factor_C
    mel_cfg = MelSpectrogramConfig()
    mapper_cfg = SemanticMapperConfig(
        z_dim=encoder_cfg.latent_dim,
        n_layers=config_dict.get("semantic_mapper", {}).get("n_layers", 6),
        hidden_dim=config_dict.get("semantic_mapper", {}).get("hidden_dim", 128),
        pretrained_model_path=mapper_checkpoint,
    )

    vae_cfg = VAEConfig(
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
        mel_spec_config=mel_cfg,
        semantic_mapper_config=mapper_cfg,
        add_semantic_distillation=False,
        add_semantic_mapper=True,
    )

    # Load VAE
    vae = VAE(vae_cfg, dtype=dtype).to(device)
    vae.from_pretrained(vae_checkpoint)
    vae.set_device(device)
    vae.set_dtype(dtype)
    vae.eval()
    logger.info(f"✓ Loaded VAE from {vae_checkpoint}")

    # Load vocoder
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    logger.info("✓ Loaded Vocos vocoder")

    return vae, vocoder


def mel_to_audio(vocoder: Vocos, mel: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert mel spectrogram to audio waveform."""
    features = mel.permute(0, 2, 1).to(device)
    waveform = vocoder.decode(features)
    waveform = waveform.float().squeeze(0).detach().cpu()
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform.view(-1)


@torch.no_grad()
def evaluate_stats(
    vae: VAE,
    dataloader: DataLoader,
    num_batches: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Compute statistics (mean, var, temporal_var) for mu, z, mapper(mu), mapper(z) over N batches.

    - mean/var: computed across all elements (all batches, all timestamps, all dimensions)
    - temporal_var: variance across timestamps within each sequence, averaged over all sequences
    """
    from tqdm import tqdm

    # Accumulators for online mean/var computation (across all elements)
    stats = {
        "mu": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
        "z": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
        "mapper_mu": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
        "mapper_z": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
    }

    # Accumulators for temporal variance (variance across timestamps within each sequence)
    temporal_var = {
        "mu": {"sum": 0.0, "count": 0},
        "z": {"sum": 0.0, "count": 0},
        "mapper_mu": {"sum": 0.0, "count": 0},
        "mapper_z": {"sum": 0.0, "count": 0},
    }

    for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Computing stats")):
        if batch_idx >= num_batches:
            break

        audios_srs = batch["output_audios_srs"]
        sr = audios_srs[0][1]
        audios_srs = [(audio.to(device, dtype=dtype), sr) for audio, sr in audios_srs]

        # Encode audio
        encoded_audios = vae.wav2mel(audios_srs)
        convformer_output = vae.encoder(
            x=encoded_audios.audio_features,
            padding_mask=encoded_audios.padding_mask,
            step=None,
        )

        mu = convformer_output.mu  # [B, T, z_dim]
        z = convformer_output.z  # [B, T, z_dim]
        padding_mask = convformer_output.padding_mask  # [B, T]

        # Map through mapper
        mapper_mu = vae.semantic_mapper(mu).y
        mapper_z = vae.semantic_mapper(z).y

        # Debug: log min/max values for first batch
        if batch_idx == 0:
            logger.info(f"DEBUG - mu:         min={mu.min().item():.4f}, max={mu.max().item():.4f}")
            logger.info(f"DEBUG - z:          min={z.min().item():.4f}, max={z.max().item():.4f}")
            logger.info(f"DEBUG - mapper(mu): min={mapper_mu.min().item():.4f}, max={mapper_mu.max().item():.4f}")
            logger.info(f"DEBUG - mapper(z):  min={mapper_z.min().item():.4f}, max={mapper_z.max().item():.4f}")

        # Compute stats only on non-padded positions
        for i in range(mu.shape[0]):
            mask = ~padding_mask[i]  # True = valid
            valid_mu = mu[i][mask].float()
            valid_z = z[i][mask].float()
            valid_mapper_mu = mapper_mu[i][mask].float()
            valid_mapper_z = mapper_z[i][mask].float()

            n = valid_mu.numel()
            stats["mu"]["sum"] += valid_mu.sum().item()
            stats["mu"]["sum_sq"] += (valid_mu**2).sum().item()
            stats["mu"]["count"] += n

            stats["z"]["sum"] += valid_z.sum().item()
            stats["z"]["sum_sq"] += (valid_z**2).sum().item()
            stats["z"]["count"] += n

            stats["mapper_mu"]["sum"] += valid_mapper_mu.sum().item()
            stats["mapper_mu"]["sum_sq"] += (valid_mapper_mu**2).sum().item()
            stats["mapper_mu"]["count"] += n

            stats["mapper_z"]["sum"] += valid_mapper_z.sum().item()
            stats["mapper_z"]["sum_sq"] += (valid_mapper_z**2).sum().item()
            stats["mapper_z"]["count"] += n

            # Temporal variance: variance across timestamps for each sequence
            # Shape: [T_valid, z_dim] -> var over dim=0 -> [z_dim] -> mean
            if valid_mu.shape[0] > 1:  # Need at least 2 frames for variance
                temporal_var["mu"]["sum"] += valid_mu.var(dim=0).mean().item()
                temporal_var["mu"]["count"] += 1

                temporal_var["z"]["sum"] += valid_z.var(dim=0).mean().item()
                temporal_var["z"]["count"] += 1

                temporal_var["mapper_mu"]["sum"] += valid_mapper_mu.var(dim=0).mean().item()
                temporal_var["mapper_mu"]["count"] += 1

                temporal_var["mapper_z"]["sum"] += valid_mapper_z.var(dim=0).mean().item()
                temporal_var["mapper_z"]["count"] += 1

    # Compute final mean and variance
    results = {}
    for name, s in stats.items():
        mean = s["sum"] / s["count"]
        var = (s["sum_sq"] / s["count"]) - (mean**2)
        results[name] = {"mean": mean, "var": var}

    # Add temporal variance (average variance across timestamps per sequence)
    for name, tv in temporal_var.items():
        if tv["count"] > 0:
            results[name]["temporal_var"] = tv["sum"] / tv["count"]
        else:
            results[name]["temporal_var"] = 0.0

    return results


@torch.no_grad()
def evaluate_invertibility(
    vae: VAE,
    vocoder: Vocos,
    dataloader: DataLoader,
    num_batches: int,
    n_steps: int,
    temperature: float,
    guidance_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    """
    Evaluate mapper invertibility by comparing reconstructions.

    For each sample:
      1. Encode audio -> z
      2. Standard: z -> decoder -> mel -> audio
      3. Via mapper: z -> y -> z_rec -> decoder -> mel -> audio
      4. Compare (should be identical with same generator seed)
    """
    results = {
        "table_rows": [],  # List of (sample_idx, sr, original_audio, standard_audio, mapper_audio)
        "z_reconstruction_errors": [],
        "mel_reconstruction_errors": [],
    }

    sample_idx = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        audios_srs = batch["output_audios_srs"]
        sr = audios_srs[0][1]
        audios_srs = [(audio.to(device, dtype=dtype), sr) for audio, sr in audios_srs]

        # Encode audio to mel and latent z
        encoded_audios = vae.wav2mel(audios_srs)
        convformer_output = vae.encode(
            audios_srs=audios_srs,
        )
        z = convformer_output.mu  # [B, T, z_dim]
        padding_mask = convformer_output.padding_mask

        # === Standard reconstruction: z -> decoder -> mel ===
        generator_standard = torch.Generator(device=device).manual_seed(seed)
        standard_mel = vae.decoder.generate(
            num_steps=n_steps,
            context_vector=z,
            temperature=temperature,
            guidance_scale=guidance_scale,
            generator=generator_standard,
            padding_mask=padding_mask,
            std=1.0,
        )
        if vae.config.mel_spec_config.normalize:
            standard_mel = vae.denormalize_mel(standard_mel)

        # === Mapper reconstruction: z -> y -> z_rec -> decoder -> mel ===
        # Forward through mapper: z -> y
        mapper_output = convformer_output.semantic_features
        y = mapper_output.y

        # Inverse through mapper: y -> z_rec
        z_rec = vae.semantic_mapper.inverse(y)

        # Compute z reconstruction error
        z_error = (z - z_rec).abs().mean().item()
        logger.info(f"Batch {batch_idx}: z reconstruction error = {z_error:.6f}")

        # Reconstruct from z_rec (using SAME seed for identical generation)
        generator_mapper = torch.Generator(device=device).manual_seed(seed)
        mapper_mel = vae.decoder.generate(
            num_steps=n_steps,
            context_vector=z_rec,
            temperature=temperature,
            guidance_scale=guidance_scale,
            generator=generator_mapper,
            padding_mask=padding_mask,
            std=0.0,
        )
        if vae.config.mel_spec_config.normalize:
            mapper_mel = vae.denormalize_mel(mapper_mel)

        # Compute mel reconstruction error
        mel_error = (standard_mel - mapper_mel).abs().mean().item()
        logger.info(f"Batch {batch_idx}: mel reconstruction error = {mel_error:.6f}")

        # Original mel for reference
        original_mel = encoded_audios.audio_features
        if vae.config.mel_spec_config.normalize:
            original_mel = vae.denormalize_mel(original_mel)

        # Process each sample in the batch
        compress_factor = vae.config.encoder_config.compress_factor_C
        for i in range(len(audios_srs)):
            # Get valid (non-padded) frames - multiply by compress_factor since
            # padding_mask is at compressed length but mel is at original length
            mask = padding_mask[i]
            valid_len_compressed = (~mask).sum().item()
            valid_len = valid_len_compressed * compress_factor

            cur_original_mel = original_mel[i][:valid_len]
            cur_standard_mel = standard_mel[i][:valid_len]
            cur_mapper_mel = mapper_mel[i][:valid_len]

            # Convert to audio
            original_audio = mel_to_audio(vocoder, cur_original_mel.unsqueeze(0), device)
            standard_audio = mel_to_audio(vocoder, cur_standard_mel.unsqueeze(0), device)
            mapper_audio = mel_to_audio(vocoder, cur_mapper_mel.unsqueeze(0), device)

            # Store for table (sample_idx, sr, original, standard, mapper)
            results["table_rows"].append((sample_idx, sr, original_audio, standard_audio, mapper_audio))

            sample_idx += 1

        results["z_reconstruction_errors"].append(z_error)
        results["mel_reconstruction_errors"].append(mel_error)

        logger.info(f"Processed batch {batch_idx + 1}/{num_batches}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate mapper invertibility")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML (contains all settings)")
    # Mode selection
    parser.add_argument(
        "--stats-only", action="store_true", help="Only compute latent statistics (no audio generation)"
    )
    # Optional CLI overrides
    parser.add_argument("--vae-checkpoint", type=str, default=None, help="Override VAE checkpoint path")
    parser.add_argument("--mapper-checkpoint", type=str, default=None, help="Override mapper checkpoint path")
    parser.add_argument("--num-batches", type=int, default=None, help="Override number of batches")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--n-steps", type=int, default=None, help="Override diffusion steps")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--wandb-project", type=str, default=None, help="Override W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Override W&B run name")

    args = parser.parse_args()

    # Load config
    config_dict = load_config(args.config)

    # Get values from config, with CLI overrides
    vae_checkpoint = args.vae_checkpoint or config_dict.get("vae_checkpoint")
    mapper_checkpoint = args.mapper_checkpoint or config_dict.get("mapper_checkpoint")
    num_batches = args.num_batches or config_dict.get("num_batches", 5)
    batch_size = args.batch_size or config_dict.get("batch_size", 4)
    n_steps = args.n_steps or config_dict.get("n_steps", 4)
    temperature = args.temperature or config_dict.get("temperature", 0.3)
    guidance_scale = args.guidance_scale or config_dict.get("guidance_scale", 1.3)
    seed = args.seed or config_dict.get("seed", 42)
    wandb_project = args.wandb_project or config_dict.get("wandb_project", "mapper-invertibility")
    wandb_run_name = args.wandb_run_name or config_dict.get("wandb_run_name")

    if not vae_checkpoint:
        raise ValueError("vae_checkpoint must be specified in config or via --vae-checkpoint")

    # Setup
    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    logger.info(f"Using device: {device}")

    # Build models
    vae, vocoder = build_model(
        config_dict=config_dict,
        vae_checkpoint=vae_checkpoint,
        mapper_checkpoint=mapper_checkpoint,
        device=device,
        dtype=dtype,
    )

    # Create dataloader
    dataset = TrainDatasetWrapper(LibriTTS(), "train")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollator(),
    )
    logger.info(f"✓ Created dataloader with {len(dataset)} samples")

    # Initialize W&B
    mode_name = "stats" if args.stats_only else "audio"
    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name or f"invertibility-{mode_name}-{num_batches}batches",
        config={
            "mode": mode_name,
            "vae_checkpoint": vae_checkpoint,
            "mapper_checkpoint": mapper_checkpoint,
            "num_batches": num_batches,
            "batch_size": batch_size,
            "n_steps": n_steps,
            "temperature": temperature,
            "guidance_scale": guidance_scale,
            "seed": seed,
        },
    )

    if args.stats_only:
        # Stats-only mode: compute mean/var of latents
        logger.info("Starting stats-only evaluation...")
        stats = evaluate_stats(
            vae=vae,
            dataloader=dataloader,
            num_batches=num_batches,
            device=device,
            dtype=dtype,
        )

        # Create stats table
        stats_table = wandb.Table(columns=["Variable", "Mean", "Variance", "Temporal Variance"])
        stats_table.add_data(
            "mu (encoder output)", stats["mu"]["mean"], stats["mu"]["var"], stats["mu"]["temporal_var"]
        )
        stats_table.add_data("z (reparameterized)", stats["z"]["mean"], stats["z"]["var"], stats["z"]["temporal_var"])
        stats_table.add_data(
            "mapper(mu)", stats["mapper_mu"]["mean"], stats["mapper_mu"]["var"], stats["mapper_mu"]["temporal_var"]
        )
        stats_table.add_data(
            "mapper(z)", stats["mapper_z"]["mean"], stats["mapper_z"]["var"], stats["mapper_z"]["temporal_var"]
        )

        wandb.log({"latent_stats": stats_table})

        # Also log individual metrics for easier comparison
        wandb.log(
            {
                "mu_mean": stats["mu"]["mean"],
                "mu_var": stats["mu"]["var"],
                "mu_temporal_var": stats["mu"]["temporal_var"],
                "z_mean": stats["z"]["mean"],
                "z_var": stats["z"]["var"],
                "z_temporal_var": stats["z"]["temporal_var"],
                "mapper_mu_mean": stats["mapper_mu"]["mean"],
                "mapper_mu_var": stats["mapper_mu"]["var"],
                "mapper_mu_temporal_var": stats["mapper_mu"]["temporal_var"],
                "mapper_z_mean": stats["mapper_z"]["mean"],
                "mapper_z_var": stats["mapper_z"]["var"],
                "mapper_z_temporal_var": stats["mapper_z"]["temporal_var"],
            }
        )

        logger.info(f"\n{'='*80}")
        logger.info("LATENT STATISTICS:")
        logger.info(
            f"  mu:         mean={stats['mu']['mean']:.6f}, var={stats['mu']['var']:.6f}, temporal_var={stats['mu']['temporal_var']:.6f}"
        )
        logger.info(
            f"  z:          mean={stats['z']['mean']:.6f}, var={stats['z']['var']:.6f}, temporal_var={stats['z']['temporal_var']:.6f}"
        )
        logger.info(
            f"  mapper(mu): mean={stats['mapper_mu']['mean']:.6f}, var={stats['mapper_mu']['var']:.6f}, temporal_var={stats['mapper_mu']['temporal_var']:.6f}"
        )
        logger.info(
            f"  mapper(z):  mean={stats['mapper_z']['mean']:.6f}, var={stats['mapper_z']['var']:.6f}, temporal_var={stats['mapper_z']['temporal_var']:.6f}"
        )
        logger.info(f"{'='*80}")

    else:
        # Audio mode: generate reconstructions
        logger.info("Starting invertibility evaluation...")
        results = evaluate_invertibility(
            vae=vae,
            vocoder=vocoder,
            dataloader=dataloader,
            num_batches=num_batches,
            n_steps=n_steps,
            temperature=temperature,
            guidance_scale=guidance_scale,
            device=device,
            dtype=dtype,
            seed=seed,
        )

        # Log to W&B
        avg_z_error = sum(results["z_reconstruction_errors"]) / len(results["z_reconstruction_errors"])
        avg_mel_error = sum(results["mel_reconstruction_errors"]) / len(results["mel_reconstruction_errors"])

        # Create table for side-by-side comparison
        table = wandb.Table(columns=["Sample", "Original", "Standard Recon (z→mel)", "Mapper Recon (z→y→z→mel)"])
        for sample_idx, sr, original_audio, standard_audio, mapper_audio in results["table_rows"]:
            table.add_data(
                sample_idx,
                wandb.Audio(original_audio.numpy(), sample_rate=sr),
                wandb.Audio(standard_audio.numpy(), sample_rate=sr),
                wandb.Audio(mapper_audio.numpy(), sample_rate=sr),
            )

        wandb.log(
            {
                "avg_z_reconstruction_error": avg_z_error,
                "avg_mel_reconstruction_error": avg_mel_error,
                "audio_comparison": table,
            }
        )

        logger.info(f"\n{'='*50}")
        logger.info(f"RESULTS:")
        logger.info(f"  Average z reconstruction error: {avg_z_error:.6f}")
        logger.info(f"  Average mel reconstruction error: {avg_mel_error:.6f}")
        logger.info(f"{'='*50}")

        if avg_z_error < 1e-4:
            logger.info("✓ Mapper is highly invertible (z error < 1e-4)")
        elif avg_z_error < 1e-2:
            logger.info("⚠ Mapper has small invertibility error (z error < 1e-2)")
        else:
            logger.info("✗ Mapper has significant invertibility error")

    run.finish()
    logger.info("Done!")


if __name__ == "__main__":
    main()

# Example usage (all settings from config):
# python eval_mapper_invertibility.py --config configs/settings/eval_mapper.yaml
#
# Stats-only mode (no audio generation, just latent statistics):
# python eval_mapper_invertibility.py --config configs/settings/eval_mapper.yaml --stats-only --num-batches 100
#
# With CLI overrides:
# python eval_mapper_invertibility.py --config configs/settings/eval_mapper.yaml --num-batches 10
