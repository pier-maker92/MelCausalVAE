#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import torch
import torchaudio


HYBRID_REPO = Path("/Users/software/Research/hybrid_tts")
HYBRID_CHECKPOINT = HYBRID_REPO / "checkpoints" / "dicodec-18-kmeans-512"
Dicodec_CHECKPOINT = Path("/Users/software/Research/MelCausalVAE/checkpoints/paper/disent/dicodec-18")
KMEANS_PATH = Path("/Users/software/Research/MelCausalVAE/kmeans/512")
AUDIO_DIR = Path(__file__).resolve().parent / "audio"
VOICE_CONDITIONS = {
    "male": AUDIO_DIR / "male.wav",
    "female": AUDIO_DIR / "female.wav",
}

SECTIONS = {
    "tts": [
        (
            "warm_sun",
            "The warm sun poured over the soft grass as I settled into my favorite hammock.",
        ),
        (
            "get_out",
            "Get out of my way, I won’t hesitate to push through anyone who stands in my path!",
        ),
        (
            "strange_noise",
            "I heard a strange noise in the dark, and my heart raced as I fumbled for the light switch.",
        ),
        (
            "faded_letter",
            "I held the faded letter in my hands and tears streaming down my face as memories of happier times flooded back.",
        ),
        (
            "shoelaces_pie",
            "I tripped over my own shoelaces and landed face-first in a pie. At least dessert was served!",
        ),
    ],
    "tts_long": [
        (
            "wandering_bean",
            "In the heart of a bustling city, where skyscrapers touch the clouds and the streets hum with life, there lies a quaint little café known as The Wandering Bean. Tucked away between a vibrant bookstore and an art gallery, this café is a hidden gem. Its rustic wooden sign sways gently in the breeze, inviting passersby to step inside and escape the chaos outside.",
        )
    ],
    "tts_multilingual": [
        (
            "french_1",
            "Ce matin, je me suis levé très tôt parce que je voulais absolument terminer un chapitre de mon livre avant que la journée ne commence vraiment.",
        ),
        (
            "french_2",
            "Si le temps reste beau demain, j’aimerais faire une longue promenade le long de la rivière et prendre quelques photos du paysage.",
        ),
        (
            "spanish_1",
            "Esta mañana me levanté muy temprano porque quería terminar un capítulo de mi libro antes de que empezara realmente el día.",
        ),
        (
            "spanish_2",
            "Si mañana hace buen tiempo, me gustaría dar un largo paseo junto al río y sacar algunas fotos del paisaje.",
        ),
        (
            "italian_1",
            "Questa mattina mi sono alzato molto presto perché volevo assolutamente finire un capitolo del mio libro prima che la giornata cominciasse davvero.",
        ),
        (
            "italian_2",
            "Se domani il tempo sarà bello, mi piacerebbe fare una lunga passeggiata lungo il fiume e scattare alcune foto del paesaggio.",
        ),
    ],
}


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_kmeans(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ckpt_path = path / "encoder_kmeans.pt" if path.is_dir() else path
    codebook = torch.load(ckpt_path, map_location="cpu")
    return codebook["centroids"].to(device=device, dtype=dtype)


def load_speaker_embedding(path: Path, dicodec: torch.nn.Module, device: torch.device) -> torch.Tensor:
    wav, sample_rate = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)
    max_abs = wav.abs().max()
    if max_abs > 0:
        wav = wav / max_abs
    speaker_embedding = dicodec.extract_speaker_embedding([(wav.to(device), sample_rate)])
    if speaker_embedding is None:
        raise RuntimeError(f"No speaker embedding returned for {path}")
    return speaker_embedding.to(device=device, dtype=next(dicodec.parameters()).dtype)


def align_continuous(z_cont, length: int, dim: int, dtype: torch.dtype, device: torch.device):
    if z_cont is None:
        return torch.zeros((1, length, dim), dtype=dtype, device=device)
    z_cont = z_cont.to(device=device, dtype=dtype)
    if z_cont.shape[1] == length:
        return z_cont
    if z_cont.shape[1] < length:
        pad = torch.zeros(
            (z_cont.shape[0], length - z_cont.shape[1], z_cont.shape[2]),
            dtype=dtype,
            device=device,
        )
        return torch.cat([z_cont, pad], dim=1)
    return z_cont[:, :length]


def synthesize(vocoder, mel: torch.Tensor, mel_mask: torch.BoolTensor, device: torch.device):
    mel = mel[0][~mel_mask[0]].unsqueeze(0).permute(0, 2, 1).float().to(device)
    wav = vocoder.decode(mel).squeeze()
    wav = wav / (wav.abs().max() + 1e-8)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav.cpu()


def main():
    sys.path.insert(0, str(HYBRID_REPO))
    os.environ.setdefault("SCRATCH", "/Users/software/Research")

    from inference import (
        clean_text_and_phonemize,
        load_hybrid_model,
        load_dicodec,
        load_vocoder,
        trim_unpaired_discrete_tokens,
    )
    from util import build_tokenizer

    device = choose_device()
    dtype = torch.float32 if device.type in {"cpu", "mps"} else torch.bfloat16
    print(f"device={device} dtype={dtype}", flush=True)

    with (HYBRID_CHECKPOINT / "config.json").open() as f:
        cfg = json.load(f)
    cfg["dicodec_checkpoint"] = str(Dicodec_CHECKPOINT)
    cfg["kmeans_path"] = str(KMEANS_PATH)

    with (HYBRID_REPO / "data" / "phoneme_vocab.json").open() as f:
        phoneme_vocab = json.load(f)

    tokenizer = build_tokenizer(cfg, pretrinaed=False)
    centroids = load_kmeans(KMEANS_PATH, device=device, dtype=dtype)
    hybrid_model = load_hybrid_model(cfg, str(HYBRID_CHECKPOINT), device, dtype, tokenizer)
    dicodec = load_dicodec(str(Dicodec_CHECKPOINT), device, dtype)
    if dicodec is None:
        raise RuntimeError(f"Could not load Dicodec from {Dicodec_CHECKPOINT}")
    if getattr(dicodec, "speaker_encoder_type", None) == "wavlm" and getattr(dicodec, "speaker_encoder", None) is not None:
        dicodec.speaker_encoder = dicodec.speaker_encoder.to(device=device, dtype=torch.float32)
    vocoder = load_vocoder(cfg.get("vocoder_checkpoint", "vocos"), device)
    if vocoder is None:
        raise RuntimeError("Could not load vocoder")

    speaker_embeddings = {
        name: load_speaker_embedding(path, dicodec, device)
        for name, path in VOICE_CONDITIONS.items()
    }

    manifest = {}
    generator = torch.Generator(device=device).manual_seed(42)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for section, rows in SECTIONS.items():
            manifest[section] = []
            for slug, text in rows:
                print(f"generating {section}/{slug}", flush=True)
                prompt_ids = clean_text_and_phonemize(text, phoneme_vocab)
                if not prompt_ids:
                    raise RuntimeError(f"Empty phoneme sequence for {section}/{slug}")
                prompt_ids.append(tokenizer.start_audio_id)
                batch = {
                    "discrete_sequence": torch.tensor([prompt_ids], dtype=torch.long, device=device),
                    "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.bool, device=device),
                }
                target_len = max(64, int(len(prompt_ids) * 2.2))
                sample = hybrid_model.sample(
                    batch=batch,
                    max_steps=target_len,
                    temperature=0.0,
                    num_steps=8,
                    diffusion_temperature=0.5,
                    guidance_scale=1.3,
                    dicodec=dicodec,
                    generator=generator,
                )

                discrete = sample["discrete_tokens"].reshape(-1).long()
                if sample.get("discrete_lengths") is not None:
                    discrete = discrete[: int(sample["discrete_lengths"][0].item())]
                discrete = discrete[discrete >= 0]
                if discrete.numel() == 0:
                    raise RuntimeError(f"No audio tokens generated for {section}/{slug}")
                z_cont = sample["continuous_tokens"][:, : discrete.numel()]
                discrete = trim_unpaired_discrete_tokens(discrete, z_cont)
                z_cont = z_cont[:, : discrete.numel()]
                if discrete.min().item() < 0 or discrete.max().item() >= centroids.shape[0]:
                    raise RuntimeError(
                        f"K-means token out of range for {section}/{slug}: "
                        f"min={discrete.min().item()} max={discrete.max().item()}"
                    )
                z_semantic = centroids.index_select(0, discrete).unsqueeze(0)
                z_acoustic = align_continuous(
                    z_cont,
                    length=discrete.numel(),
                    dim=hybrid_model.config.continuous_dim,
                    dtype=dtype,
                    device=device,
                )
                z = torch.cat([z_semantic, z_acoustic], dim=-1)
                padding_mask = torch.zeros(z.shape[:2], dtype=torch.bool, device=device)

                audio_files = {}
                for speaker, speaker_embedding in speaker_embeddings.items():
                    mel, mel_mask = dicodec.sample(
                        num_steps=8,
                        temperature=0.2,
                        guidance_scale=1.3,
                        z=z,
                        padding_mask=padding_mask,
                        speaker_embedding=speaker_embedding,
                    )
                    wav = synthesize(vocoder, mel, mel_mask, device)
                    filename = f"{section}_{slug}_{speaker}.wav"
                    torchaudio.save(str(AUDIO_DIR / filename), wav, 24000)
                    audio_files[speaker] = filename
                    print(f"saved {filename}", flush=True)

                manifest[section].append(
                    {
                        "slug": slug,
                        "text": text,
                        "audio": audio_files,
                    }
                )

    with (AUDIO_DIR / "tts_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"wrote {AUDIO_DIR / 'tts_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
