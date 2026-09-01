import dataclasses
import json
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from vocos import Vocos

_orig_get_field = dataclasses._get_field


def _patched_get_field(cls, a_name, a_type, *args, **kwargs):
    try:
        return _orig_get_field(cls, a_name, a_type, *args, **kwargs)
    except ValueError as e:
        if "mutable default" in str(e):
            default = getattr(cls, a_name, dataclasses.MISSING)
            actual_default = (
                default.default if isinstance(default, dataclasses.Field) else default
            )
            if actual_default is not dataclasses.MISSING:
                default_cls = actual_default.__class__
                orig_hash = getattr(default_cls, "__hash__", None)
                try:
                    default_cls.__hash__ = lambda self: id(self)
                except TypeError:
                    pass
                try:
                    return _orig_get_field(cls, a_name, a_type, *args, **kwargs)
                finally:
                    try:
                        default_cls.__hash__ = orig_hash
                    except TypeError:
                        pass
        raise


dataclasses._get_field = _patched_get_field

from modules.builder import build_model  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demopage"
AUDIO_DIR = DEMO_DIR / "audio"
CHECKPOINT_DIR = ROOT / "checkpoints" / "dropout" / "18"
KMEANS_PATH = ROOT / "checkpoints" / "kmeans" / "libritts-r" / "encoder_kmeans.pt"
SOURCES = {
    "male": ROOT / "ablations" / "male.wav",
    "female": ROOT / "ablations" / "female.wav",
    "pier": ROOT / "ablations" / "pier.wav",
    "fra": ROOT / "ablations" / "fra.wav",
    "podcast": ROOT / "ablations" / "podcastvoice.mp3",
}


def load_wav_mono_resampled(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    peak = wav.abs().max().clamp(min=1e-6)
    return (wav / peak).squeeze(0)


def save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    audio = audio.detach().cpu()
    audio = audio / audio.abs().max().clamp(min=1e-6)
    torchaudio.save(str(path), audio, sample_rate)


def load_kmeans_codebook(path: Path) -> dict:
    if path.is_dir():
        path = path / "encoder_kmeans.pt"
    codebook = torch.load(str(path), map_location="cpu")
    if "centroids" not in codebook:
        raise ValueError(f"K-means checkpoint has no centroids: {path}")
    if "latent_selection" not in codebook and "feature_dims" not in codebook:
        raise ValueError(
            "K-means checkpoint must contain latent_selection or legacy feature_dims."
        )
    return codebook


def kmeans_keep_dims(codebook: dict) -> int:
    selection = codebook.get("latent_selection")
    if selection is None:
        return int(codebook["feature_dims"])
    indices = selection.get("indices")
    if indices is not None:
        sorted_indices = sorted(int(index) for index in indices)
        expected = list(range(len(sorted_indices)))
        if sorted_indices != expected:
            raise ValueError(
                "kmeans-only demo supports contiguous latent selections starting at 0."
            )
        return len(sorted_indices)
    start = int(selection.get("start", 0))
    end = int(selection["end"])
    if start != 0:
        raise ValueError(
            "kmeans-only demo supports contiguous latent selections starting at 0."
        )
    return end


def reconstruct(
    model,
    vocoder,
    audios_srs,
    output_path: Path,
    speaker_embedding=None,
    first_dims: int | None = None,
    zero_first_dims: int | None = None,
    kmeans_codebook: dict | None = None,
    kmeans_only: bool = False,
) -> None:
    params = {
        "audios_srs": audios_srs,
        "num_steps": 16,
        "temperature": 0.3,
        "guidance_scale": 1.3,
    }
    if speaker_embedding is not None:
        params["speaker_embedding"] = speaker_embedding
    if first_dims is not None:
        params["chunk_size"] = 1
        params["chunk"] = first_dims
    if zero_first_dims is not None:
        params["chunk_size"] = 1
        params["exclude_start_chunk"] = zero_first_dims
    if kmeans_codebook is not None:
        params["kmeans_codebook"] = kmeans_codebook
        params["kmeans_chunk_size"] = 16384
    if kmeans_only:
        if kmeans_codebook is None:
            raise ValueError("kmeans_only requires a kmeans_codebook.")
        params["chunk_size"] = 1
        params["chunk"] = kmeans_keep_dims(kmeans_codebook)

    out = model.encode_decode(**params)
    mel = out["decoder_output"].audio_features
    audio = vocoder.decode(mel.permute(0, 2, 1))
    save_audio(output_path, audio, model.config.sample_rate)


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for old_audio in AUDIO_DIR.glob("*.wav"):
        old_audio.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    with open(CHECKPOINT_DIR / "config.json") as f:
        cfg = json.load(f)

    model = build_model(cfg)
    model.from_pretrained(str(CHECKPOINT_DIR / "model.safetensors"))
    model.eval().to(device)

    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    kmeans_codebook = load_kmeans_codebook(KMEANS_PATH)

    waves = {}
    for name, source in SOURCES.items():
        wav = load_wav_mono_resampled(source, model.config.sample_rate).to(device)
        waves[name] = wav
        save_audio(AUDIO_DIR / f"{name}.wav", wav[None, :], model.config.sample_rate)

    audios = {
        name: [(wav, model.config.sample_rate)]
        for name, wav in waves.items()
    }

    with torch.inference_mode():
        speaker_embeddings = {
            name: model.extract_speaker_embedding(audio)
            for name, audio in audios.items()
        }

        for source_name, source_audio in audios.items():
            reconstruct(
                model,
                vocoder,
                source_audio,
                AUDIO_DIR / f"{source_name}_reconstruction.wav",
            )
            reconstruct(
                model,
                vocoder,
                source_audio,
                AUDIO_DIR / f"{source_name}_first4.wav",
                first_dims=4,
            )
            reconstruct(
                model,
                vocoder,
                source_audio,
                AUDIO_DIR / f"{source_name}_dims4_64.wav",
                zero_first_dims=4,
            )
            reconstruct(
                model,
                vocoder,
                source_audio,
                AUDIO_DIR / f"{source_name}_kmeans.wav",
                kmeans_codebook=kmeans_codebook,
                kmeans_only=True,
            )
            reconstruct(
                model,
                vocoder,
                source_audio,
                AUDIO_DIR / f"{source_name}_kmeans_tail.wav",
                kmeans_codebook=kmeans_codebook,
            )

            for target_name, target_embedding in speaker_embeddings.items():
                if target_name == source_name:
                    continue
                reconstruct(
                    model,
                    vocoder,
                    source_audio,
                    AUDIO_DIR / f"{source_name}_swap_{target_name}.wav",
                    speaker_embedding=target_embedding,
                )


if __name__ == "__main__":
    main()
