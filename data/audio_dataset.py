import os
import json
import torch
import random
import warnings
from dataclasses import dataclass
import torchaudio.transforms as T
import torchaudio.sox_effects as SoxEffects
from torch.utils.data import Dataset
from typing import Optional, Sequence, Dict


class SimpleAudioDataset(Dataset):
    def __init__(self):
        self._audio_worker_pid = None

    def _process_audio(self, audio: torch.Tensor, sr: int, target_sr: int):
        if target_sr is not None:  # handle resampling
            if sr != target_sr:
                audio = T.Resample(sr, target_sr)(audio)
            sr = target_sr
        # normalize audio
        audio = audio / (audio.abs().max() + 1e-8)
        return audio, sr

    def _process_audio_component(self, audio_data, target_sr, max_duration=None):
        """Helper method to process audio components with optional duration limiting"""
        audio_array = torch.Tensor(audio_data["array"]).to(torch.float32)
        audio, sr = self._process_audio(
            audio_array, audio_data["sampling_rate"], target_sr
        )
        if max_duration and audio.shape[0] > sr * max_duration:
            audio = audio[: sr * max_duration]
        return audio, sr

    def __len__(self):
        return len(self.train_dataset)

    def _process_audio_output(
        self, data_dict, audio_data, key_name: str = "audio_output", target_sr=24000
    ):
        audio_output, sr_output = self._process_audio_component(
            audio_data,
            target_sr=target_sr,
        )
        data_dict.update({f"{key_name}": [audio_output], f"{key_name}_sr": [sr_output]})

    def _pitch_shift_audio(
        self, audio: torch.Tensor, sr: int, max_abs_semitones: float
    ) -> torch.Tensor:
        if max_abs_semitones <= 0.0:
            return audio

        worker_pid = os.getpid()
        if self._audio_worker_pid != worker_pid:
            torch.set_num_threads(1)
            self._audio_worker_pid = worker_pid

        n_steps = random.uniform(-max_abs_semitones, max_abs_semitones)
        if abs(n_steps) < 1e-6:
            return audio

        audio_in = audio
        if audio_in.ndim == 1:
            audio_in = audio_in.unsqueeze(0)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*torchaudio\.sox_effects.*deprecated.*",
                category=UserWarning,
            )
            shifted, shifted_sr = SoxEffects.apply_effects_tensor(
                audio_in,
                sr,
                effects=[
                    ["pitch", f"{n_steps * 100.0:.4f}"],
                    ["rate", str(sr)],
                ],
                channels_first=True,
            )
        if shifted_sr != sr:
            raise RuntimeError(
                f"Pitch shifting changed sample rate from {sr} to {shifted_sr}."
            )

        shifted = shifted[..., : audio_in.shape[-1]]
        if shifted.shape[-1] < audio_in.shape[-1]:
            shifted = torch.nn.functional.pad(
                shifted, (0, audio_in.shape[-1] - shifted.shape[-1])
            )
        if audio.ndim == 1:
            shifted = shifted.squeeze(0)
        shifted = shifted.to(audio.dtype)
        return shifted / (shifted.abs().max() + 1e-8)


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = dict()
        # handling etherogeneous samples in the batch, if a key is not present in the batch, add None in the index corresponding to the sample
        batch_input_audios_srs = [None] * len(instances)
        batch_output_audios_srs = [None] * len(instances)
        batch_perturbed_audios_srs = [None] * len(instances)
        batch_condition_audios_srs = [None] * len(instances)
        batch_transcription_ids = [None] * len(instances)
        batch_aligned_transcription_ids = [None] * len(instances)
        batch_transcription = [None] * len(instances)
        batch_language = [None] * len(instances)
        batch_ids = [None] * len(instances)
        batch_phoneme_alignments = [None] * len(instances)
        batch_pitch_targets = [None] * len(instances)
        for i, instance in enumerate(instances):
            if "audio_input" in instance:
                batch_input_audios_srs[i] = (
                    instance["audio_input"][0],
                    instance["audio_input_sr"][0],
                )
            if "audio_output" in instance:
                batch_output_audios_srs[i] = (
                    instance["audio_output"][0],
                    instance["audio_output_sr"][0],
                )
            if "perturbed_audio" in instance:
                batch_perturbed_audios_srs[i] = (
                    instance["perturbed_audio"][0],
                    instance["perturbed_audio_sr"][0],
                )
            if "audio_condition" in instance:
                batch_condition_audios_srs[i] = (
                    instance["audio_condition"][0],
                    instance["audio_condition_sr"][0],
                )
            if "phoneme_alignments" in instance:
                batch_phoneme_alignments[i] = instance["phoneme_alignments"]
            if "transcription_ids" in instance:
                batch_transcription_ids[i] = instance["transcription_ids"]
            if "aligned_transcription_ids" in instance:
                batch_aligned_transcription_ids[i] = instance[
                    "aligned_transcription_ids"
                ]
            if "transcription" in instance:
                batch_transcription[i] = instance["transcription"]
            if "language" in instance:
                batch_language[i] = instance["language"]
            if "ids" in instance:
                batch_ids[i] = instance["ids"]
            if "pitch_targets" in instance:
                batch_pitch_targets[i] = instance["pitch_targets"]

        # if not all none add to the batch
        def all_none(batch):
            return all([x is None for x in batch])

        def collate_pitch_targets(targets):
            log_f0 = torch.nn.utils.rnn.pad_sequence(
                [target["log_f0"] for target in targets],
                batch_first=True,
                padding_value=0.0,
            )
            voiced = torch.nn.utils.rnn.pad_sequence(
                [target["voiced"] for target in targets],
                batch_first=True,
                padding_value=False,
            )
            return {"log_f0": log_f0, "voiced": voiced}

        if not all_none(batch_input_audios_srs):
            batch["input_audios_srs"] = batch_input_audios_srs
        if not all_none(batch_output_audios_srs):
            batch["output_audios_srs"] = batch_output_audios_srs
        if not all_none(batch_perturbed_audios_srs):
            batch["perturbed_audio_srs"] = batch_perturbed_audios_srs
        if not all_none(batch_condition_audios_srs):
            batch["condition_audios_srs"] = batch_condition_audios_srs
        if not all_none(batch_transcription_ids):
            batch["transcription_ids"] = batch_transcription_ids
        if not all_none(batch_aligned_transcription_ids):
            batch["aligned_transcription_ids"] = batch_aligned_transcription_ids
        if not all_none(batch_transcription):
            batch["transcription"] = batch_transcription
        if not all_none(batch_language):
            batch["language"] = batch_language
        if not all_none(batch_ids):
            batch["ids"] = batch_ids
        if not all_none(batch_phoneme_alignments):
            batch["phoneme_alignments"] = batch_phoneme_alignments
        if not all_none(batch_pitch_targets):
            batch["pitch_targets"] = collate_pitch_targets(batch_pitch_targets)
        return batch


@dataclass
class EvalDataCollator(object):
    """Collate examples for evaluation."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = dict()
        batch_output_audios_srs = [None] * len(instances)
        batch_16k_audios = [None] * len(instances)
        batch_transcription = [None] * len(instances)
        batch_language = [None] * len(instances)
        batch_ids = [None] * len(instances)
        for i, instance in enumerate(instances):
            if "audio_output" in instance:
                batch_output_audios_srs[i] = (
                    instance["audio_output"][0],
                    instance["audio_output_sr"][0],
                )
            if "16k_audio" in instance:
                batch_16k_audios[i] = instance["16k_audio"][0]

            if "transcription" in instance:
                batch_transcription[i] = instance["transcription"]
            if "language" in instance:
                batch_language[i] = instance["language"]
            if "ids" in instance:
                batch_ids[i] = instance["ids"]

        # if not all none add to the batch
        def all_none(batch):
            return all([x is None for x in batch])

        if not all_none(batch_output_audios_srs):
            batch["output_audios_srs"] = batch_output_audios_srs
        if not all_none(batch_16k_audios):
            batch["16k_audios"] = batch_16k_audios
        if not all_none(batch_transcription):
            batch["transcription"] = batch_transcription
        if not all_none(batch_language):
            batch["language"] = batch_language
        if not all_none(batch_ids):
            batch["ids"] = batch_ids
        return batch


class TrainDatasetWrapper(SimpleAudioDataset):
    def __init__(
        self,
        dataset: SimpleAudioDataset,
        split: str,
        max_audio_len: Optional[float] = None,
        enable_perturbed_audio: bool = False,
        perturbed_pitch_shift_max_semitones: float = 8.0,
        pitch_extractor: Optional[object] = None,
    ):
        super().__init__()
        assert split in ["train", "test"], "split must be either train or test"
        self.enable_perturbed_audio = enable_perturbed_audio
        self.perturbed_pitch_shift_max_semitones = float(
            perturbed_pitch_shift_max_semitones
        )
        self.pitch_extractor = pitch_extractor
        self.dataset = getattr(dataset, f"{split}_dataset")
        if max_audio_len is not None:
            self.dataset = self.dataset.filter(
                lambda x: (
                    x.get("duration", x.get("duration_sec", len(x["audio"]["array"]) / x["audio"]["sampling_rate"]))
                ) <= max_audio_len,
                num_proc=os.cpu_count() or 1
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]
        self._process_audio_output(data_dict, data["audio"])
        if self.enable_perturbed_audio:
            clean_audio = data_dict["audio_output"][0]
            clean_sr = data_dict["audio_output_sr"][0]
            perturbed_audio = self._pitch_shift_audio(
                clean_audio,
                clean_sr,
                self.perturbed_pitch_shift_max_semitones,
            )
            data_dict["perturbed_audio"] = [perturbed_audio]
            data_dict["perturbed_audio_sr"] = [clean_sr]
        if self.pitch_extractor is not None:
            pitch_targets = self.pitch_extractor(
                [(data_dict["audio_output"][0], data_dict["audio_output_sr"][0])]
            )
            data_dict["pitch_targets"] = {
                "log_f0": pitch_targets["log_f0"].squeeze(0),
                "voiced": pitch_targets["voiced"].squeeze(0),
            }
        data_dict["ids"] = data.get("id")
        data_dict["phoneme_alignments"] = data.get("phonemes", None)
        return data_dict


class TestDatasetWrapper(SimpleAudioDataset):
    def __init__(
        self,
        dataset: SimpleAudioDataset,
        split: str,
        max_audio_len: Optional[float] = None,
        enable_perturbed_audio: bool = False,
        perturbed_pitch_shift_max_semitones: float = 8.0,
    ):
        super().__init__()
        assert split in ["test", "train"], "split must be test or train"
        self.enable_perturbed_audio = enable_perturbed_audio
        self.perturbed_pitch_shift_max_semitones = float(
            perturbed_pitch_shift_max_semitones
        )
        self.dataset = getattr(dataset, f"{split}_dataset")
        if max_audio_len is not None:
            self.dataset = self.dataset.filter(
                lambda x: (
                    x.get("duration", x.get("duration_sec", len(x["audio"]["array"]) / x["audio"]["sampling_rate"]))
                ) <= max_audio_len,
                num_proc=os.cpu_count() or 1
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]
        self._process_audio_output(data_dict, data["audio"])
        if self.enable_perturbed_audio:
            clean_audio = data_dict["audio_output"][0]
            clean_sr = data_dict["audio_output_sr"][0]
            perturbed_audio = self._pitch_shift_audio(
                clean_audio,
                clean_sr,
                self.perturbed_pitch_shift_max_semitones,
            )
            data_dict["perturbed_audio"] = [perturbed_audio]
            data_dict["perturbed_audio_sr"] = [clean_sr]
        self._process_audio_output(
            data_dict, data["audio"], key_name="16k_audio", target_sr=16000
        )

        # Robust transcription field lookup
        transcription = (
            data.get("text_normalized") or data.get("transcript") or "transcript"
        )
        self._process_transcription(data_dict, transcription)

        data_dict["language"] = data.get("language", "en")
        data_dict["ids"] = data.get("id")
        return data_dict

    def _process_transcription(self, data_dict, transcription):
        data_dict.update({"transcription": transcription})
        return data_dict
