import os
import torch
from dataclasses import dataclass
import torchaudio.transforms as T
from torch.utils.data import Dataset
from typing import Optional, Sequence, Dict


class SimpleAudioDataset(Dataset):
    def __init__(self):
        pass

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
        self, data_dict, audio_data, key_name: str = "audio_input", target_sr=24000
    ):
        audio_output, sr_output = self._process_audio_component(
            audio_data,
            target_sr=target_sr,
        )
        data_dict.update({f"{key_name}": [audio_output], f"{key_name}_sr": [sr_output]})


@dataclass
class DataCollator(object):
    """Collate audio examples using the project-wide minimal batch contract."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = dict()
        batch_audio_input_srs = [None] * len(instances)
        batch_16k_audio = [None] * len(instances)
        batch_transcription = [None] * len(instances)
        batch_ids = [None] * len(instances)
        for i, instance in enumerate(instances):
            if "audio_input" in instance:
                batch_audio_input_srs[i] = (
                    instance["audio_input"][0],
                    instance["audio_input_sr"][0],
                )
            if "16k_audio" in instance:
                batch_16k_audio[i] = instance["16k_audio"][0]
            if "transcription" in instance:
                batch_transcription[i] = instance["transcription"]
            if "ids" in instance:
                batch_ids[i] = instance["ids"]

        def all_none(batch):
            return all([x is None for x in batch])

        if not all_none(batch_audio_input_srs):
            batch["audio_input_srs"] = batch_audio_input_srs
        if not all_none(batch_16k_audio):
            batch["16k_audio"] = batch_16k_audio
        if not all_none(batch_transcription):
            batch["transcription"] = batch_transcription
        if not all_none(batch_ids):
            batch["ids"] = batch_ids
        return batch


@dataclass
class EvalDataCollator(object):
    """Compatibility alias for evaluation code."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return DataCollator()(instances)


class TrainDatasetWrapper(SimpleAudioDataset):
    def __init__(
        self,
        dataset: SimpleAudioDataset,
        split: str,
        max_audio_len: Optional[float] = None,
    ):
        super().__init__()
        assert split in ["train", "test"], "split must be either train or test"
        self.dataset = getattr(dataset, f"{split}_dataset")
        if max_audio_len is not None:
            self.dataset = self.dataset.filter(
                lambda x: (
                    x.get(
                        "duration",
                        x.get(
                            "duration_sec",
                            len(x["audio"]["array"]) / x["audio"]["sampling_rate"],
                        ),
                    )
                )
                <= max_audio_len,
                num_proc=8,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]
        self._process_audio_output(data_dict, data["audio"])
        self._process_audio_output(
            data_dict, data["audio"], key_name="16k_audio", target_sr=16000
        )
        transcription = (
            data.get("text_normalized")
            or data.get("transcription")
            or data.get("transcript")
            or data.get("text")
            or ""
        )
        data_dict["transcription"] = transcription
        data_dict["ids"] = data.get("id")
        return data_dict


class TestDatasetWrapper(SimpleAudioDataset):
    def __init__(
        self,
        dataset: SimpleAudioDataset,
        split: str,
        max_audio_len: Optional[float] = None,
    ):
        super().__init__()
        assert split in ["test", "train"], "split must be test or train"
        self.dataset = getattr(dataset, f"{split}_dataset")
        if max_audio_len is not None:
            self.dataset = self.dataset.filter(
                lambda x: (
                    x.get(
                        "duration",
                        x.get(
                            "duration_sec",
                            len(x["audio"]["array"]) / x["audio"]["sampling_rate"],
                        ),
                    )
                )
                <= max_audio_len,
                num_proc=8,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]
        self._process_audio_output(data_dict, data["audio"])
        self._process_audio_output(
            data_dict, data["audio"], key_name="16k_audio", target_sr=16000
        )

        # Robust transcription field lookup
        transcription = (
            data.get("text_normalized")
            or data.get("transcription")
            or data.get("transcript")
            or data.get("text")
            or ""
        )
        self._process_transcription(data_dict, transcription)

        data_dict["ids"] = data.get("id")
        return data_dict

    def _process_transcription(self, data_dict, transcription):
        data_dict.update({"transcription": transcription})
        return data_dict
