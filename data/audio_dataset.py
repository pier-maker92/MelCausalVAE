import os
import json
import torch
import random
from dataclasses import dataclass
import torchaudio.transforms as T
from torch.utils.data import Dataset
from typing import Optional, Sequence, Dict
from data.tokenizer import BPETokenizer


@dataclass
class BPEOutput:
    semantic_ids: torch.LongTensor
    durations: torch.LongTensor
    semantic_embeddings: Optional[torch.FloatTensor] = None


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
        audio, sr = self._process_audio(audio_array, audio_data["sampling_rate"], target_sr)
        if max_duration and audio.shape[0] > sr * max_duration:
            audio = audio[: sr * max_duration]
        return audio, sr

    def __len__(self):
        return len(self.train_dataset)

    def _process_audio_output(self, data_dict, audio_data, target_sr=24000):
        audio_output, sr_output = self._process_audio_component(
            audio_data, target_sr=target_sr, max_duration=10  # FIXME hardcoded duration
        )
        data_dict.update({"audio_output": [audio_output], "audio_output_sr": [sr_output]})


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = dict()
        # handling etherogeneous samples in the batch, if a key is not present in the batch, add None in the index corresponding to the sample
        batch_input_audios_srs = [None] * len(instances)
        batch_output_audios_srs = [None] * len(instances)
        batch_condition_audios_srs = [None] * len(instances)
        batch_transcription_ids = [None] * len(instances)
        batch_aligned_transcription_ids = [None] * len(instances)
        batch_transcription = [None] * len(instances)
        batch_language = [None] * len(instances)
        batch_ids = [None] * len(instances)
        batch_tokenized_units = [None] * len(instances)
        batch_tokenized_units_durations = [None] * len(instances)
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
            if "audio_condition" in instance:
                batch_condition_audios_srs[i] = (
                    instance["audio_condition"][0],
                    instance["audio_condition_sr"][0],
                )
            if "transcription_ids" in instance:
                batch_transcription_ids[i] = instance["transcription_ids"]
            if "aligned_transcription_ids" in instance:
                batch_aligned_transcription_ids[i] = instance["aligned_transcription_ids"]
            if "transcription" in instance:
                batch_transcription[i] = instance["transcription"]
            if "language" in instance:
                batch_language[i] = instance["language"]
            if "ids" in instance:
                batch_ids[i] = instance["ids"]
            if "tokenized_units" in instance:
                batch_tokenized_units[i] = torch.LongTensor(instance["tokenized_units"])
            if "tokenized_units_durations" in instance:
                batch_tokenized_units_durations[i] = torch.LongTensor(instance["tokenized_units_durations"])

        # if not all none add to the batch
        def all_none(batch):
            return all([x is None for x in batch])

        if not all_none(batch_input_audios_srs):
            batch["input_audios_srs"] = batch_input_audios_srs
        if not all_none(batch_output_audios_srs):
            batch["output_audios_srs"] = batch_output_audios_srs
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
        if not all_none(batch_tokenized_units) and not all_none(batch_tokenized_units_durations):
            # pad the tokenized units to the same length using rnn padding
            tokenized_units = torch.nn.utils.rnn.pad_sequence(
                batch_tokenized_units,
                batch_first=True,
                padding_value=16384,
            )
            tokenized_units_durations = torch.nn.utils.rnn.pad_sequence(
                batch_tokenized_units_durations,
                batch_first=True,
                padding_value=0,
            )
            batch["hubert_guidance"] = BPEOutput(semantic_ids=tokenized_units, durations=tokenized_units_durations)
        return batch


class TrainDatasetWrapper(SimpleAudioDataset):
    def __init__(self, dataset: SimpleAudioDataset, split: str):
        super().__init__()
        assert split in ["train", "test"], "split must be either train or test"
        self.dataset = getattr(dataset, f"{split}_dataset")
        self.tokenizer = BPETokenizer(n_initial_units=1024, target_vocab_size=16384, deduplicate=True, verbose=True)
        self.tokenizer.load("data/tokenizer.json")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]
        self._process_audio_output(data_dict, data["audio"], target_sr=24000)
        data_dict["ids"] = data.get("id")
        data_dict["language"] = data.get("language", "en")
        # FIXME creating this monstrosity to test the padding system with a A10 gpu of only 16 GB
        limit = 500
        data_dict["units"] = data.get("audio_codes", None)[:limit]
        audio_codes, durations = self.tokenizer.encode(data_dict["units"])
        data_dict["tokenized_units"] = audio_codes
        data_dict["tokenized_units_durations"] = durations

        return data_dict


class HubertDatasetWrapper(SimpleAudioDataset):
    def __init__(self, dataset: SimpleAudioDataset, split: str):
        super().__init__()
        assert split in ["train", "test"], "split must be either train or test"
        self.dataset = getattr(dataset, f"{split}_dataset")
        self.tokenizer = BPETokenizer(n_initial_units=1024, target_vocab_size=16384, deduplicate=True, verbose=True)
        self.tokenizer.load("data/tokenizer.json")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]
        data_dict["audio_codes"] = data.get("audio_codes", None)
        data_dict["audio_codes"] = self.tokenizer.encode_batch([data_dict["audio_codes"]])
        return data_dict


class TestDatasetWrapper(SimpleAudioDataset):
    def __init__(self, dataset: SimpleAudioDataset, split: str):
        super().__init__()
        assert split in ["test", "train"], "split must be test or train"
        self.dataset = getattr(dataset, f"{split}_dataset")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]
        self._process_audio_output(data_dict, data["audio"])
        self._process_transcription(data_dict, data.get("text_normalized", "transcript"))
        data_dict["language"] = data.get("language", "en")
        return data_dict

    def _process_transcription(self, data_dict, transcription):
        data_dict.update({"transcription": transcription})
        return data_dict
