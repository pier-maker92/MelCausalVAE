import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

from scripts.encode_dataset import SCHEMA, ShardWriter, encode_partitions, read_experiment


def latent_row(index):
    z = np.full((10 + index % 3, 8), index, dtype=np.float32).tolist()
    return {"z": z, "attributes": {"z_sem": z, "z_pros": z, "z_mean": [z[0]]}}


class TinyModel:
    device = torch.device("cpu")
    config = SimpleNamespace(sample_rate=24000)

    def extract_features(self, audios, audio_16khz):
        assert audios[0][0].device == self.device
        assert audio_16khz[0].device == self.device
        assert audios[0][1] == 24000
        assert audio_16khz[0].numel() == 800
        z = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        mask = torch.tensor([[False, False, True]])
        return z, mask, None, None

    def encode(self, z, mask, compute_attributes):
        assert compute_attributes
        assert not torch.is_grad_enabled()
        mean = z[:, :2].mean(dim=1, keepdim=True)
        return SimpleNamespace(z=z, padding_mask=mask, attributes=SimpleNamespace(
            z_sem=z - mean, z_pros=torch.zeros_like(z), z_mean=mean,
        ))


class EncodingTest(unittest.TestCase):
    def test_shard_file_limit_and_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = ShardWriter(tmp, 6500)
            expected = [latent_row(i) for i in range(23)]
            for row in expected:
                writer.append(row)
            writer.close()
            paths = sorted(Path(tmp).glob("*.parquet"))
            self.assertGreater(len(paths), 1)
            rows = []
            for path in paths:
                self.assertLessEqual(path.stat().st_size, 6500)
                table = pq.read_table(path)
                self.assertGreater(table.num_rows, 0)
                self.assertEqual(table.schema, SCHEMA)
                rows.extend(table.to_pylist())
            self.assertEqual(rows, expected)
            self.assertEqual(writer.rows_written, len(expected))
            self.assertFalse(list(Path(tmp).glob("*.partial")))

    def test_empty_export_does_not_create_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            ShardWriter(tmp, 6500).close()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_unrepresentable_row_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = ShardWriter(tmp, 10)
            with self.assertRaisesRegex(ValueError, "single encoded sample"):
                writer.append(latent_row(0))
            self.assertEqual(list(Path(tmp).glob("*.parquet")), [])

    def fixture(self, root):
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{}')
        (checkpoint / "model.safetensors").touch()
        source = root / "source"
        buffer = io.BytesIO()
        sf.write(buffer, np.sin(np.arange(800) * 0.05), 16000, format="WAV")
        for part in ("train_clean_100", "dev_clean", "test_other"):
            folder = source / part
            folder.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([
                {"id": str(i), "audio": {"bytes": buffer.getvalue(), "path": "unused.wav"}}
                for i in range(3)
            ]), folder / 'audio.parquet')
        config = {
            'dataset_name': 'librispeech-aligned', 'num_gpus': 1,
            'dataset_extension': 'parquet', 'dataset_partitions': ['train_clean_100', 'dev_clean'],
            'encoding': {'checkpoint': str(checkpoint), 'model_version': 'v2-ls-25',
                         'shard_size_mb': 512, 'input_root': str(source), 'output_root': str(root / 'output')},
        }
        path = root / 'experiment.yaml'
        OmegaConf.save(OmegaConf.create(config), path)
        return path

    def test_selected_partitions_roundtrip_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            _, files, destination = read_experiment(path)
            encode_partitions(TinyModel(), files, destination, 6500)
            self.assertEqual(destination.name, 'librispeech-dicodec-v2-ls-25')
            self.assertEqual(sorted(p.name for p in destination.iterdir()), ['dev_clean', 'train_clean_100'])
            for part in files:
                table = pq.read_table(destination / part)
                self.assertEqual(table.column_names, ['z', 'attributes'])
                self.assertEqual(table.num_rows, 3)
                for row in table.to_pylist():
                    self.assertEqual(len(row['z']), 2)
                    attrs = row['attributes']
                    self.assertEqual(len(attrs['z_mean']), 1)
                    np.testing.assert_allclose(row['z'], np.array(attrs['z_sem']) + np.array(attrs['z_pros']) + np.array(attrs['z_mean']))
            with self.assertRaises(FileExistsError):
                read_experiment(path)

    def test_missing_selected_input_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            cfg = OmegaConf.load(path)
            cfg.dataset_partitions = ['train_other_500']
            OmegaConf.save(cfg, path)
            with self.assertRaisesRegex(FileNotFoundError, 'train_other_500'):
                read_experiment(path)

    def test_failed_partition_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            _, files, destination = read_experiment(path)
            with patch.object(TinyModel, 'encode', side_effect=RuntimeError('test failure')):
                with self.assertRaisesRegex(RuntimeError, 'test failure'):
                    encode_partitions(TinyModel(), files, destination, 6500)
            self.assertEqual(list(destination.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
