# Speech language modeling quantizer

Learns discrete DiCodec tokens from frozen, aligned latent sequences `[B, T, D]`.
Implementation, YAML configuration, output dataclasses and tests live in this package;
`scripts/train_sm_quantizer.py` is the entry point.

## Objective

A framewise encoder maps `input[t]` into quantizer space. The causal transformer
consumes only quantized vectors, with a straight-through training path. Its context
at `t` sees tokens `0..t`; a small conditional flow matching head predicts
`target[t+1]`. A separate framewise MLP reconstructs `target[t]` with L1+L2 loss.
Both losses follow `data.target`, even when it differs from `data.input`.

The diffusion head follows the MLP formulation in `hybrid_tts/modules/diffusion_head`,
without a runtime dependency on that repository. For independently sampled noise/time:
`x(t) = (1 - (1-sigma_min)*t)*noise + t*target_next`, with velocity target
`target_next - (1-sigma_min)*noise`.

The total loss is the weighted sum of flow MSE, reconstruction L1, reconstruction
MSE, EMA commitment MSE and BSQ regularization. Each unweighted component is exposed
in the output dataclass and metrics. Only the applicable quantizer losses are active.
No continuous features bypass the quantizer. Downstream TTS quality requires
validation on actual speech tasks.

## Quantizers

Select `model.quantizer.type: vq_ema | bsq | fsq` and `codebook_size` in YAML.

- `vq_ema` (default): nearest-center assignments with online EMA clustering,
  commitment loss and optional dead-code resets. `quantizer.dim` controls its dimension.
- `bsq`: fixed binary spherical codes; dimension is `log2(codebook_size)`, requiring
  a power of two. Uses identity straight-through and the same regularizer as the
  existing quantizer trainer: negative marginal bit entropy plus squared off-diagonal
  probability covariance. Configure `quantizer.entropy_temperature` (default 1) and
  `loss.bsq_regularization` (default 0.1).
- `fsq`: fixed scalar levels; dimension is inferred from the preset below.
  Uses the shared FSQ's own rounding straight-through, preserving the `tanh`
  derivative. No commitment or BSQ regularization loss is applied.

BSQ and FSQ quantize online during training but do not update EMA centroids.
`quantizer.dim` and EMA settings have no effect on these two backends. Presets are
explicitly exported as `FSQ_LEVELS` in `fsq_levels.py`; they are checked against the
shared FSQ implementation when the model is built.

| Tokens | FSQ levels |
| --- | --- |
| 512 | `[8, 4, 4, 4]` |
| 1024 | `[8, 8, 4, 4]` |
| 2048 | `[8, 8, 8, 4]` |
| 4096 | `[8, 8, 8, 8]` |
| 8192 | `[16, 8, 8, 8]` |

For example, set `type: bsq, codebook_size: 1024` for ten binary dimensions,
or `type: fsq, codebook_size: 2048` for four scalar dimensions. Unsupported FSQ
vocabulary sizes fail at configuration loading.

## Narval Parquet datasets

The default YAML uses:

```yaml
data:
  format: parquet
  train_path: /scratch/piermel/datasets/dicodec/librispeech-dicodec-v2-ls-25
  train_partitions: [train_clean_100, train_clean_360]
  validation_path: null
  validation_partitions: []
  input: z
  target: z
  cache_dir: null
```

For the other dataset, change only `train_path` to
`/scratch/piermel/datasets/dicodec/librispeech-dicodec-v2-ls-12.5`.
Both exports have latent dimension 64 and the two training partitions above.
Use one dataset version per run; the repeated `ls-25` path is not a third dataset.

`input` and `target` independently accept `z` or `z_sem`: `z` reads the top-level
column and `z_sem` reads `attributes.z_sem`. All four combinations are supported.
To quantize full latents while reconstructing semantic latents, set `input: z` and
`target: z_sem`. There is no audio decoding or DiCodec inference in this loader.

Hugging Face Datasets (`datasets` and `pyarrow`, already in the training environment)
converts selected columns to a disk-backed Arrow cache. Set `cache_dir` to a writable
scratch directory when desired; null uses the Hugging Face default. Initial loading
requires cache disk space. The full dataset is not retained in RAM. Semantic loading
projects the `attributes` struct, which also contains the other attribute fields.

Partition selectors are explicit: only named directories are read, with sorted shards
and hidden files excluded. With an empty selector, the path must be a shard file or
a directory directly containing shards. No recursive discovery combines train/dev/test.
Validation is disabled until `validation_path` is supplied, with its own partitions.

The collator returns `LatentBatch(inputs, targets, valid_mask)`, converts both sources
to float32, checks finite values, dimensions and equal original frame counts, then
truncates both to the first `max_frames` and right-pads. Each sequence must have at
least two frames. Prechunk long utterances to train on frames beyond this prefix.
The common `model.latent_dim` applies to both sources.

## PT and custom datasets

For `.pt` data use `format: pt` and point `train_path` to a file or directory.
An item can be a tensor `[T, D]` for same-source training, or a mapping containing
`z` and either `attributes.z_sem` or top-level `z_sem`. Raw tensors cannot provide
separate input and target sources. `data.key` retains the legacy top-level key for
`z` (default `z`); the new source selectors remain `z | z_sem`.

For a map-style Python dataset set `data.factory: your_package.module:YourDataset`,
`train_kwargs: {...}` and optionally `validation_kwargs: {...}`. The callable must
return items with the same contract. Factory mode ignores format/path/partition settings.

## Training and resume

### Narval run_job preset

`configs/settings/dicodec/quantize/ema1024-zsem.yaml` is the Hydra preset for
EMA 1024, semantic input/target, and `train_clean_100` at 25 fps. The corresponding
launcher YAML is mirrored in `sm_quantizer/experiments/quantize-ls-v2-25-ema1024-zsem.yaml`
and installed at `/scratch/piermel/experiments/quantize-ls-v2-25-ema1024-zsem.yaml`.
After updating the repository on Narval, launch with:

```bash
sh /scratch/piermel/scripts/run_job.sh quantize-ls-v2-25-ema1024-zsem
```

The launcher stages `dicodec/librispeech-dicodec-v2-ls-25` under the job's
`SLURM_TMPDIR/datasets`; the trainer reads that copy and caches Arrow on the same
local disk. Checkpoints use the unique `training.output_dir` supplied by run_job.
The Hydra adapter consumes only the `sm_quantizer` namespace; shared DiCodec
settings and launcher metadata are not passed into the standalone dataclasses.

To select multiple partitions, update both `dataset_partitions` (staging) and
`extra_args` (training) in the experiment YAML, for example:

```yaml
dataset_partitions: [train_clean_100, train_clean_360]
extra_args: "sm_quantizer.data.train_partitions=[train_clean_100,train_clean_360]"
```

Keep `num_gpus: 1` and precision flags false; this trainer uses one process and
full precision. Resource defaults follow the existing Narval encoding experiment:
one 10 GB GPU slice, 8 CPUs, 16 GB RAM and 9 hours. They have not been benchmarked
on a full training run. `training.wandb_run_name` is accepted as launcher metadata;
the SM trainer writes JSONL metrics, not WandB events.

### Standalone CLI

```bash
python scripts/train_sm_quantizer.py --config dicodec/modules/sm_quantizer/default.yaml
python scripts/train_sm_quantizer.py --config my_config.yaml --resume outputs/sm_quantizer/last.pt
```

Training uses full precision AdamW on one device (`auto`: CUDA if available, otherwise
CPU), gradient clipping, optional validation and JSONL metrics. Atomic `last.pt`
checkpoints are saved after epochs and at `max_steps`, a total limit including resumed
steps. They contain model/EMA state, optimizer, config, epoch/batch cursor and
Python/PyTorch RNG states. Resume restores optimizer settings from the checkpoint.

Old EMA configurations/checkpoints receive defaults for the newly added fields and
retain the existing state-dict layout. Changing the quantizer, sources or data settings
on resume is rejected. Same-device, deterministic datasets reproduce uninterrupted
training; stochastic external datasets must manage their own RNG/worker state.
BSQ validation regularization is a frame-weighted average of batch statistics, so it
can depend on batch composition. This trainer does not implement distributed EMA.

## Python API and tests

```python
from dicodec.modules.sm_quantizer import SMQuantizer, load_config

config = load_config("my_config.yaml")
model = SMQuantizer(config.model)
output = model(inputs, valid_mask, target=targets)
output.loss.backward()

model.eval()  # freezes EMA and dropout
# Under torch.no_grad():
tokens = model.encode(inputs, valid_mask).indices
next_target = model.predict_next(inputs, valid_mask)  # [B, D], target space
```

Omitting `target` preserves same-source forward behavior. `valid_mask` is boolean
`[B, T]`, True for valid frames, with a nonempty contiguous prefix per sequence.
Token IDs at padding are -1. Sampling supports `steps`, `temperature` (including zero)
and a PyTorch `generator`, with midpoint ODE integration.

```bash
python -m unittest dicodec.modules.sm_quantizer.test_sm_quantizer
```

Tests cover all source/quantizer combinations, FSQ presets and bounding gradients,
BSQ regularization, Parquet fixtures, masking/causality, and local synthetic training
with deterministic resume for all three backends, including legacy EMA config loading.
