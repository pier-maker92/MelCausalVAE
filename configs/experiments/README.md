# LibriSpeech latent export on Narval

The three `encode-librispeech-dicodec-v2-ls-*.yaml` files are run_job experiments
for checkpoints `paper/v2/ls/25`, `12.5`, and `6.25`. Their installed copies are in
`/scratch/piermel/experiments/` on Narval.

Edit `dataset_partitions` in the installed experiment to select the splits.
The same list controls `copy_dataset.sh` and `scripts/encode_dataset.py`, through
`encoding.experiment_config`. Keep the latter path consistent if renaming an
experiment. Initially all seven original partitions are selected.

For example, to stage and encode only these two partitions:

```yaml
dataset_partitions:
  - train_clean_100
  - dev_clean
```

Launch manually when ready (no jobs were submitted while preparing these files):

```bash
sh /scratch/piermel/scripts/run_job.sh encode-librispeech-dicodec-v2-ls-25
sh /scratch/piermel/scripts/run_job.sh encode-librispeech-dicodec-v2-ls-12.5
sh /scratch/piermel/scripts/run_job.sh encode-librispeech-dicodec-v2-ls-6.25
```

Each output root is:

```text
/scratch/piermel/datasets/dicodec/librispeech-dicodec-v2-ls-<version>/
  train_clean_100/shard_00000.parquet
  dev_clean/shard_00000.parquet
  ...only selected partitions...
```

Files contain exactly `z` and the `attributes` struct (`z_sem`, `z_pros`, `z_mean`),
as float32 nested lists. Temporal padding is removed; `z_mean` keeps shape `[1,D]`.
All source utterances are preserved, in sorted source-file and original row order;
there is no duration filter, split merging, audio, ID, or transcript in the output.

`encoding.shard_size_mb: 512` is a hard **512 MiB** file-size limit (536870912 bytes,
matching the original script's convention). Files use uncompressed Parquet without
dictionaries. Sizes are approximately the configured limit; last shards are smaller.
The writer checks actual file sizes and splits at row boundaries when necessary.
A single row larger than the limit raises an error instead of violating it.

The architecture is loaded from each checkpoint's `config.json`. Encoding uses
float32 and one utterance at a time to avoid padding-dependent normalization. The
model and input tensors share the device. `num_gpus` must stay 1. Output partitions
are published only after successful completion and are never overwritten; to add
more partitions, select only those not already exported.

Validation: local synthetic audio roundtrip, exact schema, row preservation,
padding removal, on-disk shard size, selected-input validation, failed-partition
cleanup, Hydra overrides, and rsync partition filters. No real checkpoint inference
or cluster job was run. `run_job.sh` and `copy_dataset.sh` were left unchanged.
