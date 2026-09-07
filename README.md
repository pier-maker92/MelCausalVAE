# Dicodec

## Overview

Dicodec is a disentangled audio codec that can process either WavLM features or Mel spectrograms as input. Its architecture is designed to explicitly isolate different components of the audio signal through a multi-branch approach:

- **Global Branch:** Dedicated to extracting and modeling the global timbre (speaker identity).
- **Local Branch:** Focuses on modeling the semantics and prosody of the speech.

By disentangling these elements, Dicodec allows you to easily isolate and manipulate specific audio components (e.g., separating the speaker's voice from the spoken content and intonation).

## Installation

To set up the environment and install the required dependencies using `uv`, run:

```bash
uv pip install -e .
uv pip install -e ".[training,eval]"
```


## Training

Dicodec uses [Hydra](https://hydra.cc/) for robust configuration management. The base configurations are located in `configs/defaults/`, while experiment-specific overrides are stored in `configs/settings/`.

### Running an Experiment (Single GPU)
To start a training run with a specific setting file, use:
```bash
python train.py settings=your_experiment_name
```

### Multi-GPU (Accelerate)
```bash
accelerate launch train.py settings=your_experiment_name
```

### Fine-tune the decoder and speaker encoder

Start from a compatible checkpoint with the same model settings:

```bash
python train.py settings=dicodec/18 \
  training.from_pretrained=/path/to/checkpoint \
  training.finetune_decoder=true \
  training.output_dir=checkpoints/decoder-finetune
```

Only `speaker_encoder` and `decoder` parameters receive gradients. Both use
`decoder_lr`, `decoder_min_lr`, and `decoder_warmup_ratio`. Shared WavLM, the
latent encoder, feature extractors, vocoder, and external quantizer stay frozen
and in evaluation mode, including after Trainer switches back to training.
Encoder sampling, dropout, and KL loss are therefore disabled in this mode.
Training still uses audio for the speaker input and target mel spectrograms;
the latent-only Parquet export is not a decoder fine-tuning dataset.

Use `training.from_pretrained` to start this new training phase with a fresh
optimizer. Use `training.resume_from_checkpoint` only to resume a run already
started with `training.finetune_decoder=true`, keeping that flag enabled.

### DeepSpeed
```bash
accelerate launch --config_file configs/deepspeed/ds_config.yaml train.py settings=your_experiment_name
```

### Managing Configurations via CLI
Hydra allows you to override any configuration parameter directly from the command line. For example:
```bash
python train.py settings=your_experiment_name training.learning_rate=1e-4 training.per_device_train_batch_size=8
```
