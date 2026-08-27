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

### DeepSpeed
```bash
accelerate launch --config_file configs/deepspeed/ds_config.yaml train.py settings=your_experiment_name
```

### Managing Configurations via CLI
Hydra allows you to override any configuration parameter directly from the command line. For example:
```bash
python train.py settings=your_experiment_name training.learning_rate=1e-4 training.per_device_train_batch_size=8
```
