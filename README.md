# MelCausalVAE

A Variational Autoencoder (VAE) for audio mel spectrograms, featuring a causal Convformer encoder and a DiT-based Conditional Flow Matching (CFM) decoder.

## Structure Overview

- `modules/`: Core model components.
  - `VAE.py`: Main model wrapper.
  - `encoder/`: Causal Convformer architecture.
  - `decoder/`: DiT-based CFM decoder.
  - `configs.py`: Dataclass-based configurations.
- `configs/`: Hydra configuration system.
  - `defaults/`: Base configurations for encoder, decoder, and training.
  - `settings/`: Experiment-specific overrides.
- `data/`: Dataset loaders (LibriTTS, MLS, etc.).
- `train.py`: Main training script using Hugging Face Trainer and Hydra.

## Installation

```bash
pip install -r requirements.txt
```

*Note: For optimal performance, [Flash Attention](https://github.com/Dao-AILab/flash-attention) is recommended.*

## Training

The project uses [Hydra](https://hydra.cc/) for configuration management.

### Single GPU
```bash
python train.py experiment=your_experiment_name
```

### Multi-GPU (Accelerate)
```bash
accelerate launch train.py experiment=your_experiment_name
```

### DeepSpeed
```bash
accelerate launch --config_file configs/deepspeed/ds_config.yaml train.py experiment=your_experiment_name
```

Overrides can be applied directly via CLI:
```bash
python train.py training.learning_rate=1e-4 training.per_device_train_batch_size=8
```

