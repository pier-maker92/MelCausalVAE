# MelCausalVAE

**Build requirement**: Run `cd modules/Vits && python setup.py build_ext --inplace` to compile the `monotonic_align` extension.

A Variational Autoencoder (VAE) for audio mel spectrogram encoding and generation, featuring a causal Convformer encoder and Conditional Flow Matching (CFM) decoder.

## Overview

MelCausalVAE is a deep learning model designed for audio compression and generation through mel spectrogram representations. The architecture combines:

- **Causal Convformer Encoder**: A hybrid architecture using causal convolutions with dilations and transformer layers for efficient causal audio encoding
- **Conditional Flow Matching Decoder**: A DiT (Diffusion Transformer) based decoder that generates high-quality mel spectrograms from latent representations
- **Flash Attention Support**: Optimized attention mechanisms for faster training and inference
- **Scalable Training**: DeepSpeed integration for efficient multi-GPU training

### Key Features

- ✨ **Causal Architecture**: Ensures autoregressive consistency for streaming/real-time applications
- 🚀 **Flash Attention**: Multiple implementations (PyTorch SDPA, Dao-AILab flash-attn) for optimal performance
- 💾 **Memory Efficient**: DeepSpeed ZeRO-2/3 support for training large models
- 📊 **Production Ready**: Built on Hugging Face Transformers with W&B integration
- 🎵 **High Quality**: Flow matching decoder for superior audio reconstruction
- ⚡ **Fast Training**: Optimized for modern GPUs with mixed precision support

## Architecture

### Encoder: Causal Convformer

The encoder processes mel spectrograms `[B, T, 100]` through:

1. **Input Processing**: Projects mel spectrogram to higher dimensions
2. **Causal Conv Blocks**: 3 blocks with dilations (1, 2, 4) for temporal context
3. **Downsampling Stages**: Progressive temporal compression (configurable factor C)
4. **Frequency Collapse**: Reduces frequency dimension while preserving temporal information
5. **Transformer Tail**: Causal transformer layers for long-range dependencies
6. **Latent Projection**: Projects to latent space `[B, T/C, latent_dim]`

**Key Properties**:
- Time-causal: Output at time `t` depends only on inputs up to time `t`
- Configurable compression ratio (default: 8x)
- KL divergence regularization with warmup scheduling

### Decoder: Conditional Flow Matching (DiT)

The decoder generates mel spectrograms using:

1. **Flow Matching**: Continuous normalizing flow for high-quality generation
2. **DiT Architecture**: Transformer-based denoising with adaptive layer normalization
3. **Conditional Generation**: Uses encoder latent vectors as conditioning
4. **Flexible Sampling**: Adjustable number of steps, temperature, and guidance scale

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU support)
- PyTorch 2.0+ (recommended for flash attention)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/MelCausalVAE.git
cd MelCausalVAE

# Install PyTorch (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers accelerate datasets
pip install einops beartype vocos wandb pyyaml

# Optional: Install DeepSpeed for multi-GPU training
pip install deepspeed

# Optional: Install flash-attn for maximum performance (requires CUDA)
pip install flash-attn --no-build-isolation
```

## Quick Start

### Training

#### Basic Training (Single GPU)

```bash
python -m MelCausalVAE.train --exp-config MelCausalVAE/configs/settings/setting1.yaml
```

#### Multi-GPU Training with Accelerate

```bash
# First configure accelerate (one-time setup)
accelerate config

# Launch training
accelerate launch MelCausalVAE/train.py --exp-config MelCausalVAE/configs/settings/setting1.yaml
```

#### Training with DeepSpeed (Recommended for Large Models)

```bash
# Using the convenience script
cd MelCausalVAE
./launch_training.sh configs/settings/setting1.yaml --deepspeed

# Or manually
accelerate launch --deepspeed_config_file configs/deepspeed/zero3.json \
    MelCausalVAE/train.py --exp-config MelCausalVAE/configs/settings/setting1.yaml
```

See [TRAINING.md](TRAINING.md) for detailed training instructions and [DEEPSPEED_QUICKSTART.md](DEEPSPEED_QUICKSTART.md) for DeepSpeed optimization tips.

### Configuration

Training is configured through YAML files with a hierarchical structure:

- `configs/defaults/`: Default configurations
  - `train.yaml`: Training hyperparameters
  - `convformer.yaml`: Encoder architecture
  - `cfm.yaml`: Decoder architecture
- `configs/settings/`: Experiment-specific overrides
  - `setting1.yaml`: Example configuration

Example configuration override:

```yaml
# configs/settings/my_experiment.yaml
training:
  output_dir: outputs/my_experiment
  num_train_epochs: 5
  learning_rate: 1e-4
  per_device_train_batch_size: 4
  bf16: true
  report_to: wandb
  wandb_project: my-audio-vae
  wandb_run_name: experiment-1

convformer:
  compress_factor_C: 8
  tf_layers: 4
  latent_dim: 64
  
cfm:
  unet_dim: 1024
  unet_depth: 8
```

## Model Architecture Details

### Encoder Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compress_factor_C` | 8 | Temporal compression ratio |
| `tf_heads` | 8 | Number of attention heads |
| `tf_layers` | 4 | Number of transformer layers |
| `drop_p` | 0.1 | Dropout probability |
| `latent_dim` | 64 | Latent vector dimension |
| `n_residual_blocks` | 3 | Residual blocks per stage |
| `kl_loss_weight` | 1e-3 | KL divergence loss weight |
| `kl_loss_warmup_steps` | 1000 | Steps to warm up KL loss |

### Decoder Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `unet_dim` | 512 | Base model dimension |
| `unet_depth` | 6 | Number of DiT blocks |
| `unet_heads` | 8 | Number of attention heads |
| `unet_dropout_rate` | 0.0 | Dropout rate |
| `sigma` | 1e-5 | Flow matching noise scale |
| `mel_channels` | 100 | Mel spectrogram channels |

## Flash Attention

This project includes multiple flash attention implementations for optimal performance:

### 1. PyTorch SDPA (Recommended for PyTorch 2.0+)

The `Attend` module in `modules/utils.py` automatically uses PyTorch's Scaled Dot-Product Attention:

```python
from modules.utils import Attend

# Automatically uses flash attention if available
attend = Attend(dropout=0.1, flash=True)
```

### 2. Dao-AILab flash-attn

Custom implementation in `modules/flash_attn_encoder.py` using the flash-attn package:

```python
from modules.flash_attn_encoder import FlashTransformerEncoder

encoder = FlashTransformerEncoder(d_model=512, nhead=8, nlayers=4)
```

To use this, ensure flash-attn is installed:
```bash
pip install flash-attn --no-build-isolation
```

## Dataset

The model is trained on LibriTTS-R, a high-quality English speech dataset:

- **Dataset**: `parler-tts/libritts_r_filtered`
- **Subsets**: clean + other
- **Sample Rate**: 24kHz
- **Total Hours**: ~500+ hours of speech
- **Preprocessing**: Automatic conversion to mel spectrograms (100 channels)

The dataset is automatically downloaded and cached on first run.

## Monitoring and Logging

### Weights & Biases Integration

Enable W&B logging in your config:

```yaml
training:
  report_to: wandb
  wandb_project: my-project-name
  wandb_run_name: experiment-1
```

Logged metrics include:
- Training loss (total, audio loss, KL loss)
- Learning rate schedule
- Gradient norms
- Sample reconstructions (mel spectrograms + audio)

### Checkpointing

Checkpoints are saved to `output_dir` and include:
- Model weights
- Optimizer state
- Training configuration
- Resume capability

## Inference and Sampling

```python
import torch
from MelCausalVAE.modules.VAE import VAE, VAEConfig
from MelCausalVAE.modules.Encoder import ConvformerEncoderConfig
from MelCausalVAE.modules.cfm import DiTConfig
from MelCausalVAE.modules.melspecEncoder import MelSpectrogramConfig

# Load model
config = VAEConfig(
    encoder_config=ConvformerEncoderConfig(),
    decoder_config=DiTConfig(),
    mel_spec_config=MelSpectrogramConfig()
)
model = VAE(config)
model.load_state_dict(torch.load("path/to/checkpoint.pt"))
model.eval()

# Encode and reconstruct audio
# audios_srs: list of (audio_tensor, sample_rate) tuples
results = model.encode_and_sample(
    audios_srs=[(audio, 24000)],
    num_steps=10,  # More steps = higher quality
    temperature=1.0,
    guidance_scale=1.0
)

# Get reconstructed mel spectrogram
reconstructed_mel = results["reconstructed_mel"]
latent_code = results["context_vector"]
```

## Project Structure

```
MelCausalVAE/
├── configs/
│   ├── defaults/          # Default configurations
│   ├── settings/          # Experiment configs
│   └── deepspeed/         # DeepSpeed configs
├── data/
│   ├── audio_dataset.py   # Dataset base classes
│   └── libri_tts.py       # LibriTTS dataset loader
├── modules/
│   ├── Encoder.py         # Causal Convformer encoder
│   ├── Decoder.py         # DiT decoder components
│   ├── cfm.py             # Conditional flow matching
│   ├── VAE.py             # Main VAE model
│   ├── melspecEncoder.py  # Mel spectrogram preprocessing
│   ├── utils.py           # Attention and utility modules
│   └── flash_attn_encoder.py  # Flash attention implementation
├── train.py               # Training script
├── launch_training.sh     # Convenience launcher
├── TRAINING.md           # Detailed training guide
├── DEEPSPEED_QUICKSTART.md  # DeepSpeed reference
└── README.md             # This file
```
### Custom Compression Ratio

Adjust temporal compression for your use case:

```yaml
convformer:
  compress_factor_C: 16  # Higher = more compression, lower quality
```

Valid values: Powers of 2 (4, 8, 16, 32)

## Troubleshooting

Enable DeepSpeed:
   ```bash
   ./launch_training.sh configs/settings/setting1.yaml --deepspeed
   ```

### Slow Training

1. Enable flash attention (requires PyTorch 2.0+):
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. Use bf16 if supported:
   ```yaml
   training:
     bf16: true
     fp16: false
   ```

3. Increase dataloader workers:
   ```yaml
   training:
     dataloader_num_workers: 8
   ```

### Import Errors

Ensure you're running from the parent directory:
```bash
cd /home/ec2-user  # or wherever MelCausalVAE is located
python -m MelCausalVAE.train --exp-config ...
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{melcausalvae2024,
  title={MelCausalVAE: Causal Variational Autoencoder for Audio Mel Spectrograms},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/MelCausalVAE}
}
```

## License

[Specify your license here]

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://huggingface.co/transformers/)
- Flash Attention by [Dao-AILab](https://github.com/Dao-AILab/flash-attention)
- Vocos vocoder for audio generation
- LibriTTS-R dataset from [Parler-TTS](https://huggingface.co/parler-tts)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

---

**Status**: Active Development | **Version**: 0.1.0 | **Last Updated**: October 2025
