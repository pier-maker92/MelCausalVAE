# Training Guide

## Quick Start

### 1. Configure Accelerate (First Time Only)

Run the interactive configuration:
```bash
accelerate config
```

Or use the provided config:
```bash
accelerate config --config_file accelerate_config.yaml
```

### 2. Launch Training

#### Basic Usage (uses default config)
```bash
./launch_training.sh
```

#### With Custom Config
```bash
./launch_training.sh configs/settings/setting1.yaml
```

#### Single GPU Training
```bash
./launch_training.sh configs/settings/setting1.yaml --num_processes 1
```

#### Multi-GPU Training (DDP)
```bash
./launch_training.sh configs/settings/setting1.yaml --multi_gpu --num_processes 4
```

#### DeepSpeed ZeRO-3 Training (Recommended for Large Models)
```bash
./launch_training.sh configs/settings/setting1.yaml --deepspeed
```

#### DeepSpeed with Custom Config
```bash
./launch_training.sh configs/settings/setting1.yaml --deepspeed --deepspeed-config configs/deepspeed/zero2.json
```

#### With Custom Accelerate Config
```bash
./launch_training.sh configs/settings/setting1.yaml --config_file accelerate_config.yaml
```

#### Get Help
```bash
./launch_training.sh --help
```

## DeepSpeed Training

### How DeepSpeed Integration Works

This project integrates DeepSpeed seamlessly with Hugging Face Transformers:
1. Pass `--deepspeed` to `launch_training.sh`
2. The script automatically configures DeepSpeed in `TrainingArguments`
3. Transformers Trainer detects DeepSpeed and initializes it via Accelerate
4. Training runs with DeepSpeed optimizations automatically applied

**No manual Accelerate configuration needed!** Just use the `--deepspeed` flag.

### What is DeepSpeed?

DeepSpeed is a deep learning optimization library that enables:
- **Memory efficiency**: Train larger models with ZeRO (Zero Redundancy Optimizer)
- **Speed**: Faster training through optimized kernels and memory management
- **Scalability**: Better multi-GPU and multi-node training

### When to Use DeepSpeed?

- **Large Models**: Model doesn't fit in GPU memory
- **Large Batch Sizes**: Need to train with bigger batches
- **Multi-GPU**: Want optimal memory distribution across GPUs
- **Memory Constrained**: Running on GPUs with limited VRAM

### ZeRO Stages

The provided config uses **ZeRO Stage 3** (`configs/deepspeed/zero3.json`):
- **Stage 1**: Optimizer state partitioning
- **Stage 2**: Gradient partitioning
- **Stage 3**: Parameter partitioning (highest memory savings)

### DeepSpeed Configuration

Available configs:
- `configs/deepspeed/zero2.json` - ZeRO Stage 2 (good balance, no CPU offload)
- `configs/deepspeed/zero3.json` - ZeRO Stage 3 (default, best GPU memory savings)
- `configs/deepspeed/zero3_offload.json` - ZeRO Stage 3 with CPU offload (extreme memory savings, slower)

**Choosing a config:**
- Use `zero2.json` for fast multi-GPU training with moderate memory needs
- Use `zero3.json` (default) for large models that barely fit in GPU memory
- Use `zero3_offload.json` for extremely large models, offloads optimizer and params to CPU

You can create custom configs based on your needs.

## Monitoring Training

If you've enabled wandb in your config:
```yaml
training:
  report_to: wandb
  wandb_project: your-project-name
  wandb_run_name: experiment-1
```

## Output Location

Training outputs (checkpoints, logs) will be saved to the directory specified in your config:
```yaml
training:
  output_dir: outputs/setting1  # Your checkpoints go here
```

## Troubleshooting

### Import Errors
Make sure you're running from the parent directory of `MelCausalVAE`:
```bash
cd /home/ec2-user
./MelCausalVAE/launch_training.sh
```

### CUDA Out of Memory

**Option 1: Reduce batch size**
```yaml
training:
  batch_size: 32  # Reduce this value
```

**Option 2: Use DeepSpeed**
```bash
./launch_training.sh configs/settings/setting1.yaml --deepspeed
```

**Option 3: Increase gradient accumulation**
```yaml
training:
  gradient_accumulation_steps: 4  # Effective batch size = batch_size * gradient_accumulation_steps
```

### Check Accelerate Configuration
```bash
accelerate env
```

### DeepSpeed Issues

If you encounter DeepSpeed-specific errors:

1. **Check DeepSpeed installation**:
```bash
python -c "import deepspeed; print(deepspeed.__version__)"
```

2. **Verify config file syntax**:
```bash
cat configs/deepspeed/zero3.json
```

3. **Enable DeepSpeed logging**:
Add to your launch command: `--deepspeed_multinode_launcher standard`

