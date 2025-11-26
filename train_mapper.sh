#!/bin/bash

# Launch semantic mapper training using Accelerate with optional DeepSpeed
# Usage: ./scripts/train_mapper.sh <vae_checkpoint> [config_path] [--deepspeed] [--deepspeed-config deepspeed_config_path] [additional_args]

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to parent directory (where MelCausalVAE module is)
cd "$(dirname "$SCRIPT_DIR")"

# Default values
VAE_CHECKPOINT=""
CONFIG_PATH="MelCausalVAE/configs/settings/semantic_mapper.yaml"
USE_DEEPSPEED=false
DEEPSPEED_CONFIG="MelCausalVAE/configs/deepspeed/zero3.json"
ADDITIONAL_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --deepspeed)
            USE_DEEPSPEED=true
            shift
            ;;
        --deepspeed-config)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 <vae_checkpoint> [config_path] [options]"
            echo ""
            echo "Arguments:"
            echo "  vae_checkpoint               Path to pretrained VAE checkpoint (required)"
            echo "  config_path                  Path to mapper config YAML (default: configs/settings/semantic_mapper.yaml)"
            echo ""
            echo "Options:"
            echo "  --deepspeed                  Enable DeepSpeed training"
            echo "  --deepspeed-config PATH      Specify DeepSpeed config file (default: configs/deepspeed/zero3.json)"
            echo "  --multi_gpu                  Use multiple GPUs with DDP"
            echo "  --num_processes N            Number of processes (GPUs) to use"
            echo "  --config_file PATH           Specify Accelerate config file"
            echo ""
            echo "Available DeepSpeed configs:"
            echo "  - configs/deepspeed/zero2.json          (ZeRO-2: balanced speed/memory)"
            echo "  - configs/deepspeed/zero3.json          (ZeRO-3: best GPU memory savings)"
            echo "  - configs/deepspeed/zero3_offload.json  (ZeRO-3 + CPU offload: extreme savings)"
            echo ""
            echo "Examples:"
            echo "  # Basic single GPU training"
            echo "  $0 checkpoints/vae/model.safetensors"
            echo ""
            echo "  # Multi-GPU training with DDP"
            echo "  $0 checkpoints/vae/model.safetensors --multi_gpu --num_processes 4"
            echo ""
            echo "  # DeepSpeed ZeRO-3 training (default)"
            echo "  $0 checkpoints/vae/model.safetensors --deepspeed"
            echo ""
            echo "  # DeepSpeed ZeRO-2 (faster, less memory savings)"
            echo "  $0 checkpoints/vae/model.safetensors --deepspeed --deepspeed-config configs/deepspeed/zero2.json"
            echo ""
            echo "  # Custom config"
            echo "  $0 checkpoints/vae/model.safetensors configs/settings/my_mapper.yaml"
            exit 0
            ;;
        *)
            if [ -z "$VAE_CHECKPOINT" ] && [[ ! "$1" =~ ^-- ]]; then
                VAE_CHECKPOINT="$1"
            elif [ -n "$VAE_CHECKPOINT" ] && [ "$CONFIG_PATH" = "MelCausalVAE/configs/settings/semantic_mapper.yaml" ] && [[ ! "$1" =~ ^-- ]]; then
                CONFIG_PATH="$1"
            else
                ADDITIONAL_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Check if VAE checkpoint is provided
if [ -z "$VAE_CHECKPOINT" ]; then
    echo "Error: VAE checkpoint path is required!"
    echo "Run '$0 --help' for usage information"
    exit 1
fi

# Check if VAE checkpoint exists
if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "Error: VAE checkpoint file '$VAE_CHECKPOINT' not found!"
    exit 1
fi

# If config path doesn't start with MelCausalVAE/, prepend it
if [[ ! "$CONFIG_PATH" =~ ^MelCausalVAE/ ]] && [[ -f "MelCausalVAE/$CONFIG_PATH" ]]; then
    CONFIG_PATH="MelCausalVAE/$CONFIG_PATH"
fi

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file '$CONFIG_PATH' not found!"
    echo "Run '$0 --help' for usage information"
    exit 1
fi

# Prepare training script arguments
TRAIN_ARGS=("--exp-config" "$CONFIG_PATH" "--vae-checkpoint" "$VAE_CHECKPOINT")

if [ "$USE_DEEPSPEED" = true ]; then
    # If deepspeed config doesn't start with MelCausalVAE/, prepend it
    if [[ ! "$DEEPSPEED_CONFIG" =~ ^MelCausalVAE/ ]] && [[ -f "MelCausalVAE/$DEEPSPEED_CONFIG" ]]; then
        DEEPSPEED_CONFIG="MelCausalVAE/$DEEPSPEED_CONFIG"
    fi
    
    if [ ! -f "$DEEPSPEED_CONFIG" ]; then
        echo "Error: DeepSpeed config file '$DEEPSPEED_CONFIG' not found!"
        exit 1
    fi
    
    TRAIN_ARGS+=("--deepspeed" "$DEEPSPEED_CONFIG")
fi

echo "================================================"
echo "Launching Z2Y Mapper Training with Accelerate"
echo "Working directory: $(pwd)"
echo "VAE Checkpoint: $VAE_CHECKPOINT"
echo "Config: $CONFIG_PATH"
if [ "$USE_DEEPSPEED" = true ]; then
    echo "DeepSpeed: Enabled"
    echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
fi
echo "================================================"

# Launch training with accelerate
# DeepSpeed config is passed to the training script, not to accelerate launch
# Accelerate will automatically detect and use DeepSpeed from TrainingArguments
accelerate launch \
    "${ADDITIONAL_ARGS[@]}" \
    -m MelCausalVAE.train_semantic_mapper \
    "${TRAIN_ARGS[@]}"

echo "================================================"
echo "Training completed!"
echo "================================================"
