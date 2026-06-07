#!/usr/bin/env python3
import sys
import os

# Temporarily disable offline mode if it was set by sourcing init.sh
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

def main():
    print("Initializing Hugging Face WavLM downloader...")
    
    # Target model name
    model_name = "microsoft/wavlm-large"
    
    # Determine the cache directory to display
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_dir = os.path.join(hf_home, "hub")
    else:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    try:
        from transformers import WavLMModel
        print(f"Imported transformers. Downloading '{model_name}' model weights and configuration...")
        
        # This will download the model config and weights and store them in the HF cache directory
        model = WavLMModel.from_pretrained(model_name)
        
        print("\n" + "="*50)
        print("Success! WavLM model has been successfully downloaded and cached.")
        print(f"Cache directory: {cache_dir}")
        print("="*50)
        
    except ImportError:
        print("Error: 'transformers' library not found in the current Python environment.")
        print("Please run this script using the correct environment python binary, e.g.:")
        print("  ~/envs/vae/bin/python download_wavlm.py")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during model download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
