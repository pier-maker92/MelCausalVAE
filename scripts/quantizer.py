import argparse
import os
import sys
import yaml
import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.configs import VQConfig
from modules.quantizer.vq import VectorQuantizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vector Quantizer on extracted z latents.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing sample subdirectories with z.npy files")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file for VQConfig")
    parser.add_argument("--output-dir", type=str, default="quantizer_output", help="Directory to save the trained quantizer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()

class LatentDataset(Dataset):
    def __init__(self, data_dir):
        import numpy as np
        self.np = np
        self.files = glob.glob(os.path.join(data_dir, "*", "z.npy"))
        print(f"Found {len(self.files)} z.npy files in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        z = self.np.load(self.files[idx])
        return torch.from_numpy(z).float()

def collate_fn(batch):
    # batch is a list of [T, D] tensors
    # pad them to [B, max_T, D]
    lengths = [x.size(0) for x in batch]
    padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
    
    # create padding_mask [B, max_T], True where padded
    padding_mask = torch.zeros(padded.shape[:2], dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < padded.size(1):
            padding_mask[i, l:] = True
            
    return padded, padding_mask

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        
    vq_dict = cfg_dict.get("vq_config", cfg_dict) # fallback to root if vq_config key not present
    if "dim_to_quantize" in vq_dict and "vq_dim" not in vq_dict:
        vq_dict["vq_dim"] = vq_dict.pop("dim_to_quantize")
    
    # Remove unsupported keys for VQConfig dataclass if any
    import inspect
    allowed = set(inspect.signature(VQConfig).parameters.keys())
    filtered_vq_dict = {k: v for k, v in vq_dict.items() if k in allowed}
    
    vq_config = VQConfig(**filtered_vq_dict)
    print(f"Instantiated VQConfig: {vq_config}")
    
    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(filtered_vq_dict, f, indent=4)
        
    device = torch.device(args.device)
    
    # Model
    model = VectorQuantizer(config=vq_config, dim=vq_config.vq_dim).to(device)
    
    # Data
    dataset = LatentDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (z, padding_mask) in enumerate(pbar):
            z = z.to(device)
            padding_mask = padding_mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            out = model(z, padding_mask)
            
            # Optimization
            if out.loss is not None and out.loss.requires_grad:
                out.loss.backward()
                optimizer.step()
                loss_val = out.loss.item()
            else:
                loss_val = 0.0 # e.g. for EMA VQ which might not have a trainable loss term
                
            epoch_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "ppl": f"{out.stats.perplexity:.2f}"})
            
        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss / len(dataloader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"quantizer_ep{epoch+1}.pt"))
        
    torch.save(model.state_dict(), os.path.join(args.output_dir, "quantizer_final.pt"))
    print("Training finished!")

if __name__ == "__main__":
    main()
