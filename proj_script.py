import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.audio_dataset import TrainDatasetWrapper, TestDatasetWrapper, DataCollator
from modules.feature_extractor import WavLMFeatureExtractor
from modules.configs import WavLMConfig
from modules.builder import build_model


class OrthogonalProjectors(nn.Module):
    def __init__(self, wavlm_dim, components_cfg):
        """
        Args:
            wavlm_dim: Dimensione delle feature di WavLM Layer 6 (es. 1024)
            components_cfg: Dict con i nomi dei componenti e indici start/end
        """
        super().__init__()
        self.D = wavlm_dim
        
        self.projectors = nn.ModuleDict()
        self.slices = {}
        
        for comp_name, comp_info in components_cfg.items():
            start, end = comp_info["start"], comp_info["end"]
            dim = end - start
            self.projectors[comp_name] = nn.Linear(self.D, dim)
            self.slices[comp_name] = (start, end)

    def forward(self, x):
        # x: [..., T, D]
        preds = {}
        for comp_name, proj in self.projectors.items():
            preds[comp_name] = proj(x)
        return preds


def orthogonality_loss(W1, W2):
    """
    Calcola la penalità di ortogonalità tra due matrici di pesi.
    """
    W1_norm = F.normalize(W1, p=2, dim=1) # Shape: [out_features1, D]
    W2_norm = F.normalize(W2, p=2, dim=1) # Shape: [out_features2, D]
    
    dot_product = torch.matmul(W1_norm, W2_norm.T) # Shape: [out_features1, out_features2]
    return torch.sum(dot_product ** 2)


def train_projectors(dataloader, wavlm_extractor, vae_model, device, cfg):
    """
    Script per addestrare i proiettori lineari estraendo dinamicamente le feature
    da un dataset reale e allineando temporalmente WavLM a MelCausalVAE.
    """
    model = None
    optimizer = None
    
    C = vae_model.config.encoder_config.compress_factor_C
    
    print("Inizio training dei proiettori lineari...")
    
    for epoch in range(cfg.epochs):
        if model is not None:
            model.train()
        total_l_rec = 0
        total_l_ortho = 0
        total_loss = 0
        batches = 0
        
        # tqdm per visualizzare la progress bar su ogni epoca
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            # Recupero audio dal batch
            batch_size_actual = len(batch.get("ids", batch.get("id", batch.get("file", []))))
            if batch_size_actual == 0:
                continue
                
            audios_srs = []
            for b_idx in range(batch_size_actual):
                audio_tensor = batch["output_audios_srs"][b_idx][0].squeeze().to(device)
                sr = batch["output_audios_srs"][b_idx][1]
                audios_srs.append((audio_tensor, sr))
                
            if not audios_srs:
                continue

            # --- ESTRAZIONE FEATURE ---
            with torch.no_grad():
                # 1. Feature Mel per ottenere T_mel (molto veloce)
                mel_output = vae_model.feature_extractor(audios_srs)
                T_mel = mel_output.audio_features.shape[1]
                
                # 2. Estrai WavLM (una sola volta!)
                wavlm_output = wavlm_extractor(audios_srs)
                wavlm_feats_raw = wavlm_output.audio_features # [B, T_w, D]
                
                # 3. Allineamento temporale WavLM -> T_mel
                # a) Causal upsample x2 (repeat)
                wavlm_feats_aligned = wavlm_feats_raw.repeat_interleave(2, dim=1) # [B, 2*T_w, D]
                # b) Interpolate to match Mel spectrogram length
                wavlm_feats_aligned = F.interpolate(
                    wavlm_feats_aligned.float().transpose(1, 2),
                    size=T_mel,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2).to(wavlm_feats_raw.dtype) # [B, T_mel, D]
                
                enc_padding_mask = F.interpolate(
                    wavlm_output.padding_mask.float().unsqueeze(1),
                    size=T_mel,
                    mode="nearest",
                ).squeeze(1).bool()
                
                # 4. Encode per ottenere il latente Z
                encoder_output = vae_model.encode(wavlm_feats_aligned, enc_padding_mask)
                z = encoder_output.z # [B, T_z, C]
                
                # 5. Downsample WavLM per i proiettori (Avg pooling by C)
                wavlm_feats = F.avg_pool1d(wavlm_feats_aligned.transpose(1, 2), kernel_size=C, stride=C) # [B, D, T_mel // C]
                wavlm_feats = wavlm_feats.transpose(1, 2) # [B, T_mel // C, D]
                
                # Gestione di lievi discrepanze temporali dovute agli arrotondamenti
                min_len = min(z.shape[1], wavlm_feats.shape[1])
                z = z[:, :min_len, :]
                wavlm_feats = wavlm_feats[:, :min_len, :]
                
                # Applica il padding mask
                # Adattiamo il padding mask all'encoder (che è compresso di C)
                z_padding_mask = encoder_output.padding_mask[:, :min_len]

            # --- FORWARD PASS PROIETTORI ---
            # Selezioniamo solo i frame validi (non paddati)
            valid_z = z[~z_padding_mask] # [N_valid, vae_dim]
            valid_wavlm = wavlm_feats[~z_padding_mask] # [N_valid, wavlm_dim]
            
            if valid_z.shape[0] == 0:
                continue
                
            # Lazy initialization del modello basata sulla vera dimensione D
            if model is None:
                actual_wavlm_dim = valid_wavlm.shape[-1]
                print(f"Inizializzazione dinamica proiettori con WavLM dim = {actual_wavlm_dim}")
                model = OrthogonalProjectors(actual_wavlm_dim, cfg.components).to(device)
                optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
                model.train()

            optimizer.zero_grad()

            # Slicing dei target Z
            z_targets = {}
            for comp_name, (start, end) in model.slices.items():
                z_targets[comp_name] = valid_z[:, start:end]
            
            # Predizioni
            preds = model(valid_wavlm)
            
            # --- LOSS CALCULATION ---
            # 3A. Loss di Ricostruzione (MSE)
            L_rec = 0
            for comp_name in preds.keys():
                L_rec += F.mse_loss(preds[comp_name], z_targets[comp_name])
            
            # 3B. Loss di Ortogonalità (Soft Constraint)
            L_ortho = 0
            comp_names = list(model.projectors.keys())
            for i in range(len(comp_names)):
                for j in range(i + 1, len(comp_names)):
                    W1 = model.projectors[comp_names[i]].weight
                    W2 = model.projectors[comp_names[j]].weight
                    L_ortho += orthogonality_loss(W1, W2)
            
            # Loss Totale
            L_total = L_rec + cfg.lambda_ortho * L_ortho
            
            # Backpropagation
            L_total.backward()
            optimizer.step()
            
            # Accumulo metriche
            total_l_rec += L_rec.item()
            total_l_ortho += L_ortho.item()
            total_loss += L_total.item()
            batches += 1
            
            pbar.set_postfix({"Loss": L_total.item(), "L_rec": L_rec.item(), "L_ortho": L_ortho.item()})
            
        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            avg_loss = total_loss / max(1, batches)
            avg_rec = total_l_rec / max(1, batches)
            avg_ortho = total_l_ortho / max(1, batches)
            print(f"Epoch {epoch+1}/{cfg.epochs} | Avg L_total: {avg_loss:.4f} | Avg L_rec: {avg_rec:.4f} | Avg L_ortho: {avg_ortho:.4f}")
            
    return model


@hydra.main(version_base=None, config_path="configs", config_name="proj_script")
def main(cfg: DictConfig):
    print("=========================================")
    print("Running with config:")
    print(OmegaConf.to_yaml(cfg))
    print("=========================================")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Caricamento Dataset
    print(f"Loading dataset {cfg.dataset_name}...")
    if cfg.dataset_name == "mls":
        from data.mls import MLSDataset
        base_dataset = MLSDataset()
    elif cfg.dataset_name == "libritts":
        from data.libri_tts import LibriTTS
        base_dataset = LibriTTS()
    elif cfg.dataset_name in ["librispeech_aligned", "librispeech-aligned"]:
        from data.librispeech_align import LibriSpeechAlignDataset
        base_dataset = LibriSpeechAlignDataset(debug=False)
    else:
        from datasets import load_dataset
        print(f"Fallback to HF dataset {cfg.dataset_name}...")
        base_dataset = load_dataset(cfg.dataset_name)
    
    dataset = TrainDatasetWrapper(base_dataset, "train")
    collator = DataCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # 2. Caricamento Modelli
    print(f"Loading MelCausalVAE from {cfg.checkpoint_dir}...")
    config_path = os.path.join(cfg.checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    vae_model = build_model(cfg_dict)
    
    checkpoint_path = os.path.join(cfg.checkpoint_dir, "model.safetensors")
    vae_model.from_pretrained(checkpoint_path)
    vae_model.to(device)
    vae_model.eval()
    
    print("Loading WavLMFeatureExtractor...")
    if getattr(vae_model.config, "wavlm_config", None) is not None:
        print(f"Usando WavLM config dal checkpoint: {vae_model.config.wavlm_config.pretrained_model_name}")
        wavlm_extractor = WavLMFeatureExtractor(config=vae_model.config.wavlm_config).to(device)
    else:
        print("Usando WavLM config di default")
        wavlm_config = WavLMConfig()
        wavlm_extractor = WavLMFeatureExtractor(config=wavlm_config).to(device)
    wavlm_extractor.eval()
    
    # 3. Training
    trained_model = train_projectors(
        dataloader=dataloader,
        wavlm_extractor=wavlm_extractor,
        vae_model=vae_model,
        device=device,
        cfg=cfg
    )
    print("Training completato.")
    
    # Salvataggio pesi dei proiettori
    save_path = "orthogonal_projectors.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Modello salvato in {save_path}")


if __name__ == "__main__":
    main()
