import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importiamo la parametrizzazione nativa di PyTorch per la Hard Orthogonality
from torch.nn.utils.parametrizations import orthogonal

from data.audio_dataset import TrainDatasetWrapper, TestDatasetWrapper, DataCollator
from modules.feature_extractor import WavLMFeatureExtractor
from modules.configs import WavLMConfig
from modules.builder import build_model


class OrthogonalProjectors(nn.Module):
    def __init__(
        self,
        wavlm_dim,
        components_cfg,
        mlp_layers=0,
        mlp_hidden_dim=1024,
        use_soft_ortho=False,
    ):
        """
        Args:
            wavlm_dim: Dimensione delle feature di WavLM Layer 6 (es. 1024)
            components_cfg: Dict con i nomi dei componenti e indici start/end
            mlp_layers: Numero di strati dell'MLP (se 0, l'MLP non viene usato)
            mlp_hidden_dim: Dimensione nascosta dell'MLP
        """
        super().__init__()
        self.D = wavlm_dim

        self.mlp = None
        if mlp_layers > 0:
            layers = []
            in_dim = self.D
            for _ in range(mlp_layers):
                layers.append(nn.Linear(in_dim, mlp_hidden_dim))
                layers.append(nn.GELU())
                in_dim = mlp_hidden_dim
            self.mlp = nn.Sequential(*layers)
            self.D = mlp_hidden_dim

        # Calcoliamo la dimensione totale necessaria (es. chunk semantico + prosodia + timbro)
        total_dim = 0
        self.local_slices = {}  # Mappa interna per affettare l'output della proiezione
        self.global_slices = {}  # Mappa dei target (indici reali sul latente Z del VAE)

        for comp_name, comp_info in components_cfg.items():
            start, end = comp_info["start"], comp_info["end"]
            dim = end - start

            self.local_slices[comp_name] = (total_dim, total_dim + dim)
            self.global_slices[comp_name] = (start, end)
            total_dim += dim

        # Creiamo un SINGOLO strato lineare con righe ortonormali.
        self.proj = nn.Linear(self.D, total_dim, bias=True)

        # IMPONIAMO LA HARD ORTHOGONALITY sulle righe della matrice dei pesi.
        # Questo garantisce ortogonalità inter-gruppo (e intra-gruppo come side-effect).
        if not use_soft_ortho:
            orthogonal(self.proj, name="weight")

        # Scaling diagonale apprendibile per ogni componente.
        # Le righe ortonormali hanno norma unitaria, quindi non possono scalare
        # liberamente l'output. Questi parametri compensano, permettendo al modello
        # di adattare la varianza per ogni dimensione del target Z.
        self.scales = nn.ParameterDict()
        for comp_name, (l_start, l_end) in self.local_slices.items():
            dim = l_end - l_start
            self.scales[comp_name] = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [..., T, D]
        orig_dtype = x.dtype
        x = x.to(self.proj.weight.dtype)

        if self.mlp is not None:
            x = self.mlp(x)

        # Unica proiezione globale (garantita ortogonale in ogni sua sottomatrice)
        out = self.proj(x)  # [..., T, total_dim]

        preds = {}
        for comp_name, (l_start, l_end) in self.local_slices.items():
            # Affettiamo l'output e applichiamo lo scaling diagonale
            preds[comp_name] = (out[..., l_start:l_end] * self.scales[comp_name]).to(
                orig_dtype
            )

        return preds


def mel_to_audio(vocoder, mel, device, vocoder_type="vocos"):
    """Converte mel spectrogram in waveform usando il vocoder."""
    if vocoder_type == "bigvgan":
        target_dtype = next(vocoder.parameters()).dtype
    else:
        target_dtype = next(vocoder.backbone.parameters()).dtype
    features = mel.permute(0, 2, 1).to(device=device, dtype=target_dtype)
    if vocoder_type == "bigvgan":
        waveform = vocoder(features)
    else:
        waveform = vocoder.decode(features)
    waveform = waveform.float().squeeze().detach().cpu()
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform.view(-1)


@torch.no_grad()
def decode_and_save(
    proj_model,
    vae_model,
    vocoder,
    vocoder_type,
    wavlm_feats,
    z_padding_mask,
    device,
    output_dir,
    epoch,
    sr=24000,
):
    """
    Assembla Z dalle proiezioni, decodifica attraverso il VAE decoder freezato,
    e salva gli audio risultanti.
    """
    proj_model.eval()
    latent_dim = vae_model.config.encoder_config.latent_dim
    B, T = wavlm_feats.shape[0], wavlm_feats.shape[1]
    B = min(B, 4)  # Limita il decoding a 4 sample
    wavlm_feats = wavlm_feats[:B]
    z_padding_mask = z_padding_mask[:B]

    # Forward attraverso i proiettori
    preds = proj_model(wavlm_feats)

    # Assembla il vettore Z completo, mediando le regioni sovrapposte
    assembled_z = torch.zeros(B, T, latent_dim, device=device, dtype=wavlm_feats.dtype)
    counts = torch.zeros(latent_dim, device=device)
    for comp_name, (start, end) in proj_model.global_slices.items():
        assembled_z[:, :, start:end] += preds[comp_name]
        counts[start:end] += 1
    counts = counts.clamp(min=1)
    assembled_z = assembled_z / counts
    assembled_z = assembled_z.to(dtype=vae_model.dtype)

    # Decodifica mel attraverso il VAE decoder (freezato)
    reconstructed_mel, _ = vae_model.sample(
        z=assembled_z,
        padding_mask=z_padding_mask,
        num_steps=4,
    )

    # Salva ogni sample come file audio
    step_dir = os.path.join(output_dir, f"step_{epoch+1}")
    os.makedirs(step_dir, exist_ok=True)
    C = vae_model.config.encoder_config.compress_factor_C

    def _save_audio(mel, mask, path):
        """Helper per salvare un singolo audio da mel."""
        mel_i = mel.clone()
        if mask is not None:
            valid_len = (~mask).sum().item()
            mel_valid_len = min(valid_len * C, mel_i.shape[1])
            mel_i = mel_i[:, :mel_valid_len, :]
        waveform = mel_to_audio(vocoder, mel_i, device, vocoder_type)
        torchaudio.save(path, waveform.unsqueeze(0).to(torch.float32), sample_rate=sr)

    # --- Decodifica completa (tutti i componenti) ---
    reconstructed_mel, _ = vae_model.sample(
        z=assembled_z,
        padding_mask=z_padding_mask,
        num_steps=4,
    )
    for i in range(B):
        _save_audio(
            reconstructed_mel[i : i + 1],
            z_padding_mask[i],
            os.path.join(step_dir, f"sample_{i}.wav"),
        )

    # --- Decodifica per singolo componente ---
    for comp_name, (start, end) in proj_model.global_slices.items():
        comp_dir = os.path.join(step_dir, comp_name)
        os.makedirs(comp_dir, exist_ok=True)

        # Z con solo questo componente attivo, il resto azzerato
        comp_z = torch.zeros_like(assembled_z)
        comp_z[:, :, start:end] = preds[comp_name].to(dtype=vae_model.dtype)

        comp_mel, _ = vae_model.sample(
            z=comp_z,
            padding_mask=z_padding_mask,
            num_steps=4,
        )
        for i in range(B):
            _save_audio(
                comp_mel[i : i + 1],
                z_padding_mask[i],
                os.path.join(comp_dir, f"sample_{i}.wav"),
            )

    print(
        f"  -> Salvati {B} audio (full + {len(proj_model.global_slices)} componenti) in {step_dir}"
    )
    proj_model.train()


def train_projectors(
    dataloader,
    wavlm_extractor,
    vae_model,
    vocoder,
    vocoder_type,
    device,
    cfg,
    target_dtype=torch.float32,
):
    """
    Script per addestrare i proiettori lineari estraendo dinamicamente le feature
    da un dataset reale e allineando temporalmente WavLM a MelCausalVAE.
    """
    model = None
    optimizer = None

    C = vae_model.config.encoder_config.compress_factor_C
    decode_every = cfg.get("decode_every", 0)
    output_dir = "proj_output_training"

    print("Inizio training dei proiettori lineari ortogonali (Hard Orthogonality)...")
    global_step = 0

    for epoch in range(cfg.epochs):
        if model is not None:
            model.train()

        total_l_rec = 0
        total_r2 = {}
        total_ortho = 0.0
        batches = 0

        # tqdm per visualizzare la progress bar su ogni epoca
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            # Recupero audio dal batch
            batch_size_actual = len(
                batch.get("ids", batch.get("id", batch.get("file", [])))
            )
            if batch_size_actual == 0:
                continue

            audios_srs = []
            for b_idx in range(batch_size_actual):
                audio_tensor = (
                    batch["output_audios_srs"][b_idx][0]
                    .squeeze()
                    .to(device=device, dtype=target_dtype)
                )
                sr = batch["output_audios_srs"][b_idx][1]
                audios_srs.append((audio_tensor, sr))

            if not audios_srs:
                continue

            # --- ESTRAZIONE FEATURE ---
            with torch.no_grad():
                # 1. Feature VAE (incluso WavLM interno già allineato temporalmente a T_mel)
                enc_features, enc_padding_mask, _, _, _ = vae_model.extract_features(
                    audios_srs
                )

                # 2. Encode per ottenere il latente Z
                encoder_output = vae_model.encode(enc_features, enc_padding_mask)
                z = encoder_output.z  # [B, T_z, C_channels]

                # 3. Downsample WavLM per i proiettori
                wavlm_feats = F.avg_pool1d(
                    enc_features.transpose(1, 2), kernel_size=C, stride=C
                )
                wavlm_feats = wavlm_feats.transpose(1, 2)  # [B, T_mel // C, D]

                # Gestione di lievi discrepanze temporali dovute agli arrotondamenti
                min_len = min(z.shape[1], wavlm_feats.shape[1])
                z = z[:, :min_len, :]
                wavlm_feats = wavlm_feats[:, :min_len, :]

                # Applica il padding mask
                z_padding_mask = encoder_output.padding_mask[:, :min_len]

            # --- FORWARD PASS PROIETTORI ---
            # Selezioniamo solo i frame validi (non paddati)
            valid_z = z[~z_padding_mask]  # [N_valid, vae_dim]
            valid_wavlm = wavlm_feats[~z_padding_mask]  # [N_valid, wavlm_dim]

            if valid_z.shape[0] == 0:
                continue

            # Lazy initialization del modello basata sulla vera dimensione D
            if model is None:
                actual_wavlm_dim = valid_wavlm.shape[-1]
                print(
                    f"Inizializzazione dinamica proiettori con WavLM dim = {actual_wavlm_dim}"
                )
                # PyTorch's orthogonal parametrization uses QR decomposition (orgqr)
                # which is not implemented for BFloat16 on CUDA. So we force float32.
                model = OrthogonalProjectors(
                    actual_wavlm_dim,
                    cfg.components,
                    mlp_layers=cfg.get("mlp_layers", 0),
                    mlp_hidden_dim=cfg.get("mlp_hidden_dim", 1024),
                    use_soft_ortho=cfg.get("use_soft_orthogonality", False),
                ).to(device=device, dtype=torch.float32)
                optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
                model.train()

            optimizer.zero_grad()

            # Slicing dei target Z usando la mappa globale (dove Z è già tagliato correttamente)
            z_targets = {}
            for comp_name, (start, end) in model.global_slices.items():
                z_targets[comp_name] = valid_z[:, start:end]

            # Predizioni
            preds = model(valid_wavlm)

            # --- LOSS CALCULATION ---
            # Loss di Ricostruzione (MSE) + calcolo R² per componente
            L_rec = 0
            batch_r2 = {}
            for comp_name in preds.keys():
                mse_loss = F.mse_loss(preds[comp_name], z_targets[comp_name])
                L_rec += mse_loss
                target_var = torch.var(z_targets[comp_name])
                r2_score = 1.0 - (mse_loss / target_var)
                batch_r2[comp_name] = r2_score.item()

            total_loss = L_rec
            batch_ortho_val = 0.0
            if cfg.get("use_soft_orthogonality", False):
                W = model.proj.weight
                M = torch.mm(W, W.t())

                # Creiamo una maschera per ignorare l'ortogonalità intra-gruppo
                if not hasattr(model, "inter_group_mask"):
                    mask = torch.ones_like(M)
                    for start, end in model.local_slices.values():
                        mask[start:end, start:end] = (
                            0.0  # Ignora il blocco di questo componente
                        )
                    model.inter_group_mask = mask

                # Penalizziamo solo i dot product tra righe di componenti diversi
                inter_group_M = M * model.inter_group_mask
                ortho_penalty = torch.sum(inter_group_M**2) / max(
                    1.0, model.inter_group_mask.sum().item()
                )
                batch_ortho_val = ortho_penalty.item()

                lambda_ortho = cfg.get("soft_ortho_lambda")
                total_loss += lambda_ortho * ortho_penalty

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            # Accumulo metriche
            total_l_rec += L_rec.item()
            total_ortho += batch_ortho_val
            for comp_name, r2_val in batch_r2.items():
                total_r2[comp_name] = total_r2.get(comp_name, 0.0) + r2_val
            batches += 1
            global_step += 1

            postfix = {"MSE": f"{L_rec.item():.4f}"}
            if cfg.get("use_soft_orthogonality", False):
                postfix["Orth"] = f"{batch_ortho_val:.4f}"
            for comp_name, r2_val in batch_r2.items():
                postfix[f"R²_{comp_name}"] = f"{r2_val:.3f}"
            pbar.set_postfix(postfix)

            # --- DECODING PERIODICO (ogni N steps) ---
            if decode_every > 0 and global_step > 0 and global_step % decode_every == 0:
                decode_and_save(
                    proj_model=model,
                    vae_model=vae_model,
                    vocoder=vocoder,
                    vocoder_type=vocoder_type,
                    wavlm_feats=wavlm_feats,
                    z_padding_mask=z_padding_mask,
                    device=device,
                    output_dir=output_dir,
                    epoch=global_step,
                )

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            avg_rec = total_l_rec / max(1, batches)
            avg_ortho = total_ortho / max(1, batches)
            r2_str = " | ".join(
                f"R²_{k}: {v / max(1, batches):.4f}" for k, v in total_r2.items()
            )
            ortho_str = (
                f" | Avg Orth: {avg_ortho:.4f}"
                if cfg.get("use_soft_orthogonality", False)
                else ""
            )
            print(
                f"Epoch {epoch+1}/{cfg.epochs} | Avg MSE: {avg_rec:.4f}{ortho_str} | {r2_str}"
            )

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

    use_bfloat16 = cfg.get("use_bfloat16", False)
    if device.type == "cuda" and use_bfloat16:
        target_dtype = torch.bfloat16
        print("Using bfloat16")
    else:
        target_dtype = torch.float32

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
    if target_dtype != torch.float32:
        vae_model.to(dtype=target_dtype)
    vae_model.eval()

    print("Impostazione WavLMFeatureExtractor dal VAE...")
    if getattr(vae_model, "wavlm_extractor", None) is not None:
        wavlm_extractor = vae_model.wavlm_extractor
        print(
            f"Usando WavLM config dal checkpoint: {vae_model.config.wavlm_config.pretrained_model_name}"
        )
    else:
        print("Il VAE non usa WavLM come encoder_input. Fallback.")
        wavlm_config = WavLMConfig()
        wavlm_extractor = WavLMFeatureExtractor(config=wavlm_config).to(device)
    # WavLMFeatureExtractor forces inputs to .float() internally.
    # To avoid dtype mismatch, its weights must remain in float32.
    wavlm_extractor.to(dtype=torch.float32)
    wavlm_extractor.eval()

    # 3. Caricamento Vocoder per il decoding periodico
    mel_cfg = vae_model.config.mel_spectrogram_config
    if getattr(mel_cfg, "use_bigvgan_mel", False):
        import sys
        from pathlib import Path

        base_dir = Path(__file__).parent.absolute()
        bigvgan_path = str(base_dir / "bigvgan" / "bigvgan_v2_24khz_100band_256x")
        if bigvgan_path not in sys.path:
            sys.path.append(bigvgan_path)
        import bigvgan

        vocoder = bigvgan.BigVGAN.from_pretrained(bigvgan_path, use_cuda_kernel=False)
        vocoder_type = "bigvgan"
    else:
        from vocos import Vocos

        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        vocoder_type = "vocos"
    vocoder.to(device)
    vocoder.eval()
    print(f"Vocoder caricato: {vocoder_type}")

    # 4. Training
    trained_model = train_projectors(
        dataloader=dataloader,
        wavlm_extractor=wavlm_extractor,
        vae_model=vae_model,
        vocoder=vocoder,
        vocoder_type=vocoder_type,
        device=device,
        cfg=cfg,
        target_dtype=target_dtype,
    )
    print("Training completato.")

    # Salvataggio pesi dei proiettori
    save_path = "orthogonal_projectors.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Modello salvato in {save_path}")


if __name__ == "__main__":
    main()
