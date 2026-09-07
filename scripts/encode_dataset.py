import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# Aggiunge la root directory al path per poter importare train
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import get_dataset
from dicodec.modules.builder import build_model
from dicodec.data.audio_dataset import DataCollator

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    training_cfg = cfg_dict["training"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name, train_dataset, test_dataset = get_dataset(training_cfg)
    
    model = build_model(cfg_dict)
    from_pretrained = training_cfg.get("from_pretrained")
    if from_pretrained:
        model.from_pretrained(from_pretrained)
    
    model.to(device)
    model.eval()

    collator = DataCollator()
    dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=training_cfg.get("per_device_train_batch_size", 4), 
        collate_fn=collator,
        num_workers=4
    )

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoded_dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    shard_size_limit = 512 * 1024 * 1024 # 512 MB
    
    # Definizione dello schema pyarrow per il salvataggio in parquet
    schema = pa.schema([
        ('id', pa.string()),
        ('z', pa.list_(pa.list_(pa.float32()))),
        ('z_sem', pa.list_(pa.list_(pa.float32()))),
        ('z_pros', pa.list_(pa.list_(pa.float32()))),
        ('z_mean', pa.list_(pa.list_(pa.float32()))),
        ('padding_mask', pa.list_(pa.bool_()))
    ])
    
    shard_idx = 0
    writer = None
    current_size = 0
    
    def get_writer(idx):
        return pq.ParquetWriter(os.path.join(output_dir, f"shard_{idx:04d}.parquet"), schema)

    writer = get_writer(shard_idx)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding Dataset"):
            audios_srs = batch.get("audio_input_srs")
            if audios_srs is None:
                continue
                
            audio_16khz = batch.get("16k_audio")
            
            # Estrazione feature
            enc_features, enc_padding_mask, _, _ = model.extract_features(
                audios_srs, 
                target_audios_srs=audios_srs,
                audio_16khz=audio_16khz
            )
            
            # Solo encode con compute_attributes=True, come richiesto
            encoder_output = model.encode(
                enc_features, 
                enc_padding_mask, 
                compute_attributes=True
            )
            
            # Prepariamo i dati per pyarrow
            z = encoder_output.z.cpu().numpy()
            padding_mask = encoder_output.padding_mask.cpu().numpy()
            
            z_sem = encoder_output.attributes.z_sem.cpu().numpy()
            z_pros = encoder_output.attributes.z_pros.cpu().numpy()
            z_mean = encoder_output.attributes.z_mean.cpu().numpy()
            
            ids = batch.get("ids")
            if ids is None:
                ids = [f"sample_{shard_idx}_{i}" for i in range(len(z))]
            else:
                ids = [str(x) for x in ids]

            batch_data = []
            for i in range(len(z)):
                batch_data.append({
                    'id': ids[i],
                    'z': z[i].tolist(),
                    'z_sem': z_sem[i].tolist(),
                    'z_pros': z_pros[i].tolist(),
                    'z_mean': z_mean[i].tolist(),
                    'padding_mask': padding_mask[i].tolist()
                })
            
            table = pa.Table.from_pylist(batch_data, schema=schema)
            writer.write_table(table)
            
            # Accumula la dimensione per rispettare il limite di shard (512 MB)
            current_size += table.nbytes
            
            if current_size >= shard_size_limit:
                writer.close()
                shard_idx += 1
                writer = get_writer(shard_idx)
                current_size = 0
                
    if writer:
        writer.close()

if __name__ == "__main__":
    main()
