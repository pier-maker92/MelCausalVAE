import argparse
from pathlib import Path
import torch
import torchaudio
import json

from dicodec.modules.dicodec import Dicodec
from dicodec.modules.configs import DicodecConfig

from dicodec.modules.quantizer.std_vq import StandardVectorQuantizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", type=Path, default=Path("/workspace/MelCausalVAE/checkpoints/dual_branch_ae/model_epoch_1_std_vq.pt"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("/workspace/MelCausalVAE/checkpoints/18-denc128-novq"))
    parser.add_argument("--audio", type=Path, required=True, help="Input audio file")
    parser.add_argument("--output", type=Path, default=Path("reconstructed.wav"), help="Output audio file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--quantizer", type=str, default="std_vq")
    parser.add_argument("--input-latent", type=str, choices=["z_sem", "z"], default="z_sem", help="Input to the semantic encoder")
    parser.add_argument("--pros-encoder", action="store_true", help="Use the model with prosody encoder")
    args = parser.parse_args()
    
    if args.pros_encoder:
        from train_dual_branch_pros_ae import DualBranchQuantizedAE
    else:
        from train_dual_branch_ae import DualBranchQuantizedAE
        
    device = torch.device(args.device)
    
    from dicodec.modules.builder import load_pretrained_model
    
    print(f"Loading base model from {args.base_checkpoint}...")
    dicodec = load_pretrained_model(str(args.base_checkpoint))
    dicodec.to(device)
    dicodec.eval()
    
    print(f"Loading DualBranch AE from {args.ae_checkpoint}...")
    latent_dim = dicodec.encoder.config.latent_dim
    quant_dim = latent_dim
    
    # Load state dict FIRST to auto-infer hyperparameters
    state_dict = torch.load(args.ae_checkpoint, map_location=device)
    
    # Auto-infer quantizer type and codebook size
    dim = state_dict["sem_encoder.out_proj.bias"].shape[0]
    
    if "quantizer.quantizer_module.embedding.weight" in state_dict:
        args.quantizer = "std_vq"
        inferred_cb_size = state_dict["quantizer.quantizer_module.embedding.weight"].shape[0]
        args.codebook_size = inferred_cb_size
        print(f"Auto-inferred VQ codebook size: {inferred_cb_size}")
    elif "quantizer.quantizer_module.embedding" in state_dict:
        args.quantizer = "vq_ema"
        inferred_cb_size = state_dict["quantizer.quantizer_module.embedding"].shape[0]
        args.codebook_size = inferred_cb_size
        print(f"Auto-inferred VQ EMA codebook size: {inferred_cb_size}")
    elif dim == 4:
        print("Auto-inferred quantizer type: fsq (based on dim=4)")
        args.quantizer = "fsq"
        # We can't know the exact codebook size for FSQ without knowing the levels, 
        # but the default 1024 uses [8,8,4,4] which is 4 dimensions.
        args.codebook_size = 1024 
        print(f"Falling back to default FSQ codebook size: {args.codebook_size}")
    else:
        print(f"Auto-inferred quantizer type: bsq (based on dim={dim})")
        args.quantizer = "bsq"
        args.codebook_size = 2 ** dim
        print(f"Auto-inferred BSQ codebook size: {args.codebook_size}")
        
    sem_block_keys = set([k.split('.')[3] for k in state_dict.keys() if k.startswith("sem_encoder.resnet.blocks.")])
    if sem_block_keys:
        inferred_sem = len(sem_block_keys)
        print(f"Auto-inferred num_sem_blocks: {inferred_sem}")
        args.num_sem_blocks = inferred_sem
    else:
        args.num_sem_blocks = 2 # default fallback
        
    dec_block_keys = set([k.split('.')[3] for k in state_dict.keys() if k.startswith("decoder.resnet.blocks.")])
    if dec_block_keys:
        inferred_dec = len(dec_block_keys)
        print(f"Auto-inferred num_dec_blocks: {inferred_dec}")
        args.num_dec_blocks = inferred_dec
    else:
        args.num_dec_blocks = 2 # default fallback
    
    class QuantizerWrapper(torch.nn.Module):
        def __init__(self, quantizer_module, quantizer_type, args):
            super().__init__()
            self.quantizer_module = quantizer_module
            self.quantizer_type = quantizer_type
            self.args = args
            
        def forward(self, x, valid_mask=None):
            toks, codes = self.quantizer_module(x)
            if self.args.pros_encoder:
                from train_dual_branch_pros_ae import QuantizerOutput
            else:
                from train_dual_branch_ae import QuantizerOutput
            return QuantizerOutput(z_q=codes, loss=torch.tensor(0.0).to(x.device), indices=toks)

    if args.quantizer == "std_vq":
        from dicodec.modules.quantizer.std_vq import StandardVectorQuantizer
        base_quantizer = StandardVectorQuantizer(dim=quant_dim, codebook_size=args.codebook_size).to(device)
        quantizer_wrapper = QuantizerWrapper(base_quantizer, args.quantizer, args)
    elif args.quantizer == "vq_ema":
        from dicodec.modules.quantizer.vq_ema import EMAVectorQuantizer
        base_quantizer = EMAVectorQuantizer(dim=quant_dim, codebook_size=args.codebook_size).to(device)
        quantizer_wrapper = QuantizerWrapper(base_quantizer, args.quantizer, args)
    elif args.quantizer == "fsq":
        from dicodec.modules.quantizer.fsq import FiniteScalarQuantizer
        base_quantizer = FiniteScalarQuantizer(codebook_size=args.codebook_size).to(device)
        quantizer_wrapper = QuantizerWrapper(base_quantizer, args.quantizer, args)
        quant_dim = base_quantizer.dim
    elif args.quantizer == "bsq":
        from dicodec.modules.quantizer.bsq import BinarySphericalQuantizer
        base_quantizer = BinarySphericalQuantizer(codebook_size=args.codebook_size).to(device)
        quantizer_wrapper = QuantizerWrapper(base_quantizer, args.quantizer, args)
        quant_dim = base_quantizer.dim
    else:
        raise NotImplementedError(f"Quantizer {args.quantizer} not implemented in inference script yet.")

    model = DualBranchQuantizedAE(
        dim=latent_dim,
        quantizer=quantizer_wrapper,
        quant_dim=quant_dim,
        hidden_dim=args.hidden_dim,
        num_sem_blocks=args.num_sem_blocks,
        num_dec_blocks=args.num_dec_blocks,
    ).to(device)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded AE checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    model.eval()
    
    print(f"Processing audio {args.audio}...")
    wav, sr = torchaudio.load(args.audio)
    if wav.dim() > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    import torchaudio.transforms as T
    if sr != dicodec.config.sample_rate:
        wav = T.Resample(sr, dicodec.config.sample_rate)(wav)
        sr = dicodec.config.sample_rate
        
    wav = wav / (wav.abs().max() + 1e-8)
    wav = wav.to(device)
    audios_srs = [(wav.squeeze(0), sr)]
    
    with torch.inference_mode():
        # 1. Base Encoder Features via Dicodec
        enc_features, enc_padding_mask, dec_features, dec_padding_mask = dicodec.extract_features(audios_srs)
        encoder_output = dicodec.encode(enc_features, enc_padding_mask)
        z = encoder_output.z
        padding_mask = encoder_output.padding_mask
        
        # 2. Encode Attributes via Dicodec
        attrs = dicodec.encode_attributes(z, padding_mask=padding_mask)
        z_sem, z_pros, z_mean = attrs.z_sem, attrs.z_pros, attrs.z_mean
        
        # 3. Dual Branch AE - Full
        valid_mask = ~padding_mask if padding_mask is not None else None
        
        if args.pros_encoder:
            ae_out = model(z_sem=z_sem, z_pros=z_pros, z_mean=z_mean, valid_mask=valid_mask)
            z_variants = {
                "ae_full": ae_out.z_rec
            }
        else:
            if getattr(args, "input_latent", "z_sem") == "z":
                ae_out = model(z, valid_mask=valid_mask)
            else:
                ae_out = model(z_sem, valid_mask=valid_mask)
            z_sem_rec = ae_out.z_rec
            
            # To get the full z for the codec, we MUST add back the unquantized z_pros and z_mean
            z_rec_full = z_sem_rec + z_pros + z_mean
            
            # 3b. Only Quantized (Reconstructed semantics + mean, but NO prosody)
            z_rec_only_quantized = z_sem_rec + z_mean
            
            # 3c. Only Prosody (Original prosody + mean, NO semantics)
            z_rec_only_prosody = z_pros + z_mean
            
            z_variants = {
                "ae_full": z_rec_full,
                "only_quantized": z_rec_only_quantized,
                "only_prosody": z_rec_only_prosody
            }
        
    # 4. Decode
    # To guarantee 100% parity with inference.py, we will use its own load function and encode_decode
    import sys
    if "/workspace/MelCausalVAE" not in sys.path:
        sys.path.insert(0, "/workspace/MelCausalVAE")
    from inference import load_wav_mono_resampled
    
    with torch.inference_mode():
        wav_infer = load_wav_mono_resampled(str(args.audio), dicodec.config.sample_rate).to(device)
        audios_srs_infer = [(wav_infer, dicodec.config.sample_rate)]
        
        # Generiamo z_orig usando ESATTAMENTE encode_decode come fa inference.py
        out_orig = dicodec.encode_decode(
            audios_srs=audios_srs_infer,
            num_steps=16,
            temperature=0.3,
            guidance_scale=1.3
        )
        audio_orig = out_orig["audio_waveform"]
        torchaudio.save(str(args.output).replace(".wav", "_z_orig.wav"), audio_orig.cpu(), 24000)
        print(f"Saved z_orig reconstruction to {str(args.output).replace('.wav', '_z_orig.wav')}")
        
        speaker_embedding = dicodec.extract_speaker_embedding(audios_srs_infer)
        if speaker_embedding is not None:
            print(f"Extracted speaker embedding with shape: {speaker_embedding.shape}")
            
        for name, z_to_test in z_variants.items():
            reconstructed_mel, _ = dicodec.sample(
                z=z_to_test,
                num_steps=16,
                temperature=0.3,
                guidance_scale=1.3,
                padding_mask=padding_mask,
                speaker_embedding=speaker_embedding
            )
            audio = dicodec.vocoder.decode(reconstructed_mel.permute(0, 2, 1))
            audio = audio / (audio.abs().max() + 1e-8)
            out_path = str(args.output).replace(".wav", f"_{name}.wav")
            torchaudio.save(out_path, audio.cpu(), 24000)
            print(f"Saved {name} reconstruction to {out_path}")

if __name__ == "__main__":
    main()
