# Speech language modeling quantizer

Learns discrete DiCodec tokens from frozen latent sequences `z: [B, T, D]`.
All implementation, configuration and output dataclasses live in this package;
`scripts/train_sm_quantizer.py` is only an entry point.

## Objective

1. A framewise MLP maps each `z[t]` to quantizer space.
2. The existing `EMAVectorQuantizer` assigns nearest centers and updates them online.
   Padding is excluded; a straight-through estimator carries task gradients to the encoder.
3. A decoder-only causal transformer consumes quantized vectors (the embeddings of
   the discrete IDs). Context `h[t]` can see only tokens `0..t`.
4. A small framewise adaptive MLP predicts `z[t+1]` using conditional flow matching,
   following the formulation in `hybrid_tts/modules/diffusion_head/cfm.py` and `mlp.py`.
   It has no runtime dependency on that repository. Each pair draws independent noise
   and time: `x(t) = (1 - (1-sigma_min)*t)*noise + t*z[t+1]`,
   with velocity target `z[t+1] - (1-sigma_min)*noise`.
5. A separate framewise MLP reconstructs `z[t]` from its quantized vector.
   L1 and L2 compare this reconstruction with the original latent, since integer
   token IDs themselves cannot be compared meaningfully with latent vectors.

`loss = flow_weight * flow_MSE + l1_weight * reconstruction_L1
      + l2_weight * reconstruction_MSE + commitment_weight * commitment_MSE`

No continuous encoder features bypass the quantizer. The flow objective is next-frame
prediction, not same-frame denoising conditioned on that frame's token. The same-frame
reconstruction objective preserves information in each token. Suitability for downstream
TTS still needs evaluation on actual speech data; this objective alone does not establish it.

## Dataset and training

Default input: a directory of `.pt` files, each containing a floating point `[T, D]`
tensor or `{"z": tensor}`. Files are independent sequences; they are never concatenated.
Each must have at least two frames. The collator right-pads and truncates to the first
`data.max_frames`; prechunk long utterances if all their frames should be used.
Set `model.latent_dim` to the actual DiCodec dimension (64 is an example).

```bash
python scripts/train_sm_quantizer.py --config dicodec/modules/sm_quantizer/default.yaml
python scripts/train_sm_quantizer.py --config my_config.yaml --resume outputs/sm_quantizer/last.pt
```

Every setting is exposed in `default.yaml`; unknown keys are rejected. Paths are
relative to the working directory. For an existing map-style dataset, set
`data.factory: your_package.your_module:YourDataset`, `data.train_kwargs: {...}`
and optionally `data.validation_kwargs: {...}`. The callable must return items with
the same tensor contract. When using a factory, the path settings are unused.

Training uses full precision AdamW on one device (`auto`: CUDA when available,
otherwise CPU), gradient clipping, optional validation, JSONL metrics and an atomic
`last.pt` checkpoint after each epoch and at `max_steps`. `max_steps` is a total step
limit including resumed steps. Checkpoints contain model/EMA state, optimizer, config,
epoch/batch cursor and Python/PyTorch RNG state. Resume restores optimizer settings
from the checkpoint. Deterministic datasets with the same data and device reproduce
the uninterrupted sequence; stochastic external datasets must manage their own RNG
and worker state. This trainer does not implement distributed EMA synchronization.

## Python API

```python
import torch
from dicodec.modules.sm_quantizer import SMQuantizer
from dicodec.modules.sm_quantizer.configs import ModelConfig, from_dict

checkpoint = torch.load("outputs/sm_quantizer/last.pt", map_location="cpu", weights_only=True)
model = SMQuantizer(from_dict(ModelConfig, checkpoint["config"]["model"]))
model.load_state_dict(checkpoint["model"])
model.eval()  # freezes clustering and dropout
with torch.no_grad():
    tokens = model.encode(z, valid_mask).indices  # [B, T], -1 at padding
    next_latent = model.predict_next(z, valid_mask)  # [B, D], midpoint ODE sampling
```

`valid_mask` is boolean `[B, T]`, True for valid frames. Only contiguous valid prefixes
and right padding are accepted. Forward outputs include each unweighted loss,
quantized vectors, token IDs, reconstruction, causal contexts and the next-frame mask.
Sampling supports `steps`, `temperature` (including zero), and a PyTorch `generator`.

## Verification

```bash
python -m unittest dicodec.modules.sm_quantizer.test_sm_quantizer
```

Tests cover future-token isolation, shifted targets, flow gradients through the
quantizer, padding exclusion from losses/EMA/gradients, frozen inference, YAML,
and training/checkpoint resume equivalence with validation.
