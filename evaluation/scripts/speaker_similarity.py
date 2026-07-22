# ==============================================================================
# Copyright 2025 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Cosine similarity between speaker embeddings."""

import torch
import torchaudio
from typing import List, Optional
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from speechbrain.utils.metric_stats import MetricStats
from transformers import AutoModelForAudioXVector

__all__ = ["SpkSimWavLM"]


SAMPLE_RATE = 16000


class SpkSimWavLM(MetricStats):
    def __init__(
        self, model_hub, save_path=HUGGINGFACE_HUB_CACHE, model=None, device=None
    ):
        self.model = model
        if model is None:
            self.model = AutoModelForAudioXVector.from_pretrained(
                model_hub, cache_dir=save_path
            )
        self.clear()
        self.device = device

    @torch.no_grad()
    def append(
        self,
        hyp_sr: int,
        ref_sr: int,
        ids: List[str],
        hyp_sig: List[torch.FloatTensor],
        ref_sig: List[torch.FloatTensor],
    ):
        # handling sr
        if hyp_sr != SAMPLE_RATE:
            hyp_sig = [
                torchaudio.functional.resample(x, hyp_sr, SAMPLE_RATE) for x in hyp_sig
            ]
        if ref_sr != SAMPLE_RATE:
            ref_sig = [
                torchaudio.functional.resample(x, ref_sr, SAMPLE_RATE) for x in ref_sig
            ]

        self.model.to(self.device)
        self.model.eval()

        lens = [torch.tensor(x.shape[-1]) for x in hyp_sig + ref_sig]

        # Attention mask
        attention_mask = None
        sig = torch.nn.utils.rnn.pad_sequence(
            hyp_sig + ref_sig, batch_first=True, padding_value=0.0
        )

        attention_mask = torch.ones(sig.shape, dtype=torch.long, device=sig.device)
        for i, l in enumerate(lens):
            attention_mask[i, l:] = 0

        # Forward
        embs = self.model(
            input_values=sig,
            attention_mask=attention_mask,
            output_attentions=False,
        ).embeddings

        hyp_embs, ref_embs = embs.split([len(hyp_sig), len(ref_sig)])
        scores = torch.nn.functional.cosine_similarity(hyp_embs, ref_embs, dim=-1)

        self.ids += ids
        self.scores += scores.cpu().tolist()


if __name__ == "__main__":
    sample_rate = 24000
    ids = ["A", "B"]
    hyp_sig = torch.randn(2, 2 * sample_rate)
    ref_sig = torch.randn(2, 2 * sample_rate)

    # spk_sim = SpkSimECAPATDNN("speechbrain/spkrec-ecapa-voxceleb", sample_rate)
    # spk_sim.append(ids, hyp_sig, ref_sig)
    # print(spk_sim.summarize("average"))

    spk_sim = SpkSimWavLM("microsoft/wavlm-base-sv", sample_rate)
    spk_sim.append(ids, hyp_sig, ref_sig)
    print(spk_sim.summarize("average"))
