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

"""UTokyo-SaruLab System for VoiceMOS Challenge 2022 (UTMOS) (see https://arxiv.org/abs/2204.02152)."""

import torch
import torchaudio
from speechbrain.utils.metric_stats import MetricStats

__all__ = ["UTMOS"]


SAMPLE_RATE = 16000


class UTMOS(MetricStats):
    def __init__(self, sample_rate: int, model: str = None, device=None):
        self.sample_rate = sample_rate
        self.model = model
        if model is None:
            self.model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            )
        self.clear()
        self.device = device

    @torch.no_grad()
    def append(self, ids: list, sig: torch.FloatTensor):

        # Resample
        if self.sample_rate != SAMPLE_RATE:
            sig = [
                torchaudio.functional.resample(s, self.sample_rate, SAMPLE_RATE)
                for s in sig
            ]

        self.model.to(self.device)
        self.model.eval()

        # Forward
        for s, id_ in zip(sig, ids):
            scores = self.model(s.unsqueeze(0), SAMPLE_RATE)

            self.ids += [id_]
            self.scores += scores.cpu().tolist()


if __name__ == "__main__":
    sample_rate = 24000
    ids = ["A", "B"]
    hyp_sig = torch.randn(2, 2 * sample_rate)

    utmos = UTMOS(sample_rate)
    utmos.append(ids, hyp_sig)
    print(utmos.summarize("average"))
