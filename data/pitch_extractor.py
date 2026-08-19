from typing import Any, Optional

import torch
import torchaudio.functional as AF


class PitchExtractor:
    def __init__(
        self,
        encoder_config: Any,
        sampling_rate: int,
        hop_length: int,
    ):
        self.config = encoder_config
        self.pitch_loss_config = getattr(encoder_config, "pitch_loss_config", None)
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

    @torch.no_grad()
    def __call__(
        self,
        audios_srs,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if self.pitch_loss_config is None:
            return None

        import torchcrepe

        cfg = self.pitch_loss_config
        decoder = getattr(torchcrepe.decode, cfg.torchcrepe_decoder)
        device = device or torch.device("cpu")
        hop_length = cfg.torchcrepe_hop_length or self.hop_length

        waveforms = []
        for audio, sample_rate in audios_srs:
            waveform = audio.to(device=device, dtype=torch.float32)
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=0)
            if sample_rate != self.sampling_rate:
                waveform = AF.resample(waveform, sample_rate, self.sampling_rate)
            waveforms.append(waveform)

        padded = torch.nn.utils.rnn.pad_sequence(
            waveforms, batch_first=True, padding_value=0.0
        )
        f0, periodicity = torchcrepe.predict(
            padded,
            self.sampling_rate,
            hop_length,
            cfg.fmin,
            cfg.fmax,
            cfg.torchcrepe_model,
            decoder=decoder,
            batch_size=cfg.torchcrepe_batch_size,
            device=device,
            return_periodicity=True,
        )
        voiced = periodicity >= cfg.periodicity_threshold
        log_f0 = torch.log(f0.clamp_min(1.0))

        return {
            "log_f0": log_f0.to(device=device, dtype=dtype),
            "voiced": voiced.to(device=device),
        }
