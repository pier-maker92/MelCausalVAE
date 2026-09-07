"""Exercise the real Dicodec forward with small modules and no downloads."""
import types
import unittest

import torch
from torch import nn

from dicodec.modules.dicodec import Dicodec


class TinyEncoder(nn.Linear):
    def forward(self, x, padding_mask, step=None):
        z = super().forward(x)
        return types.SimpleNamespace(
            z=z, mu=z, padding_mask=padding_mask,
            kl_loss=z.square().mean() if self.training else None,
        )


class TinySpeaker(nn.Linear):
    def forward(self, audio):
        return super().forward(audio.mean(dim=1))


class TinyDecoder(nn.Linear):
    def forward(self, target, target_padding_mask, context_vector, speaker_embedding):
        prediction = super().forward(context_vector + speaker_embedding[:, None])
        return types.SimpleNamespace(loss=(prediction - target).square().mean())


def tiny_model():
    model = Dicodec.__new__(Dicodec)
    nn.Module.__init__(model)
    model.decoder_finetuning = False
    model.wavlm = nn.Linear(4, 4)
    model.wavlm_extractor = nn.BatchNorm1d(4)
    model.feature_extractor = nn.BatchNorm1d(4)
    model.encoder = TinyEncoder(4, 4)
    model.speaker_encoder = TinySpeaker(4, 4)
    # Match the production speaker's unregistered reference to shared WavLM.
    object.__setattr__(model.speaker_encoder, "_wavlm", model.wavlm)
    model.decoder = TinyDecoder(4, 4)
    model.lowpass_filter = nn.Linear(4, 4)
    model.vocoder = nn.Linear(4, 4)
    model.external_semantic_quantizer = None

    def extract_features(self, audio, **kwargs):
        features = self.feature_extractor(audio.transpose(1, 2)).transpose(1, 2)
        mask = torch.zeros(audio.shape[:2], dtype=torch.bool)
        return features, mask, audio, mask

    model.extract_features = types.MethodType(extract_features, model)
    return model


class DecoderFinetuningTest(unittest.TestCase):
    def test_only_speaker_and_decoder_change_after_step(self):
        torch.manual_seed(7)
        model = tiny_model()
        model.configure_decoder_finetuning()
        model.eval()
        model.train()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        output = model(torch.randn(2, 6, 4))
        self.assertIsNone(output.kl_loss)
        output.audio_loss.backward()
        for name, param in model.named_parameters():
            trainable = name.startswith(("speaker_encoder.", "decoder."))
            self.assertEqual(param.requires_grad, trainable, name)
            self.assertEqual(param.grad is not None, trainable, name)
        optimizer.step()
        for name, value in model.state_dict().items():
            changed = not torch.equal(value, before[name])
            self.assertEqual(
                changed, name.startswith(("speaker_encoder.", "decoder.")), name
            )

    def test_modes_survive_repeated_train_eval_transitions(self):
        model = tiny_model()
        model.external_semantic_quantizer = nn.Linear(4, 4)
        model.configure_decoder_finetuning()
        for mode in (False, True, False, True):
            model.train(mode)
            for name, module in model.named_children():
                self.assertEqual(
                    module.training,
                    mode and name in {"speaker_encoder", "decoder"}, name,
                )

    def test_default_training_keeps_encoder_trainable(self):
        model = tiny_model().train()
        self.assertTrue(model.encoder.training)
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))

    def test_missing_speaker_fails_before_freezing(self):
        model = tiny_model()
        model.speaker_encoder = None
        with self.assertRaisesRegex(ValueError, "speaker_encoder"):
            model.configure_decoder_finetuning()
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
