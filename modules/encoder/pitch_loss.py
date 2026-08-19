from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float):
    return _GradientReverse.apply(x, lambda_)


class PitchLoss(nn.Module):
    def __init__(self, config, semantic_dim: int, acoustic_dim: int):
        super().__init__()
        self.config = config
        self.acoustic_f0_head = self._make_pitch_head(acoustic_dim)
        self.semantic_f0_head = self._make_pitch_head(semantic_dim)

    @staticmethod
    def _make_pitch_head(input_dim: int):
        hidden_dim = max(32, min(256, input_dim * 2))
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        semantic_quantized: torch.Tensor,
        acoustic: torch.Tensor,
        pitch_targets,
        padding_mask: Optional[torch.BoolTensor],
        step: Optional[int] = None,
    ):
        if pitch_targets is None:
            zero = semantic_quantized.new_zeros(())
            return {
                "loss": zero,
                "acoustic_f0_loss": None,
                "semantic_f0_adv_loss": None,
                "pitch_voiced_loss": None,
                "pitch_contour_loss": None,
            }

        log_f0, voiced = self._align_pitch_targets(
            pitch_targets,
            acoustic.shape[1],
            device=acoustic.device,
            dtype=acoustic.dtype,
        )
        if padding_mask is None:
            valid = torch.ones(
                acoustic.shape[:2], device=acoustic.device, dtype=torch.bool
            )
        else:
            valid = ~padding_mask
        voiced_valid = valid & voiced

        acoustic_pred = self.acoustic_f0_head(acoustic)
        acoustic_log_f0 = acoustic_pred[..., 0]
        acoustic_voiced_logits = acoustic_pred[..., 1]

        acoustic_f0_loss = self._masked_smooth_l1(
            acoustic_log_f0, log_f0, voiced_valid
        )
        pitch_voiced_loss = self._masked_bce(
            acoustic_voiced_logits, voiced.to(acoustic_voiced_logits.dtype), valid
        )
        pitch_contour_loss = self._masked_contour_l1(
            acoustic_log_f0, log_f0, voiced, valid
        )

        semantic_reversed = grad_reverse(
            semantic_quantized, self._pitch_grl_lambda(step)
        )
        semantic_pred = self.semantic_f0_head(semantic_reversed)
        semantic_log_f0 = semantic_pred[..., 0]
        semantic_voiced_logits = semantic_pred[..., 1]
        semantic_f0_loss = self._masked_smooth_l1(
            semantic_log_f0, log_f0, voiced_valid
        )
        semantic_voiced_loss = self._masked_bce(
            semantic_voiced_logits, voiced.to(semantic_voiced_logits.dtype), valid
        )
        semantic_f0_adv_loss = (
            semantic_f0_loss + self.config.voiced_loss_weight * semantic_voiced_loss
        )

        loss = (
            self.config.acoustic_loss_weight * acoustic_f0_loss
            + self.config.semantic_adv_loss_weight * semantic_f0_adv_loss
            + self.config.voiced_loss_weight * pitch_voiced_loss
            + self.config.contour_loss_weight * pitch_contour_loss
        )
        return {
            "loss": loss,
            "acoustic_f0_loss": acoustic_f0_loss,
            "semantic_f0_adv_loss": semantic_f0_adv_loss,
            "pitch_voiced_loss": pitch_voiced_loss,
            "pitch_contour_loss": pitch_contour_loss,
        }

    def _pitch_grl_lambda(self, step: Optional[int]):
        if self.config.grl_warmup_steps == 0 or step is None:
            return self.config.grl_lambda
        return self.config.grl_lambda * min(
            1.0, float(step) / float(self.config.grl_warmup_steps)
        )

    def _align_pitch_targets(
        self,
        pitch_targets,
        target_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        log_f0 = F.interpolate(
            pitch_targets["log_f0"].to(device=device, dtype=torch.float32).unsqueeze(1),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        voiced = (
            F.interpolate(
                pitch_targets["voiced"].to(device=device, dtype=torch.float32).unsqueeze(1),
                size=target_length,
                mode="nearest",
            ).squeeze(1)
            > 0.5
        )
        return log_f0.to(dtype=dtype), voiced

    def _masked_smooth_l1(self, pred, target, mask):
        if not mask.any():
            return pred.new_zeros(())
        return F.smooth_l1_loss(pred[mask].float(), target[mask].float())

    def _masked_bce(self, logits, target, mask):
        if not mask.any():
            return logits.new_zeros(())
        return F.binary_cross_entropy_with_logits(
            logits[mask].float(), target[mask].float()
        )

    def _masked_contour_l1(self, pred, target, voiced_mask, valid_mask):
        if pred.shape[1] < 2:
            return pred.new_zeros(())
        pair_mask = (
            voiced_mask[:, 1:]
            & voiced_mask[:, :-1]
            & valid_mask[:, 1:]
            & valid_mask[:, :-1]
        )
        if not pair_mask.any():
            return pred.new_zeros(())
        pred_delta = pred[:, 1:] - pred[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        return F.l1_loss(pred_delta[pair_mask].float(), target_delta[pair_mask].float())


pitchLoss = PitchLoss
