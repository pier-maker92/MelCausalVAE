import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolateRegulator(nn.Module):
    def __init__(
        self,
        depth: int,  # <-- number of conv-GN-Mish stacks
        in_channels: int,
        channels: int,
        out_channels: int,
        groups: int = 1,
        is_causal: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.is_causal = is_causal

        layers = nn.ModuleList([])

        # ----- convolution blocks -----
        for i in range(depth):
            conv = nn.Conv1d(
                in_channels if i == 0 else channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=0 if is_causal else 1,  # symmetric vs causal
            )
            layers.extend(
                [
                    conv,
                    nn.GroupNorm(groups, channels),
                    nn.Mish(),
                ]
            )

        # ----- final projection -----
        layers.append(nn.Conv1d(channels if depth > 0 else in_channels, out_channels, 1))

        self.model = nn.Sequential(*layers)

    def _apply_conv_stack(self, x):
        """
        Applies the conv → norm → activation stack.
        Uses left-padding for Conv1d when causal.
        """
        for layer in self.model:
            if isinstance(layer, nn.Conv1d) and layer.kernel_size[0] > 1:
                if self.is_causal:
                    # causal left padding: pad = kernel_size - 1
                    k = layer.kernel_size[0]
                    x = F.pad(x, (k - 1, 0))
                    x = layer(x)
                else:
                    x = layer(x)
            else:
                x = layer(x)
        return x

    def _causal_interpolate(self, x, out_length):
        """
        Nearest-neighbor causal upsampling:
        Only uses *past* frames (zero-order hold).
        """
        B, C, T = x.shape
        if T == out_length:
            return x

        scale = out_length / T
        idx = torch.floor(torch.arange(out_length, device=x.device) / scale).long()
        idx = torch.clamp(idx, 0, T - 1)
        return x[:, :, idx]

    @staticmethod
    def masked_mean(x, mask):
        # mask: 1 means real, 0 means ignore
        return (x * mask).sum() / mask.sum()

    def forward(
        self,
        guidance,  # semantic_guidance.feature (B, T_g, D)
        guidance_mask,  # semantic_guidance.padding_mask (B, T_g) 1 = pad, 0 = real
        target,  # (B, T_target, D_target)
        target_padding_mask,  # (B, T_target) 1 = pad, 0 = real
    ):
        """
        Interpolates guidance features and mask to match target length,
        combines masks, and computes masked cosine distillation loss.
        """

        B, T_target = target.shape[:2]

        # ----------------- INTERPOLATE guidance -----------------
        x = guidance.transpose(1, 2)  # (B, D, T_g)

        if self.is_causal:
            x = self._causal_interpolate(x, T_target)
        else:
            x = F.interpolate(x, size=T_target, mode="linear", align_corners=False)
        out = self._apply_conv_stack(x)  # (B, T_target, D_proj)
        out = out.transpose(1, 2)  # (B, T_target, D_proj)

        # ----------------- INTERPOLATE guidance_mask -----------------
        gmask = (guidance_mask == 0).float().unsqueeze(1)  # (B, 1, T_g)
        if self.is_causal:
            gmask_interp = self._causal_interpolate(gmask, T_target)
        else:
            gmask_interp = F.interpolate(gmask, size=T_target, mode="linear", align_corners=False)
        gmask_interp = gmask_interp.squeeze(1)  # (B, T_target)
        gmask_interp = (gmask_interp > 0.5).float()  # threshold → boolean mask

        # ----------------- COMBINE MASKS -----------------
        # 1 = valid, 0 = ignore
        final_mask = ((target_padding_mask == 0) & (gmask_interp == 1)).float()  # (B, T_target)

        # ----------------- COMPUTE LOSS -----------------
        proj_loss = 0.0
        for i in range(B):
            cos_sim = F.cosine_similarity(out[i], target[i], dim=-1)  # (T_target,)
            proj_loss += self.masked_mean(1 - cos_sim, final_mask[i])

        proj_loss = proj_loss / B
        return proj_loss
