import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class Aligner(nn.Module):
    def __init__(self, hdim, stride, sigma_square=5.0, ver_f=False, causal=False):
        super(Aligner, self).__init__()
        self.stride = stride
        self.score = nn.Conv1d(hdim, 1, 1, bias=False)
        self.score.weight.data.zero_()

        self.sigma_square = sigma_square
        self.ver_f = ver_f
        self.causal = causal
        self.register_buffer("indices", torch.arange(4096).unsqueeze(0).unsqueeze(0))
        self.register_buffer("z", torch.zeros(1, hdim, 4096))

    def forward(self, x, x_mask, x_lengths):
        B, D, T = x.size()
        L = math.ceil(T / self.stride)
        z_lengths = torch.ceil(x_lengths / self.stride).long()
        z_mask = x_mask[:, :: self.stride]

        score = self.score(x).exp().squeeze(1) * x_mask
        cum_score = torch.cumsum(score, dim=-1)  # B, T
        cum_score_norm = (
            ((cum_score - cum_score[:, 0:1]) / (cum_score[:, -1:] - cum_score[:, 0:1])) * (z_lengths.unsqueeze(-1) - 1)
        ).unsqueeze(-1)
        score_loss = self.get_score_loss(cum_score_norm, x_mask, x_lengths)

        if self.ver_f == False:
            distance = -self.sigma_square * ((self.indices[:, :, :L] - cum_score_norm) ** 2).transpose(
                1, 2
            )  # B, T/2, T
            distance = distance.masked_fill(~x_mask.unsqueeze(1), -np.inf)

            if self.causal:
                # Causal mask: output position l can only attend to
                # input positions t where t < (l + 1) * stride
                t_idx = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)
                l_idx = torch.arange(L, device=x.device).unsqueeze(1)   # (L, 1)
                causal_mask = t_idx < (l_idx + 1) * self.stride          # (L, T)
                distance = distance.masked_fill(~causal_mask.unsqueeze(0), -np.inf)

            alignment = F.softmax(distance, dim=-1) * z_mask.unsqueeze(-1)

            z = torch.bmm(alignment, x.transpose(1, 2)).transpose(1, 2)

            return z, z_mask, z_lengths, alignment, score_loss

        elif self.ver_f == True:
            dist_target = torch.round(cum_score_norm).long()

            if self.causal:
                # Causal clamp: input position t must map to output position >= t // stride
                t_positions = torch.arange(T, device=x.device).view(1, -1, 1)
                min_target = t_positions // self.stride
                dist_target = torch.max(dist_target, min_target)

            # Ensure target indices stay within [0, L-1]
            dist_target = dist_target.clamp(0, L - 1)

            distance = -self.sigma_square * ((dist_target - cum_score_norm) ** 2)  # B, T, 1

            exp_D = torch.exp(distance) * x_mask.unsqueeze(-1)  # B, T, 1
            sum_exp_D = exp_D.new_zeros(B, L)  # B, L
            sum_exp_D.scatter_add_(1, dist_target.squeeze(-1), exp_D.squeeze(-1))
            sum_exp_denom = torch.gather(sum_exp_D, 1, dist_target.squeeze(-1))

            x_weights = exp_D.transpose(1, 2) / sum_exp_denom.unsqueeze(1)  # B, 1, T

            indices = dist_target.squeeze(-1).unsqueeze(1).repeat(1, x.size(1), 1)
            z = self.z[:, :, :L].repeat(B, 1, 1)
            z = z.scatter_add(2, indices, x * x_weights)

            # Build an (approximate) alignment matrix for downstream code.
            alignment = x_weights.transpose(1, 2).new_zeros(B, L, T)
            alignment.scatter_add_(1, dist_target.transpose(1, 2), x_weights)

            return z, z_mask, z_lengths, self.z[:, :, :L].repeat(B, 1, 1), indices, x_weights, alignment, score_loss

    def get_score_loss(self, cum_score_norm, x_mask, x_lengths):
        score_loss = (cum_score_norm[:, 1:] - cum_score_norm[:, :-1]).squeeze(-1)
        score_loss = ((torch.relu(score_loss - 1.0) * x_mask[:, 1:]).sum(dim=-1) / (x_lengths - 1)).mean()
        return score_loss
