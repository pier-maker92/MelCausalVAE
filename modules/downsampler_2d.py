import math
import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
from dataclasses import dataclass


class CausalDownsamplingBlock(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        n_residual_blocks=2,
        compress_freq=4,
        drop_p=0.1,
        separable=False,
    ):
        super().__init__()
        # residual blocks
        dilations = [1, 2, 4, 8]
        self.residual_blocks = nn.ModuleList(
            [
                PreNormResCausalBlock(
                    c_in,
                    c_in,
                    kt=3,
                    kf=5,
                    dt=dilation,
                    df=1,
                    st=1,
                    sf=1,
                    drop_p=drop_p,
                    separable=separable,
                )
                for dilation in dilations[:n_residual_blocks]
            ]
        )
        # downsampler
        self.downsampling = PreNormResCausalBlock(
            c_in,
            c_out,
            kt=5,
            kf=7,
            dt=1,
            df=1,
            st=2,
            sf=compress_freq,
            drop_p=drop_p,
            separable=separable,
        )

    def forward(self, x):
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return self.downsampling(x)


class ChannelLastLayerNorm(nn.Module):
    # LayerNorm over channels for tensors in [B, C, T, F]
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):  # x: [B, C, T, F]
        x = rearrange(x, "b c t f -> b t f c")  # [B, T, F, C]
        x = self.ln(x)  # normalize over C
        return rearrange(x, "b t f c -> b c t f")  # [B, C, T, F]


class TimeCausalConv2d(nn.Module):
    # Causal in time (left pad only); "same" in frequency (symmetric left/right)
    def __init__(self, c_in, c_out, kt, kf, dt=1, df=1, st=1, sf=1, separable=False):
        super().__init__()
        self.kt, self.kf, self.dt, self.df, self.st, self.sf = kt, kf, dt, df, st, sf
        self.separable = separable

        if separable:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    c_in,
                    c_in,
                    (kt, kf),
                    dilation=(dt, df),
                    stride=(st, sf),
                    padding=0,
                    groups=c_in,
                ),
                nn.Conv2d(c_in, c_out, 1),
            )
        else:
            self.conv = nn.Conv2d(
                c_in, c_out, (kt, kf), dilation=(dt, df), stride=(st, sf), padding=0
            )

    def _same_freq_pad(self, F_in):
        # Solve for total pad so L_out = ceil(F_in / sf)
        # L_out = floor((F_in + p - df*(kf-1) - 1)/sf + 1)
        target = (F_in + self.sf - 1) // self.sf
        p_total = max(0, (target - 1) * self.sf + self.df * (self.kf - 1) + 1 - F_in)
        left = p_total // 2
        right = p_total - left
        return left, right

    def forward(self, x):  # x: [B, C, T, F]
        t_left = (self.kt - 1) * self.dt  # causal time pad (left only)
        f_left, f_right = self._same_freq_pad(x.shape[-1])
        x = torch.nn.functional.pad(x, (f_left, f_right, t_left, 0))
        return self.conv(x)


class PreNormResCausalBlock(nn.Module):
    # LN -> act -> causal conv; skip mirrors stride if T or F changes, else 1x1
    def __init__(
        self,
        c_in,
        c_out,
        *,
        kt=3,
        kf=7,
        dt=1,
        df=1,
        st=1,
        sf=1,
        act=nn.GELU,
        drop_p=0.1,
        separable=False,
    ):
        super().__init__()
        self.ln = ChannelLastLayerNorm(c_in)
        self.act = act()
        self.main = TimeCausalConv2d(
            c_in, c_out, kt=kt, kf=kf, dt=dt, df=df, st=st, sf=sf, separable=separable
        )
        shape_changes = (c_in != c_out) or (st != 1) or (sf != 1)
        if shape_changes:
            # Match spatial sizes when stride or non-same padding is used
            self.skip = TimeCausalConv2d(
                c_in, c_out, kt=1, kf=1, dt=1, df=1, st=st, sf=sf, separable=separable
            )
        else:
            if c_in != c_out:
                self.skip = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1)
            else:
                self.skip = nn.Identity()
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        h = self.act(self.ln(x))
        h = self.dropout(h)
        return self.main(h) + self.skip(x)


@dataclass
class DownSamplerConfig:
    d_model: int
    compress_factor: int = 1
    drop_p: float = 0.1
    n_residual_blocks: int = 3
    separable: bool = False
    out_dim: Optional[int] = None


class DownSampler(nn.Module):
    def __init__(self, config: DownSamplerConfig):
        """
        Downsampler module for compressing the time-frequency representation of audio.
        Args:
            d_model: Hidden dimension size
            compress_factor: Factor by which to compress the time-frequency representation
            drop_p: Dropout probability
            n_residual_blocks: Number of residual blocks
            separable: Whether to use separable convolutions
        """
        super().__init__()
        self.compress_factor = config.compress_factor

        self.freq_mixer = nn.Sequential(
            PreNormResCausalBlock(
                1,
                config.d_model // 8,
                kt=7,
                kf=7,
                dt=1,
                df=1,
                st=1,
                sf=1,
                drop_p=config.drop_p,
                separable=config.separable,
            ),
            PreNormResCausalBlock(
                config.d_model // 8,
                config.d_model // 4,
                kt=5,
                kf=5,
                dt=1,
                df=4,
                st=1,
                sf=1,
                drop_p=config.drop_p,
                separable=config.separable,
            ),
            PreNormResCausalBlock(
                config.d_model // 4,
                config.d_model // 2,
                kt=3,
                kf=3,
                dt=1,
                df=8,
                st=1,
                sf=1,
                drop_p=config.drop_p,
                separable=config.separable,
            ),
        )
        # downsample operation
        self.downsampling = nn.ModuleDict()
        downsampling_stages = int(math.log2(self.compress_factor))
        for i in range(max(0, downsampling_stages)):
            d_in = config.d_model // 2 if not i else config.d_model
            compress_freq = 10 if not i else 1
            self.downsampling[f"downsample@{2**(i+2+1)}"] = CausalDownsamplingBlock(
                d_in,
                config.d_model,
                n_residual_blocks=config.n_residual_blocks,
                compress_freq=compress_freq,
                drop_p=config.drop_p,
                separable=config.separable,
            )
        self.freq_out_proj = nn.Linear(10, 1)
        d_out = config.d_model if config.out_dim is None else config.out_dim
        self.out_proj = nn.Linear(config.d_model, d_out)

    def forward(
        self, x: torch.FloatTensor, padding_mask: Optional[torch.BoolTensor] = None
    ):
        # x: shape [B, T, D]
        x = x.unsqueeze(1)
        x = self.freq_mixer(x)
        for layer in self.downsampling.values():
            x = layer(x)
        x = self.freq_out_proj(x).squeeze(-1).permute(0, 2, 1)
        x = self.out_proj(x)

        if padding_mask is not None:
            # Downsample padding_mask.
            # Since each downsampling block has stride 2 in time, we subsample by 2 at each stage.
            # Total downsampling factor is self.compress_factor.
            # We use interpolation to ensure it matches the target length exactly.
            target_length = x.shape[1]
            padding_mask = (
                torch.nn.functional.interpolate(
                    padding_mask.unsqueeze(1).float(),
                    size=target_length,
                    mode="nearest",
                ).squeeze(1)
                > 0.5
            )
            return x, padding_mask

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Benchmark DownSampler module")
    parser.add_argument(
        "-b",
        "--benchmark",
        action="store_true",
        help="Run speed comparison benchmark",
    )
    parser.add_argument(
        "-d",
        "--d_model",
        type=int,
        default=512,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "-c",
        "--compress_factor",
        type=int,
        default=1,
        help="Compression factor",
    )
    parser.add_argument(
        "-r",
        "--n_residual_blocks",
        type=int,
        default=3,
        help="Number of residual blocks",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if args.benchmark:
        print(f"Benchmarking on {device}...")

        # Setup models
        config = DownSamplerConfig(
            d_model=args.d_model,
            separable=False,
            n_residual_blocks=args.n_residual_blocks,
            compress_factor=args.compress_factor,
        )
        down_std = DownSampler(config).to(device)
        config.separable = True
        down_sep = DownSampler(config).to(device)

        def benchmark(model, label, iterations=50):
            # Warmup
            for _ in range(10):
                x = torch.randn(1, 400, 100).to(device)
                model(x)

            if device.type == "cuda":
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                starter.record()
                for _ in range(iterations):
                    model(x)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) / iterations
            else:
                # CPU or MPS
                start = time.perf_counter()
                for _ in range(iterations):
                    model(x)
                    if device.type == "mps":
                        torch.mps.synchronize()
                curr_time = (time.perf_counter() - start) * 1000 / iterations  # ms

            params = count_parameters(model)
            print(
                f"{label:12} | Params: {params:10,d} | Time/iter: {curr_time:8.2f} ms"
            )
            return curr_time

        print("-" * 60)
        std_time = benchmark(down_std, "Standard")
        sep_time = benchmark(down_sep, "Separable")
        print("-" * 60)
        print(f"Speedup: {std_time / sep_time:.2f}x")
    else:
        # Default run: just show shape and parameters
        config = DownSamplerConfig(
            d_model=args.d_model,
            separable=True,
            n_residual_blocks=args.n_residual_blocks,
            compress_factor=args.compress_factor,
        )
        down = DownSampler(config).to(device)
        x = torch.randn(1, 400, 100).to(device)
        y = down(x)
        print(f"\nModel (separable=True) params: {count_parameters(down):,d}")
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {y.shape}")
        print("\nUse --benchmark to run the speed comparison.")
