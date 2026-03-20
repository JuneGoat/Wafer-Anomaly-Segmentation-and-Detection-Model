from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ConvAutoencoderConfig:
    in_channels: int = 3
    base_channels: int = 64
    num_down: int = 3
    out_channels: int = 3


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, cfg: ConvAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        enc: list[nn.Module] = []
        pools: list[nn.Module] = []
        skip_channels: list[int] = []
        ch_in = cfg.in_channels
        ch = cfg.base_channels
        for _ in range(cfg.num_down):
            enc.append(_ConvBlock(ch_in, ch))
            pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            skip_channels.append(ch)
            ch_in = ch
            ch *= 2
        self.encoder_blocks = nn.ModuleList(enc)
        self.pools = nn.ModuleList(pools)

        self.bottleneck = _ConvBlock(ch_in, ch_in)

        dec_up: list[nn.Module] = []
        dec_blocks: list[nn.Module] = []
        ch = ch_in
        for skip_ch in reversed(skip_channels):
            up_out = ch // 2
            dec_up.append(nn.ConvTranspose2d(ch, up_out, kernel_size=2, stride=2))
            dec_blocks.append(_ConvBlock(up_out + skip_ch, up_out))
            ch = up_out
        self.decoder_up = nn.ModuleList(dec_up)
        self.decoder_blocks = nn.ModuleList(dec_blocks)

        self.out_conv = nn.Conv2d(ch, cfg.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = x
        for block, pool in zip(self.encoder_blocks, self.pools):
            h = block(h)
            skips.append(h)
            h = pool(h)

        h = self.bottleneck(h)

        for up, block, skip in zip(self.decoder_up, self.decoder_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        return self.out_conv(h)

    @torch.no_grad()
    def reconstruction_error_map(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        err = (recon - x).pow(2)
        if err.shape[1] > 1:
            err = err.mean(dim=1, keepdim=True)
        return err

    @staticmethod
    def loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, target)
