from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def conv_block(in_ch: int, out_ch: int, stride: int = 2, use_sn: bool = True) -> nn.Sequential:
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)
    if use_sn:
        conv = spectral_norm(conv)
    return nn.Sequential(
        conv,
        nn.LeakyReLU(0.2, inplace=True),
    )


class PatchDiscriminator(nn.Module):
    """Lightweight PatchGAN discriminator for SR GAN training.

    Produces a patch-wise real/fake map instead of a single scalar.
    Design inspired by ESRGAN/RelativisticGAN discriminators (no batch norm).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, use_sn: bool = True):
        super().__init__()
        c = base_channels
        self.body = nn.Sequential(
            conv_block(in_channels, c, stride=2, use_sn=use_sn),  # H/2
            conv_block(c, c, stride=1, use_sn=use_sn),
            conv_block(c, c * 2, stride=2, use_sn=use_sn),        # H/4
            conv_block(c * 2, c * 2, stride=1, use_sn=use_sn),
            conv_block(c * 2, c * 4, stride=2, use_sn=use_sn),    # H/8
            conv_block(c * 4, c * 4, stride=1, use_sn=use_sn),
            conv_block(c * 4, c * 8, stride=2, use_sn=use_sn),    # H/16
            conv_block(c * 8, c * 8, stride=1, use_sn=use_sn),
        )
        head = nn.Conv2d(c * 8, 1, kernel_size=3, stride=1, padding=1)
        self.head = spectral_norm(head) if use_sn else head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        out = self.head(h)
        return out
