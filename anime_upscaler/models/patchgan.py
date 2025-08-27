from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
    )


class PatchDiscriminator(nn.Module):
    """Lightweight PatchGAN discriminator.

    Produces a map of realism scores. Use hinge loss by default.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_layers: int = 4):
        super().__init__()
        layers = [conv_block(in_channels, base_channels, stride=2)]
        ch = base_channels
        for i in range(1, num_layers):
            layers += [conv_block(ch, ch * 2, stride=2)]
            ch *= 2
        layers += [nn.Conv2d(ch, 1, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


