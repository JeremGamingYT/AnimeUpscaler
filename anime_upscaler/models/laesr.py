from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def sobel_kernel(device: torch.device) -> torch.Tensor:
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device)
    kernel = torch.stack([kx, ky], dim=0)  # (2,3,3)
    return kernel


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return identity + out * 0.2


class FrequencySeparation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Depthwise separable convolution to approximate learnable low/high-pass
        self.depthwise = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels)
        self.pointwise_low = nn.Conv2d(channels, channels, 1)
        self.pointwise_high = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base = self.depthwise(x)
        low = self.pointwise_low(base)
        high = self.pointwise_high(x - base)
        # Learnable mixing gates
        g = self.gate(torch.mean(x, dim=1, keepdim=True))
        low = low * (1.0 - g)
        high = high * g
        return low, high


class LAESR(nn.Module):
    """Line-Aware Edge and Spectrum Super-Resolution.

    Lightweight model tailored for anime content: strong edges and flat colors.
    Uses three branches: content, line, and spectrum, then fuses and upsamples.
    """

    def __init__(self, scale: int = 4, base_channels: int = 64, num_blocks: int = 12):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, base_channels, 3, 1, 1)

        # Content branch
        self.trunk = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])

        # Line branch: fixed Sobel -> refinement
        self.edge_refine = nn.Sequential(
            nn.Conv2d(2, base_channels // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels, 3, 1, 1),
        )

        # Spectrum branch: frequency separation + refinement
        self.freq = FrequencySeparation(base_channels)
        self.freq_refine = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
        )

        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
        )

        # Upsample with pixel shuffle
        up_layers = []
        up_factor = scale
        while up_factor > 1:
            up_layers += [
                nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ]
            up_factor //= 2
        self.upsample = nn.Sequential(*up_layers)
        self.tail = nn.Conv2d(base_channels, 3, 3, 1, 1)

        # Register Sobel buffers
        k = sobel_kernel(torch.device("cpu"))
        self.register_buffer("sobel", k.view(2, 1, 3, 3), persistent=False)

    def extract_edges(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale for edge extraction
        gray = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        edges = F.conv2d(gray, self.sobel.to(gray.dtype), padding=1)
        return edges

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)

        # Content
        content = self.trunk(feat)

        # Line
        edges = self.extract_edges(x)
        line = self.edge_refine(edges)

        # Spectrum
        low, high = self.freq(content)
        spec = self.freq_refine(torch.cat([low, high], dim=1))

        fused = self.fuse(torch.cat([content, line, spec], dim=1))
        up = self.upsample(fused)
        out = self.tail(up)
        # Global residual to stabilize training and avoid dark outputs
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        out = out + base
        return out


