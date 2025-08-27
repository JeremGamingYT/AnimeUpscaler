from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiBandingLoss(nn.Module):
    """Penalize staircase/banding artifacts in flat regions while preserving edges.

    Idea: Detect flat zones via low gradient magnitude and penalize histogram/DCT-like repetitive steps
    by measuring second derivative energy, masked by flatness.
    """

    def __init__(self, grad_threshold: float = 0.02):
        super().__init__()
        self.grad_threshold = grad_threshold
        # Simple Laplacian kernel for second derivative energy
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("lap", k.view(1, 1, 3, 3))
        # Sobel magnitude for flatness mask
        sx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("sobelx", sx.view(1, 1, 3, 3))
        self.register_buffer("sobely", sy.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to luma to focus on intensity banding
        luma = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gx = F.conv2d(luma, self.sobelx.to(device=luma.device, dtype=luma.dtype), padding=1)
        gy = F.conv2d(luma, self.sobely.to(device=luma.device, dtype=luma.dtype), padding=1)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        flat_mask = (grad_mag < self.grad_threshold).float()

        lap = F.conv2d(luma, self.lap.to(device=luma.device, dtype=luma.dtype), padding=1)
        loss = (lap.abs() * flat_mask).mean()
        return loss



class ChromaTVLoss(nn.Module):
    """Total variation on chroma channels only (Cb/Cr) to keep color flats clean.

    Operates in YCbCr space; encourages smooth chroma while leaving luma edges.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
        # x in [0,1]
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y  = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        return torch.cat([y, cb, cr], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ycbcr = self._rgb_to_ycbcr(x)
        chroma = ycbcr[:, 1:3]
        dx = chroma[:, :, :, 1:] - chroma[:, :, :, :-1]
        dy = chroma[:, :, 1:, :] - chroma[:, :, :-1, :]
        loss = dx.abs().mean() + dy.abs().mean()
        return loss


class AntiHaloLoss(nn.Module):
    """Penalize overshoot (ringing/halo) around strong edges.

    Heuristic: compute luma detail via unsharp mask; penalize detail magnitude
    only in regions with strong gradients (edge mask). Keep weight small.
    """

    def __init__(self, edge_threshold: float = 0.05, blur_kernel: int = 5):
        super().__init__()
        k = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        self.register_buffer("sobelx", k.view(1, 1, 3, 3))
        self.register_buffer("sobely", k.t().contiguous().view(1, 1, 3, 3))
        self.edge_threshold = edge_threshold
        self.blur_kernel = max(1, int(blur_kernel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Luma for edge detection
        luma = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gx = F.conv2d(luma, self.sobelx.to(device=luma.device, dtype=luma.dtype), padding=1)
        gy = F.conv2d(luma, self.sobely.to(device=luma.device, dtype=luma.dtype), padding=1)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        edge_mask = (grad_mag > self.edge_threshold).float()

        # Unsharp detail: x - blur(x)
        k = self.blur_kernel
        if k > 1:
            pad = (k // 2, k // 2, k // 2, k // 2)
            lpad = F.pad(luma, (pad[0], pad[1], pad[2], pad[3]), mode="reflect")
            blur = F.avg_pool2d(lpad, kernel_size=k, stride=1)
        else:
            blur = luma
        detail = luma - blur
        loss = (detail.abs() * edge_mask).mean()
        return loss


class TotalVariationLoss(nn.Module):
    """Simple total variation loss on all RGB channels.

    Helps suppress isolated pixel noise and small artifacts by encouraging
    spatial smoothness.  Applied with a very small weight so that edges are
    preserved while minor irregularities are reduced.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx.abs().mean() + dy.abs().mean()
