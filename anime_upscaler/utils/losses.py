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


