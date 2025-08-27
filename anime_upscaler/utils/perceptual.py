from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPSLike(nn.Module):
	def __init__(self):
		super().__init__()
		self.enabled = False
		self.lpips = None
		self.fallback: Optional[nn.Sequential] = None
		try:
			import lpips  # type: ignore
			self.lpips = lpips.LPIPS(net='vgg')
			self.enabled = True
		except Exception:
			self.enabled = False
			# Lightweight perceptual fallback: fixed random conv features (no grad)
			layers = [
				nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(inplace=True),
				nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
				nn.AvgPool2d(2),
				nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(inplace=True),
			]
			self.fallback = nn.Sequential(*layers)
			for p in self.fallback.parameters():
				p.requires_grad_(False)

	def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		if self.enabled and self.lpips is not None:
			# expects [-1,1]
			xn = x * 2 - 1
			yn = y * 2 - 1
			return self.lpips(xn, yn).mean()
		# Deterministic non-zero fallback perceptual distance in a fixed feature space
		assert self.fallback is not None
		fx = self.fallback.to(device=x.device, dtype=x.dtype)(x)
		fy = self.fallback.to(device=y.device, dtype=y.dtype)(y)
		return F.l1_loss(fx, fy)
