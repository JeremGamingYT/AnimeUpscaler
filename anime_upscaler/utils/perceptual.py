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
		try:
			import lpips  # type: ignore
			self.lpips = lpips.LPIPS(net='vgg')
			self.enabled = True
		except Exception:
			self.enabled = False

	def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		if self.enabled and self.lpips is not None:
			# expects [-1,1]
			xn = x * 2 - 1
			yn = y * 2 - 1
			return self.lpips(xn, yn).mean()
		# Fallback to L2 in feature space using small convs (lightweight)
		f = nn.Conv2d(3, 16, 3, 1, 1).to(x.device, x.dtype)
		with torch.no_grad():
			f.weight.fill_(0)
			f.bias.fill_(0)
		xf = F.relu(f(x))
		yf = F.relu(f(y))
		return F.l1_loss(xf, yf)
