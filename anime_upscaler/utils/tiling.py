from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def upscale_tensor_tiled(model: torch.nn.Module, lr: torch.Tensor, scale: int, tile_size: int, tile_overlap: int, device: torch.device) -> torch.Tensor:
    """
    Args:
        lr: (1,3,H,W) in [0,1]
        returns: (1,3,H*scale,W*scale)
    """
    assert lr.dim() == 4 and lr.size(0) == 1
    _, _, h, w = lr.shape
    rh, rw = h * scale, w * scale
    out = torch.zeros((1, 3, rh, rw), device=device)
    weight = torch.zeros_like(out[:, :1])

    stride = tile_size - tile_overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y0 = y
            x0 = x
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            in_tile = lr[:, :, y0:y1, x0:x1]
            # Reflect pad to tile_size
            pad_bottom = tile_size - (y1 - y0)
            pad_right = tile_size - (x1 - x0)
            # Use 'replicate' to allow padding larger than the current tile size
            in_tile = F.pad(in_tile, (0, pad_right, 0, pad_bottom), mode="replicate")
            sr_tile = model(in_tile)
            # Remove padding from sr_tile
            sy1 = (y1 - y0) * scale
            sx1 = (x1 - x0) * scale
            sr_tile = sr_tile[:, :, :sy1, :sx1]

            oy0, ox0 = y0 * scale, x0 * scale
            oy1, ox1 = oy0 + sy1, ox0 + sx1

            # Feathering mask to reduce seams
            wy = torch.ones((sy1,), device=device)
            wx = torch.ones((sx1,), device=device)
            edge = tile_overlap * scale
            if edge > 0:
                rampy = torch.linspace(0, 1, steps=min(edge, sy1), device=device)
                rampx = torch.linspace(0, 1, steps=min(edge, sx1), device=device)
                wy[: rampy.numel()] = rampy
                wy[-rampy.numel() :] = rampy.flip(0)
                wx[: rampx.numel()] = rampx
                wx[-rampx.numel() :] = rampx.flip(0)
            mask = (wy.view(1, 1, -1, 1) * wx.view(1, 1, 1, -1)).clamp(1e-3, 1.0)

            out[:, :, oy0:oy1, ox0:ox1] += sr_tile * mask
            weight[:, :, oy0:oy1, ox0:ox1] += mask

    out = out / weight.clamp(min=1e-6)
    return out


