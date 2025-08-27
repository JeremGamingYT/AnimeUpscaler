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


@torch.no_grad()
def edge_aware_sharpen(x: torch.Tensor, amount: float = 0.3, radius: int = 1, threshold: float = 0.0) -> torch.Tensor:
    """Simple edge-aware sharpening in tensor space.

    Unsharp mask variant: x + amount * (x - blur(x)), with edge thresholding.
    """
    if amount <= 1e-6:
        return x
    # Gaussian approx by box blur (fast) using avgpool
    k = max(1, int(radius))
    if k > 1:
        pad = (k // 2, k // 2, k // 2, k // 2)
        x_pad = F.pad(x, (pad[0], pad[1], pad[2], pad[3]), mode="reflect")
        blur = F.avg_pool2d(x_pad, kernel_size=k, stride=1)
    else:
        blur = x
    mask = x - blur
    if threshold > 0:
        m = (mask.abs() > threshold).float()
        mask = mask * m
    y = (x + amount * mask).clamp(0, 1)
    return y


@torch.no_grad()
def upscale_tensor_tiled_tta(model: torch.nn.Module, lr: torch.Tensor, scale: int, tile_size: int, tile_overlap: int, device: torch.device) -> torch.Tensor:
    """x8 Test-Time Augmentation (D4 group: rotations + flips) with tiled upscale.

    Returns averaged output. Slower but can improve detail and reduce artifacts.
    """
    def t_identity(x: torch.Tensor) -> torch.Tensor:
        return x

    def t_hflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-1])

    def t_vflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-2])

    def t_transpose(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(-1, -2)

    T = [
        (t_identity, t_identity),
        (t_hflip, t_hflip),
        (t_vflip, t_vflip),
        (lambda x: t_hflip(t_vflip(x)), lambda y: t_hflip(t_vflip(y))),
        (t_transpose, t_transpose),
        (lambda x: t_hflip(t_transpose(x)), lambda y: t_transpose(t_hflip(y))),
        (lambda x: t_vflip(t_transpose(x)), lambda y: t_transpose(t_vflip(y))),
        (lambda x: t_hflip(t_vflip(t_transpose(x))), lambda y: t_transpose(t_vflip(t_hflip(y)))),
    ]

    acc = None
    for fwd, inv in T:
        lr_t = fwd(lr)
        sr_t = upscale_tensor_tiled(model, lr_t, scale=scale, tile_size=tile_size, tile_overlap=tile_overlap, device=device)
        sr_t = inv(sr_t)
        acc = sr_t if acc is None else acc + sr_t
    out = acc / float(len(T))
    return out


