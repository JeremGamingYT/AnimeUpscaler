import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, channel: int | None = None, size_average=True) -> torch.Tensor:
    # Minimal SSIM implementation for training-time monitoring (not fully reference-accurate)
    import math
    device = pred.device
    dtype = pred.dtype
    # Ensure target matches pred dtype/device to avoid AMP dtype mismatches
    if target.dtype != dtype or target.device != device:
        target = target.to(device=device, dtype=dtype)
    if channel is None:
        channel = int(pred.size(1))

    def gaussian(window_size, sigma):
        vals = [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
        gauss = torch.tensor(vals, device=device, dtype=dtype)
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    window = (_1D_window @ _1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.flatten(1).mean(-1)


