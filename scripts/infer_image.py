import os
import random
from typing import List

import yaml
import torch
import numpy as np
from PIL import Image
from rich.console import Console

from anime_upscaler.models.laesr import LAESR, sobel_kernel
from anime_upscaler.utils.perceptual import LPIPSLike
from anime_upscaler.utils.ckpt import find_latest_checkpoint, load_model_from_checkpoint
from anime_upscaler.utils.tiling import (
    upscale_tensor_tiled,
    edge_aware_sharpen,
    upscale_tensor_tiled_tta,
    upscale_tensor_tiled_autoscale,
    upscale_tensor_tiled_tta_autoscale,
)
import torch.nn.functional as F
import torch.optim as optim


console = Console()


def list_images(path: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files: List[str] = []
    if os.path.isdir(path):
        for root, _, fnames in os.walk(path):
            for f in fnames:
                if os.path.splitext(f)[1].lower() in exts:
                    files.append(os.path.join(root, f))
    elif os.path.isfile(path) and os.path.splitext(path)[1].lower() in exts:
        files = [path]
    return files


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dx + dy


def self_finetune_on_image(model: LAESR, lr_img: torch.Tensor, scale: int, device: torch.device, cfg: dict) -> None:
    """Quick per-image finetune (ZSSR-style):
    Downscale input by 1/scale and train the last layers to reconstruct the original.
    Trains only a small subset of params for ~seconds, improving local sharpness.
    """
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return

    steps = int(cfg.get("steps", 200))
    in_patch = int(cfg.get("in_patch", 128))
    lr_rate = float(cfg.get("lr", 1e-4))
    edge_weight = float(cfg.get("edge_weight", 0.4))
    tv_weight = float(cfg.get("tv_weight", 0.0))
    lpips_weight = float(cfg.get("lpips_weight", 0.0))
    edge_focus = bool(cfg.get("edge_focus", False))
    edge_focus_ratio = float(cfg.get("edge_focus_ratio", 0.5))

    # Prepare training pairs: input is downscaled by 1/scale, target is original lr_img
    down_factor = 1.0 / float(scale)
    with torch.no_grad():
        lr_small = F.interpolate(lr_img, scale_factor=down_factor, mode="bicubic", align_corners=False)
        target = lr_img

    # Freeze all, unfreeze only lightweight heads for quick adaptation
    for p in model.parameters():
        p.requires_grad = False
    trainable_modules = ["fuse", "tail"]
    # Optionally adapt edge and spectrum refiners for line content
    if bool(cfg.get("adapt_edges", True)):
        trainable_modules.append("edge_refine")
    if bool(cfg.get("adapt_spectrum", False)):
        trainable_modules.append("freq_refine")

    params = []
    for name in trainable_modules:
        mod = getattr(model, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True
            params += list(mod.parameters())

    if len(params) == 0:
        return

    model.train().to(device)
    # Use FP32 during quick finetune for stability
    model.float()
    opt = optim.Adam(params, lr=lr_rate, betas=(0.9, 0.99))
    k = sobel_kernel(device).to(torch.float32).view(2, 1, 3, 3)
    lpips = LPIPSLike()

    _, _, hs, ws = lr_small.shape
    patch = max(16, min(in_patch, hs, ws))
    use_patches = (hs >= patch and ws >= patch)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Precompute edge map for edge-focused sampling
    edge_energy_map = None
    if use_patches and edge_focus:
        with torch.no_grad():
            gray_small = 0.299 * lr_small[:, :1] + 0.587 * lr_small[:, 1:2] + 0.114 * lr_small[:, 2:3]
            edges_map = torch.nn.functional.conv2d(gray_small, k, padding=1)
            edge_energy_map = edges_map.abs().sum(dim=1, keepdim=True)  # (1,1,H,W)
    for _ in range(max(1, steps)):
        if use_patches:
            # Optionally bias sampling towards edge-rich patches
            if edge_energy_map is not None and (random.random() < edge_focus_ratio):
                candidates = 12
                best_score = None
                best_xy = (0, 0)
                for _ in range(candidates):
                    cy = random.randint(0, hs - patch)
                    cx = random.randint(0, ws - patch)
                    score = edge_energy_map[0, 0, cy:cy + patch, cx:cx + patch].mean().item()
                    if (best_score is None) or (score > best_score):
                        best_score = score
                        best_xy = (cy, cx)
                y0, x0 = best_xy
            else:
                y0 = random.randint(0, hs - patch)
                x0 = random.randint(0, ws - patch)
            inp = lr_small[:, :, y0:y0 + patch, x0:x0 + patch]
            ty0, tx0 = y0 * scale, x0 * scale
            tgt = target[:, :, ty0:ty0 + patch * scale, tx0:tx0 + patch * scale]
        else:
            inp = lr_small
            tgt = target

        with torch.cuda.amp.autocast(enabled=True):
            out = model(inp)
            base_loss = F.smooth_l1_loss(out, tgt)
            # Edge consistency in Y space
            def edges(x: torch.Tensor) -> torch.Tensor:
                gray = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
                return F.conv2d(gray, k, padding=1)
            e_loss = F.smooth_l1_loss(edges(out), edges(tgt))
            loss = base_loss + edge_weight * e_loss
            if lpips_weight > 0.0:
                # LPIPSLike runs in fp32 internally
                loss = loss + lpips_weight * lpips(out.clamp(0, 1).float(), tgt.clamp(0, 1).float())
            if tv_weight > 0.0:
                loss = loss + tv_weight * _total_variation(out)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    model.eval()

def main() -> None:
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    device = torch.device("cuda")

    inp = input("Chemin d\'entrée (image ou dossier): ").strip().strip('"')
    out_dir = os.path.join("outputs", "images")
    os.makedirs(out_dir, exist_ok=True)
    images = list_images(inp)
    if not images:
        console.print("[red]Aucune image trouvée.[/red]")
        return

    mcfg = conf.get("model", {})
    model = LAESR(
        scale=conf["scale"],
        base_channels=int(mcfg.get("base_channels", 64)),
        num_blocks=int(mcfg.get("num_blocks", 12)),
    ).to(device).eval()
    latest = find_latest_checkpoint(conf["paths"]["checkpoints_dir"]) 
    if latest:
        load_model_from_checkpoint(model, latest, device)
        console.print(f"[green]Checkpoint chargé:[/green] {latest}")
    else:
        console.print("[yellow]Aucun checkpoint trouvé, modèle non entraîné utilisé.[/yellow]")

    tile = conf["infer"]["tile_size"]
    overlap = conf["infer"]["tile_overlap"]
    autotile = bool(conf["infer"].get("autotile", True))
    min_tile = int(conf["infer"].get("min_tile", max(64, tile // 4)))
    scale = conf["scale"]
    use_half = conf["infer"].get("half_precision", True)

    for p in images:
        img = Image.open(p).convert("RGB")
        lr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        lr = lr.unsqueeze(0).to(device)
        # Optional quick self-finetune per image (before inference)
        ft_cfg = conf.get("infer", {}).get("self_finetune", {})
        try:
            self_finetune_on_image(model, lr.float(), scale=scale, device=device, cfg=ft_cfg)
        except Exception as e:
            console.print(f"[yellow]Self-finetune ignoré (erreur):[/yellow] {e}")

        with torch.no_grad():
            if use_half:
                model.half()
                lr = lr.half()
            use_tta = bool(conf["infer"].get("tta", False))
            if use_tta:
                if autotile:
                    sr = upscale_tensor_tiled_tta_autoscale(model, lr, scale=scale, tile_size=tile, min_tile=min_tile, tile_overlap=overlap, device=device)
                else:
                    sr = upscale_tensor_tiled_tta(model, lr, scale=scale, tile_size=tile, tile_overlap=overlap, device=device)
            else:
                if autotile:
                    sr = upscale_tensor_tiled_autoscale(model, lr, scale=scale, tile_size=tile, min_tile=min_tile, tile_overlap=overlap, device=device)
                else:
                    sr = upscale_tensor_tiled(model, lr, scale=scale, tile_size=tile, tile_overlap=overlap, device=device)
            # Optional chroma preserve from bicubic baseline to avoid color shifts
            if bool(conf["infer"].get("preserve_chroma", True)):
                base_up = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
                # Replace Cb/Cr from base_up, keep Y from sr
                def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
                    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
                    y  = 0.299 * r + 0.587 * g + 0.114 * b
                    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
                    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
                    return torch.cat([y, cb, cr], dim=1)
                def _ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
                    y, cb, cr = x[:, 0:1], x[:, 1:2]-0.5, x[:, 2:3]-0.5
                    r = y + 1.402 * cr
                    g = y - 0.344136 * cb - 0.714136 * cr
                    b = y + 1.772 * cb
                    return torch.cat([r, g, b], dim=1)
                ycbcr_sr = _rgb_to_ycbcr(sr)
                ycbcr_base = _rgb_to_ycbcr(base_up)
                y_keep = ycbcr_sr[:, 0:1]
                # Slightly blend chroma to avoid abrupt changes (80% base chroma)
                cb_keep = 0.8 * ycbcr_base[:, 1:2] + 0.2 * ycbcr_sr[:, 1:2]
                cr_keep = 0.8 * ycbcr_base[:, 2:3] + 0.2 * ycbcr_sr[:, 2:3]
                sr = _ycbcr_to_rgb(torch.cat([y_keep, cb_keep, cr_keep], dim=1))
            # Optional sharpening
            ps = conf["infer"].get("post_sharpen", {})
            if ps.get("enabled", False):
                sr = edge_aware_sharpen(
                    sr,
                    amount=float(ps.get("amount", 0.3)),
                    radius=int(ps.get("radius", 3)),
                    threshold=float(ps.get("threshold", 0.0)),
                )
        sr = (sr.clamp(0, 1) * 255.0).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + f"_x{scale}.png")
        Image.fromarray(sr).save(out_path)
        console.print(f"[green]Enregistré:[/green] {out_path}")


if __name__ == "__main__":
    main()


