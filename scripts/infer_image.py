import os
from typing import List

import yaml
import torch
import numpy as np
from PIL import Image
from rich.console import Console

from anime_upscaler.models.laesr import LAESR
from anime_upscaler.utils.ckpt import find_latest_checkpoint, load_model_from_checkpoint
from anime_upscaler.utils.tiling import (
    upscale_tensor_tiled,
    edge_aware_sharpen,
    upscale_tensor_tiled_tta,
    upscale_tensor_tiled_autoscale,
    upscale_tensor_tiled_tta_autoscale,
)
import torch.nn.functional as F


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
                cb_keep = ycbcr_base[:, 1:2]
                cr_keep = ycbcr_base[:, 2:3]
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


