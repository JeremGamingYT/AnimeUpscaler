import os
import yaml
import cv2
import torch
import numpy as np
from rich.console import Console

from anime_upscaler.models.laesr import LAESR
from anime_upscaler.utils.ckpt import find_latest_checkpoint, load_model_from_checkpoint
from anime_upscaler.utils.tiling import (
    upscale_tensor_tiled,
    upscale_tensor_tiled_autoscale,
)


console = Console()


def main() -> None:
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    device = torch.device("cuda")

    inp = input("Chemin de la vidéo d\'entrée: ").strip().strip('"')
    if not os.path.isfile(inp):
        console.print("[red]Fichier vidéo introuvable.[/red]")
        return
    out_dir = os.path.join("outputs", "videos")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(inp))[0]
    out_path = os.path.join(out_dir, base + f"_x{conf['scale']}.mp4")

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        console.print("[red]Impossible d\'ouvrir la vidéo.[/red]")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = conf["scale"]
    out_w, out_h = w * scale, h * scale
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    model = LAESR(scale=scale).to(device).eval()
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
    use_half = bool(conf["infer"].get("half_precision", True))
    smooth = conf["infer"].get("temporal_smoothing", False)
    alpha = conf["infer"].get("smoothing_alpha", 0.7)
    post = conf["infer"].get("post_sharpen", {})
    preserve_chroma = bool(conf["infer"].get("preserve_chroma", True))
    prev_sr = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lr = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        lr = lr.unsqueeze(0).to(device)
        with torch.no_grad():
            if use_half:
                model.half()
                lr = lr.half()
            if autotile:
                sr = upscale_tensor_tiled_autoscale(model, lr, scale=scale, tile_size=tile, min_tile=min_tile, tile_overlap=overlap, device=device)
            else:
                sr = upscale_tensor_tiled(model, lr, scale=scale, tile_size=tile, tile_overlap=overlap, device=device)
            # Optional chroma preservation via bicubic baseline
            if preserve_chroma:
                import torch.nn.functional as F
                base_up = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
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
            # Optional post sharpen
            if post.get("enabled", False):
                from anime_upscaler.utils.tiling import edge_aware_sharpen
                sr = edge_aware_sharpen(
                    sr,
                    amount=float(post.get("amount", 0.3)),
                    radius=int(post.get("radius", 2)),
                    threshold=float(post.get("threshold", 0.0)),
                )
            sr_np = (sr.clamp(0, 1) * 255.0).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()

        if smooth and prev_sr is not None:
            # Simple temporal smoothing by alpha blend
            sr_np = (alpha * sr_np + (1 - alpha) * prev_sr).astype(np.uint8)

        prev_sr = sr_np
        bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

        frame_idx += 1
        if frame_idx % 30 == 0:
            console.print(f"Traitement frame {frame_idx}")

    writer.release()
    cap.release()
    console.print(f"[green]Vidéo enregistrée:[/green] {out_path}")


if __name__ == "__main__":
    main()


