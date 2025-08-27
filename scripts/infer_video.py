import os
import yaml
import cv2
import torch
import numpy as np
from rich.console import Console

from anime_upscaler.models.laesr import LAESR
from anime_upscaler.utils.ckpt import find_latest_checkpoint, load_model_from_checkpoint
from anime_upscaler.utils.tiling import upscale_tensor_tiled


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
    smooth = conf["infer"].get("temporal_smoothing", False)
    alpha = conf["infer"].get("smoothing_alpha", 0.7)
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
            sr = upscale_tensor_tiled(model, lr, scale=scale, tile_size=tile, tile_overlap=overlap, device=device)
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


