import os
import math
import time
import yaml
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn

from anime_upscaler.models.laesr import LAESR
from anime_upscaler.models.patchgan import PatchDiscriminator
from anime_upscaler.data.dataset import AnimeSRDataset, AnimeSRTileDataset, list_images
from anime_upscaler.utils.metrics import psnr, ssim
from anime_upscaler.utils.ema import EMA
from anime_upscaler.utils.perceptual import LPIPSLike
from anime_upscaler.utils.losses import AntiBandingLoss, ChromaTVLoss, AntiHaloLoss
from anime_upscaler.utils.tiling import upscale_tensor_tiled
from anime_upscaler.utils.ckpt import load_model_from_checkpoint, find_latest_checkpoint


@dataclass
class TrainConfig:
    config_path: str = "configs/default.yaml"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def charbonnier_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((x - y) ** 2 + eps * eps).mean()


def quality_score(psnr_val: float, ssim_val: float) -> float:
    # Normalize PSNR roughly to [0,1] for 20–40 dB range
    psnr_n = max(0.0, min(1.0, (psnr_val - 20.0) / 20.0))
    return 0.5 * (psnr_n + float(ssim_val))


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def chroma_preservation_loss(sr: torch.Tensor, base_up: torch.Tensor) -> torch.Tensor:
    ycbcr_sr = rgb_to_ycbcr(sr)
    ycbcr_base = rgb_to_ycbcr(base_up)
    cb_sr, cr_sr = ycbcr_sr[:, 1:2], ycbcr_sr[:, 2:3]
    cb_b, cr_b = ycbcr_base[:, 1:2], ycbcr_base[:, 2:3]
    return (cb_sr - cb_b).abs().mean() + (cr_sr - cr_b).abs().mean()


def main(cfg: TrainConfig | None = None) -> None:
    if cfg is None:
        cfg = TrainConfig()
    conf = load_config(cfg.config_path)

    device = torch.device("cuda")
    torch.manual_seed(conf.get("seed", 42))
    cudnn.benchmark = True

    runs_dir = conf["paths"]["runs_dir"]
    ckpt_dir = conf["paths"]["checkpoints_dir"]
    logs_dir = conf["paths"]["logs_dir"]
    samples_dir = conf["paths"]["samples_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Model capacity from config
    model_cfg = conf.get("model", {})
    model = LAESR(
        scale=conf["scale"],
        base_channels=int(model_cfg.get("base_channels", 64)),
        num_blocks=int(model_cfg.get("num_blocks", 12)),
    ).to(device)
    # Safe compile: skip if Triton is unavailable to avoid runtime TritonMissing
    want_compile = bool(conf["train"].get("compile", False))
    if want_compile:
        triton_ok = True
        try:
            import triton  # type: ignore
            _ = triton.__version__
        except Exception:
            triton_ok = False
        if not triton_ok:
            print("[TorchCompile] Triton introuvable: torch.compile désactivé.")
        else:
            try:
                model = torch.compile(model)  # default inductor backend
                print("[TorchCompile] Modèle compilé.")
            except Exception as e:
                print(f"[TorchCompile] Échec compilation: {e}")
    try:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(f"[GPU] {gpu_name} | CUDA {torch.version.cuda} | Torch {torch.__version__} | VRAM {total_mem_gb:.1f} GB", flush=True)
    except Exception:
        print("[GPU] Impossible de lire les informations GPU, mais CUDA est activé.", flush=True)

    # Optional resume from checkpoint
    resume_from = conf.get("train", {}).get("resume_from", None)
    resume_latest = bool(conf.get("train", {}).get("resume_latest", False))
    if isinstance(resume_from, str) and len(resume_from) > 0 and os.path.isfile(resume_from):
        obj = load_model_from_checkpoint(model, resume_from, device)
        print(f"[RESUME] Modèle initialisé depuis: {resume_from}")
    elif resume_latest:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            obj = load_model_from_checkpoint(model, latest, device)
            print(f"[RESUME] Modèle initialisé depuis dernier checkpoint: {latest}")

    # Resolve training HR directory with fallback to commons images
    hr_dir = conf["data"]["hr_dir"]
    hr_list = list_images(hr_dir)
    if len(hr_list) == 0:
        commons_dir = os.path.join(conf["data"]["commons_dir"], "images")
        if os.path.isdir(commons_dir):
            alt_list = list_images(commons_dir)
            if len(alt_list) > 0:
                hr_dir = commons_dir
            else:
                raise RuntimeError(f"No images found in {hr_dir} or {commons_dir}.")
        else:
            raise RuntimeError(f"No images found in {hr_dir} and commons directory missing: {commons_dir}.")

    print(f"[DATA] Dossier HR utilisé: {hr_dir}", flush=True)
    if bool(conf["data"].get("use_tile_dataset", False)):
        dataset = AnimeSRTileDataset(
            hr_dir=hr_dir,
            scale_choices=tuple(conf["data"]["scale_choices"]),
            crop_size=conf["data"]["crop_size"],
            tile_stride=int(conf["data"].get("tile_stride", conf["data"]["crop_size"])),
            jpeg_q_range=tuple(conf["data"]["jpeg_q_range"]),
            noise_sigma_range=tuple(conf["data"]["noise_sigma_range"]),
            blur_kernel=conf["data"]["blur_kernel_size"],
            blur_sigma_range=tuple(conf["data"]["blur_sigma_range"]),
        )
        print(f"[DATA] Tile dataset activé: crop {conf['data']['crop_size']} stride {conf['data'].get('tile_stride', conf['data']['crop_size'])}")
    else:
        dataset = AnimeSRDataset(
            hr_dir=hr_dir,
            scale_choices=tuple(conf["data"]["scale_choices"]),
            crop_size=conf["data"]["crop_size"],
            jpeg_q_range=tuple(conf["data"]["jpeg_q_range"]),
            noise_sigma_range=tuple(conf["data"]["noise_sigma_range"]),
            blur_kernel=conf["data"]["blur_kernel_size"],
            blur_sigma_range=tuple(conf["data"]["blur_sigma_range"]),
        )
    loader = DataLoader(dataset, batch_size=conf["batch_size"], shuffle=True, num_workers=conf["num_workers"], pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=conf["optimizer"]["lr"], betas=tuple(conf["optimizer"]["betas"]), weight_decay=conf["optimizer"]["weight_decay"])
    use_gan = bool(conf["loss"].get("use_gan", False))
    if use_gan:
        disc = PatchDiscriminator().to(device)
        opt_d = torch.optim.Adam(disc.parameters(), lr=conf["optimizer"]["lr"] * 0.5, betas=tuple(conf["optimizer"]["betas"]))

    total_steps = conf["max_steps"]
    warmup = conf["scheduler"]["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    scaler = torch.amp.GradScaler("cuda", enabled=conf["train"]["mixed_precision"])
    ema = EMA(model, decay=conf["train"]["ema_decay"]) if conf["train"].get("ema", True) else None

    writer = SummaryWriter(log_dir=logs_dir)
    # Track simple in-training performance history for a final graph
    perf_steps: list[int] = []
    perf_psnr_list: list[float] = []
    perf_ssim_list: list[float] = []
    perc_loss = LPIPSLike()
    antiband = AntiBandingLoss(grad_threshold=float(conf["loss"].get("antibanding_grad_thr", 0.02)))
    chroma_tv = ChromaTVLoss()
    anti_halo = AntiHaloLoss(edge_threshold=float(conf["loss"].get("antihalo_edge_thr", 0.05)),
                             blur_kernel=int(conf["loss"].get("antihalo_blur", 5)))
    # Prepare one eval HR path for full-image validation
    eval_hr_path = None
    try:
        hr_candidates = list_images(hr_dir)
        if len(hr_candidates) > 0:
            eval_hr_path = hr_candidates[0]
    except Exception:
        pass

    step = 0
    model.train()
    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            gan_d_val = None

            with torch.amp.autocast("cuda", enabled=conf["train"]["mixed_precision"]):
                sr = model(lr)
                base = charbonnier_loss(sr, hr) * conf["loss"]["charbonnier_weight"]
                # Edge consistency using Sobel on outputs and targets
                from anime_upscaler.models.laesr import sobel_kernel
                k = sobel_kernel(device).to(sr.dtype).view(2, 1, 3, 3)
                def edges(x):
                    gray = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
                    return torch.nn.functional.conv2d(gray, k, padding=1)
                e_loss = charbonnier_loss(edges(sr), edges(hr)) * conf["loss"]["edge_weight"]
                p_loss = perc_loss(sr, hr) * conf["loss"].get("perceptual_weight", 0.0)
                ab_loss = antiband(sr) * float(conf["loss"].get("antibanding_weight", 0.0))
                ctv_loss = chroma_tv(sr) * float(conf["loss"].get("chroma_tv_weight", 0.0))
                ah_loss = anti_halo(sr) * float(conf["loss"].get("antihalo_weight", 0.0))
                # Bicubic baseline upsample for reference chroma
                with torch.no_grad():
                    base_up = F.interpolate(lr, scale_factor=int(conf["scale"]), mode="bicubic", align_corners=False)
                chroma_keep = chroma_preservation_loss(sr, base_up) * float(conf["loss"].get("chroma_keep_weight", 0.0))
                loss = base + e_loss + p_loss + ab_loss + ctv_loss + ah_loss + chroma_keep

            if use_gan and step >= conf["loss"].get("gan_warmup_steps", 0):
                # Discriminator update (R1 regularization)
                disc.train()
                for p in disc.parameters():
                    p.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=conf["train"]["mixed_precision"]):
                    pred_real = disc(hr)
                    pred_fake = disc(sr.detach())
                    # Hinge loss
                    loss_d = torch.relu(1.0 - pred_real).mean() + torch.relu(1.0 + pred_fake).mean()
                    # R1 penalty on real
                    if conf["loss"].get("r1_gamma", 0.0) > 0:
                        hr.requires_grad_(True)
                        pred_real_r = disc(hr)
                        grads = torch.autograd.grad(outputs=pred_real_r.sum(), inputs=hr, create_graph=True)[0]
                        r1 = grads.pow(2).reshape(grads.size(0), -1).sum(1).mean() * (conf["loss"]["r1_gamma"] * 0.5)
                        loss_d = loss_d + r1
                        hr = hr.detach()
                scaler.scale(loss_d).backward()
                scaler.step(opt_d)
                gan_d_val = float(loss_d.item())

                # Generator adversarial term
                with torch.amp.autocast("cuda", enabled=conf["train"]["mixed_precision"]):
                    pred_fake_g = disc(sr)
                    loss_gan = -pred_fake_g.mean() * conf["loss"].get("gan_weight", 0.0)
                    loss = loss + loss_gan

            scaler.scale(loss).backward()
            if conf["train"]["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

            if ema is not None:
                ema.update(model)

            if step % 20 == 0:
                with torch.no_grad():
                    p = psnr(sr.detach(), hr)
                    s = ssim(sr.detach(), hr)
                    # Baseline bicubic metrics for reference
                    base_up_m = F.interpolate(lr.detach(), scale_factor=int(conf["scale"]), mode="bicubic", align_corners=False)
                    pb = psnr(base_up_m, hr)
                    sb = ssim(base_up_m, hr)
                writer.add_scalar("loss/train", loss.item(), step)
                writer.add_scalar("metrics/psnr", p.item(), step)
                writer.add_scalar("metrics/ssim", s.item(), step)
                writer.add_scalar("metrics_bicubic/psnr", pb.item(), step)
                writer.add_scalar("metrics_bicubic/ssim", sb.item(), step)
                perf_steps.append(step)
                perf_psnr_list.append(float(p.item()))
                perf_ssim_list.append(float(s.item()))
                if use_gan and gan_d_val is not None:
                    writer.add_scalar("loss/gan_d", gan_d_val, step)
                try:
                    mem_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"step {step}/{total_steps} | loss {loss.item():.4f} | PSNR {p.item():.2f} | mem {mem_mb:.0f} MB | lr {lr_now:.2e}", flush=True)
                except Exception:
                    pass

            # Periodic full-image tiled validation and sample export
            if eval_hr_path is not None and conf.get("val_interval", 0) > 0 and step % conf["val_interval"] == 0 and step > 0:
                try:
                    from PIL import Image as _Image
                    model.eval()
                    hr_img = _Image.open(eval_hr_path).convert("RGB")
                    w, h = hr_img.size
                    sc = int(conf["scale"])
                    lr_img = hr_img.resize((max(1, w // sc), max(1, h // sc)), _Image.BICUBIC)
                    # To tensors
                    lr_t = torch.from_numpy(np.array(lr_img)).permute(2, 0, 1).float() / 255.0
                    lr_t = lr_t.unsqueeze(0).to(device)
                    hr_t = torch.from_numpy(np.array(hr_img)).permute(2, 0, 1).float() / 255.0
                    hr_t = hr_t.unsqueeze(0).to(device)
                    tile = int(conf["infer"].get("tile_size", 256))
                    overlap = int(conf["infer"].get("tile_overlap", 16))
                    with torch.no_grad():
                        sr_t = upscale_tensor_tiled(model, lr_t, scale=sc, tile_size=tile, tile_overlap=overlap, device=device)
                        ps = conf["infer"].get("post_sharpen", {})
                        if ps.get("enabled", False):
                            from anime_upscaler.utils.tiling import edge_aware_sharpen
                            sr_t = edge_aware_sharpen(
                                sr_t,
                                amount=float(ps.get("amount", 0.3)),
                                radius=int(ps.get("radius", 3)),
                                threshold=float(ps.get("threshold", 0.0)),
                            )
                    # Metrics
                    with torch.no_grad():
                        pf = psnr(sr_t, hr_t).item()
                        sf = ssim(sr_t, hr_t).item()
                    writer.add_scalar("metrics_full/psnr", pf, step)
                    writer.add_scalar("metrics_full/ssim", sf, step)
                    # Save visual
                    lr_up = np.array(lr_img.resize((w, h), _Image.BICUBIC))
                    sr_img = (sr_t.clamp(0, 1) * 255.0).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
                    hr_np = np.array(hr_img)
                    cat = np.concatenate([lr_up, sr_img, hr_np], axis=1)
                    os.makedirs(samples_dir, exist_ok=True)
                    out_path = os.path.join(samples_dir, f"sample_full_step{step}.png")
                    _Image.fromarray(cat).save(out_path)
                    print(f"[VAL] step {step} | full PSNR {pf:.2f} SSIM {sf:.3f} | saved {out_path}", flush=True)
                finally:
                    model.train()

            if step % conf["save_interval"] == 0 and step > 0:
                path = os.path.join(ckpt_dir, f"laesr_step{step}.pt")
                save_obj = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
                if ema is not None:
                    save_obj["ema"] = ema.shadow
                torch.save(save_obj, path)

            step += 1
            if step >= total_steps:
                break

    writer.close()

    # Save final checkpoint regardless of save_interval
    path_last = os.path.join(ckpt_dir, "laesr_last.pt")
    save_obj = {"model": model.state_dict(), "step": step}
    if ema is not None:
        save_obj["ema"] = ema.shadow
    torch.save(save_obj, path_last)
    print(f"[CKPT] Sauvegardé: {path_last}", flush=True)

    # Export a quick sample triplet (LR upscaled bicubic, SR, HR)
    try:
        os.makedirs(samples_dir, exist_ok=True)
        # Full-image final export on a real HR image
        if eval_hr_path is not None:
            model.eval()
            hr_img = Image.open(eval_hr_path).convert("RGB")
            w, h = hr_img.size
            sc = int(conf["scale"])
            lr_img = hr_img.resize((max(1, w // sc), max(1, h // sc)), Image.BICUBIC)
            lr_t = torch.from_numpy(np.array(lr_img)).permute(2, 0, 1).float() / 255.0
            lr_t = lr_t.unsqueeze(0).to(device)
            tile = int(conf["infer"].get("tile_size", 256))
            overlap = int(conf["infer"].get("tile_overlap", 16))
            with torch.no_grad():
                sr_t = upscale_tensor_tiled(model, lr_t, scale=sc, tile_size=tile, tile_overlap=overlap, device=device)
                ps = conf["infer"].get("post_sharpen", {})
                if ps.get("enabled", False):
                    from anime_upscaler.utils.tiling import edge_aware_sharpen
                    sr_t = edge_aware_sharpen(
                        sr_t,
                        amount=float(ps.get("amount", 0.3)),
                        radius=int(ps.get("radius", 3)),
                        threshold=float(ps.get("threshold", 0.0)),
                    )
            lr_up = np.array(lr_img.resize((w, h), Image.BICUBIC))
            sr_img = (sr_t.clamp(0, 1) * 255.0).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
            hr_np = np.array(hr_img)
            cat = np.concatenate([lr_up, sr_img, hr_np], axis=1)
            sample_path = os.path.join(samples_dir, "sample_full_last.png")
            Image.fromarray(cat).save(sample_path)
            print(f"[SAMPLE] Exporté: {sample_path}", flush=True)
        else:
            # Fallback: keep the patch-based sample
            sample = dataset[0]
            lr_s = sample["lr"].unsqueeze(0).to(device)
            hr_s = sample["hr"].to(device)
            model.eval()
            with torch.no_grad():
                sr_s = model(lr_s).squeeze(0).clamp(0, 1).cpu()
            lr_img = (sample["lr"].clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
            hr_img = (hr_s.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
            sr_img = (sr_s * 255.0).byte().permute(1, 2, 0).numpy()
            lr_up = np.array(Image.fromarray(lr_img).resize((hr_img.shape[1], hr_img.shape[0]), Image.BICUBIC))
            cat = np.concatenate([lr_up, sr_img, hr_img], axis=1)
            sample_path = os.path.join(samples_dir, "sample_last.png")
            Image.fromarray(cat).save(sample_path)
            print(f"[SAMPLE] Exporté: {sample_path}", flush=True)
    except Exception as e:
        print(f"[SAMPLE] Échec de l'export d'exemple: {e}", flush=True)

    # Simple end-of-training graph: performance over steps
    try:
        import json
        import matplotlib.pyplot as plt
        # Read TensorBoard scalars is non-trivial; instead, infer from saved logs is complex.
        # For simplicity, we create a tiny CSV of last N evaluations during training could be stored.
        # If not available, we at least create a placeholder with final PSNR/SSIM measured just now on eval_hr_path.
        perf_png = os.path.join(samples_dir, "training_performance.png")
        # If we have history, plot it
        if len(perf_steps) > 0:
            qs = [quality_score(p, s) for p, s in zip(perf_psnr_list, perf_ssim_list)]
            plt.figure(figsize=(7, 4))
            plt.plot(perf_steps, qs, label="Qualité", color="#f28e2b")
            plt.plot(perf_steps, perf_psnr_list, label="PSNR(dB)", color="#4e79a7", alpha=0.6)
            plt.plot(perf_steps, perf_ssim_list, label="SSIM", color="#59a14f", alpha=0.6)
            plt.xlabel("Steps")
            plt.ylabel("Score")
            plt.title("Performance pendant l'entraînement")
            plt.legend()
            plt.tight_layout()
            plt.savefig(perf_png)
            print(f"[GRAPH] Enregistré: {perf_png}")
            plt.close()
            return
        # Compute final full-image metrics if eval_hr_path exists
        if eval_hr_path is not None:
            from PIL import Image as _Image
            model.eval()
            hr_img = _Image.open(eval_hr_path).convert("RGB")
            w, h = hr_img.size
            sc = int(conf["scale"])
            lr_img = hr_img.resize((max(1, w // sc), max(1, h // sc)), _Image.BICUBIC)
            lr_t = torch.from_numpy(np.array(lr_img)).permute(2, 0, 1).float() / 255.0
            lr_t = lr_t.unsqueeze(0).to(device)
            with torch.no_grad():
                sr_t = upscale_tensor_tiled(model, lr_t, scale=sc, tile_size=int(conf["infer"].get("tile_size", 256)), tile_overlap=int(conf["infer"].get("tile_overlap", 16)), device=device)
            hr_t = torch.from_numpy(np.array(hr_img)).permute(2, 0, 1).float() / 255.0
            hr_t = hr_t.unsqueeze(0).to(device)
            final_psnr = psnr(sr_t, hr_t).item()
            final_ssim = ssim(sr_t, hr_t).item()
        else:
            final_psnr = 0.0
            final_ssim = 0.0
        final_quality = quality_score(final_psnr, final_ssim)
        # Make a very simple bar chart: PSNR, SSIM, Quality
        names = ["PSNR(dB)", "SSIM", "Qualité"]
        vals = [final_psnr, final_ssim, final_quality]
        plt.figure(figsize=(6, 4))
        bars = plt.bar(names, vals, color=["#4e79a7", "#59a14f", "#f28e2b"])
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        plt.ylim(0, max(40.0, max(vals) * 1.2))
        plt.title("Performance finale (approx.)")
        plt.tight_layout()
        plt.savefig(perf_png)
        print(f"[GRAPH] Enregistré: {perf_png}")
    except Exception as e:
        print(f"[GRAPH] Impossible de générer le graph de performance: {e}")


if __name__ == "__main__":
    main()


