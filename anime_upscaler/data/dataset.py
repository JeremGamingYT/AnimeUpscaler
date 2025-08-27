from __future__ import annotations

import os
import random
from typing import Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def list_images(directory: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files: List[str] = []
    if not os.path.isdir(directory):
        return files
    for root, _, fnames in os.walk(directory):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))
    return files


def random_crop_pair(hr: Image.Image, scale: int, crop_size: int) -> Tuple[Image.Image, Image.Image]:
    w, h = hr.size
    lr_crop = crop_size // scale
    if w < crop_size or h < crop_size:
        hr = hr.resize((max(crop_size, w), max(crop_size, h)), Image.BICUBIC)
        w, h = hr.size
    x = random.randint(0, w - crop_size)
    y = random.randint(0, h - crop_size)
    hr_patch = hr.crop((x, y, x + crop_size, y + crop_size))
    lr_patch = hr_patch.resize((lr_crop, lr_crop), Image.BICUBIC)
    return lr_patch, hr_patch


def degrade(lr_img: Image.Image, jpeg_q_range=(60, 95), noise_sigma_range=(0.0, 4.0), blur_kernel=7, blur_sigma_range=(0.2, 2.0)) -> Image.Image:
    # Apply blur (random isotropic or anisotropic approximation)
    if blur_kernel > 1:
        from PIL import ImageFilter
        sigma = random.uniform(*blur_sigma_range)
        if random.random() < 0.5:
            lr_img = lr_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        else:
            # Approximate anisotropic by sequential directional blurs
            lr_img = lr_img.filter(ImageFilter.GaussianBlur(radius=sigma * 0.5))
            lr_img = lr_img.filter(ImageFilter.BoxBlur(radius=max(0.1, sigma * 0.3)))
    # Add noise
    sigma_noise = random.uniform(*noise_sigma_range)
    if sigma_noise > 0:
        arr = np.array(lr_img).astype(np.float32)
        noise_g = np.random.normal(0, sigma_noise, arr.shape).astype(np.float32)
        # Poisson-like noise
        scale = max(1.0, 30.0 / 255.0)
        noise_p = np.random.poisson(arr * scale) / scale - arr
        noise = noise_g + 0.3 * noise_p
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        lr_img = Image.fromarray(arr)
    # Downscale (non-bicubic) then JPEG to simulate real degradations
    if random.random() < 0.7:
        w, h = lr_img.size
        scale = random.choice([0.5, 0.75])
        ds = lr_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), random.choice([Image.BILINEAR, Image.NEAREST]))
        lr_img = ds.resize((w, h), random.choice([Image.BILINEAR, Image.NEAREST]))

    # JPEG compression with randomized chroma subsampling via PIL internal
    q = random.randint(*jpeg_q_range)
    from io import BytesIO
    buff = BytesIO()
    lr_img.save(buff, format="JPEG", quality=q, subsampling=random.choice([0, 1, 2]))
    buff.seek(0)
    lr_img = Image.open(buff).convert("RGB")

    # Light banding/ringing approximation by slight posterization/dither
    if random.random() < 0.3:
        lr_img = lr_img.convert("P", palette=Image.ADAPTIVE, colors=random.choice([128, 192, 224])).convert("RGB")
    return lr_img


class AnimeSRDataset(Dataset):
    def __init__(self, hr_dir: str, scale_choices=(2, 3, 4), crop_size=192, jpeg_q_range=(60, 95), noise_sigma_range=(0.0, 4.0), blur_kernel=7, blur_sigma_range=(0.2, 2.0)):
        super().__init__()
        self.hr_paths = list_images(hr_dir)
        if len(self.hr_paths) == 0:
            raise RuntimeError(f"No images found in {hr_dir}.")
        self.scale_choices = scale_choices
        self.crop_size = crop_size
        self.jpeg_q_range = jpeg_q_range
        self.noise_sigma_range = noise_sigma_range
        self.blur_kernel = blur_kernel
        self.blur_sigma_range = blur_sigma_range

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int):
        hr = Image.open(self.hr_paths[idx]).convert("RGB")
        scale = random.choice(self.scale_choices)
        lr_patch, hr_patch = random_crop_pair(hr, scale=scale, crop_size=self.crop_size)
        lr_patch = degrade(lr_patch, self.jpeg_q_range, self.noise_sigma_range, self.blur_kernel, self.blur_sigma_range)

        # To tensor [0,1]
        lr_t = torch.from_numpy(np.array(lr_patch)).permute(2, 0, 1).float() / 255.0
        hr_t = torch.from_numpy(np.array(hr_patch)).permute(2, 0, 1).float() / 255.0
        return {"lr": lr_t, "hr": hr_t, "scale": scale}


class AnimeSRTileDataset(Dataset):
    """Dataset that iterates deterministically over all non-overlapping HR tiles.

    Generates LR/HR pairs by cropping the HR image into tiles of size crop_size,
    then downscaling each tile by a random scale from scale_choices.
    """

    def __init__(self, hr_dir: str, scale_choices=(2, 3, 4), crop_size=192, tile_stride: int | None = None,
                 jpeg_q_range=(60, 95), noise_sigma_range=(0.0, 4.0), blur_kernel=7, blur_sigma_range=(0.2, 2.0)):
        super().__init__()
        self.hr_paths = list_images(hr_dir)
        if len(self.hr_paths) == 0:
            raise RuntimeError(f"No images found in {hr_dir}.")
        self.scale_choices = scale_choices
        self.crop_size = crop_size
        self.tile_stride = tile_stride or crop_size
        self.jpeg_q_range = jpeg_q_range
        self.noise_sigma_range = noise_sigma_range
        self.blur_kernel = blur_kernel
        self.blur_sigma_range = blur_sigma_range

        # Precompute tile index: (img_idx, x, y)
        self.index: List[Tuple[int, int, int]] = []
        for idx, p in enumerate(self.hr_paths):
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            sx = self.tile_stride
            sy = self.tile_stride
            max_x = max(0, w - crop_size)
            max_y = max(0, h - crop_size)
            x_positions = list(range(0, max_x + 1, sx))
            y_positions = list(range(0, max_y + 1, sy))
            # Ensure last tile reaches the border
            if x_positions[-1] != max_x:
                x_positions.append(max_x)
            if y_positions[-1] != max_y:
                y_positions.append(max_y)
            for y in y_positions:
                for x in x_positions:
                    self.index.append((idx, x, y))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        img_idx, x, y = self.index[i]
        hr = Image.open(self.hr_paths[img_idx]).convert("RGB")
        crop_size = self.crop_size
        # Safe-guard in case of boundary issues
        x = min(max(0, x), max(0, hr.size[0] - crop_size))
        y = min(max(0, y), max(0, hr.size[1] - crop_size))
        hr_patch = hr.crop((x, y, x + crop_size, y + crop_size))
        scale = random.choice(self.scale_choices)
        lr_patch = hr_patch.resize((crop_size // scale, crop_size // scale), Image.BICUBIC)
        lr_patch = degrade(lr_patch, self.jpeg_q_range, self.noise_sigma_range, self.blur_kernel, self.blur_sigma_range)

        lr_t = torch.from_numpy(np.array(lr_patch)).permute(2, 0, 1).float() / 255.0
        hr_t = torch.from_numpy(np.array(hr_patch)).permute(2, 0, 1).float() / 255.0
        return {"lr": lr_t, "hr": hr_t, "scale": scale, "tile_xy": (x, y), "img_idx": img_idx}

