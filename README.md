AnimeUpscaler (LAESR) — Line-Aware Edge and Spectrum Super-Resolution

Overview

This project provides a complete GPU-first pipeline to train and run a novel anime upscaler named LAESR (Line-Aware Edge and Spectrum SR). LAESR focuses on preserving crisp line art and flat color regions characteristic of anime by combining three branches:

- Content branch: lightweight residual feature extractor
- Line branch: fixed Sobel edge detector with learned refinement to sharpen contours
- Spectrum branch: frequency-aware enhancement via learnable low/high-frequency separation

It includes:

- Interactive CLI (single command) to download sample data, train, and run inference on images and videos
- Mixed precision training, gradient accumulation, EMA, checkpointing, metrics (PSNR/SSIM)
- Tiled inference for large images
- Optional temporal smoothing for video

Hardware

- Target: NVIDIA RTX 5070 with 12GB VRAM. Training and inference are GPU-only; CPU is used minimally for I/O and video decoding.
- Windows 10/11 supported.

Install

1) Install PyTorch for your GPU (CUDA 12.1) on Windows (do NOT install CPU-only):

   - Follow the official selector at `https://pytorch.org/get-started/locally/` and choose: Windows, Pip, Python, CUDA 12.1.
   - Example (do not run here, copy to your terminal):

     pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

2) Install remaining dependencies:

     pip install -r requirements.txt

Usage (single command)

Run the interactive menu:

     python main.py

From the menu you can:

- Download a small safe sample dataset from Wikimedia Commons (CC BY/CC BY-SA/CC0)
- Train LAESR on your GPU using synthetic degradations
- Upscale images (with tiling) and videos (optional temporal smoothing)

Data

- Provided: `scripts/download_commons.py` fetches anime-style images from Wikimedia Commons under permissive licenses (CC BY, CC BY-SA, CC0). It also stores a `metadata.json` with attribution data (source URLs and authors). Please review licenses for redistribution.
- Own data: put your high-resolution anime images under `data/hr/`. The training pipeline will generate low-resolution inputs via realistic degradations.

Training

- Edit `configs/default.yaml` if needed.
- Start training from the menu, or via:

     python -m scripts.train --config configs/default.yaml

Outputs

- Checkpoints: `runs/checkpoints/`
- Logs/metrics: `runs/logs/`
- Samples: `runs/samples/`

Notes

- Perceptual loss (VGG19) is optional and enabled automatically if `torchvision` is available; otherwise training uses Charbonnier + edge consistency losses.
- Video I/O uses OpenCV; temporal smoothing with optical flow is optional.
- All heavy compute runs on GPU; if CUDA is not available, the CLI will warn and stop.

License

- Code: MIT
- Downloaded media: subject to original licenses; attribution saved in `data/commons/metadata.json`.


