from __future__ import annotations

import os
from typing import Optional, Dict, Any

import torch


def find_latest_checkpoint(directory: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    ckpts = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".pt") or f.endswith(".pth")
    ]
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return ckpts[0]


def load_model_from_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> Dict[str, Any]:
    obj = torch.load(path, map_location=device)
    state = obj.get("ema") or obj.get("model") or obj
    if isinstance(state, dict):
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(obj, strict=False)
    return obj


