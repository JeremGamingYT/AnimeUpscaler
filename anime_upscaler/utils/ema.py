from __future__ import annotations

import copy
from typing import Iterator

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_avg = (1.0 - self.decay) * param.detach() + self.decay * self.shadow[name]
            self.shadow[name] = new_avg.clone()

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])


