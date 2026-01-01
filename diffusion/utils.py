# nb_sim/diffusion/utils.py
from __future__ import annotations
import os, random
import numpy as np
import torch
from pathlib import Path

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        i = 0
        for p in model.parameters():
            if not p.requires_grad: continue
            self.shadow[i].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))
            i += 1

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        i = 0
        for p in model.parameters():
            if not p.requires_grad: continue
            p.data.copy_(self.shadow[i]); i += 1

def save_ckpt(path: str | Path, model, opt, epoch: int, step: int, ema: EMA | None = None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step}
    if ema is not None:
        obj["ema"] = [t.cpu() for t in ema.shadow]
        obj["ema_decay"] = ema.decay
    torch.save(obj, path); print(f"[ckpt] saved to {path}")

def load_ckpt_ema_only(path: str | Path, model, device: torch.device):
    obj = torch.load(path, map_location=device)
    model.load_state_dict(obj["model"])
    if "ema" in obj:
        ema_shadow = obj["ema"]
        i = 0
        with torch.no_grad():
            for p in model.parameters():
                if not p.requires_grad: continue
                p.data.copy_(ema_shadow[i].to(device)); i += 1
    print(f"[ckpt] loaded (EMA) from {path}")

def load_ckpt_raw(path: str | Path, model, device: torch.device, strict: bool = True):
    obj = torch.load(path, map_location=device)
    model.load_state_dict(obj["model"], strict=strict)
    print(f"[ckpt] loaded (RAW) from {path}")
    return obj