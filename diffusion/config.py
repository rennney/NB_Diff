# nb_sim/diffusion/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class DiffusionConfig:
    # Data
    pairs_json: str = "pairs.json"           # [{"apo": "...", "holo": "..."}]
    out_dir: str = "runs/diff_egnn"
    n_modes: int = 64
    cache_dir: str = "runs/cache"

    # Block graph construction
    edge_radius: float = 30.0                # Å: connect block COMs within this distance

    # Model
    hidden_dim: int = 256
    n_layers: int = 6
    time_dim: int = 128
    dropout: float = 0.0

    # Noise schedule
    sigma_min: float = 1e-3
    sigma_max: float = 1.0

    # Train
    epochs: int = 200
    batch_size: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.01
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0
    amp: bool = True
    log_interval: int = 50
    ckpt_interval: int = 500
    t_repeats: int = 8        # <-- number of random t’s per batch step

    
    # Loss
    cartesian_loss_weight: float = 0.1    # λ_cart
    cartesian_variant: str = "x0"         # "x0" or "t"

    # Sample
    n_samples_per_pair: int = 5
    sample_steps: int = 200
    sampler: str = "ddim"  # "ddim" or "pf_ode"

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
