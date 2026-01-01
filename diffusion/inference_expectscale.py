# nb_sim/diffusion/inference.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F

from .config import DiffusionConfig
from .model import ModalEGNN
from .dataset import ModalPairsDataset, collate_batch
from .noise import alpha_sigma, t_from_logsnr
from .utils import load_ckpt_ema_only
from .adapters import save_traj


def _to_device(batch: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device) if v.dtype == torch.bool else v.to(device=device, dtype=dtype)
        else:
            out[k] = v
    return out


@torch.no_grad()
def _ddim_step_eps(
    model: ModalEGNN,
    z_t: torch.Tensor,            # [B, M]
    eigvals: torch.Tensor,        # [B, M]
    mode_mask: torch.Tensor,      # [B, M]
    t: torch.Tensor,              # [B]
    t_prev: torch.Tensor,         # [B]
    x: torch.Tensor,              # [B, N, 3]
    h: torch.Tensor,              # [B, N, Hf]
    node_mask: torch.Tensor,      # [B, N]
    edge_radius: float,           # scalar
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard ε-pred DDIM (η=0):
      ε̂ = model(...)
      x0̂ = (z_t − σ_t ε̂) / α_t
      z_{t′} = α_{t′} * x0̂ + σ_{t′} * ε̂
    """
    alpha_t, sigma_t = alpha_sigma(t)
    alpha_s, sigma_s = alpha_sigma(t_prev)

    # broadcast scalars to modal shape
    while alpha_t.dim() < z_t.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
        alpha_s = alpha_s.unsqueeze(-1)
        sigma_s = sigma_s.unsqueeze(-1)

    eps_hat = model(z_t, eigvals, mode_mask, t, x, h, node_mask, None, edge_radius)
    eps_hat = eps_hat * mode_mask

    x0_hat = ((z_t - sigma_t * eps_hat) / (alpha_t + 1e-8)) * mode_mask
    z_prev = (alpha_s * x0_hat + sigma_s * eps_hat) * mode_mask
    return z_prev, x0_hat, eps_hat


@torch.no_grad()
def sample_trajectories(cfg: DiffusionConfig, ckpt_path, out_dir, warm_start: bool = True):
    """
    Generate samples and save a multi-MODEL PDB per pair.
    warm_start=True: initialize z_t0 = α0*z* + σ0*ξ (useful for single-pair sanity).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    ds = ModalPairsDataset(
        cfg.pairs_json, cfg.n_modes, cfg.cache_dir, device, dtype,
        edge_radius=cfg.edge_radius,
    )

    # Model (EMA weights)
    model = ModalEGNN(
        node_feat_dim=3,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        time_emb_dim=cfg.time_dim,
    ).to(device)
    model.eval()
    load_ckpt_ema_only(ckpt_path, model, device)

    # -------- Schedule: high noise -> clean --------
    steps = cfg.sample_steps
    u_min, u_max = -4.0, +4.0
    u_seq  = torch.linspace(u_min, u_max, steps, device=device)  # increasing logSNR
    ts_seq = t_from_logsnr(u_seq)                                # decreases ~1 → 0
    ts     = torch.cat([ts_seq, ts_seq.new_tensor([0.0])])       # exact 0 at the end
    print(f"[sched] t0={float(ts[0]):.3f}  t_last={float(ts[-1]):.3f}  steps={steps}")

    for idx in range(len(ds)):
        item  = ds[idx]
        modes = ds.modesets[idx]
        frames: List[torch.Tensor] = []

        # one-item batch (reused)
        raw = collate_batch([item], cfg.edge_radius)
        raw = _to_device(raw, device, dtype)
        radius = float(raw["edge_radius"][0].item())
        mask   = raw["mode_mask"]
        z_star = raw["z0"]  # [1, M]

        for s_idx in range(cfg.n_samples_per_pair):
            batch = raw

            # --- init z at t≈1
            a0, s0 = alpha_sigma(ts[0])
            if warm_start:
                # sanity/debug path: start near the forward process of the *known* target
                z = (a0.view(1,1) * z_star + s0.view(1,1) * torch.randn_like(z_star)) * mask
            else:
                # pure noise init
                z = s0.view(1,1) * torch.randn_like(z_star) * mask

            last_x0 = None
            # DDIM: t ~ 1 → ... → 0
            for i in range(steps):
                t      = ts[i].expand(z.shape[0])
                t_prev = ts[i + 1].expand(z.shape[0])
                z, x0_hat, eps_hat = _ddim_step_eps(
                    model, z, batch["eigvals"], mask, t, t_prev,
                    batch["x"], batch["h"], batch["node_mask"], radius
                )
                last_x0 = x0_hat

                # light logging
                if i in (0, steps//3, 2*steps//3):
                    print(f"step {i:04d} t={float(t[0]):.3f} |x0_hat|={float(x0_hat.norm()):.2f}")

            # Final clean modal
            z0_sample = last_x0.squeeze(0)

            # Debug vs fitted target
            m = mask.squeeze(0).bool()
            zt = z0_sample[m]; zs = z_star.squeeze(0)[m]
            cos = float(F.cosine_similarity(zt, zs, dim=0))
            print(f"[dbg] warm={warm_start} |z0|={float(z0_sample.norm()):.3f} |z*|={float(z_star.norm()):.3f} cos={cos:.3f}")

            # Decode to Cartesian
            coords = modes.to_cartesian(z0_sample.to(dtype=modes.dtype, device=modes.device))
            frames.append(coords)

        # Save
        pair = ds.pairs[idx]
        apo_path = pair.apo if hasattr(pair, "apo") else pair["apo"]
        out_pair = out_dir / f"pair_{idx:05d}"
        out_pair.mkdir(parents=True, exist_ok=True)
        save_traj(
            frames=frames,
            mol=getattr(modes, "_m", modes).mol,
            out_path=out_pair / "samples.pdb",
            template_path=apo_path,
        )
        print(f"[sample] pair {idx} -> {out_pair/'samples.pdb'}")
