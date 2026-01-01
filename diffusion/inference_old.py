from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F

import math
from .config import DiffusionConfig
from .model import ModalEGNN
from .dataset import ModalPairsDataset, collate_batch
from .noise import alpha_sigma_from_t, t_from_logsnr, logsnr_from_t, alpha_sigma_from_u
from .utils import load_ckpt_ema_only, load_ckpt_raw
from .adapters import save_traj


def _to_device(batch: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device) if v.dtype == torch.bool else v.to(device=device, dtype=dtype)
        else:
            out[k] = v
    return out

def make_ts_from_rstart(r_start: float,
                        t_end: float,
                        steps: int,
                        device=None,
                        dtype=None) -> torch.Tensor:
    """
    Build a DDIM t-grid by starting at a target noise ratio r_start = σ/α
    and stepping uniformly in logSNR to t_end. Robust to bad dtype/device.
    """
    # ---- dtype/device guards ----
    if dtype is None or dtype is torch.dtype or not isinstance(dtype, torch.dtype):
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")

    # u = log(α^2 / σ^2);  r = σ/α = exp(-u/2) ⇒ u = -2 * log(r)
    u0 = -2.0 * math.log(max(float(r_start), 1e-6))

    t_end_t = torch.tensor([float(t_end)], device=device, dtype=dtype)
    a_end, s_end = alpha_sigma_from_t(t_end_t)
    u1 = torch.log((a_end**2) / (s_end**2 + 1e-12)).item()

    us = torch.linspace(u0, u1, steps + 1, device=device, dtype=dtype)
    ts = t_from_logsnr(us)  # uses the same schedule as training
    return ts


def make_u_grid_for_ddim(steps: int,
                         t_start: float = 0.99,
                         t_end: float   = 0.01,
                         u_train_min: float = -8.3,
                         u_train_max: float = 7.5,
                         device=None, dtype=None) -> torch.Tensor:
    """
    Return a length-(steps+1) tensor of u values, linear in u, clamped to the u-range used in training.
    """
    # map requested t-range to u-range with the same cosine schedule as training (s=0.008 inside logsnr_from_t)
    u0 = logsnr_from_t(torch.tensor(t_start, device=device, dtype=dtype))
    u1 = logsnr_from_t(torch.tensor(t_end,   device=device, dtype=dtype))
    # clamp to training coverage
    u0 = torch.clamp(u0, min=u_train_min, max=u_train_max)
    u1 = torch.clamp(u1, min=u_train_min, max=u_train_max)
    # build a monotone grid matching the direction from t_start→t_end
    return torch.linspace(u0.item(), u1.item(), steps + 1, device=device, dtype=dtype)

def make_t_grid_from_u(u_grid: torch.Tensor) -> torch.Tensor:
    """Elementwise invert u→t using the same schedule as training."""
    return t_from_logsnr(u_grid)


@torch.no_grad()
def _ddim_step_v(
    model, z_t, eigvals, mode_mask, t, t_prev,
    x, h, node_mask, edge_radius
):
    alpha_t, sigma_t = alpha_sigma_from_t(t)
    alpha_s, sigma_s = alpha_sigma_from_t(t_prev)
    while alpha_t.dim() < z_t.dim():
        alpha_t = alpha_t.unsqueeze(-1); sigma_t = sigma_t.unsqueeze(-1)
        alpha_s = alpha_s.unsqueeze(-1); sigma_s = sigma_s.unsqueeze(-1)

    m = mode_mask
    # IMPORTANT: don't pre-mask v_pred – mirror training 
    v_pred = model(z_t, eigvals, m, t, x, h, node_mask, None, edge_radius)

    denom = (alpha_t.pow(2) + sigma_t.pow(2)) + 1e-8  # ~1.0
    x0_hat  = (alpha_t * z_t - sigma_t * v_pred) / denom
    eps_hat = (alpha_t * v_pred + sigma_t * z_t) / denom

    # Only mask what you carry forward
    z_prev = (alpha_s * x0_hat + sigma_s * eps_hat) * m
    return z_prev, x0_hat, v_pred

@torch.no_grad()
def _ddim_step_v_stochastic(
    model, z_t, eigvals, mode_mask, t, t_prev,
    x, h, node_mask, edge_radius, *, eta: float = 0.2,
    clip_v_std: float | None = None
):
    """
    Variance-preserving stochastic DDIM for v-pred.
      z_{s} = α_s x0̂ + σ_s ( √(1-η^2) ε̂ + η ξ ),  ξ~N(0,I)
    Notes:
      • Do NOT pre-mask v_pred (match training). Only mask z that you carry forward.
      • eta ∈ [0,1]. Start small (0.05–0.3).
      • Optional: clip v_pred's global std to avoid explosions on OOD latents.
    """
    alpha_t, sigma_t = alpha_sigma_from_t(t)
    alpha_s, sigma_s = alpha_sigma_from_t(t_prev)

    while alpha_t.dim() < z_t.dim():
        alpha_t = alpha_t.unsqueeze(-1); sigma_t = sigma_t.unsqueeze(-1)
        alpha_s = alpha_s.unsqueeze(-1); sigma_s = sigma_s.unsqueeze(-1)

    m = mode_mask

    # v̂(z_t, t | apo)
    v_pred = model(z_t, eigvals, m, t, x, h, node_mask, None, edge_radius)

    # (optional) tame outliers if you see occasional spikes
    if clip_v_std is not None and clip_v_std > 0:
        v_std = torch.sqrt(((v_pred * m)**2).sum() / m.sum().clamp_min(1))
        scale = (clip_v_std / (v_std + 1e-8)).clamp(max=1.0)
        v_pred = v_pred * scale

    denom = (alpha_t.pow(2) + sigma_t.pow(2)).clamp_min(1.0 - 1e-8)  # ~1.0
    x0_hat  = (alpha_t * z_t - sigma_t * v_pred) / denom
    eps_hat = (alpha_t * v_pred + sigma_t * z_t) / denom

    # variance-preserving mix of predicted noise and fresh noise
    eta = float(max(0.0, min(1.0, eta)))
    coeff_pred = math.sqrt(max(0.0, 1.0 - eta * eta))
    noise = torch.randn_like(z_t)

    z_prev = (alpha_s * x0_hat + sigma_s * (coeff_pred * eps_hat + eta * noise)) * m

    # Guards against numerical issues
    if not torch.isfinite(z_prev).all():
        # Soft fallback: shrink eta and sanitize
        z_prev = (alpha_s * x0_hat + sigma_s * (0.95 * eps_hat + 0.312 * noise)) * m
        z_prev = torch.nan_to_num(z_prev, nan=0.0, posinf=1e6, neginf=-1e6)

    return z_prev, x0_hat, v_pred

@torch.no_grad()
def sanity_mid_t(model, raw, t_scalar=0.5, trials=8):
    dev = raw['z0'].device
    t = torch.full((1,), t_scalar, device=dev)
    a,s = alpha_sigma_from_t(t); a=a.view(1,1); s=s.view(1,1)
    errs=[]
    for _ in range(trials):
        eps = torch.randn_like(raw['z0'])
        zt  = (a*raw['z0'] + s*eps)
        vgt = (a*eps - s*raw['z0'])
        vph = model(zt, raw['eigvals'], raw['mode_mask'], t,
                    raw['x'], raw['h'], raw['node_mask'], None,
                    float(raw['edge_radius'][0].item()))
        x0h = (a*zt - s*vph) / ((a*a + s*s)+1e-8)
        mse = (((x0h - raw['z0']) * raw['mode_mask'])**2).sum() / raw['mode_mask'].sum().clamp_min(1)
        errs.append(float(mse))
    print(f"[sanity] t={t_scalar}  mean z0̂-MSE={sum(errs)/len(errs):.4f}")

@torch.no_grad()
def probe_exact(model, raw):
    from .losses import v_probe_metrics
    # IMPORTANT: mimic training-time probe mode for normalization layers
    was_training = model.training
    model.train()    # use BatchNorm with batch stats (training did not call eval())
    t_mid = torch.full((raw["z0"].shape[0],), 0.5, device=raw["z0"].device)
    rep = v_probe_metrics(model, raw, t_mid)
    if not was_training:
        model.eval()
    print(f"[probe(train-mode)] z0_hat@t=0.5: MSE={rep['mse']:.4f} MAE={rep['mae']:.4f} "
          f"corr={rep['corr']:.3f}")
    print(f"         stds: eps={rep['eps_std']:.3f} eps_hat={rep['eps_hat_std']:.3f} "
          f"v_pred={rep['v_pred_std']:.3f} v_tgt={rep['v_tgt_std']:.3f}")
    return rep

@torch.no_grad()
def _rmsd(P: torch.Tensor, Q: torch.Tensor) -> float:
    """Kabsch-aligned RMSD between two [N,3] tensors (on same device)."""
    Pc = P - P.mean(dim=0, keepdim=True); Qc = Q - Q.mean(dim=0, keepdim=True)
    H = Pc.t() @ Qc
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.t() @ U.t()
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.t() @ U.t()
    P_aligned = Pc @ R
    return float(torch.sqrt(((P_aligned - Qc)**2).mean()).item())

@torch.no_grad()
def _ddim_step_x0(
    model, z_t, eigvals, mode_mask, t, t_prev,
    x, h, node_mask, edge_radius, eta: float = 0.0,
):
    """
    DDIM step consistent with an x0-prediction model.

    Update:
      x0_hat  = model(z_t, ...)                 # clean prediction
      eps_hat = (z_t - α_t * x0_hat) / σ_t
      z_{t'}  = α_{t'} * x0_hat + σ_{t'} * eps_hat              (η = 0)
              = α_{t'} * x0_hat
                + (√(1-η^2) * σ_{t'}) * eps_hat + (η * σ_{t'}) * ξ   (η ∈ [0,1])
    Only the *state* you carry forward is masked; don’t pre-mask predictions.
    """
    alpha_t, sigma_t = alpha_sigma_from_t(t)
    alpha_s, sigma_s = alpha_sigma_from_t(t_prev)
    while alpha_t.dim() < z_t.dim():
        alpha_t = alpha_t.unsqueeze(-1); sigma_t = sigma_t.unsqueeze(-1)
        alpha_s = alpha_s.unsqueeze(-1); sigma_s = sigma_s.unsqueeze(-1)

    m = mode_mask

    # 1) Predict clean sample directly (mirror training: no pre-mask here)
    x0_hat = model(z_t, eigvals, m, t, x, h, node_mask, None, edge_radius)

    # 2) Derive score/noise estimate from x0_hat
    eps_hat = (z_t - alpha_t * x0_hat) / (sigma_t + 1e-8)

    # 3) Deterministic (η=0) or stochastic (η>0) DDIM update
    if eta and float(eta) > 0.0:
        # Split σ_s between deterministic piece and fresh noise
        c = torch.as_tensor(float(eta), device=z_t.device, dtype=z_t.dtype)
        det = sigma_s * torch.sqrt(torch.clamp(1.0 - c * c, min=0.0))
        stc = sigma_s * c
        xi  = torch.randn_like(z_t)
        z_prev = (alpha_s * x0_hat + det * eps_hat + stc * xi) * m
    else:
        z_prev = (alpha_s * x0_hat + sigma_s * eps_hat) * m

    return z_prev, x0_hat, eps_hat

@torch.no_grad()
def _ddim_step_eps(
    model, z_t, eigvals, mode_mask, t, t_prev,
    x, h, node_mask, edge_radius, eta: float = 0.0,
):
    """
    DDIM step consistent with an ε-prediction model.

    Model semantics:
      eps_hat = model(z_t, ...)               # score/noise prediction (unit-ish scale)
      x0_hat  = (z_t - σ_t * eps_hat) / α_t   # clean reconstruction implied by eps_hat

    Update (η ∈ [0,1]):
      z_{t'}  = α_{t'} * x0_hat
                + (√(1-η^2) * σ_{t'}) * eps_hat + (η * σ_{t'}) * ξ
      Deterministic DDIM is η = 0.

    Notes:
      • Keep α,σ exactly consistent with training’s schedule.
      • Only the *state* you carry forward is masked; don’t pre-mask predictions.
    """
    # α, σ for current and previous times
    alpha_t, sigma_t = alpha_sigma_from_t(t)
    alpha_s, sigma_s = alpha_sigma_from_t(t_prev)
    while alpha_t.dim() < z_t.dim():
        alpha_t = alpha_t.unsqueeze(-1); sigma_t = sigma_t.unsqueeze(-1)
        alpha_s = alpha_s.unsqueeze(-1); sigma_s = sigma_s.unsqueeze(-1)

    m = mode_mask
    # numerical guard for division by tiny α at very high t
    alpha_safe = alpha_t.clamp_min(1e-5)
    sigma_safe = sigma_t.clamp_min(1e-8)

    # 1) Predict ε̂ (do NOT pre-mask)
    eps_hat = model(z_t, eigvals, m, t, x, h, node_mask, None, edge_radius)

    # 2) Clean estimate implied by ε̂
    x0_hat = (z_t - sigma_safe * eps_hat) / alpha_safe

    # 3) DDIM update (deterministic if eta == 0)
    if eta and float(eta) > 0.0:
        c = torch.as_tensor(float(eta), device=z_t.device, dtype=z_t.dtype)
        det = sigma_s * torch.sqrt(torch.clamp(1.0 - c * c, min=0.0))
        stc = sigma_s * c
        xi  = torch.randn_like(z_t)
        z_prev = (alpha_s * x0_hat + det * eps_hat + stc * xi) * m
    else:
        z_prev = (alpha_s * x0_hat + sigma_s * eps_hat) * m

    return z_prev, x0_hat, eps_hat


@torch.no_grad()
def _ddim_step_eps_v2(
    model, z_t, eigvals, mode_mask, t, t_prev,
    x, h, node_mask, edge_radius, eta: float = 0.0,
):
    # α,σ for current and next time
    alpha_t, sigma_t = alpha_sigma_from_t(t)
    alpha_s, sigma_s = alpha_sigma_from_t(t_prev)
    while alpha_t.dim() < z_t.dim():
        alpha_t = alpha_t.unsqueeze(-1); sigma_t = sigma_t.unsqueeze(-1)
        alpha_s = alpha_s.unsqueeze(-1); sigma_s = sigma_s.unsqueeze(-1)

    m = mode_mask

    #print("current t : ", (sigma_t/alpha_t)[:5])
    #print("previous t : ", (sigma_s/alpha_s)[:5])

    # 1) predict ε̂ (do NOT pre-mask)
    eps_hat = model(z_t, eigvals, m, t, x, h, node_mask, None, edge_radius)

    # 2) state update via ratio form (no explicit division by tiny α)
    #    z_{t'} = (α_{t'}/α_t) * (z_t - σ_t ε̂) + √(1-η^2) σ_{t'} ε̂ + η σ_{t'} ξ
    ratio = (alpha_s / alpha_t)
    base  = ratio * (z_t - sigma_t * eps_hat)
    if eta and float(eta) > 0.0:
        c   = torch.as_tensor(float(eta), device=z_t.device, dtype=z_t.dtype)
        det = sigma_s * torch.sqrt(torch.clamp(1.0 - c * c, min=0.0))
        stc = sigma_s * c
        xi  = torch.randn_like(z_t)
        z_prev = (base + det * eps_hat + stc * xi) * m
    else:
        z_prev = (base + sigma_s * eps_hat) * m

    # 3) you can still compute x0̂ for logging only (safe clamp here is fine)
    alpha_safe = alpha_t.clamp_min(1e-5); sigma_safe = sigma_t.clamp_min(1e-8)
    x0_hat = (z_t - sigma_safe * eps_hat) / alpha_safe

    # diagnostics (catches schedule mismatches)
    recon = alpha_t * x0_hat + sigma_t * eps_hat
    resid = (((recon - z_t)**2).sum() / (z_t.pow(2).sum() + 1e-12)).item()
    return z_prev, x0_hat, eps_hat #, resid, float((sigma_t/alpha_t).mean().item())

@torch.no_grad()
def _ddim_step_eps_u(
    model, z_t, eigvals, mode_mask, u, u_prev,
    x, h, node_mask, edge_radius, eta: float = 0.0,
):
    """
    DDIM step for an ε-prediction model using *u = logSNR* for α,σ.
    The model still takes t (same mapping used during training).
    """
    # α,σ from u (training semantics guarantee α^2+σ^2=1)
    alpha_t, sigma_t = alpha_sigma_from_u(u)
    alpha_s, sigma_s = alpha_sigma_from_u(u_prev)
    while alpha_t.dim() < z_t.dim():
        alpha_t = alpha_t.unsqueeze(-1); sigma_t = sigma_t.unsqueeze(-1)
        alpha_s = alpha_s.unsqueeze(-1); sigma_s = sigma_s.unsqueeze(-1)

    m = mode_mask

    # Model time input must be the same mapping used in training
    t = make_t_grid_from_u(u)

    # 1) predict ε̂ (do NOT pre-mask)
    eps_hat = model(z_t, eigvals, m, t, x, h, node_mask, None, edge_radius) * 0.8

    # 2) implied clean estimate (guard tiny α)
    alpha_safe = alpha_t.clamp_min(1e-5)
    sigma_safe = sigma_t.clamp_min(1e-8)
    x0_hat = (z_t - sigma_safe * eps_hat) / alpha_safe

    # 3) DDIM update (deterministic if eta=0)
    if eta and float(eta) > 0.0:
        c = torch.as_tensor(float(eta), device=z_t.device, dtype=z_t.dtype)
        det = sigma_s * torch.sqrt(torch.clamp(1.0 - c * c, min=0.0))
        xi  = torch.randn_like(z_t)
        z_prev = (alpha_s * x0_hat + det * eps_hat + sigma_s * c * xi) * m
    else:
        z_prev = (alpha_s * x0_hat + sigma_s * eps_hat) * m

    # reconstruction sanity: ||α_t x0̂ + σ_t ε̂ − z_t|| / ||z_t||
    recon = alpha_t * x0_hat + sigma_t * eps_hat
    resid = (((recon - z_t)**2).sum() / (z_t.pow(2).sum().clamp_min(1e-12))).item()

    return z_prev, x0_hat, eps_hat, resid




@torch.no_grad()
def sample_trajectories(cfg: DiffusionConfig, ckpt_path, out_dir, warm_start: bool = False):
    """
    Generate samples and save a multi-MODEL PDB per pair.
    Also logs Cartesian RMSDs vs apo/holo to verify the modal→Cartesian mapping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ds = ModalPairsDataset(
        cfg.pairs_json, cfg.n_modes, cfg.cache_dir, device, dtype,
        edge_radius=cfg.edge_radius,
    )

    model = ModalEGNN(
        node_feat_dim=3,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        time_emb_dim=cfg.time_dim,
    ).to(device)
    model.eval()
    
    load_ckpt_ema_only(ckpt_path, model, device)
    #load_ckpt_raw(ckpt_path, model, device)
    # schedule: ~1 → 0
    steps = cfg.sample_steps
    t_start, t_end = 0.98, 0.01   

    u_grid = make_u_grid_for_ddim(
        steps=steps,
        t_start=t_start, t_end=t_end,
        u_train_min=-9.71, u_train_max=7.8,
        device=device, dtype=dtype
    )                               # shape [steps+1]
    ts = make_t_grid_from_u(u_grid) # shape [steps+1], decreasing
    #ts = torch.linspace(t_start, t_end, steps + 1, device=device, dtype=dtype)
    #ts = make_ts_from_rstart(r_start=50.0, t_end=0.01, steps=cfg.sample_steps,                     device=device, dtype=torch.dtype)
    #print(f"[sched] t0={float(ts[0]):.3f}  t_last={float(ts[-1]):.3f}  steps={steps}")
    #print(f"[sched] u0={float(u_grid[0]):.3f}  u_last={float(u_grid[-1]):.3f}  (clamped to training range)")

    for idx in range(len(ds)):
        item  = ds[idx]
        modes = ds.modesets[idx]  # has to_cartesian(), .mol, device/dtype
        frames: List[torch.Tensor] = []

        raw = collate_batch([item], cfg.edge_radius)
        raw = _to_device(raw, device, dtype)
        radius = float(raw["edge_radius"][0].item())
        mask   = raw["mode_mask"]
        z_star = raw["z0"]  # [1,M]

        # -------- ε̂ scale calibration (once per pair/model) --------
        #gamma = _estimate_eps_scale(model, raw, device)

        # -------- Decode the target once (sanity) --------
        z_star_cart = modes.to_cartesian(z_star.squeeze(0).to(dtype=modes.dtype, device=modes.device))

        # Apo coords: decode zero modal amplitudes (reference structure)
        z_zero   = torch.zeros_like(z_star.squeeze(0), device=modes.device, dtype=modes.dtype)
        apo_cart = modes.to_cartesian(z_zero)
        print(f"[cart] RMSD(decode(z*)) vs apo ≈ {_rmsd(z_star_cart, apo_cart):.2f} Å")  # apo coords as parsed
        # if you have the holo path in the pair, load it exactly; otherwise this serves as a proxy:
        # (your dataset likely aligned holo→apo when fitting z*)
        # For logging, compare the *deformed apo* to the decoded target z*
        rmsd_star_vs_apo = _rmsd(z_star_cart, apo_cart)
        print(f"[cart] RMSD(decode(z*)) vs apo ≈ {rmsd_star_vs_apo:.2f} Å")
        sanity_mid_t(model, raw, t_scalar=0.5, trials=8)
        #rep = probe_exact(model, raw)
        for s_idx in range(cfg.n_samples_per_pair):
            batch = raw
            a0, s0 = alpha_sigma_from_u(u_grid[0].view(1))
            if warm_start:
                z = (a0.view(1,1) * z_star + s0.view(1,1) * torch.randn_like(z_star)) * mask
            else:
                z = (s0.view(1,1) * torch.randn_like(z_star)) * mask

            last_x0 = None
            for i in range(steps):
                u      = u_grid[i].expand(z.shape[0])
                u_prev = u_grid[i + 1].expand(z.shape[0])

                # deterministic first (η=0.0) to debug schedule alignment
                z, x0_hat, eps_hat, resid = _ddim_step_eps_u(
                    model, z, batch["eigvals"], mask, u, u_prev,
                    batch["x"], batch["h"], batch["node_mask"], radius, eta=0.0,
                )


                
                last_x0 = z
                #print(z[0,:5])

            # helpful prints: show u, σ/α, |x0̂|, and the reconstruction residual
            # σ/α = exp(-u/2) for logSNR u = log(α^2/σ^2)
            s_over_a = float(torch.exp(-0.5 * u[0]).item())
            print(f"step {i:04d} u={float(u[0]):+6.3f}  σ/α≈{s_over_a:6.2f}  |x0_hat|={float(x0_hat.norm()):7.2f}  resid={resid:.2e}")


            z0_sample = last_x0.squeeze(0)

            # modal-space debug
            m = mask.squeeze(0).bool()
            cos = F.cosine_similarity(z0_sample[m], z_star.squeeze(0)[m], dim=0).item()
            #print(f"[dbg] z0_sample={z0_sample[:5]}")
            #print(f"[dbg] |z0|={float(z0_sample.norm()):.3f} |z*|={float(z_star.norm()):.3f} cos={cos:.3f}")

            # decode & RMSDs
            pred_cart = modes.to_cartesian(z0_sample.to(dtype=modes.dtype, device=modes.device))
            rmsd_pred_vs_apo  = _rmsd(pred_cart, apo_cart)
            rmsd_pred_vs_star = _rmsd(pred_cart, z_star_cart)
            print(f"[cart] RMSD(pred) vs apo: {rmsd_pred_vs_apo:.2f} Å   vs decode(z*): {rmsd_pred_vs_star:.2f} Å")

            frames.append(pred_cart)

        # also save the target decode for visual comparison
        pair = ds.pairs[idx]
        apo_path = pair.apo if hasattr(pair, "apo") else pair["apo"]
        out_pair = out_dir / f"pair_{idx:05d}"
        out_pair.mkdir(parents=True, exist_ok=True)

        save_traj(
            frames=[z_star_cart],  # single MODEL of target decode
            mol=getattr(modes, "_m", modes).mol,
            out_path=out_pair / "decode_zstar.pdb",
            template_path=apo_path,
        )
        save_traj(
            frames=frames,
            mol=getattr(modes, "_m", modes).mol,
            out_path=out_pair / "samples.pdb",
            template_path=apo_path,
        )
        print(f"[sample] pair {idx} → {out_pair/'samples.pdb'}   (target decode at decode_zstar.pdb)")
