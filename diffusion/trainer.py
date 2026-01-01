from __future__ import annotations
from pathlib import Path
import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from .config import DiffusionConfig
from .dataset import ModalPairsDataset
from .losses import cartesian_eps_loss
from .model import ModalEGNN
from .utils import seed_all, save_ckpt, EMA
from .noise import (
    t_from_logsnr,
    alpha_sigma_from_t,
    logsnr_from_t,
    s_over_a_from_u,
    alpha_sigma_from_u,
)
from .adapters import _apply_rtb_vec


# ---------------- Collate wrapper ----------------
@dataclass
class CollateWithRadius:
    edge_radius: float

    def __call__(self, batch):
        from .dataset import collate_batch as _collate_batch
        return _collate_batch(batch, self.edge_radius)


def batch_to_device(batch, device, non_blocking: bool = True):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def _w_eps(i, u_i, K_early=12, u0=-9.1, u_ramp_end=-3.0):
    """
    Early gate for the ε-path. Returns a scalar in [0,1].
    - 0 for the first K_early steps
    - then linearly ramps from u0 → u_ramp_end
    """
    try:
        u_scalar = float(u_i) if not hasattr(u_i, "item") else float(u_i.item())
    except Exception:
        u_scalar = float(u_i)

    if i < K_early:
        return 0.0

    denom = (u_ramp_end - u0)
    if abs(denom) < 1e-6:
        return 1.0

    t = (u_scalar - u0) / denom
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return t


def _u_bounds(epoch: int):
    """Start narrow in log-SNR and widen to full range by ~100 epochs."""
    widen = min(1.0, epoch / 100.0)
    a = -2.0 - (5.0 - 2.0) * widen
    b = 2.0 + (5.0 - 2.0) * widen
    return a, b


def _early_unbias_eps(z_flat, s, eps_hat_flat, step, K_early=12, gamma_max=1.0):
    """
    Generic early unbias in a flattened space:
      - z_flat: [B, D] current state (here flattened y_t)
      - s     : [B, 1] σ at this step
      - eps_hat_flat: [B, D]
    Aligns ε̂ to the analytic baseline b = z_flat / σ during early steps,
    then gradually removes the baseline component.
    """
    # b: [B, D]
    b = z_flat / s.clamp_min(1e-8)
    num = (eps_hat_flat * b).sum(1, keepdim=True)            # <ε̂, b>
    den = (b.pow(2).sum(1, keepdim=True) + 1e-8)             # ||b||^2
    proj = (num / den) * b                                   # proj_b(ε̂)

    if K_early <= 0:
        gamma = 0.0
    else:
        t = min(max(step / float(max(1, K_early - 1)), 0.0), 1.0)
        gamma = float(gamma_max) * t

    eps_unbiased_flat = eps_hat_flat - gamma * proj
    return eps_unbiased_flat


def _guard_rho(z_flat, s, eps_hat_flat, rho_min=0.80):
    """
    Guard against over-cancellation:
      ρ = ||z_flat - σ ε̂|| / ||z_flat||.
    If ρ is too small, shrink ε̂ to bring ρ up to rho_min.
    """
    numer = (z_flat - s * eps_hat_flat)
    rho = numer.norm(dim=1, keepdim=True) / (
        z_flat.norm(dim=1, keepdim=True) + 1e-12
    )
    k = torch.where(
        rho < rho_min,
        (rho_min / (rho + 1e-12)).clamp(max=1.0),
        torch.ones_like(rho),
    )
    return eps_hat_flat * k


# =======================
# Helpers to build Cartesian displacements/noise
# =======================
def _build_y0_and_eps_y(batch, decoder_bank):
    """
    Build RTB-DOF state for probes:

      y0    : [B, N, 6] clean RTB DOFs (apo→holo)
      eps_y : [B, N, 6] RTB-DOF noise (along RTB modes)

    This mirrors the construction in cartesian_eps_loss, but is used only
    for debugging probes (cartesian_probe_metrics_u, ddim_cold_probe_u).
    """
    z0        = batch["z0"]           # [B,M]
    mode_mask = batch["mode_mask"]    # [B,M]
    x_apo     = batch["x"]            # [B,N,3]
    node_mask = batch["node_mask"]    # [B,N]
    idxs      = batch["idxs"]         # [B]

    device = x_apo.device
    dtype  = x_apo.dtype

    B, M = z0.shape
    _, N, _ = x_apo.shape

    m     = mode_mask
    eps_q = torch.randn_like(z0) * m  # [B,M]

    y0_list:   list[torch.Tensor] = []
    eps_y_list:list[torch.Tensor] = []

    for b in range(B):
        idx = int(idxs[b].item())
        ms   = decoder_bank[idx]
        mset = getattr(ms, "_m", ms)

        z0_b   = (z0[b]   * m[b]).to(device=mset.device, dtype=mset.dtype)
        eps_qb = (eps_q[b]* m[b]).to(device=mset.device, dtype=mset.dtype)

        V     = mset.eigvecs_rtb
        s_dec = mset.decode_scales

        rtb0    = V @ (s_dec * z0_b)    # [R]
        rtb_eps = V @ (s_dec * eps_qb)  # [R]

        n_blocks = len(mset.blocks)
        v0_ms    = rtb0.view(n_blocks, -1)    # [Nb,6]
        eps_v_ms = rtb_eps.view(n_blocks, -1) # [Nb,6]

        y0_list.append(v0_ms.to(device=device, dtype=dtype))
        eps_y_list.append(eps_v_ms.to(device=device, dtype=dtype))

    dof_dim = y0_list[0].shape[-1]
    y0    = x_apo.new_zeros(B, N, dof_dim)
    eps_y = x_apo.new_zeros(B, N, dof_dim)

    for b in range(B):
        valid = node_mask[b].bool()
        n_i = int(valid.sum().item())
        y0[b, valid]    = y0_list[b][:n_i]
        eps_y[b, valid] = eps_y_list[b][:n_i]

    return y0, eps_y



# =======================
# Cartesian probe at fixed u (replaces v_probe_metrics_u)
# =======================
@torch.no_grad()
def cartesian_probe_metrics_u(model, batch, decoder_bank, u_val: float):
    """
    Probe training quality at a fixed logSNR u in Cartesian displacement space.

    Builds:
      y_t = α y0 + σ ε_y
    then predicts ε̂_y, reconstructs y0̂, and returns metrics like:
      - mse, mae
      - cosine corr(y0̂, y0)
      - amp_ratio = ||y0̂|| / ||y0||
      - eps_dir   = corr(ε̂_y, ε_y)
      - s_over_a  = σ/α for this u
    """
    device = batch["x"].device
    z0 = batch["z0"]
    node_mask = batch["node_mask"]
    B = z0.shape[0]

    # build displacements and noise
    y0, eps_y = _build_y0_and_eps_y(batch, decoder_bank)  # [B,N,3]

    # scalar u for all items
    u = torch.full((B,), float(u_val), device=device, dtype=z0.dtype)
    a, s = alpha_sigma_from_u(u)  # [B]
    a_b3 = a.view(B, 1, 1)
    s_b3 = s.view(B, 1, 1)

    # forward diffusion
    y_t = a_b3 * y0 + s_b3 * eps_y

    # predict ε_y
    x_apo = batch["x"]
    h = batch["h"]
    edge_tensor = batch.get("edge_radius", torch.tensor([0.0], device=device))
    edge_rad = float(edge_tensor[0].item()) if isinstance(edge_tensor, torch.Tensor) else float(edge_tensor)

    eps_y_hat = model.forward_cartesian(
        y_t=y_t,
        x_apo=x_apo,
        nfeat=h,
        node_mask=node_mask,
        u=u,
        W=None,
        edge_radius=edge_rad,
    )  # [B,N,3]

    # reconstruct y0̂
    a_safe = a.clamp_min(1e-2)
    a_safe3 = a_safe.view(B, 1, 1)
    y0_hat = (y_t - s_b3 * eps_y_hat) / a_safe3

    nm = node_mask.float().unsqueeze(-1)          # [B,N,1]
    denom = nm.sum(dim=(1, 2)).clamp_min(1.0)     # [B]

    # metrics
    diff = (y0_hat - y0) * nm
    mse = (diff.pow(2).sum(dim=(1, 2)) / denom).mean().item()
    mae = (diff.abs().sum(dim=(1, 2)) / denom).mean().item()

    # cosine correlation
    def _corr(a, b):
        a = (a * nm).reshape(B, -1)
        b = (b * nm).reshape(B, -1)
        a = a - a.mean(1, keepdim=True)
        b = b - b.mean(1, keepdim=True)
        num = (a * b).sum(1)
        den = a.norm(dim=1) * b.norm(dim=1) + 1e-12
        return float((num / den).mean().item())

    cos = _corr(y0_hat, y0)

    # amplitude ratio
    y0_norm = ((y0 * nm).reshape(B, -1).norm(dim=1) + 1e-12)
    y0h_norm = ((y0_hat * nm).reshape(B, -1).norm(dim=1) + 1e-12)
    amp_ratio = float((y0h_norm / y0_norm).mean().item())

    # ε direction
    eps_dir = _corr(eps_y_hat, eps_y)
    s_over_a_val = float((s / a).mean().item())

    return dict(
        mse=mse,
        mae=mae,
        cos=cos,
        amp_ratio=amp_ratio,
        eps_dir=eps_dir,
        s_over_a=s_over_a_val,
    )


# =======================
# In-training cold-start DDIM probe in Cartesian space
# =======================
@torch.no_grad()
def ddim_cold_probe_u(
    model,
    batch,
    decoder_bank,
    *,
    steps: int = 50,
    u_start: float = -7.2,
    u_end: float = +7.2,
    seed: int = 1234,
    return_all: bool = True,
):
    """
    Cold-start DDIM probe in Cartesian displacement space:
      - start from pure noise along RTB modes at u_start
      - run DDIM-style updates y_t → y_{t-1}
      - measure final y0̂ vs true y0 in displacement space, and RMSD vs apo/holo.
    """
    # --- unpack batch ---
    z0 = batch["z0"]
    x_apo = batch["x"]
    h = batch["h"]
    node_mask = batch["node_mask"]
    edge_tensor = batch.get("edge_radius", torch.tensor([0.0], device=z0.device))
    edge_rad = float(edge_tensor[0].item()) if isinstance(edge_tensor, torch.Tensor) else float(edge_tensor)

    device, dtype = z0.device, z0.dtype
    B = z0.shape[0]

    # --- build true displacement y0 and mode-based noise eps_y (fixed seed) ---
    torch.manual_seed(int(seed))
    y0, eps_y = _build_y0_and_eps_y(batch, decoder_bank)  # [B,N,3]

    # --- u grid ---
    u0 = torch.tensor(float(u_start), device=device, dtype=dtype)
    u1 = torch.tensor(float(u_end),   device=device, dtype=dtype)
    us = torch.linspace(u0, u1, steps + 1, device=device, dtype=dtype)

    # --- cold start: y_t = σ_0 * eps_y (pure mode-based noise in displacement space) ---
    a0, s0 = alpha_sigma_from_u(us[0].expand(B))
    s0_b3 = s0.view(B, 1, 1)
    y_t = s0_b3 * eps_y  # [B,N,3]

    for i in range(steps):
        ui, uj = us[i].expand(B), us[i + 1].expand(B)
        ai, si = alpha_sigma_from_u(ui)
        aj, sj = alpha_sigma_from_u(uj)

        ai = ai.clamp_min(1e-5)
        aj = aj.clamp_min(1e-5)

        ai3 = ai.view(B, 1, 1)
        si3 = si.view(B, 1, 1)
        aj3 = aj.view(B, 1, 1)
        sj3 = sj.view(B, 1, 1)

        # 1) predict ε_y
        eps_y_hat = model.forward_cartesian(
            y_t=y_t,
            x_apo=x_apo,
            nfeat=h,
            node_mask=node_mask,
            u=ui,
            W=None,
            edge_radius=edge_rad,
        )  # [B,N,3]

        # 2) early unbias against analytic baseline in flattened space
        nm = node_mask.float().unsqueeze(-1)  # [B,N,1]
        y_flat = (y_t * nm).reshape(B, -1)
        eps_flat = (eps_y_hat * nm).reshape(B, -1)
        s_flat = si.view(B, 1)

        eps_flat = _early_unbias_eps(
            y_flat, s_flat, eps_flat, step=i, K_early=12, gamma_max=1.0
        )

        # 3) guard against over-cancellation in the first few steps
        if i < 6:
            eps_flat = _guard_rho(y_flat, s_flat, eps_flat, rho_min=0.85)

        eps_y_hat = eps_flat.reshape_as(eps_y_hat)

        # 4) compute y0̂ with de-biased ε̂
        y0_hat = (y_t - si3 * eps_y_hat) / ai3

        # 5) gate ε path in the state update
        w = _w_eps(
            i,
            float(ui[0].item()),
            K_early=12,
            u0=float(us[0].item()),
            u_ramp_end=-3.0,
        )
        y_t = aj3 * y0_hat + w * sj3 * eps_y_hat

        if i == 0:
            # quick baseline diagnostics at the very first step
            eps_base = (y_t / si3.clamp_min(1e-8)) * nm

            def _corr(a, b):
                a = (a * nm).reshape(B, -1)
                b = (b * nm).reshape(B, -1)
                a = a - a.mean(1, keepdim=True)
                b = b - b.mean(1, keepdim=True)
                num = (a * b).sum(1)
                den = a.norm(dim=1) * b.norm(dim=1) + 1e-12
                return float((num / den).mean().item())

            print("[cold-step0] corr(ε̂, ε_base)=", _corr(eps_y_hat, eps_base))
            num = ((y_t - si3 * eps_y_hat) * nm).reshape(B, -1).norm(dim=1)
            den = (y_t * nm).reshape(B, -1).norm(dim=1) + 1e-12
            print("[cold-step0] |y_t - σ·ε̂| / |y_t| =", float((num / den).mean().item()))

    # --- final y0̂ for metrics (at us[-1]) ---
    ae, se = alpha_sigma_from_u(us[-1].expand(B))
    ae = ae.clamp_min(1e-5).view(B, 1, 1)
    se = se.view(B, 1, 1)

    eps_hat_end = model.forward_cartesian(
        y_t=y_t,
        x_apo=x_apo,
        nfeat=h,
        node_mask=node_mask,
        u=us[-1].expand(B),
        W=None,
        edge_radius=edge_rad,
    )
    y0_hat_end = (y_t - se * eps_hat_end) / ae

    nm = node_mask.float().unsqueeze(-1)
    denom_pix = nm.sum().clamp_min(1.0)
    diff = (y0_hat_end - y0) * nm
    mse = float((diff.pow(2).sum() / denom_pix).item())
    mae = float((diff.abs().sum() / denom_pix).item())

    def _corr(a, b):
        a = (a * nm).reshape(B, -1)
        b = (b * nm).reshape(B, -1)
        a = a - a.mean(1, keepdim=True)
        b = b - b.mean(1, keepdim=True)
        num = (a * b).sum(1)
        den = a.norm(dim=1) * b.norm(dim=1) + 1e-12
        return float((num / den).mean().item())

    cos = _corr(y0_hat_end, y0)

    # --- RMSD in Cartesian coordinates vs apo and vs holo (full-atom via RTB decode) ---
    idxs = batch["idxs"]
    rmsd_apo_list = []
    rmsd_holo_list = []

    for b in range(B):
        idx = int(idxs[b].item())
        ms = decoder_bank[idx]
        mset = getattr(ms, "_m", ms)

        valid = node_mask[b].bool()
        # Use as many blocks as the ModeSet actually has
        n_blocks = len(mset.blocks)
        v0_b = y0[b, valid][:n_blocks].to(device=mset.device, dtype=mset.dtype)        # [Nb, dof_dim]
        v0_hat_b = y0_hat_end[b, valid][:n_blocks].to(device=mset.device, dtype=mset.dtype)

        rtb0 = v0_b.reshape(-1)       # [6 * Nb]
        rtb_hat = v0_hat_b.reshape(-1)

        # Apo coords (baseline)
        coords_apo = mset.mol.coords.to(mset.device)  # [Natoms,3]
        # Holo coords from true RTB DOFs
        coords_holo = _apply_rtb_vec(mset.mol, mset.blocks, rtb0, mset.block_dofs, mset.device)
        # Predicted coords from y0_hat_end
        coords_pred = _apply_rtb_vec(mset.mol, mset.blocks, rtb_hat, mset.block_dofs, mset.device)

        diff_apo = coords_pred - coords_apo
        diff_holo = coords_pred - coords_holo

        rmsd_apo_b = torch.sqrt((diff_apo.pow(2).sum(dim=1)).mean())
        rmsd_holo_b = torch.sqrt((diff_holo.pow(2).sum(dim=1)).mean())
        rmsd_apo_list.append(rmsd_apo_b)
        rmsd_holo_list.append(rmsd_holo_b)

    rmsd_apo = float(torch.stack(rmsd_apo_list).mean().item())
    rmsd_holo = float(torch.stack(rmsd_holo_list).mean().item())


    out = dict(
        steps=steps,
        u0=float(us[0].item()),
        u_last=float(us[-1].item()),
        metrics=dict(
            mse=mse,
            mae=mae,
            corr=cos,
            rmsd_pred_apo=rmsd_apo,
            rmsd_pred_holo=rmsd_holo,
        ),
        y0_hat=y0_hat_end,
        y_t=y_t,
        y_star=y0,
    )
    return out if return_all else y0_hat_end



# =======================
# Main training loop (Cartesian)
# =======================
def train(cfg: DiffusionConfig):
    seed_all(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # reserved for future

    # Dataset / Loader
    ds = ModalPairsDataset(
        cfg.pairs_json,
        cfg.n_modes,
        cfg.cache_dir,
        device,
        dtype,
        edge_radius=cfg.edge_radius,
    )
    num_workers = int(getattr(cfg, "num_workers", 0))
    micro_bs = max(1, min(cfg.batch_size, len(ds)))
    dl = DataLoader(
        ds,
        batch_size=micro_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=CollateWithRadius(cfg.edge_radius),
        drop_last=False,
    )
    decoder_bank = ds.modesets  # used to build Cartesian displacements/noise

    # Model / Opt
    model = ModalEGNN(
        node_feat_dim=3,            # apo block coords + extra features come via dataset
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        time_emb_dim=cfg.time_dim,
    ).to(device)
    ema = EMA(model, decay=cfg.ema_decay)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=(device.type == "cuda") and bool(cfg.amp))

    scheduler = CosineAnnealingLR(
        opt, T_max=max(2000, cfg.epochs), eta_min=cfg.lr * 0.1
    )

    log_interval = max(1, int(getattr(cfg, "log_interval", 50)))
    ckpt_interval = max(1, int(getattr(cfg, "ckpt_interval", 500)))
    grad_clip = float(getattr(cfg, "grad_clip", 1.0))
    K_t = int(cfg.t_repeats)

    last_batch = None
    saved_t = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        # You can still narrow u-bounds per epoch if you want:
        # u_min, u_max = _u_bounds(epoch)
        u_min, u_max = -9.2, 7.2
        epoch_loss_sum = 0.0
        n_iters = 0

        for it, batch in enumerate(dl, start=1):
            batch = batch_to_device(batch, device)
            last_batch = batch

            B, M = batch["z0"].shape
            opt.zero_grad(set_to_none=True)

            loss_sum_this_iter = 0.0

            for _ in range(K_t):
                u = torch.empty(B, device=device).uniform_(u_min, u_max)

                with autocast(
                    enabled=(device.type == "cuda") and bool(cfg.amp)
                ):
                    loss, dbg = cartesian_eps_loss(
                        model,
                        batch,
                        u,
                        decoder_bank=decoder_bank,
                    )

                scaler.scale(loss / K_t).backward()
                loss_sum_this_iter += float(loss)

            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            epoch_loss_sum += (loss_sum_this_iter / K_t)
            n_iters += 1

        scheduler.step()

        # ------------- epoch logging & probes -------------
        if epoch % 250 == 0 and (last_batch is not None):
            loss_avg = epoch_loss_sum / max(1, n_iters)
            print(f"[epoch {epoch:04d}] loss={loss_avg:.6f}")

            # Cartesian probes at fixed u
            print("  [Cartesian probe @ u=0.0]")
            rep = cartesian_probe_metrics_u(
                model, last_batch, decoder_bank, u_val=0.0
            )
            print(
                f"    MSE={rep['mse']:.4f}  MAE={rep['mae']:.4f}  cos={rep['cos']:.3f}  amp={rep['amp_ratio']:.3f}"
            )
            print("  [Cartesian probe @ u=+6.0 (very clean)]")
            rep = cartesian_probe_metrics_u(
                model, last_batch, decoder_bank, u_val=+6.0
            )
            print(
                f"    MSE={rep['mse']:.4f}  MAE={rep['mae']:.4f}  cos={rep['cos']:.3f}  amp={rep['amp_ratio']:.3f}"
            )
            print("  [Cartesian probe @ u=-6.0 (high noise)]")
            rep = cartesian_probe_metrics_u(
                model, last_batch, decoder_bank, u_val=-6.0
            )
            print(
                f"    eps_dir={rep['eps_dir']:.3f}  s/a≈{rep['s_over_a']:.2f}  amp={rep['amp_ratio']:.3f}"
            )

            # Cold-start DDIM probe (Cartesian)
            model.eval()
            try:
                probe_steps = int(getattr(cfg, "sample_steps", 50))
            except Exception:
                probe_steps = 50

            cold = ddim_cold_probe_u(
                model,
                last_batch,
                decoder_bank,
                steps=getattr(cfg, "sample_steps", 40),
                u_start=-9.1,
                u_end=7.2,
                seed=1234,
                return_all=True,
            )
            print(
                f"[DDIM(cold,u)] steps={cold['steps']} u0={cold['u0']:+.3f} "
                f"u_last={cold['u_last']:+.3f} "
                f"MSE={cold['metrics']['mse']:.4f} "
                f"MAE={cold['metrics']['mae']:.4f} "
                f"corr={cold['metrics']['corr']:.3f}"
                f"RMSD(pred,apo)={cold['metrics']['rmsd_pred_apo']:.3f}Å "
                f"RMSD(pred,holo)={cold['metrics']['rmsd_pred_holo']:.3f}Å"
            )

            # Dump a tiny head (first 5 valid blocks) for inspection
            nm = last_batch["node_mask"][0].bool()
            idx = nm.nonzero(as_tuple=True)[0][:5]
            if idx.numel() > 0:
                y_t_head = cold["y_t"][0, idx].detach().cpu()
                ystar_head = cold["y_star"][0, idx].detach().cpu()
                y0h_head = cold["y0_hat"][0, idx].detach().cpu()

                def fmt_vec3(v):
                    return "[" + ", ".join(f"{float(x):+.4f}" for x in v.tolist()) + "]"

                print("           (final) y_t  :", [fmt_vec3(v) for v in y_t_head])
                print("           (target) y*  :", [fmt_vec3(v) for v in ystar_head])
                print("           (final) y0̂  :", [fmt_vec3(v) for v in y0h_head])
            model.train()

        # ------------- checkpoints -------------
        if epoch % ckpt_interval == 0:
            save_ckpt(
                out_dir / f"ckpt_e{epoch:03d}.pt",
                model,
                opt,
                epoch,
                epoch,
                ema,
            )

    save_ckpt(
        out_dir / "ckpt_final.pt",
        model,
        opt,
        cfg.epochs,
        cfg.epochs,
        ema,
    )
    print("[train] done.")
