from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F

from .config import DiffusionConfig
from .model import ModalEGNN
from .dataset import ModalPairsDataset, collate_batch
from .noise import alpha_sigma_from_u
from .utils import load_ckpt_ema_only
from .adapters import save_traj, _apply_rtb_vec


# -------------------------------
# Basic helpers
# -------------------------------

def _to_device(batch: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.dtype == torch.bool:
                out[k] = v.to(device)
            else:
                out[k] = v.to(device=device, dtype=dtype)
        else:
            out[k] = v
    return out


def _masked_corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Cosine correlation between a and b under a mask.

    a, b: [B, ..., D]
    mask: broadcastable to a/b (1 where valid, 0 otherwise)
    """
    a = a * mask
    b = b * mask
    a_flat = a.flatten(1)
    b_flat = b.flatten(1)

    a_mu = a_flat.mean(dim=1, keepdim=True)
    b_mu = b_flat.mean(dim=1, keepdim=True)

    a0 = a_flat - a_mu
    b0 = b_flat - b_mu

    num = (a0 * b0).sum(dim=1)
    den = torch.sqrt((a0.pow(2).sum(dim=1) + 1e-12) * (b0.pow(2).sum(dim=1) + 1e-12))
    corr = num / den
    return float(corr.mean().item())


# -------------------------------
# Early unbias and guard (same logic as training)
# -------------------------------

def _early_unbias_eps(
    y_flat: torch.Tensor,          # [B, D]
    sigma: torch.Tensor,           # [B, 1]
    eps_flat: torch.Tensor,        # [B, D]
    step_idx: int,
    K_early: int,
    gamma_max: float = 1.0,
) -> torch.Tensor:
    """
    Early unbias in flattened space:
      - y_flat:   current state (here flattened y_t)
      - sigma:    σ at this step (broadcastable)
      - eps_flat: predicted ε̂_y
    Align ε̂_y to the analytic baseline b = y_flat / σ at very early steps,
    then gradually remove that baseline component as steps progress.
    """
    # Analytic baseline b = y / σ
    b = y_flat / sigma.clamp_min(1e-8)
    num = (eps_flat * b).sum(1, keepdim=True)        # <ε̂, b>
    den = (b.pow(2).sum(1, keepdim=True) + 1e-8)     # ||b||^2
    proj = (num / den) * b

    if K_early <= 1:
        t = 1.0
    else:
        t = min(max(step_idx / float(K_early - 1), 0.0), 1.0)
    gamma = float(gamma_max) * t

    return eps_flat - gamma * proj


def _guard_rho(
    y_flat: torch.Tensor,          # [B, D]
    sigma: torch.Tensor,           # [B, 1]
    eps_flat: torch.Tensor,        # [B, D]
    rho_min: float = 0.85,
) -> torch.Tensor:
    """
    Guard against over-cancellation:

      rho = ||y_t − σ ε_hat|| / ||y_t||.

    If rho is too small, shrink ε_hat so that rho >= rho_min.
    """
    numer = (y_flat - sigma * eps_flat)
    rho = numer.norm(dim=1, keepdim=True) / (y_flat.norm(dim=1, keepdim=True) + 1e-12)
    scale = torch.where(
        rho < rho_min,
        (rho_min / (rho + 1e-12)).clamp_max(1.0),
        torch.ones_like(rho),
    )
    return eps_flat * scale


# -------------------------------
# Build y0 and eps_y (identical to training)
# -------------------------------

@torch.no_grad()
def _build_y0_and_eps_y(batch: Dict[str, torch.Tensor], decoder_bank) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build RTB-DOF state for sampling:

      y0    : [B, N, 6] clean RTB DOFs (apo→holo)
      eps_y : [B, N, 6] RTB-DOF noise (along RTB modes)

    This mirrors the construction used in training.
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



# -------------------------------
# One DDIM step in Cartesian displacement space
# (identical logic to training's inner loop)
# -------------------------------

@torch.no_grad()
def _ddim_step_cartesian_u(
    model: ModalEGNN,
    y_t: torch.Tensor,             # [B, N, 3] current displacements
    u: torch.Tensor,               # [B]
    u_prev: torch.Tensor,          # [B]
    x_apo: torch.Tensor,           # [B, N, 3] apo block coords
    h: torch.Tensor,               # [B, N, F] node scalar features
    node_mask: torch.Tensor,       # [B, N]
    edge_radius: float,
    u_start: float,
    *,
    eta: float,
    step_idx: int,
    K_early_unbias: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    DDIM step in displacement space (y_t) with epsilon prediction.

        y0_hat = (y_t - σ_t * ε̂) / α_t
        y_prev = α_prev * y0_hat + w * σ_prev * ε̂         (eta=0)
    """
    B, N, _ = y_t.shape
    device, dtype = y_t.device, y_t.dtype

    alpha_t, sigma_t = alpha_sigma_from_u(u)         # [B]
    alpha_p, sigma_p = alpha_sigma_from_u(u_prev)    # [B]

    alpha_t3 = alpha_t.view(B, 1, 1)
    sigma_t3 = sigma_t.view(B, 1, 1)
    alpha_p3 = alpha_p.view(B, 1, 1)
    sigma_p3 = sigma_p.view(B, 1, 1)

    # 1) Predict ε_y at current step
    eps_hat = model.forward_cartesian(
        y_t=y_t,
        x_apo=x_apo,
        nfeat=h,
        node_mask=node_mask,
        u=u,
        W=None,
        edge_radius=edge_radius,
    )  # [B,N,3]

    # 2) Flatten and apply early unbias + guard (same as training)
    y_flat = y_t.flatten(1)                          # [B, 3N]
    eps_flat = eps_hat.flatten(1)                    # [B, 3N]
    sigma_mat = sigma_t.view(B, 1)                   # [B,1]

    # analytic baseline for diagnostics / unbias
    eps_base_flat = y_flat / sigma_mat.clamp_min(1e-8)

    # Early unbias relative to baseline, only for first K_early steps
    if K_early_unbias > 0:
        eps_flat = _early_unbias_eps(
            y_flat=y_flat,
            sigma=sigma_mat,
            eps_flat=eps_flat,
            step_idx=step_idx,
            K_early=K_early_unbias,
            gamma_max=1.0,
        )

    # Guard against over-cancellation: keep ρ above threshold during early region
    if u[0].item() >= (u_start - 1.0):
        eps_flat = _guard_rho(
            y_flat=y_flat,
            sigma=sigma_mat,
            eps_flat=eps_flat,
            rho_min=0.80,
        )

    eps_hat = eps_flat.view_as(y_t)

    # 3) Compute y0_hat at this step
    y0_hat = (y_t - sigma_t3 * eps_hat) / alpha_t3.clamp_min(1e-8)

    # 4) DDIM update: deterministic path + optional noise (eta)
    if eta == 0.0:
        # pure DDIM (deterministic)
        y_prev = alpha_p3 * y0_hat + sigma_p3 * eps_hat
    else:
        # add scaled noise term (stochastic DDIM)
        z = torch.randn_like(y_t)
        sigma_extra = eta * sigma_p3
        y_prev = alpha_p3 * y0_hat + sigma_p3 * eps_hat + sigma_extra * z

    # 5) reconstruction residual for debugging
    recon = alpha_t3 * y0_hat + sigma_t3 * eps_hat
    resid = (((recon - y_t) ** 2).sum() /
             (y_t.pow(2).sum().clamp_min(1e-12))).item()

    return y_prev, y0_hat, eps_hat, float(resid)


# -------------------------------
# Public API: sampling trajectories (Cartesian cold start)
# -------------------------------

@torch.no_grad()
def sample_trajectories(
    cfg: DiffusionConfig,
    ckpt_path: str | Path,
    out_dir: str | Path,
    *,
    eta: float = 0.7,
    u_start: float = -9.1,
    u_end: float = 7.2,
    K_early_unbias: int = 12,
    print_debug: bool = True,
    seed: int | None = None,
) -> None:
    """
    Cold-start DDIM sampling in Cartesian displacement space, mirroring the
    training `ddim_cold_probe_u` logic.

    For each apo–holo pair:
      - Build y0 (apo→holo) and eps_y (noise along modes) for nodes.
      - Start from y_t = σ(u_start) * eps_y (pure noise).
      - Run DDIM in y-space to get y0_hat at u_end.
      - Decode final node displacements to full-atom coords via blocks and save.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset (apo–holo pairs + modesets)
    ds = ModalPairsDataset(
        pairs_json=cfg.pairs_json,
        n_modes=cfg.n_modes,
        cache_dir=cfg.cache_dir,
        device=device,
        dtype=dtype,
        edge_radius=cfg.edge_radius,
    )

    # Simple decoder bank
    decoder_bank = ds.modesets

    # Model & weights (EMA)
    model = ModalEGNN(
        node_feat_dim=3,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        time_emb_dim=cfg.time_dim,
    ).to(device)
    model.eval()
    load_ckpt_ema_only(ckpt_path, model, device)

    steps = int(cfg.sample_steps)

    # u-grid identical to training's Cartesian DDIM probe
    u0 = torch.tensor(float(u_start), device=device, dtype=dtype)
    u1 = torch.tensor(float(u_end),   device=device, dtype=dtype)
    u_grid = torch.linspace(u0, u1, steps + 1, device=device, dtype=dtype)
    u0_start = float(u_grid[0].item())

    if print_debug:
        print(f"[sched] u0={float(u_grid[0]):+.3f}  u_last={float(u_grid[-1]):+.3f}  steps={steps}")

    # Optional reproducibility
    if seed is not None:
        torch.manual_seed(int(seed))

    # ------------------------------------------------------------------
    # Loop over apo–holo pairs
    # ------------------------------------------------------------------
    for idx in range(len(ds)):
        pair = ds.pairs[idx]
        apo_path = pair.apo if hasattr(pair, "apo") else pair["apo"]
        modes = ds.modesets[idx]

        # Collate a single-item batch, as in training
        raw = collate_batch([ds[idx]], cfg.edge_radius)
        raw = _to_device(raw, device, dtype)

        x_nodes = raw["x"]                # [1, N_nodes, 3] apo block coords for EGNN
        h_nodes = raw["h"]                # [1, N_nodes, F]
        node_mask = raw["node_mask"]      # [1, N_nodes]
        z_star = raw["z0"]                # [1, M] target in mode basis (for metrics only)
        radius = float(raw["edge_radius"][0].item())

        B, N_nodes, _ = x_nodes.shape
        assert B == 1, "sample_trajectories currently assumes batch size 1 per pair"

        # Build reference y0 and eps_y on nodes (as in training)
        batch = {
            "x": x_nodes,
            "h": h_nodes,
            "node_mask": node_mask,
            "z0": z_star,
            "eigvals": raw["eigvals"],
            "mode_mask": raw["mode_mask"],
            "edge_radius": raw["edge_radius"],
            "idxs": raw["idxs"],
        }
        y0_ref, eps_y = _build_y0_and_eps_y(batch, decoder_bank)   # [1,N_nodes,3]

        # Decode z* and apo in full Cartesian coords (for RMSD / saving)
        with torch.no_grad():
            z_star_cart_full = modes.to_cartesian(
                z_star.squeeze(0).to(device=modes.device, dtype=modes.dtype)
            )  # [N_cart, 3]
            z_zero = torch.zeros_like(
                z_star.squeeze(0), device=modes.device, dtype=modes.dtype
            )
            apo_cart_full = modes.to_cartesian(z_zero)  # [N_cart, 3]

        # Align nodes with the first N_nodes positions of the modes geometry
        valid_nodes = node_mask[0].bool()
        n_valid = int(valid_nodes.sum().item())
        if n_valid > apo_cart_full.shape[0]:
            raise RuntimeError(
                f"Node count {n_valid} exceeds modes_cart length {apo_cart_full.shape[0]}"
            )

        apo_nodes_cart = apo_cart_full[:n_valid].to(device=device, dtype=dtype)          # [n_valid,3]
        star_nodes_cart = z_star_cart_full[:n_valid].to(device=device, dtype=dtype)      # [n_valid,3]

        # Sanity: y0_ref on nodes should match apo→holo via modes
        if print_debug:
            holo_rmsd = float(torch.sqrt(((z_star_cart_full - apo_cart_full) ** 2).mean()).item())
            print(f"[pair {idx}] RMSD(decode(z*)) vs apo ≈ {holo_rmsd:.2f} Å")
            if x_nodes.shape[1] == apo_nodes_cart.shape[0]:
                diff_rmsd = float(torch.sqrt(((x_nodes[0, valid_nodes].cpu() -
                                               apo_nodes_cart.cpu()) ** 2).mean()).item())
                print(f"[pair {idx}] RMSD(dataset apo vs modes apo nodes) ≈ {diff_rmsd:.3f} Å")

        frames: List[torch.Tensor] = []

        # -------------------------------
        # For each stochastic sample (from same apo)
        # -------------------------------
        n_samples = int(getattr(cfg, "n_samples_per_pair", 1))
        for s_idx in range(n_samples):

            # --- Cold start in y-space: y_t = σ(u0) * eps_y ---
            u0_batch = u_grid[0].view(1).to(device)
            alpha0, sigma0 = alpha_sigma_from_u(u0_batch)  # [1]
            sigma0_b3 = sigma0.view(1, 1, 1)
            y_t = sigma0_b3 * eps_y.to(device=device, dtype=dtype)

            if print_debug:
                # Baseline sanity check at step 0 (exact analogue of training cold probe)
                eps0 = model.forward_cartesian(
                    y_t=y_t,
                    x_apo=x_nodes,
                    nfeat=h_nodes,
                    node_mask=node_mask,
                    u=u0_batch,
                    W=None,
                    edge_radius=radius,
                )
                base0 = y_t / sigma0_b3.clamp_min(1e-8)
                corr0 = _masked_corr(eps0, base0, node_mask.unsqueeze(-1).float())
                rho0 = float((y_t - sigma0_b3 * eps0).flatten(1).norm(dim=1) /
                             (y_t.flatten(1).norm(dim=1) + 1e-12))
                print(f"[pair {idx} sample {s_idx}] cold-step0: "
                      f"corr(ε_hat, ε_base)={corr0:.3f}, "
                      f"|y_t - σ ε_hat|/|y_t|={rho0:.3f}")

            last_y0_hat = None

            # --- DDIM sweep in y-space (identical to training inner loop) ---
            for i in range(steps):
                u_i = u_grid[i].view(1).to(device)
                u_prev = u_grid[i + 1].view(1).to(device)

                y_t, y0_hat, eps_hat, resid = _ddim_step_cartesian_u(
                    model=model,
                    y_t=y_t,
                    u=u_i,
                    u_prev=u_prev,
                    x_apo=x_nodes,
                    h=h_nodes,
                    node_mask=node_mask,
                    edge_radius=radius,
                    u_start=u0_start,
                    eta=eta,
                    step_idx=i,
                    K_early_unbias=K_early_unbias,
                )

                last_y0_hat = y0_hat

                if print_debug:
                    nm = node_mask.float().unsqueeze(-1)
                    diff = (y0_hat - y0_ref.to(device)) * nm
                    denom = nm.sum(dim=(1, 2)).clamp_min(1.0)
                    mse_y = float((diff.pow(2).sum(dim=(1, 2)) / denom).mean().item())
                    cos_y = _masked_corr(y0_hat, y0_ref.to(device), nm)
                    print(f"[pair {idx} sample {s_idx}] step {i:03d}: "
                          f"MSE(y0̂,y0)={mse_y:.4f}, corr={cos_y:.3f}, "
                          f"recon_resid={resid:.3e}")

            if last_y0_hat is None:
                last_y0_hat = y_t

            y_pred_nodes = last_y0_hat          # [1, N_nodes, 3]

            # Cartesian displacement metrics vs y0_ref on nodes (same as training)
            nm = node_mask.float().unsqueeze(-1)
            diff = (y_pred_nodes - y0_ref.to(device)) * nm
            denom = nm.sum(dim=(1, 2)).clamp_min(1.0)
            mse_y = float((diff.pow(2).sum(dim=(1, 2)) / denom).mean().item())
            mae_y = float((diff.abs().sum(dim=(1, 2)) / denom).mean().item())
            cos_y = _masked_corr(y_pred_nodes, y0_ref.to(device), nm)

            if print_debug:
                print(f"[pair {idx} sample {s_idx}] final node-displacement metrics: "
                      f"MSE={mse_y:.4f}  MAE={mae_y:.4f}  corr={cos_y:.3f}")

            # ---- Decode to full Cartesian coords for saving ----
            #
            # We now interpret each node as a rigid block (as in build_block_graph)
            # and apply the predicted block displacement to *all atoms* in that block.
            # This way the full molecule moves, instead of only the first n_valid atoms.
            ms = getattr(modes, "_m", modes)
            blocks = getattr(ms, "blocks", None)
            if blocks is None:
                raise RuntimeError("Modes object has no `blocks` attribute; cannot map nodes to atoms.")

            if len(blocks) < n_valid:
                raise RuntimeError(
                    f"Number of blocks {len(blocks)} is smaller than n_valid={n_valid} "
                    "(this should not happen; check dataset / modes construction)."
                )

            # y_pred_nodes: [1, N_nodes, 6] RTB DOFs; restrict to valid nodes
            v_pred_nodes = y_pred_nodes[0, valid_nodes].detach().to(
                device=modes.device, dtype=modes.dtype
            )  # [n_valid, 6]

            # Use underlying ModeSet to apply full RTB vector
            mset = getattr(modes, "_m", modes)
            rtb_pred = v_pred_nodes.reshape(-1)  # [R = 6 * n_blocks]

            pred_cart_full = _apply_rtb_vec(
                mol=mset.mol,
                blocks=mset.blocks,
                rtb_vec=rtb_pred,
                block_dofs=mset.block_dofs,
                device=mset.device,
            )
            print("[debug] first few DOFs:", v_pred_nodes[:5])

            print(apo_cart_full[:6])
            print(z_star_cart_full[:6])
            print(pred_cart_full[:6])
            # Quick RMSDs vs apo and decode(z*)
            # NOTE: averaged over all atoms (will look smaller than node-only RMSDs)
            apo_rmsd = float(torch.sqrt(((pred_cart_full - apo_cart_full) ** 2).mean()).item())
            star_rmsd = float(torch.sqrt(((pred_cart_full - z_star_cart_full) ** 2).mean()).item())
            if print_debug:
                print(f"[pair {idx} sample {s_idx}] RMSD(pred, apo) ≈ {apo_rmsd:.2f} Å, "
                      f"RMSD(pred, decode(z*)) ≈ {star_rmsd:.2f} Å")

            frames.append(pred_cart_full.detach().cpu())

        # ------------------------------------------------------------------
        # Save trajectory and reference decode(z*)
        # ------------------------------------------------------------------
        out_pair = out_dir / f"pair_{idx:03d}"
        out_pair.mkdir(parents=True, exist_ok=True)

        save_traj(
            frames=[z_star_cart_full],
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
        print(f"[sample] pair {idx} → {out_pair / 'samples.pdb'}   "
              f"(target decode at decode_zstar.pdb)")
