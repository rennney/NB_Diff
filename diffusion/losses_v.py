# nb_sim/diffusion/losses.py
from __future__ import annotations
import torch
from .noise import add_noise, alpha_sigma_from_t
import torch.nn.functional as F


def v_from(z0, eps, t):
    alpha_t, sigma_t = alpha_sigma_from_t(t)
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
    return alpha_t * eps - sigma_t * z0  # v = α ε − σ z0

@torch.no_grad()
def _std_over_mask(x, m):
    # scalar std over all valid entries
    denom = m.sum().clamp_min(1)
    return torch.sqrt(((x * m) ** 2).sum() / denom)


def x0_loss(model, batch, t, huber_delta=1.0):
    z0        = batch["z0"]          # [B,M] holo-in-apo chart (your z*)
    eigvals   = batch["eigvals"]     # [B,M]
    mode_mask = batch["mode_mask"]   # [B,M] 1/0
    x         = batch["x"]           # [B,N,3]
    h         = batch["h"]           # [B,N,F]
    node_mask = batch["node_mask"]   # [B,N]
    edge_radius = float(batch["edge_radius"])

    # forward sample
    eps = torch.randn_like(z0)
    alpha_t, sigma_t = alpha_sigma_from_t(t)      # [B]
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
    zt = alpha_t * z0 + sigma_t * eps             # [B,M]

    # --- model now returns x0_pred ---
    x0_pred = model(zt, eigvals, mode_mask, t, x, h, node_mask,
                    W=None, edge_radius=edge_radius) * mode_mask   # [B,M]

    # primary objective: SNR-weighted x0 loss
    snr   = (alpha_t**2) / (sigma_t**2 + 1e-12)                    # [B,1]
    w_x0  = snr / (1.0 + snr)                                      # emphasize low/mid noise
    per   = F.smooth_l1_loss(x0_pred, z0, reduction="none", beta=huber_delta)
    per   = (per * mode_mask).sum(1) / mode_mask.sum(1).clamp_min(1)
    L_x0  = ((w_x0.squeeze(-1) * per)).mean()

    # consistency: derive eps_hat, v_hat (used by probes/sampler)
    eps_hat = (zt - alpha_t * x0_pred) / (sigma_t + 1e-12)
    v_hat   = alpha_t * eps_hat - sigma_t * x0_pred

    # small auxiliary smooth v error (only to stabilize high-t a bit)
    v_tgt = alpha_t * eps - sigma_t * z0
    w_v   = 0.1 * (1.0 / (1.0 + snr))                               # small weight, emphasize high-t
    per_v = F.smooth_l1_loss(v_hat, v_tgt, reduction="none", beta=huber_delta)
    per_v = (per_v * mode_mask).sum(1) / mode_mask.sum(1).clamp_min(1)
    L_v   = ((w_v.squeeze(-1) * per_v)).mean()

    loss = L_x0 + L_v

    dbg = {
        "z0_hat": x0_pred.detach(),
        "v_hat": v_hat.detach(),
        "v_tgt": v_tgt.detach(),
        "eps": eps.detach(),
        "eps_hat": eps_hat.detach(),
        "zt": zt.detach(),
    }
    return loss, dbg

def v_loss(model, batch, t, huber_delta=1.0, var_reg=0.05, mean_reg=0.01):
    """
    Robust v-prediction loss with variance & mean regularization.
    Returns: loss, dbg
    dbg contains: z0_hat, v_pred, v_tgt, eps, eps_pred, zt
    """
    z0        = batch["z0"]
    eigvals   = batch["eigvals"]
    mode_mask = batch["mode_mask"]
    x         = batch["x"]
    h         = batch["h"]
    node_mask = batch["node_mask"]
    edge_radius = float(batch["edge_radius"][0].item()) if isinstance(batch["edge_radius"], torch.Tensor) else float(batch["edge_radius"])

    alpha_t, sigma_t = alpha_sigma_from_t(t)
    # sample noise & z_t
    eps = torch.randn_like(z0)
    zt  = add_noise(z0, t, eps)

    # target in v-space
    v_tgt = v_from(z0, eps, t)

    use_cold = (torch.rand((), device=z0.device) < 0.2)
    zt_in = zt.clone()
    if use_cold:
        # cold *but consistent*: same eps you drew above, not a fresh randn
        zt_in = (sigma_t * eps)

    # predict v on the exact z_t you feed in
    v_pred = model(zt_in, eigvals, mode_mask, t, x, h, node_mask, W=None, edge_radius=edge_radius)

    # reconstruct with the SAME z_t (v-parameterization identities)
    z0_hat  = (alpha_t * zt_in - sigma_t * v_pred) * mode_mask
    eps_hat = (alpha_t * v_pred + sigma_t * zt_in) * mode_mask #* mode_mask



    # scale regularizer: pull std(eps_pred) toward std(eps)
    with torch.no_grad():
        eps_std = _std_over_mask(eps, mode_mask)
    eps_pred_std = _std_over_mask(eps_hat, mode_mask)
    var_pen = torch.abs(eps_pred_std - eps_std)

    # mean regularizer: keep eps_pred mean ~ 0 over masked entries
    mu = ((eps_hat * mode_mask).sum() / mode_mask.sum().clamp_min(1))
    mean_pen = mu.pow(2)

    per = F.smooth_l1_loss(v_pred, v_tgt, reduction="none", beta=huber_delta)
    per = (per * mode_mask).sum(1) / mode_mask.sum(1).clamp_min(1)  # [B]

    snr  = (alpha_t**2) / (sigma_t**2 + 1e-12)
    w_v  = 10.0 / (1.0 + snr)
    base = (w_v * per).mean()   # <-- keep weighting *inside* the mean

    # one SNR-aware x0 anchor (simple and stable)
    w_x0 = snr / (1.0 + snr)    # large at low-noise, small at high-noise
    L_x0 = ((w_x0 * ((z0_hat - z0)**2).sum(1) / mode_mask.sum(1).clamp_min(1))).mean()

    cos  = F.cosine_similarity(z0_hat * mode_mask, z0 * mode_mask, dim=-1).clamp(-1, 1)
    L_dir = (1.0 - cos).mean()

    loss = base + 0.5 * L_x0 + 0.1 * L_dir
    dbg = {"z0_hat": z0_hat, "v_pred": v_pred, "v_tgt": v_tgt,
           "eps": eps, "eps_hat": eps_hat, "zt": zt}
    return loss, dbg


    
def kabsch_align(P: torch.Tensor, Q: torch.Tensor):
    """
    Align coordinates P to Q (both [N,3]) with Kabsch; returns aligned P.
    Detached from rotation/translation params to keep the graph simple—gradients
    keep flowing through P.
    """
    # subtract centroids
    Pc = P - P.mean(dim=0, keepdim=True)
    Qc = Q - Q.mean(dim=0, keepdim=True)
    H = Pc.T @ Qc
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    # correct possible reflection
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Q.mean(dim=0, keepdim=True) - Pc.mean(dim=0, keepdim=True) @ R.T
    return Pc @ R.T + Q.mean(dim=0, keepdim=True)



@torch.no_grad()
def v_probe_metrics(model, batch, t):
    """
    One-shot probe at time t. Builds a synthetic z_t from z0 and random ε,
    runs the model, reconstructs z0̂ via v-parameterization, and returns
    scalar diagnostics expected by trainer.py.
    """
    z0        = batch["z0"]            # [B,M]
    eigvals   = batch["eigvals"]       # [B,M]
    mode_mask = batch["mode_mask"]     # [B,M]
    x         = batch["x"]             # [B,N,3]
    h         = batch["h"]             # [B,N,Hf]
    node_mask = batch["node_mask"]     # [B,N]
    edge_radius = float(batch["edge_radius"][0].item()) if isinstance(batch["edge_radius"], torch.Tensor) else float(batch["edge_radius"])

    # forward sample
    eps = torch.randn_like(z0)
    alpha_t, sigma_t = alpha_sigma_from_t(t)
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
    zt    = alpha_t * z0 + sigma_t * eps
    v_tgt = alpha_t * eps - sigma_t * z0

    # predict v, reconstruct x0̂ and ε̂
    v_pred = model(zt, eigvals, mode_mask, t, x, h, node_mask, W=None, edge_radius=edge_radius) * mode_mask
    denom  = (alpha_t.pow(2) + sigma_t.pow(2)) + 1e-8
    z0_hat = (alpha_t * zt - sigma_t * v_pred) / denom
    eps_hat = (alpha_t * v_pred + sigma_t * zt) / denom

    m = mode_mask
    denom_modes = m.sum(1).clamp_min(1)

    mse = (( (z0_hat - z0).pow(2) * m ).sum(1) / denom_modes).mean().item()
    mae = (( (z0_hat - z0).abs()    * m ).sum(1) / denom_modes).mean().item()
    cos = F.cosine_similarity(z0_hat * m, z0 * m, dim=1).mean().item()

    # aggregate stds over valid entries
    def _std(tens):
        return float(torch.sqrt(((tens * m)**2).sum() / m.sum().clamp_min(1)).item())

    out = dict(
        mse=mse,
        mae=mae,
        cos=cos,
        corr=cos,  # alias expected by your print
        eps_std=_std(eps),
        eps_hat_std=_std(eps_hat),
        v_pred_std=_std(v_pred),
        v_tgt_std=_std(v_tgt),
        z0_hat_norm=float(( (z0_hat * m).norm(dim=1).mean() ).item()),
        z0_norm=float(( (z0 * m).norm(dim=1).mean() ).item()),
        alpha_sigma_sq=float((alpha_t.pow(2) + sigma_t.pow(2)).mean().item()),
    )
    return out

@torch.no_grad()
def probe_at_t(model, batch, t_scalar: float, *, seed: int = 1234, min_alpha: float = 1e-3):

    z0         = batch["z0"];       eigvals   = batch["eigvals"]
    mode_mask  = batch["mode_mask"]; x        = batch["x"]
    h          = batch["h"];         node_mask = batch["node_mask"]
    edge_rad   = float(batch["edge_radius"][0].item())

    if mode_mask.dim() == 1:
        mode_mask = mode_mask.view(1, -1)
    m = mode_mask.to(z0.dtype)

    B, M = z0.shape
    device = z0.device
    t = torch.full((B,), float(t_scalar), device=device)

    alpha, sigma = alpha_sigma_from_t(t)
    alpha = alpha.clamp_min(min_alpha).view(B, 1)
    sigma = sigma.view(B, 1)

    torch.manual_seed(int(seed))
    eps = torch.randn_like(z0)
    z_t = (alpha * z0 + sigma * eps) * m

    # v-pred forward (your model predicts v)
    v_pred = model(z_t, eigvals, mode_mask, t, x, h, node_mask, W=None, edge_radius=edge_rad) * m

    # v-consistent reconstruction (α²+σ²=1 for your schedule)
    x0_hat = (alpha * z_t - sigma * v_pred)  # /(alpha.pow(2)+sigma.pow(2)) == identity

    # Metrics (true cosine with L2)
    denom_modes = m.sum(1).clamp_min(1)
    mse = (( (x0_hat - z0).pow(2) * m ).sum(1) / denom_modes).mean().item()
    mae = (( (x0_hat - z0).abs()    * m ).sum(1) / denom_modes).mean().item()
    cos = F.cosine_similarity(x0_hat * m, z0 * m, dim=1).mean().item()

    return {"z_t": z_t, "z_star": z0, "x0_hat": x0_hat, "metrics": {"mse": mse, "mae": mae, "corr": cos}}


@torch.no_grad()
def ddim_probe(model, batch, *, steps: int = 20,
               t_start: float = 0.995, t_end: float = 0.01,
               seed: int = 1234, start_mode: str = "cold",
               return_all: bool = True):
    """
    Deterministic DDIM (η=0) using the model's v-parameterization.
    - Linear stepping in t: ts = linspace(t_start, t_end, steps+1)
    - Update uses: x0_hat = α_t * z_t - σ_t * v_pred
                   ε_hat  = α_t * v_pred + σ_t * z_t
                   z_{t'} = α_{t'} * x0_hat + σ_{t'} * ε_hat
    Returns either final x0_hat or a dict with metrics (when return_all=True).
    """


    z0        = batch["z0"]
    eigvals   = batch["eigvals"]
    mode_mask = batch["mode_mask"]
    x         = batch["x"]
    h         = batch["h"]
    node_mask = batch["node_mask"]
    edge_rad  = float(batch.get("edge_radius", torch.tensor([0.0], device=z0.device))[0].item())

    device, dtype = z0.device, z0.dtype
    B = z0.shape[0]
    m = mode_mask.to(dtype)

    # Linear t schedule (to mirror training)
    ts = torch.linspace(float(t_start), float(t_end), steps + 1, device=device, dtype=dtype)

    # Initialize z_t at t_start
    a0, s0 = alpha_sigma_from_t(ts[0].expand(B))
    A0 = a0.clamp_min(1e-5).view(B, 1)
    S0 = s0.view(B, 1)

    torch.manual_seed(int(seed))
    if start_mode == "cold":
        # cold start: pure noise at t_start
        z_t = (S0 * torch.randn_like(z0)) * m
    else:
        # warm start: sample from p(z_t | z0)
        eps = torch.randn_like(z0)
        z_t = (A0 * z0 + S0 * eps) * m

    # DDIM loop (η=0), v-parameterization throughout
    for i in range(steps):
        t_i = ts[i].expand(B)
        t_j = ts[i + 1].expand(B)

        Ai, Si = alpha_sigma_from_t(t_i); Ai = Ai.clamp_min(1e-5).view(B, 1); Si = Si.view(B, 1)
        Aj, Sj = alpha_sigma_from_t(t_j); Aj = Aj.clamp_min(1e-5).view(B, 1); Sj = Sj.view(B, 1)

        # model predicts v
        v_pred = model(z_t, eigvals, mode_mask, t_i, x, h, node_mask, W=None, edge_radius=edge_rad) * m

        # reconstructions (v-consistent)
        x0_hat  = (Ai * z_t - Si * v_pred)
        eps_hat = (Ai * v_pred + Si * z_t)

        # deterministic DDIM update
        z_t = (Aj * x0_hat + Sj * eps_hat) * m

    # Recompute x0̂ at the final t for reporting
    te = ts[-1].expand(B)
    Ae, Se = alpha_sigma_from_t(te); Ae = Ae.clamp_min(1e-5).view(B, 1); Se = Se.view(B, 1)
    v_end = model(z_t, eigvals, mode_mask, te, x, h, node_mask, W=None, edge_radius=edge_rad) * m
    x0_hat_end = (Ae * z_t - Se * v_end)

    if not return_all:
        return x0_hat_end

    # Masked metrics vs. z*
    denom = m.sum().clamp_min(1.0)
    mse = float(((x0_hat_end - z0).pow(2) * m).sum() / denom)
    mae = float(((x0_hat_end - z0).abs()    * m).sum() / denom)
    cos = F.cosine_similarity((x0_hat_end * m).view(B, -1), (z0 * m).view(B, -1), dim=1).mean().item()

    return {
        "z_t": z_t,
        "z_star": z0,
        "x0_hat": x0_hat_end,
        "metrics": {"mse": mse, "mae": mae, "corr": cos},
    }