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

    p_cold = 0.7
    use_cold = torch.rand((), device=z0.device) < p_cold
    if use_cold:
    # replace only the model's input; keep v_tgt as-is (computed from the true pair)
    # cold view at the same t:
        zt_model = (sigma_t * torch.randn_like(z0))
    else:
        zt_model = zt

    # predict eps
    v_pred = model(zt_model, eigvals, mode_mask, t, x, h, node_mask, W=None, edge_radius=edge_radius)


    z0_hat = (alpha_t*zt-sigma_t*v_pred) * mode_mask
    eps_hat = (alpha_t * v_pred + sigma_t * zt) #* mode_mask

    # robust (Huber) loss with mask
    # SmoothL1Loss uses 'beta' (a.k.a. delta) in PyTorch 1.10+ via 'beta' or 'delta' depending on version;
    # in F.smooth_l1_loss: 'beta' kw may be 'beta' or 'delta'. We'll use the functional form with reduction='none'.
    per = F.smooth_l1_loss(v_pred, v_tgt, reduction='none', beta=huber_delta)
    per = (per * mode_mask).sum(1) / mode_mask.sum(1).clamp_min(1)
    base = per.mean()

    # scale regularizer: pull std(eps_pred) toward std(eps)
    with torch.no_grad():
        eps_std = _std_over_mask(eps, mode_mask)
    eps_pred_std = _std_over_mask(eps_hat, mode_mask)
    var_pen = torch.abs(eps_pred_std - eps_std)

    # mean regularizer: keep eps_pred mean ~ 0 over masked entries
    mu = ((eps_hat * mode_mask).sum() / mode_mask.sum().clamp_min(1))
    mean_pen = mu.pow(2)
    #print("Current at time ",t[0])
    #print("v_pred = ",v_pred[0,:5])
    #print("v_tgt  = ", v_tgt[0,:5])
    #print("eps_hat= ", eps_hat[0,:5])
    #print("eps   = ", eps[0,:5])
    snr = alpha_t**2/(sigma_t**2+1e-12)
    w = alpha_t**2/(sigma_t**2+1e-12)
    w = 1.0/(1+w)
    w_hi = torch.minimum(snr, torch.full_like(snr, 5.0))
    L_x0_hi = (w_hi * (z0_hat - z0)**2 * mode_mask).sum() / mode_mask.sum().clamp_min(1)

    w_lo = (1.0 / (snr + 1e-12)).clamp_max(5.0)
    L_x0_lo = (w_lo * (z0_hat - z0).pow(2) * mode_mask).sum() / mode_mask.sum().clamp_min(1)

    cos = F.cosine_similarity(z0_hat*mode_mask, z0*mode_mask, dim=-1).clamp(-1, 1)

    L_dir = (1 - cos).mean()

    loss = w*base + 0.5*L_x0_hi+0.5*L_x0_lo + 0.1*L_dir #+ var_reg * var_pen + mean_reg * mean_pen

    dbg = {"z0_hat": z0_hat, "v_pred": v_pred, "v_tgt": v_tgt,
           "eps": eps, "eps_hat": eps_hat, "zt": zt}
    return loss, dbg

def sample_noisy_modal(batch, t: torch.Tensor):
    z0 = batch["z0"]
    eps = torch.randn_like(z0)
    zt = add_noise(z0, t, eps)
    return eps, zt

def epsilon_loss_given(model, batch, t: torch.Tensor, eps: torch.Tensor, zt: torch.Tensor):
    """
    Standard epsilon MSE in modal space.
    Uses collate keys: x, h, node_mask, edge_radius.
    """
    eig = batch["eigvals"]
    mode_mask = batch["mode_mask"]
    x = batch["x"]
    h = batch["h"]
    node_mask = batch["node_mask"]
    radius = float(batch["edge_radius"].mean().item())  # single float for the forward

    eps_pred = model(zt, eig, mode_mask, t, x, h, node_mask, None, radius)  # W=None
    diff = (eps_pred - eps) * mode_mask
    loss = (diff.pow(2).sum(dim=1) / mode_mask.sum(dim=1).clamp_min(1)).mean()
    return loss, eps_pred


def epsilon_loss(model, batch, t: torch.Tensor):
    """
    Epsilon MSE in modal space using keys produced by collate_batch():
      x, h, node_mask, z0, eigvals, mode_mask, edge_radius
    Returns (loss, dbg) where dbg contains z0_hat, eps, zt for debugging.
    """
    # Modal targets
    z0        = batch["z0"]           # [B,M]
    eigvals   = batch["eigvals"]      # [B,M]
    mode_mask = batch["mode_mask"]    # [B,M] (0/1)

    # Graph conditioning
    x         = batch["x"]            # [B,N,3]
    h         = batch["h"]            # [B,N,F]  (F=3 in your config)
    node_mask = batch["node_mask"]    # [B,N]
    er = batch["edge_radius"]         # [B]

    # scalar cutoff for now (per-item radius support can be added later)
    edge_radius = float(er[0].item()) if isinstance(er, torch.Tensor) else float(er)

    # Sample noise and build z_t
    eps = torch.randn_like(z0)
    zt  = add_noise(z0, t, eps)       # z_t = α_t z0 + σ_t ε

    # Predict ε
    eps_pred = model(
        zt, eigvals, mode_mask, t,
        x, h, node_mask,
        W=None, edge_radius=edge_radius
    )

    # Masked MSE over modes
    denom = mode_mask.sum(dim=1).clamp_min(1)
    loss_b = ((eps_pred - eps).pow(2) * mode_mask).sum(dim=1) / denom
    loss = loss_b.mean()

    # Reconstruct z0_hat for debugging:  z0_hat = (z_t - σ_t ε̂) / α_t
    alpha_t, sigma_t = alpha_sigma_from_t(t)         # [B]
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
    z0_hat = ((zt - sigma_t * eps_pred) / (alpha_t + 1e-8)) * mode_mask

    return loss, {"z0_hat": z0_hat, "eps": eps, "zt": zt}
    
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


def v_loss_direct(
    model, batch, t,
    *, huber_delta=1.0, lambda_eps: float = 0.05, mean_reg: float = 0.01,
    p_zero: float = 0.0, var_reg_eps: float = 0.0, var_reg_v: float = 0.0
):
    z0        = batch["z0"]
    eigvals   = batch["eigvals"]
    mode_mask = batch["mode_mask"]
    x         = batch["x"]
    h         = batch["h"]
    node_mask = batch["node_mask"]
    edge_radius = float(batch["edge_radius"][0].item()) if isinstance(batch["edge_radius"], torch.Tensor) else float(batch["edge_radius"])

    B, M = z0.shape

    # optional z0 dropout (disabled when p_zero=0)
    if p_zero > 0:
        drop = (torch.rand_like(z0) < p_zero) & (mode_mask > 0)
        z0_eff = z0.masked_fill(drop, 0.0)
    else:
        z0_eff = z0

    # sample forward
    eps = torch.randn_like(z0)
    alpha_t, sigma_t = alpha_sigma_from_t(t)
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
    #print("Epoch Step")
    #print("Values : ", "t : ",t, "alpha_t[0,:5]", alpha_t[0,:5], "sigma_t[0,:5]", sigma_t[0,:5])
    #print("z_0_eff : ", z0_eff[0,:5])
    zt    = alpha_t * z0_eff + sigma_t * eps
    v_tgt = alpha_t * eps     - sigma_t * z0_eff

    # predict v directly
    v_pred = model(zt, eigvals, mode_mask, t, x, h, node_mask, W=None, edge_radius=edge_radius)
    #print("zt : ", zt[0,:5])
    #print("v_tgt : ", v_tgt[0,:5])
    #print("v_pred : ", v_pred[0,:5])
    # robust masked Huber on v
    per   = F.smooth_l1_loss(v_pred, v_tgt, reduction='none', beta=huber_delta)
    denom = mode_mask.sum(1).clamp_min(1)
    base  = ((per * mode_mask).sum(1) / denom).mean()

    # tiny ε-aux (stabilizer)
    denom_as = (alpha_t.pow(2) + sigma_t.pow(2))
    eps_hat  = (alpha_t * v_pred + sigma_t * zt) / (denom_as + 1e-8)
    eps_mse  = (((eps_hat - eps).pow(2) * mode_mask).sum(1) / denom).mean()
    mu       = ((eps_hat * mode_mask).sum() / mode_mask.sum().clamp_min(1))
    mean_pen = mu.pow(2) * mean_reg

    loss = base + lambda_eps * eps_mse + mean_pen

    # (optional) variance anchors — keep OFF for now
    if var_reg_eps > 0.0:
        # match ε̂ std to 1.0 without exploding scale
        eps_std = torch.sqrt(((eps_hat * mode_mask) ** 2).sum() / mode_mask.sum().clamp_min(1))
        loss = loss + var_reg_eps * (eps_std - 1.0).abs()

    if var_reg_v > 0.0:
        v_pred_std = torch.sqrt(((v_pred * mode_mask) ** 2).sum() / mode_mask.sum().clamp_min(1))
        v_tgt_std  = torch.sqrt(((v_tgt  * mode_mask) ** 2).sum() / mode_mask.sum().clamp_min(1))
        loss = loss + var_reg_v * (v_pred_std - v_tgt_std).abs()

    with torch.no_grad():
        z0_hat = (alpha_t * zt - sigma_t * v_pred) / (denom_as + 1e-8)

    dbg = {"z0_hat": z0_hat * mode_mask, "v_pred": v_pred, "v_tgt": v_tgt,
           "eps": eps, "eps_hat": eps_hat, "zt": zt}
    return loss, dbg


@torch.no_grad()
def v_diagnostics(model, batch, t):
    """
    Run a single forward consistent with v-training and return scalars
    needed for logging (uses the same α,σ maps as the loss).
    """
    z0        = batch["z0"]
    eigvals   = batch["eigvals"]
    mode_mask = batch["mode_mask"]
    x         = batch["x"]
    h         = batch["h"]
    node_mask = batch["node_mask"]
    edge_radius = float(batch["edge_radius"][0].item()) if isinstance(batch["edge_radius"], torch.Tensor) else float(batch["edge_radius"])

    # fresh noise for diagnostics
    eps = torch.randn_like(z0)
    alpha_t, sigma_t = alpha_sigma_from_t(t)
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)

    zt    = alpha_t * z0 + sigma_t * eps
    v_tgt = alpha_t * eps - sigma_t * z0

    v_pred = model(zt, eigvals, mode_mask, t, x, h, node_mask, W=None, edge_radius=edge_radius)

    # invert maps for x0̂ and ε̂ (consistent with v-training)
    denom_as = (alpha_t.pow(2) + sigma_t.pow(2))
    z0_hat   = (alpha_t * zt - sigma_t * v_pred) / (denom_as + 1e-8)
    eps_hat  = (alpha_t * v_pred + sigma_t * zt) / (denom_as + 1e-8)

    m = mode_mask
    denom = m.sum(1).clamp_min(1)
    mse  = (((z0_hat - z0).pow(2) * m).sum(1) / denom).mean()
    mae  = (((z0_hat - z0).abs()   * m).sum(1) / denom).mean()
    cos  = torch.nn.functional.cosine_similarity((z0_hat*m), (z0*m), dim=1).mean()

    # stds
    def _std_over_mask(x): return torch.sqrt(((x*m)**2).sum() / m.sum().clamp_min(1))
    eps_std      = _std_over_mask(eps).item()
    eps_hat_std  = _std_over_mask(eps_hat).item()
    v_pred_std   = _std_over_mask(v_pred).item()
    v_tgt_std    = _std_over_mask(v_tgt).item()

    return {
        "z0_hat": z0_hat, "eps": eps, "eps_hat": eps_hat, "zt": zt,
        "v_pred": v_pred, "v_tgt": v_tgt,
        "mse": float(mse.item()), "mae": float(mae.item()), "cos": float(cos.item()),
        "eps_std": eps_std, "eps_hat_std": eps_hat_std,
        "v_pred_std": v_pred_std, "v_tgt_std": v_tgt_std,
    }

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