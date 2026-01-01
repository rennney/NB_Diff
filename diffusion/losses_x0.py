# nb_sim/diffusion/losses.py
from __future__ import annotations
import torch
from .noise import add_noise, alpha_sigma_from_t
import torch.nn.functional as F

import torch
import torch.nn.functional as F

# ---------- Ranks & weights ----------
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

def _cartesian_tether(x0_pred, z0, mode_mask, idxs, hi, decoder_bank,
                      huber_delta=1.0, w_cart=0.2):
    """
    High-t Cartesian supervision using per-item decoders.
    x0_pred, z0, mode_mask: [B,M]
    idxs: [B] dataset indices (long)
    hi: [B,1] high-t gate (≈1 at t~1, 0 otherwise)
    decoder_bank: list of ds.modesets with .to_cartesian(modal_vector)
    """
    B, M = x0_pred.shape
    total = x0_pred.new_zeros(())
    for b in range(B):
        idx = int(idxs[b].item())
        ms  = decoder_bank[idx]  # ModesProxy for this row

        # mask modes & decode; ensure dtype/device match the modeset
        z_pred_b = (x0_pred[b] * mode_mask[b]).to(device=ms.device, dtype=ms.dtype)
        z_tgt_b  = (z0[b]      * mode_mask[b]).to(device=ms.device, dtype=ms.dtype)

        xyz_pred = ms.to_cartesian(z_pred_b)  # [Nb,3]
        xyz_tgt  = ms.to_cartesian(z_tgt_b)   # [Nb,3]

        # smooth-L1 in Cartesian coords
        Lb = F.smooth_l1_loss(xyz_pred, xyz_tgt, reduction="mean", beta=huber_delta)
        #print("RSMD from cartesian loss: ", _rmsd(xyz_pred, xyz_tgt))
        # gate by high-t for this sample and accumulate on the *loss* device/dtype
        total = total + Lb.to(x0_pred.device, x0_pred.dtype) * hi[b, 0]

    # average over batch and scale
    return (total / max(1, B)) * w_cart

def _rank_by_value(eigvals: torch.Tensor, mode_mask: torch.Tensor) -> torch.Tensor:
    """
    Percentile ranks in (0,1] by *value* among valid modes, per item.
    Returns float ranks with padded modes = 0 (masked out).
    """
    m = mode_mask.to(eigvals.dtype)
    B, M = eigvals.shape
    # push padded entries to +inf so they sort last
    inf = torch.tensor(float('inf'), device=eigvals.device, dtype=eigvals.dtype)
    masked = torch.where(m > 0, eigvals, inf)

    # ascending order by λ
    order = torch.argsort(masked, dim=1, stable=True)            # [B,M], Long
    # inverse permutation: rank position for each original index
    ranks = torch.empty_like(masked)                              # Float
    src = torch.arange(M, device=eigvals.device, dtype=ranks.dtype).view(1, M).expand(B, M)
    ranks.scatter_(1, order, src)                                 # Float <- Float OK

    denom = m.sum(1, keepdim=True).clamp_min(1)                   # valid count per item
    r = (ranks + 0.5) / denom                                     # (0,1]; padded become >1 but we zero by mask next
    return r * (m > 0)                                            # zero-out padded

def stiff_weight(eigvals: torch.Tensor, mode_mask: torch.Tensor,
                 w_lo: float = 0.5, w_hi: float = 2.0, gamma: float = 2.0) -> torch.Tensor:
    """
    Bounded (w_lo..w_hi) weight that grows with stiffness (larger λ), scale-free.
    """
    r = _rank_by_value(eigvals, mode_mask)                        # 0..1
    w = w_lo + (w_hi - w_lo) * r.pow(gamma)
    return (w * mode_mask).detach()                               # detach: no grads through weights

def stiff_weight_log(eigvals: torch.Tensor, mode_mask: torch.Tensor,
                     w_lo: float = 0.5, w_hi: float = 2.0, a: float = 2.0) -> torch.Tensor:
    """
    Alternative: sigmoid of z-scored log λ (also scale-free).
    """
    eps = 1e-12
    m = mode_mask.to(eigvals.dtype)
    loglam = eigvals.clamp_min(eps).log()
    cnt = m.sum(1, keepdim=True).clamp_min(1)
    mu  = (loglam * m).sum(1, keepdim=True) / cnt
    sd  = ((loglam - mu).pow(2) * m).sum(1, keepdim=True).div(cnt).clamp_min(1e-8).sqrt()
    z   = (loglam - mu) / sd
    s   = torch.sigmoid(a * z)
    w   = w_lo + (w_hi - w_lo) * s
    return (w * m).detach()

# ---------- Subset exact supervision (vectorized) ----------

def _sample_subset_mask(probs: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Vectorized per-row top-K sampling without Python loops.
    probs: [B,M] nonnegative, rows sum to 1 over valid modes; K: [B] ints >=1
    Returns binary mask S: [B,M] with exactly K[b] ones per row.
    """
    B, M = probs.shape
    # Gumbel-top-K trick
    g = -torch.log(-torch.log(torch.rand_like(probs).clamp_min(1e-12)))
    scores = torch.log(probs.clamp_min(1e-20)) + g
    # different K per row → take max K over batch, then mask extras
    Kmax = int(K.max().item())
    topk_scores, topk_idx = torch.topk(scores, k=Kmax, dim=1)
    S = torch.zeros_like(probs)
    arangeB = torch.arange(B, device=probs.device)
    # Build a per-row mask that keeps only the first K[b] indices
    keep = (torch.arange(Kmax, device=probs.device).view(1, Kmax) <
            K.view(B, 1)).to(S.dtype)                               # [B,Kmax]
    S[arangeB.unsqueeze(1), topk_idx] = keep                        # broadcasted assignment
    return S

def subset_exact_loss(x0_pred: torch.Tensor, z0: torch.Tensor,
                      mode_mask: torch.Tensor, eigvals: torch.Tensor, hi: torch.Tensor,
                      frac: float = 0.10, w_subset: float = 1.0,
                      huber_delta: float = 1.0, favor_stiff: bool = True) -> torch.Tensor:
    """
    Exact per-mode loss on a random subset (~frac of valid modes) per sample, gated to high-t by `hi`.
    """
    m = mode_mask.to(x0_pred.dtype)
    B, M = x0_pred.shape
    Mv = m.sum(1)                                                  # [B]
    K  = torch.clamp((Mv * frac).long(), min=1)                    # [B]

    if favor_stiff:
        r = _rank_by_value(eigvals, m)                             # [B,M]
        probs = (r.pow(2.0) + 1e-8) * m
    else:
        probs = (m + 1e-8)

    probs = probs / probs.sum(1, keepdim=True).clamp_min(1e-12)
    S = _sample_subset_mask(probs, K) * m                          # [B,M] binary

    per = F.smooth_l1_loss(x0_pred, z0, reduction="none", beta=huber_delta)  # [B,M]
    L_subset = (per * S).sum(1) / S.sum(1).clamp_min(1)                        # mean over subset
    return (hi.squeeze(-1) * L_subset).mean() * w_subset


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


def x0_loss_unused(model, batch, t, huber_delta=1.0,
            lambda_hi=0.30,        # high-t floor for x0
            p_cold=0.30,           # fraction of *pure-noise* inputs (reuse same eps)
            w_eps=0.50,            # high-t ε̂ supervision
            w_dir=0.20,            # high-t directional (cosine) term
            w_norm=0.05,           # high-t norm match
            w_v=2.0):              # high-t v̂ supervision (with stop-grad)

    z0        = batch["z0"]          # [B,M]
    eigvals   = batch["eigvals"]     # [B,M]
    mode_mask = batch["mode_mask"]   # [B,M]
    x         = batch["x"]           # [B,N,3]
    h         = batch["h"]           # [B,N,F]
    node_mask = batch["node_mask"]   # [B,N]
    edge_radius = float(batch["edge_radius"])

    # --- forward sample ---
    eps = torch.randn_like(z0)
    alpha_t, sigma_t = alpha_sigma_from_t(t)  # [B]
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)

    m    = mode_mask
    snr  = (alpha_t**2) / (sigma_t**2 + 1e-12)     # [B,1]
    # HARD high-noise gate (≈ t>=~0.985 for your schedule; tune 5e-3 ↔ 1e-2)
    high_t = (snr < 5e-3).float()                  # [B,1]

    # Cold augmentation: ALWAYS cold when high_t, otherwise keep your 0.3
    p_cold = 0.30
    do_cold = (torch.rand(z0.size(0), 1, device=z0.device) < p_cold).float()
    do_cold = torch.where(high_t.bool(), torch.ones_like(do_cold), do_cold)  # force cold @ high-t

    zt_base = (alpha_t * z0 + sigma_t * eps) * m
    zt_cold = (sigma_t * eps) * m
    zt = do_cold * zt_cold + (1.0 - do_cold) * zt_base

    # --- model predicts clean x0 ---
    x0_pred = model(zt, eigvals, mode_mask, t, x, h, node_mask,
                    W=None, edge_radius=edge_radius) * m

    # ---------- Primary x0 objective ----------
    # Keep your SNR weighting + a small floor that DOES NOT bleed into mid/low t.
    w_x0_base = (snr / (1.0 + snr))                          # [B,1]
    w_x0      = w_x0_base + (0.6 * high_t)                   # floor only when high_t
    per_x0    = F.smooth_l1_loss(x0_pred, z0, reduction="none", beta=huber_delta)
    per_x0    = (per_x0 * m).sum(1) / m.sum(1).clamp_min(1)
    L_x0      = (w_x0.squeeze(-1) * per_x0).mean()

    # ---------- High-t ONLY: finish amplitude & velocity; NO ε loss ----------
    # (A) v-from-x0 supervision (stop-grad through x0 so it can't fight L_x0)
    #     At α≈0, v̂ ≈ −σ·x0̂ and v_tgt=−σ·z0  ⇒  σ^2‖x0̂−z0‖^2
    v_tgt = (alpha_t * eps - sigma_t * z0) * m
    v_hat_det = (- sigma_t * x0_pred.detach()) * m
    per_v = F.smooth_l1_loss(v_hat_det, v_tgt, reduction="none", beta=huber_delta)
    per_v = (per_v * m).sum(1) / m.sum(1).clamp_min(1)
    L_v_hi = (high_t.squeeze(-1) * per_v).mean()
    #eps_hat = ((zt - alpha_t * x0_pred) / sigma_safe) * m
    #v_hat   = (alpha_t * eps_hat - sigma_t * x0_pred) * m
    #v_tgt   = (alpha_t * eps - sigma_t * z0) * m

    #per_v = F.smooth_l1_loss(v_hat, v_tgt, reduction="none", beta=huber_delta)
    #per_v = (per_v * m).sum(1) / m.sum(1).clamp_min(1)
    #L_v_hi = (hi.squeeze(-1) * per_v).mean() * w_v

    # (B) per-mode sign + amplitude, but ONLY on dominant modes & ONLY at high-t
    abs_z0 = (z0.abs() * m)
    # focus on top-K modes per item (K~min(32, M//8)); simple percentile mask:
    K = max(8, min(32, abs_z0.shape[1] // 8))
    thr, _ = torch.kthvalue(abs_z0 + (1 - m) * 1e9, k=abs_z0.shape[1] - K, dim=1, keepdim=True)
    focus = (abs_z0 >= thr).to(z0.dtype) * m

    # smooth sign match with tanh (gives gradient; sign() is dead)
    beta = 3.0
    s_hat = torch.tanh(beta * x0_pred)
    s_gt  = torch.tanh(beta * z0)
    L_sign_hi = ( ( (s_hat - s_gt).abs() * focus ).sum(1) / focus.sum(1).clamp_min(1) )
    L_sign_hi = (high_t.squeeze(-1) * L_sign_hi).mean()

    # absolute amplitude match on the same focused modes
    L_abs_hi = ( ( (x0_pred.abs() - z0.abs()).abs() * focus ).sum(1) / focus.sum(1).clamp_min(1) )
    L_abs_hi = (high_t.squeeze(-1) * L_abs_hi).mean()

    # ---------- total (ε̂ term intentionally removed at high-t) ----------
    loss = L_x0 + 2.0 * L_v_hi + 0.4 * L_sign_hi + 0.1 * L_abs_hi

    dbg = {
        "z0_hat": x0_pred.detach(),
        "zt": zt.detach(),
        "v_tgt": v_tgt.detach(),
    }
    return loss, dbg

def x0_loss_v2(model, batch, t, huber_delta=1.0,
            lambda_hi=0.3,        # high-t floor for x0
            p_cold=0.30,           # fraction of *pure-noise* inputs (reuse same eps)
            w_eps=0.25,            # high-t ε̂ supervision
            w_dir=0.5,            # high-t directional (cosine) term
            w_norm=0.1,           # high-t norm match
            w_v=1.0,
            w_sign=0.15,
            decoder_bank=None,            # modes for Cart Loss
            w_cart=3.0               # weight for Cartesian term
):              # high-t v̂ supervision (with stop-grad)

    z0        = batch["z0"]
    eigvals   = batch["eigvals"]
    mode_mask = batch["mode_mask"]
    x         = batch["x"]
    h         = batch["h"]
    node_mask = batch["node_mask"]
    edge_radius = float(batch["edge_radius"])
    idxs      = batch["idxs"].long()  

    # --- sample noise & make both z_t variants using the SAME eps ---
    eps = torch.randn_like(z0)
    alpha_t, sigma_t = alpha_sigma_from_t(t)
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)

    m = mode_mask
    zt_base = (alpha_t * z0 + sigma_t * eps) * m                # standard αz0 + σε
    zt_cold = (sigma_t * eps) * m                                # pure noise (α≈0 behaviour)

    B = z0.shape[0]
    cold_mask = (torch.rand(B, 1, device=z0.device) < p_cold).to(z0.dtype)
    zt = cold_mask * zt_cold + (1.0 - cold_mask) * zt_base       # per-sample mix (cold or standard)


    eps_in = torch.randn_like(z0)             # independent noise (NOT the one used in targets)
    z_in = (alpha_t * z0 + sigma_t * eps_in) * m

    hi_mask = (t >= 0.97).unsqueeze(-1)       # only scramble at very high-t
    # at very high-t, remove α z0 term and keep only scrambled noise
    z_in = torch.where(hi_mask.bool(), (sigma_t * eps_in) * m, z_in)

    # normalize scale across t so the net sees O(1) magnitudes
    z_in = z_in / (sigma_t + 1e-8)
    # --- predict x0 and derive ε̂, v̂ ---
    x0_pred = model(zt, eigvals, mode_mask, t, x, h, node_mask,
                    W=None, edge_radius=edge_radius) * m

    # weights over t
    snr = (alpha_t**2) / (sigma_t**2 + 1e-12)
    hi  = 1.0 / (1.0 + snr)                                      # ~1 at high-t
    w_x0 = (0.2+lambda_hi * hi).clamp(max=1.0).detach()#(snr / (1.0 + snr)) + lambda_hi * hi                  # never vanishes at high-t

    # primary x0 (Huber) with mask
    #w_mode = (1.0 / torch.sqrt(1.0 + eigvals)).detach()  # [B,M], NN-friendly
    #per_elem = F.smooth_l1_loss(x0_pred, z0, reduction="none", beta=huber_delta) * m
    #per_x0 = (per_elem * w_mode).sum(1) / (w_mode * m).sum(1).clamp_min(1)
    per_x0 = F.smooth_l1_loss(x0_pred, z0, reduction="none", beta=huber_delta)
    per_x0 = (per_x0 * m).sum(1) / m.sum(1).clamp_min(1)
    L_x0   = (w_x0.squeeze(-1) * per_x0).mean()

    # derive ε̂ and v̂ *for diagnostics and eps/v losses*
    sigma_safe = sigma_t.clamp_min(1e-8)
    eps_hat = ((zt - alpha_t * x0_pred) / sigma_safe) * m
    v_tgt   = (alpha_t * eps - sigma_t * z0) * m

    # ---- High-t ε̂ supervision (stabilizes σ/α amplification) ----
    per_eps = F.smooth_l1_loss(eps_hat, eps, reduction="none", beta=huber_delta)
    per_eps = (per_eps * m).sum(1) / m.sum(1).clamp_min(1)
    L_eps_hi = (hi.squeeze(-1) * per_eps).mean() * w_eps

    # ---- High-t directional cosine on modes (shape match even if amplitude off) ----
    cos = F.cosine_similarity((x0_pred * m), (z0 * m), dim=1).clamp(-1, 1)
    L_dir_hi = ((1.0 - cos) * hi.squeeze(-1)).mean() * w_dir

    # ---- High-t norm match (global amplitude calibration) ----
    n_hat = (x0_pred * m).norm(dim=1)
    n_gt  = (z0 * m).norm(dim=1)
    L_norm_hi = (( (n_hat - n_gt).pow(2) / m.sum(1).clamp_min(1) ) * hi.squeeze(-1)).mean() * w_norm

    # ---- High-t v̂ supervision with stop-grad through x0 (stability) ----
    # use the *same* zt, but cut gradients through x0_pred so v-term doesn't fight x0-term
    eps_hat_det = ((zt - alpha_t * x0_pred.detach()) / sigma_safe) * m
    v_hat_det   = (alpha_t * eps_hat_det - sigma_t * x0_pred.detach()) * m
    per_v = F.smooth_l1_loss(v_hat_det, v_tgt, reduction="none", beta=huber_delta)
    per_v = (per_v * m).sum(1) / m.sum(1).clamp_min(1)
    w_v = w_v * (snr/(1.0+snr)).detach()  # downweight mid/low-t
    L_v_hi = (hi.squeeze(-1) * per_v).mean() * w_v


    # per-mode sign hinge at high-t
    w_mode = 1.0 / (z0.abs() + 1e-2)          # upweight small modes
    prod = x0_pred * z0                        # [B,M]
    L_sign = (F.relu(-prod) * w_mode * m).sum(1) / ( (w_mode * m).sum(1).clamp_min(1) )
    L_sign_hi = (hi.squeeze(-1) * L_sign).mean()

    den_rel = (z0.abs() + 1e-2)
    per_rel = F.smooth_l1_loss((x0_pred - z0) / den_rel,
                            torch.zeros_like(z0),
                            reduction="none", beta=0.5)
    per_rel = (per_rel * m).sum(1) / m.sum(1).clamp_min(1)
    L_rel_hi = (hi.squeeze(-1) * per_rel).mean()

    def _normed_energy(v):
        e = (v.pow(2) * m)
        e = e / e.sum(1, keepdim=True).clamp_min(1e-8)
        return e

    p = _normed_energy(z0).clamp_min(1e-8)
    q = _normed_energy(x0_pred).clamp_min(1e-8)
    kl = (p * (p.log() - q.log())).sum(1)                # KL(p||q) per item
    L_energy_hi = (hi.squeeze(-1) * kl).mean()

    L_subset_hi = subset_exact_loss(
    x0_pred, z0, mode_mask, eigvals, hi,
    frac=1, w_subset=0.5, huber_delta=huber_delta, favor_stiff=False
    )

    if decoder_bank is not None:
        L_cart_hi = _cartesian_tether(
            x0_pred=x0_pred, z0=z0, mode_mask=m, idxs=idxs, hi=hi,
            decoder_bank=decoder_bank, huber_delta=0.5, w_cart=w_cart
        )
    else:
        L_cart_hi = x0_pred.new_zeros(())

    loss = L_x0 + L_eps_hi + L_dir_hi + L_norm_hi + 2.0*L_v_hi + w_sign*L_sign_hi + L_cart_hi + 0.0*L_rel_hi + 0.0*L_energy_hi + 0.0*L_subset_hi

    #print(f"Total Loss : {loss.item():.4f},Loss components: L_x0={L_x0.item():.4f}, L_eps_hi={L_eps_hi.item():.4f}, L_dir_hi={L_dir_hi.item():.4f}, L_norm_hi={L_norm_hi.item():.4f}, L_v_hi={L_v_hi.item():.4f}, L_sign_hi={L_sign_hi.item():.4f}, L_cart_hi={L_cart_hi.item():.4f}, L_rel_hi={L_rel_hi.item():.4f}, L_energy_hi={L_energy_hi.item():.4f}, L_subset_hi={L_subset_hi.item():.4f}")

    dbg = {
        "z0_hat": x0_pred.detach(),
        "eps_hat": eps_hat.detach(),
        "eps": eps.detach(),
        "v_tgt": v_tgt.detach(),
        "zt": zt.detach(),
    }
    return loss, dbg


def x0_loss(model, batch, t, huber_delta=1.0,
            lambda_hi=0.30,         # floor on high-t weight inside L_x0
            p_cold=0.30,            # % of *pure-noise* inputs (cold start)
            w_dir=0.50,             # high-t directional cosine
            w_norm=0.10,            # high-t norm match
            w_v=1.00,               # high-t v-teacher
            w_sign=0.0,            # high-t sign stabilizer on focused modes
            decoder_bank=None,      # optional per-item decoder for Cartesian tether
            w_cart=0.0):            # Cartesian tether weight (if decoder_bank given)

    z0        = batch["z0"]          # [B,M]
    eigvals   = batch["eigvals"]     # [B,M]
    mode_mask = batch["mode_mask"]   # [B,M]
    x         = batch["x"]           # [B,N,3]
    h         = batch["h"]           # [B,N,F]
    node_mask = batch["node_mask"]   # [B,N]
    edge_radius = float(batch["edge_radius"])
    idxs      = batch.get("idxs", torch.arange(z0.shape[0], device=z0.device)).long()

    # schedules
    alpha_t, sigma_t = alpha_sigma_from_t(t)  # [B]
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
    m = mode_mask

    # build z_t (standard and cold) with the SAME eps
    eps = torch.randn_like(z0)
    zt_base = (alpha_t * z0 + sigma_t * eps) * m
    zt_cold = (sigma_t * eps) * m
    B = z0.shape[0]
    cold_mask = (torch.rand(B, 1, device=z0.device) < p_cold).to(z0.dtype)
    zt = cold_mask * zt_cold + (1.0 - cold_mask) * zt_base

    # -------- model predicts x0_hat directly --------
    x0_hat = model(zt, eigvals, m, t, x, h, node_mask, W=None, edge_radius=edge_radius) * m

    # targets in v-space (teacher) and an *implied* v̂ from x0̂
    v_tgt = (alpha_t * eps - sigma_t * z0) * m
    sigma_safe = sigma_t.clamp_min(1e-3)  # only used for v̂ to avoid blow-ups at low t
    v_hat = (alpha_t / sigma_safe) * zt - (1.0 / sigma_safe) * x0_hat
    v_hat = v_hat * m

    # SNR gates
    snr = (alpha_t**2) / (sigma_t**2 + 1e-12)  # [B,1]
    hi  = 1.0 / (1.0 + snr)                    # ~1 near t≈1
    lo  = snr / (1.0 + snr)                    # ~1 at mid/low-t

    # ---- losses ----
    denom_modes = m.sum(1).clamp_min(1)

    # (A) main x0 with SNR weighting + a high-t floor lambda_hi
    per_x0 = F.smooth_l1_loss(x0_hat, z0, reduction="none", beta=huber_delta) * m
    L_x0 = ((lo + lambda_hi * hi).squeeze(-1) * (per_x0.sum(1) / denom_modes)).mean()

    # (B) high-t teacher in v-space (uses v̂ derived from x0̂)
    per_v = F.smooth_l1_loss(v_hat, v_tgt, reduction="none", beta=huber_delta) * m
    L_v_hi = (hi.squeeze(-1) * (per_v.sum(1) / denom_modes)).mean()

    # (C) high-t directional cosine (shape) on x0̂ vs z0
    cos = F.cosine_similarity(x0_hat * m, z0 * m, dim=1).clamp(-1, 1)
    L_dir_hi = (hi.squeeze(-1) * (1.0 - cos)).mean()

    # (D) high-t norm match (amplitude)
    def _masked_norm(u):
        return torch.sqrt(((u * m)**2).sum(1) / denom_modes + 1e-12)
    L_norm_hi = (hi.squeeze(-1) *
                 F.smooth_l1_loss(_masked_norm(x0_hat), _masked_norm(z0),
                                   reduction="none", beta=1.0)).mean()

    # (E) sign stabilizer on focused modes (top-20% |z0| per item)
    with torch.no_grad():
        # compute per-item threshold among *valid* modes only
        z0_abs = (z0.abs() * m) + (~m.bool()).float() * (-1e9)
        k = (0.2 * denom_modes).long().clamp_min(1)
        # topk with per-item k: approximate by taking max-k then masking below threshold
        kk = int(k.max().item())
        vals, _ = torch.topk(z0_abs, k=kk, dim=1, largest=True)
        thresh = vals[:, kk-1:kk]  # [B,1]
        focus = ((z0.abs() >= thresh) * m).to(z0.dtype)
    beta = 3.0
    s_hat = torch.tanh(beta * x0_hat)
    s_gt  = torch.tanh(beta * z0)
    L_sign_hi = (hi.squeeze(-1) *
                 ((s_hat - s_gt).abs() * focus).sum(1) /
                 focus.sum(1).clamp_min(1)).mean()

    # (F) optional Cartesian tether (only if you pass decoder_bank)
    if decoder_bank is not None:
        L_cart_hi = _cartesian_tether(
            x0_pred=x0_hat, z0=z0, mode_mask=m, idxs=idxs, hi=hi,
            decoder_bank=decoder_bank, huber_delta=0.5, w_cart=w_cart
        )
    else:
        L_cart_hi = x0_hat.new_zeros(())

    loss = L_x0 + (w_v * L_v_hi) + (w_dir * L_dir_hi) + (w_norm * L_norm_hi) + (w_sign * L_sign_hi) + L_cart_hi

    dbg = {
        "z0_hat": x0_hat.detach(),
        "zt": zt.detach(),
        "v_tgt": v_tgt.detach(),
        "v_hat": v_hat.detach(),
    }
    return loss, dbg

def x0_loss_nogood(model, batch, t, huber_delta=1.0,
            lambda_hi=0.30,         # floor on high-t weight inside L_x0
            p_cold=0.30,            # % of *pure-noise* inputs (cold start)
            w_dir=0.50,             # high-t directional cosine
            w_norm=0.10,            # high-t norm match
            w_v=1.00,               # high-t v-teacher
            w_sign=0.15,            # high-t sign stabilizer on focused modes
            decoder_bank=None,      # optional per-item decoder for Cartesian tether
            w_cart=3.0):            # Cartesian tether weight (if decoder_bank given)

    z0        = batch["z0"]          # [B,M]
    eigvals   = batch["eigvals"]     # [B,M]
    mode_mask = batch["mode_mask"]   # [B,M]
    x         = batch["x"]           # [B,N,3]
    h         = batch["h"]           # [B,N,F]
    node_mask = batch["node_mask"]   # [B,N]
    edge_radius = float(batch["edge_radius"])
    idxs      = batch.get("idxs", torch.arange(z0.shape[0], device=z0.device)).long()  # used by _cartesian_tether

    # schedules
    alpha_t, sigma_t = alpha_sigma_from_t(t)  # [B]
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
    m = mode_mask

    # use the SAME eps to build both z_t views (standard and cold)
    eps = torch.randn_like(z0)
    zt_base = (alpha_t * z0 + sigma_t * eps) * m
    zt_cold = (sigma_t * eps) * m
    B = z0.shape[0]
    cold_mask = (torch.rand(B, 1, device=z0.device) < p_cold).to(z0.dtype)
    zt = cold_mask * zt_cold + (1.0 - cold_mask) * zt_base

    # model predicts v-per-mode (matches your probes/train loops)
    v_pred = model(zt, eigvals, m, t, x, h, node_mask, W=None, edge_radius=edge_radius) * m

    # targets in v-space (teacher) and reconstructions
    v_tgt  = (alpha_t * eps - sigma_t * z0) * m
    denom  = (alpha_t.pow(2) + sigma_t.pow(2)).clamp_min(1e-8)
    z0_hat = ((alpha_t * zt - sigma_t * v_pred) / denom) * m

    # SNR gates
    snr   = (alpha_t**2) / (sigma_t**2 + 1e-12)  # [B,1]
    hi    = 1.0 / (1.0 + snr)                    # ~1 near t≈1
    lo    = snr / (1.0 + snr)                    # ~1 at mid/low-t

    # ---- losses ----
    # (A) main x0 with SNR weighting + a high-t floor lambda_hi
    per_x0 = F.smooth_l1_loss(z0_hat, z0, reduction="none", beta=huber_delta) * m
    denom_modes = m.sum(1).clamp_min(1)
    L_x0 = ((lo + lambda_hi * hi).squeeze(-1) * (per_x0.sum(1) / denom_modes)).mean()

    # (B) high-t teacher in v-space
    per_v = F.smooth_l1_loss(v_pred, v_tgt, reduction="none", beta=huber_delta) * m
    L_v_hi = (hi.squeeze(-1) * (per_v.sum(1) / denom_modes)).mean()

    # (C) high-t directional cosine (shape)
    cos = F.cosine_similarity(z0_hat * m, z0 * m, dim=1).clamp(-1, 1)
    L_dir_hi = (hi.squeeze(-1) * (1.0 - cos)).mean()

    # (D) high-t norm match
    def _masked_norm(u):
        return torch.sqrt(((u * m)**2).sum(1) / denom_modes + 1e-12)
    L_norm_hi = (hi.squeeze(-1) * F.smooth_l1_loss(_masked_norm(z0_hat), _masked_norm(z0), reduction="none", beta=1.0)).mean()

    # (E) sign stabilizer on focused modes (top-20% |z0| per item)
    with torch.no_grad():
        k = (0.2 * denom_modes).long().clamp_min(1)  # per item
        thresh = torch.topk((z0.abs() * m), k.max().item(), dim=1, largest=True).values[
            torch.arange(B, device=z0.device), k.max().item()-1
        ].unsqueeze(1)
        focus = (z0.abs() >= thresh) * m
    beta = 3.0
    s_hat = torch.tanh(beta * z0_hat)
    s_gt  = torch.tanh(beta * z0)
    L_sign_hi = (hi.squeeze(-1) * ((s_hat - s_gt).abs() * focus).sum(1) / focus.sum(1).clamp_min(1)).mean()

    # (F) optional Cartesian tether (only if you pass decoder_bank)
    if decoder_bank is not None:
        L_cart_hi = _cartesian_tether(
            x0_pred=z0_hat, z0=z0, mode_mask=m, idxs=idxs, hi=hi,
            decoder_bank=decoder_bank, huber_delta=0.5, w_cart=w_cart
        )
    else:
        L_cart_hi = z0_hat.new_zeros(())

    loss = L_x0 + (w_v * L_v_hi) + (w_dir * L_dir_hi) + (w_norm * L_norm_hi) + (w_sign * L_sign_hi) + L_cart_hi

    dbg = {
        "z0_hat": z0_hat.detach(),
        "zt": zt.detach(),
        "v_tgt": v_tgt.detach(),
    }
    return loss, dbg


def x0v_loss(model, batch, t, huber_delta=1.0,
             p_cold=0.30,
             w_v_hi=1.0,          # main high-t teacher
             w_x0_lo=0.30,        # small mid/low-t anchor
             w_dir_hi=0.10,       # tiny directional shape at high-t
             w_norm_hi=0.05,      # tiny norm match at high-t
             w_sign_hi=0.05,      # sign stabilization on focused modes
             w_subset_hi=0.50,    # exact per-mode on random subset
             frac_subset=0.25,    # 10–40% works; start at 0.25
             use_eps_band=False,  # keep False unless you want a light band loss
             w_eps_band=0.10):

    z0        = batch["z0"]          # [B,M]
    eigvals   = batch["eigvals"]     # [B,M]
    mode_mask = batch["mode_mask"]   # [B,M]
    x         = batch["x"]           # [B,N,3]
    h         = batch["h"]           # [B,N,F]
    node_mask = batch["node_mask"]   # [B,N]
    edge_radius = float(batch["edge_radius"])

    # ---------- noise & schedules ----------
    eps = torch.randn_like(z0)
    alpha_t, sigma_t = alpha_sigma_from_t(t)  # [B]
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)

    m   = mode_mask
    snr = (alpha_t**2) / (sigma_t**2 + 1e-12)         # [B,1]
    hi  = 1.0 / (1.0 + snr)                           # ~1 at high-t
    lo  = (snr / (1.0 + snr))                         # ~1 at mid/low-t

    # hard high-t gate (roughly t≳0.985 for your schedule; tune if needed)
    hi_hard = (snr < 5e-3).float()                    # [B,1]

    # ---------- sampling: force cold at high-t + tiny mode dropout ----------
    B, M = z0.shape
    do_cold = (torch.rand(B, 1, device=z0.device) < p_cold).float()
    do_cold = torch.where(hi_hard.bool(), torch.ones_like(do_cold), do_cold)  # always cold at extreme high-t

    zt_base = (alpha_t * z0 + sigma_t * eps) * m
    zt_cold = (sigma_t * eps) * m
    zt = do_cold * zt_cold + (1.0 - do_cold) * zt_base

    # "mode dropout" only when high-t: randomly blank 20% modes of z_t (forces per-mode reasoning)
    with torch.no_grad():
        drop_p = 0.20
        D = (torch.rand_like(m) < drop_p).to(m.dtype) * m
    zt = torch.where((hi_hard > 0).expand_as(m).bool(), zt * (1.0 - D), zt)

    # ---------- predict x0, derive ε̂ and v̂ (NO detach) ----------
    x0_pred = model(zt, eigvals, mode_mask, t, x, h, node_mask,
                    W=None, edge_radius=edge_radius) * m

    sigma_safe = sigma_t.clamp_min(1e-8)
    eps_hat = ((zt - alpha_t * x0_pred) / sigma_safe) * m
    v_hat   = (alpha_t * eps_hat - sigma_t * x0_pred) * m
    v_tgt   = (alpha_t * eps - sigma_t * z0) * m

    # ---------- main high-t teacher: v ----------
    per_v = F.smooth_l1_loss(v_hat, v_tgt, reduction="none", beta=huber_delta)
    per_v = (per_v * m).sum(1) / m.sum(1).clamp_min(1)
    L_v_hi = (hi.squeeze(-1) * per_v).mean() * w_v_hi

    # ---------- mid/low-t anchor: x0 ----------
    per_x0 = F.smooth_l1_loss(x0_pred, z0, reduction="none", beta=huber_delta)
    per_x0 = (per_x0 * m).sum(1) / m.sum(1).clamp_min(1)
    L_x0_lo = (lo.squeeze(-1) * per_x0).mean() * w_x0_lo

    # ---------- small shape/norm stabilizers at high-t ----------
    cos = F.cosine_similarity((x0_pred * m), (z0 * m), dim=1).clamp(-1, 1)
    L_dir_hi  = ((1.0 - cos) * hi.squeeze(-1)).mean() * w_dir_hi

    n_hat = (x0_pred * m).norm(dim=1)
    n_gt  = (z0 * m).norm(dim=1)
    L_norm_hi = (((n_hat - n_gt).pow(2) / m.sum(1).clamp_min(1)) * hi.squeeze(-1)).mean() * w_norm_hi

    # ---------- focused sign + amplitude on dominant modes (high-t only) ----------
    abs_z0 = (z0.abs() * m)
    # pick top-K modes per item by |z0|; K ~ max(8, M//8) (cap at 32)
    K = max(8, min(32, z0.shape[1] // 8 if z0.shape[1] >= 8 else 8))
    # kthvalue over masked array: push pads large so they aren't selected
    big = 1e9
    thr, _ = torch.kthvalue(abs_z0 + (1 - m) * big, k=max(1, abs_z0.shape[1] - K), dim=1, keepdim=True)
    focus = (abs_z0 >= thr).to(z0.dtype) * m

    beta = 3.0
    s_hat = torch.tanh(beta * x0_pred)
    s_gt  = torch.tanh(beta * z0)
    L_sign_hi = (((s_hat - s_gt).abs() * focus).sum(1) / focus.sum(1).clamp_min(1))
    L_sign_hi = (hi.squeeze(-1) * L_sign_hi).mean() * w_sign_hi

    # ---------- subset exact per-mode supervision (random; favors stiff by rank if desired) ----------
    L_subset_hi = subset_exact_loss(
        x0_pred, z0, mode_mask, eigvals, hi,
        frac=frac_subset, w_subset=w_subset_hi, huber_delta=huber_delta, favor_stiff=True
    )

    # ---------- optional: tiny ε̂ band (not at extreme high-t) ----------
    if use_eps_band:
        band = ((snr >= 5e-3) & (snr <= 5e-2)).float()  # a modest high-t band only
        per_eps = F.smooth_l1_loss(eps_hat, eps, reduction="none", beta=huber_delta)
        per_eps = (per_eps * m).sum(1) / m.sum(1).clamp_min(1)
        L_eps_band = (band.squeeze(-1) * per_eps).mean() * w_eps_band
    else:
        L_eps_band = torch.zeros((), device=z0.device, dtype=z0.dtype)

    # ---------- total ----------
    loss = L_v_hi + L_x0_lo + L_dir_hi + L_norm_hi + L_sign_hi + L_subset_hi + L_eps_band

    dbg = {
        "z0_hat": x0_pred.detach(),
        "eps_hat": eps_hat.detach(),
        "eps": eps.detach(),
        "v_tgt": v_tgt.detach(),
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
    One-shot probe at time t for an x0-predicting model.
    Builds z_t ~ α z0 + σ ε, runs the model to get x0̂,
    then derives ε̂ and v̂ for diagnostics.
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
    alpha_t, sigma_t = alpha_sigma_from_t(t)  # shapes [B] or []
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)

    m   = mode_mask
    zt  = (alpha_t * z0 + sigma_t * eps) * m
    v_tgt = (alpha_t * eps - sigma_t * z0) * m

    # --- MODEL RETURNS x0_pred ---
    x0_pred = model(zt, eigvals, mode_mask, t, x, h, node_mask, W=None, edge_radius=edge_radius) * m

    # derive ε̂ and v̂ from x0̂
    sigma_safe = sigma_t.clamp_min(1e-8)
    eps_hat = ((zt - alpha_t * x0_pred) / sigma_safe) * m
    v_hat   = (alpha_t * eps_hat - sigma_t * x0_pred) * m

    # metrics
    denom_modes = m.sum(1).clamp_min(1)
    mse = (((x0_pred - z0).pow(2) * m).sum(1) / denom_modes).mean().item()
    mae = (((x0_pred - z0).abs()    * m).sum(1) / denom_modes).mean().item()
    cos = F.cosine_similarity(x0_pred * m, z0 * m, dim=1).mean().item()

    def _std(tens):
        return float(torch.sqrt(((tens * m) ** 2).sum() / m.sum().clamp_min(1)).item())

    out = dict(
        mse=mse,
        mae=mae,
        cos=cos,
        corr=cos,  # alias expected by your print
        eps_std=_std(eps),
        eps_hat_std=_std(eps_hat),
        v_pred_std=_std(v_hat),   # keep key name "v_pred_std" for compatibility
        v_tgt_std=_std(v_tgt),
        z0_hat_norm=float(((x0_pred * m).norm(dim=1).mean()).item()),
        z0_norm=float(((z0 * m).norm(dim=1).mean()).item()),
        alpha_sigma_sq=float(((alpha_t.pow(2) + sigma_t.pow(2)).mean()).item()),
    )
    return out


@torch.no_grad()
def probe_at_t(model, batch, t_scalar: float, *, seed: int = 1234, min_alpha: float = 1e-3):
    """
    Single-time probe for an x0-predicting model.
    Returns z_t, z*, x0̂, and scalar metrics.
    """
    z0         = batch["z0"];         eigvals    = batch["eigvals"]
    mode_mask  = batch["mode_mask"];  x          = batch["x"]
    h          = batch["h"];          node_mask  = batch["node_mask"]
    edge_rad   = float(batch["edge_radius"][0].item()) if isinstance(batch["edge_radius"], torch.Tensor) else float(batch["edge_radius"])

    if mode_mask.dim() == 1:
        mode_mask = mode_mask.view(1, -1)
    m = mode_mask.to(z0.dtype)

    B, M   = z0.shape
    device = z0.device
    t = torch.full((B,), float(t_scalar), device=device)

    alpha, sigma = alpha_sigma_from_t(t)
    alpha = alpha.clamp_min(min_alpha).view(B, 1)
    sigma = sigma.view(B, 1)

    torch.manual_seed(int(seed))
    eps = torch.randn_like(z0)
    z_t = (alpha * z0 + sigma * eps) * m

    # --- MODEL RETURNS x0_pred ---
    x0_hat = model(z_t, eigvals, mode_mask, t, x, h, node_mask, W=None, edge_radius=edge_rad) * m

    # metrics
    denom_modes = m.sum(1).clamp_min(1)
    mse = (((x0_hat - z0).pow(2) * m).sum(1) / denom_modes).mean().item()
    mae = (((x0_hat - z0).abs()    * m).sum(1) / denom_modes).mean().item()
    cos = F.cosine_similarity(x0_hat * m, z0 * m, dim=1).mean().item()

    return {"z_t": z_t, "z_star": z0, "x0_hat": x0_hat, "metrics": {"mse": mse, "mae": mae, "corr": cos}}


@torch.no_grad()
def ddim_probe(model, batch, *, steps: int = 20,
               t_start: float = 0.995, t_end: float = 0.01,
               seed: int = 1234, start_mode: str = "cold",
               return_all: bool = True):
    """
    Deterministic DDIM (η=0) for an x0-predicting model.
    Update uses:
        ε̂  = (z_t - α_t x0̂) / σ_t
        v̂  = α_t ε̂ - σ_t x0̂
        z_{t'} = α_{t'} x0̂ + σ_{t'} ε̂
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

    from .noise import alpha_sigma_from_t, t_from_logsnr  # add t_from_logsnr import at top

    # build logSNR-linear schedule
    def _logsnr(tvals):
        a, s = alpha_sigma_from_t(tvals)
        return torch.log((a**2) / (s**2 + 1e-12))

    u_start = _logsnr(torch.as_tensor([t_start], device=device, dtype=dtype)).item()
    u_end   = _logsnr(torch.as_tensor([t_end],   device=device, dtype=dtype)).item()
    us = torch.linspace(u_start, u_end, steps + 1, device=device, dtype=dtype)
    ts = t_from_logsnr(us)
    # init z_t at t_start
    a0, s0 = alpha_sigma_from_t(ts[0].expand(B))
    A0 = a0.clamp_min(1e-5).view(B, 1)
    S0 = s0.view(B, 1)

    torch.manual_seed(int(seed))
    if start_mode == "cold":
        z_t = (S0 * torch.randn_like(z0)) * m
    else:
        eps = torch.randn_like(z0)
        z_t = (A0 * z0 + S0 * eps) * m

    # DDIM loop (η=0)
    for i in range(steps):
        t_i = ts[i].expand(B)
        t_j = ts[i + 1].expand(B)

        Ai, Si = alpha_sigma_from_t(t_i); Ai = Ai.clamp_min(1e-5).view(B, 1); Si = Si.view(B, 1)
        Aj, Sj = alpha_sigma_from_t(t_j); Aj = Aj.clamp_min(1e-5).view(B, 1); Sj = Sj.view(B, 1)

        # --- MODEL RETURNS x0_pred ---
        x0_hat  = model(z_t, eigvals, mode_mask, t_i, x, h, node_mask, W=None, edge_radius=edge_rad) * m
        eps_hat = ((z_t - Ai * x0_hat) / Si.clamp_min(1e-8)) * m  # derived
        # v_hat  = (Ai * eps_hat - Si * x0_hat) * m  # not needed for the update, but correct if you log it

        # deterministic DDIM update
        z_t = (Aj * x0_hat + Sj * eps_hat) * m

    # final report at t_end
    te = ts[-1].expand(B)
    Ae, Se = alpha_sigma_from_t(te); Ae = Ae.clamp_min(1e-5).view(B, 1); Se = Se.view(B, 1)
    x0_hat_end = model(z_t, eigvals, mode_mask, te, x, h, node_mask, W=None, edge_radius=edge_rad) * m

    if not return_all:
        return x0_hat_end

    denom = m.sum().clamp_min(1.0)
    mse = float((((x0_hat_end - z0) ** 2) * m).sum() / denom)
    mae = float((((x0_hat_end - z0).abs()) * m).sum() / denom)
    cos = F.cosine_similarity((x0_hat_end * m).view(B, -1), (z0 * m).view(B, -1), dim=1).mean().item()

    return {
        "z_t": z_t,
        "z_star": z0,
        "x0_hat": x0_hat_end,
        "metrics": {"mse": mse, "mae": mae, "corr": cos},
    }