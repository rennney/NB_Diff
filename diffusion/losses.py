# nb_sim/diffusion/losses.py
from __future__ import annotations
import torch
from .noise import alpha_sigma_from_t, t_from_logsnr, logsnr_from_t, alpha_sigma_from_u,s_over_a_from_u
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


def cartesian_eps_loss(
    model,
    batch,
    u,
    decoder_bank,
    w_eps: float = 1.0,
    w_decomp: float = 1.0,
    w_x0: float = 1.0,
    w_lock: float = 0.25,
    huber_delta: float = 1.0,
):
    """RTB-DOF diffusion loss.

    State:
        v_t : [B, N, 6]  block RTB DOFs at time t (3 trans + 3 rot per block).
    Construction:
        - Start from clean modal amplitudes z0 (apo→holo) in mode space.
        - Sample Gaussian noise eps_q in mode space.
        - Map both to RTB DOF space via eigvecs_rtb and decode_scales:
              rtb0    = V @ (s * z0)
              rtb_eps = V @ (s * eps_q)
          and reshape to [N_blocks, 6] per item.
        - Run diffusion v_t = α v0 + σ eps_v in DOF space.
        - Model predicts eps_v (noise in DOF space) per block.

    This is a drop-in replacement for the previous Cartesian loss:
    only the internal representation changes from centroids to RTB DOFs.
    """
    z0        = batch["z0"]           # [B,M]
    eigvals   = batch["eigvals"]      # [B,M] (unused but kept for API)
    mode_mask = batch["mode_mask"]    # [B,M]
    x_apo     = batch["x"]            # [B,N,3] (used for device / shapes)
    h         = batch["h"]            # [B,N,F]
    node_mask = batch["node_mask"]    # [B,N]
    idxs      = batch["idxs"]         # [B]
    device    = x_apo.device
    dtype     = x_apo.dtype

    B, M = z0.shape
    _, N, _ = x_apo.shape

    # diffusion schedule in logSNR u
    alpha_t, sigma_t = alpha_sigma_from_u(u)        # [B]
    a_scalar = alpha_t
    s_scalar = sigma_t

    # sample noise in mode space (respecting mask)
    m = mode_mask
    eps_q = torch.randn_like(z0) * m                # [B,M]

    # build v0, eps_v per item using the underlying ModeSet
    v0_list: list[torch.Tensor] = []
    eps_v_list: list[torch.Tensor] = []

    for b in range(B):
        idx = int(idxs[b].item())
        ms = decoder_bank[idx]                      # ModesProxy
        mset = getattr(ms, "_m", ms)                # ModeSet

        z0_b   = (z0[b]   * m[b]).to(device=mset.device, dtype=mset.dtype)    # [Mb]
        eps_qb = (eps_q[b]* m[b]).to(device=mset.device, dtype=mset.dtype)    # [Mb]

        V     = mset.eigvecs_rtb                    # [R,Mb]
        s_dec = mset.decode_scales                  # [Mb]

        rtb0    = V @ (s_dec * z0_b)                # [R]
        rtb_eps = V @ (s_dec * eps_qb)              # [R]

        # assume standard RTB: 6 DOFs per block
        n_blocks = len(mset.blocks)
        v0_ms    = rtb0.view(n_blocks, -1)          # [Nb,6]
        eps_v_ms = rtb_eps.view(n_blocks, -1)       # [Nb,6]

        v0_list.append(v0_ms.to(device=device, dtype=dtype))
        eps_v_list.append(eps_v_ms.to(device=device, dtype=dtype))

    # pad to [B,N,6] using node_mask
    dof_dim = v0_list[0].shape[-1]
    v0    = x_apo.new_zeros(B, N, dof_dim)
    eps_v = x_apo.new_zeros(B, N, dof_dim)

    for b in range(B):
        valid = node_mask[b].bool()
        n_i = int(valid.sum().item())
        v0[b, valid]    = v0_list[b][:n_i]
        eps_v[b, valid] = eps_v_list[b][:n_i]

    # diffusion in RTB-DOF space: v_t = α v0 + σ eps_v
    a_b = a_scalar.view(B, 1, 1)
    s_b = s_scalar.view(B, 1, 1)
    v_t = a_b * v0 + s_b * eps_v

    # model prediction: ε̂_v in DOF space
    eps_v_hat = model.forward_cartesian(
        y_t=v_t,
        x_apo=x_apo,
        nfeat=h,
        node_mask=node_mask,
        u=u,
        W=None,
        edge_radius=float(batch["edge_radius"].detach().cpu().reshape(-1)[0].item()),
    )  # [B,N,6]

    # --- masks & normalizers ---
    nm    = node_mask.float().unsqueeze(-1)           # [B,N,1]
    denom = nm.sum(dim=(1, 2)).clamp_min(1.0)         # [B]

    # 1) High-noise ε_v MSE (L_eps)
    w_hi = ((s_scalar / a_scalar).clamp_min(1.0))**2  # [B]
    per_eps = ((eps_v_hat - eps_v)**2 * nm).sum(dim=(1, 2)) / denom
    L_eps = (per_eps * w_hi).mean()

    # 2) Baseline+residual decomposition (L_decomp)
    # baseline in v-space: b = v_t / σ, detached
    s_b_safe = s_b.clamp_min(1e-8)
    b = (v_t / s_b_safe).detach()                     # [B,N,6]

    B_, N_, Dofs = v_t.shape
    D = N_ * Dofs

    b_flat       = b.view(B_, D)
    eps_v_flat   = eps_v.view(B_, D)
    eps_hat_flat = eps_v_hat.view(B_, D)
    m_flat       = nm.expand(B_, N_, Dofs).reshape(B_, D)

    b_norm2 = (b_flat.pow(2) * m_flat).sum(1, keepdim=True).clamp_min(1e-8)

    s_star = ((eps_v_flat * b_flat * m_flat).sum(1, keepdim=True) / b_norm2).detach()
    s_pred = ((eps_hat_flat * b_flat * m_flat).sum(1, keepdim=True) / b_norm2)

    q_pred_flat = (eps_hat_flat - s_pred * b_flat) * m_flat

    L_scale = (s_pred - s_star).pow(2).mean()
    qb_dot  = (q_pred_flat * b_flat * m_flat).sum(1, keepdim=True)
    L_ortho = ((qb_dot ** 2) / b_norm2).mean()
    L_decomp = L_scale + L_ortho

    # 3) Lock-breaking (L_lock): avoid ε̂ being exactly collinear with b at very high noise
    eps_hat_n = eps_hat_flat / (eps_hat_flat.norm(dim=1, keepdim=True) + 1e-8)
    b_n       = b_flat       / (b_flat.norm(dim=1,       keepdim=True) + 1e-8)
    hi = ((s_scalar / a_scalar) > 10.0).float()
    cos2 = ((eps_hat_n * b_n).sum(1).clamp(-1, 1))**2
    L_lock = (cos2 * hi).mean()

    # 4) Mid/low-noise x0 anchor (L_x0) in DOF space
    a_safe = a_scalar.clamp_min(1e-2)
    a_safe_b = a_safe.view(B, 1, 1)

    v0_hat = (v_t - s_b * eps_v_hat) / a_safe_b

    midlow = ((a_scalar / s_scalar) > 1.0).float()
    per_x0 = F.smooth_l1_loss(v0_hat, v0, reduction="none", beta=huber_delta) * nm
    per_x0 = per_x0.sum(dim=(1, 2)) / denom
    L_x0   = (per_x0 * midlow).mean()

    loss = (
        w_eps    * L_eps
      + w_decomp * L_decomp
      + w_x0    * L_x0
      + w_lock  * L_lock
    )

    return loss, {
        "L_eps":    L_eps.detach(),
        "L_decomp": L_decomp.detach(),
        "L_x0":     L_x0.detach(),
        "L_lock":   L_lock.detach(),
    }
