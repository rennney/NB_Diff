# nb_sim/diffusion/model.py
from __future__ import annotations
import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from .noise import alpha_sigma_from_u


def _lam_features(eigvals: torch.Tensor, mode_mask: torch.Tensor):
    # eigvals, mode_mask: [B, M]
    eps = 1e-12
    loglam = eigvals.clamp_min(eps).log()                         # [-inf, ...] -> finite
    m = mode_mask
    cnt = m.sum(1, keepdim=True).clamp_min(1)

    mu  = (loglam * m).sum(1, keepdim=True) / cnt                 # per-sample mean in log-space
    var = ((loglam - mu).pow(2) * m).sum(1, keepdim=True) / cnt
    lam_z = (loglam - mu) / (var.clamp_min(1e-8).sqrt())          # z-score log-λ, ~N(0,1)

    return lam_z

def _rank_pos(eigvals: torch.Tensor, mode_mask: torch.Tensor):
    # percentile by *index* (fast, stable), not by value; works across pairs with variable M
    B, M = eigvals.shape
    r = (torch.arange(M, device=eigvals.device, dtype=eigvals.dtype) + 0.5) / float(M)
    return r.view(1, M).expand(B, M)   # [B, M]


def fourier_mode_id(M, K=8, device=None, dtype=None):
    i = torch.linspace(0, 1, M, device=device, dtype=dtype).view(1, M, 1)
    k = torch.arange(1, K+1, device=device, dtype=dtype).view(1, 1, K)
    a = 2*torch.pi*k*i
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)  # [1,M,2K]

def _build_mode_feats(eigvals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Build per-mode features that do NOT vanish at high-t.
    Returns [B, M, 3]: [log1p(λ), 1/sqrt(1+λ), percentile-by-value]
    """
    lam = eigvals.clamp_min(0.0)
    f1 = torch.log1p(lam)
    f2 = 1.0 / torch.sqrt(1.0 + lam)

    # percentile rank among valid modes (by value)
    B, M = lam.shape
    big = torch.where(mask > 0, lam, lam.max(dim=1, keepdim=True).values + 1.0)
    order = torch.argsort(big, dim=1)                                  # [B,M]
    inv   = torch.argsort(order, dim=1)                                # rank index
    counts = mask.sum(1, keepdim=True).clamp_min(1.0)                  # valid per item
    perc = inv.to(lam.dtype) / (counts - 1.0).clamp_min(1.0)           # [0,1]

    return torch.stack([f1, f2, perc], dim=-1)  # [B,M,3]

# ----------------- Time embedding -----------------
class TimeEmbedding(nn.Module):
    """
    Sinusoidal embedding of logSNR u (not t). Input: u [B] ~ [-10, +8].
    """
    def __init__(self, dim: int, u_scale: float = 4.0):
        super().__init__()
        self.dim = dim
        self.u_scale = u_scale
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = u.device
        # frequency range ~ [1, 1000]
        freqs  = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=device))
        angles = (u / self.u_scale)[:, None] * freqs[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


# ----------------- EGNN layer -----------------
class EGNNLayer(nn.Module):
    """
    A lightweight, equivariant message-passing layer:
      - Messages depend on node features and squared distances (no angles).
      - Coordinates updated by normalized edge-weighted relative vectors.
      - Optional FiLM conditioning from a per-graph context vector.

    Inputs:
      x: [N, 3] coordinates
      h: [N, H] node embeddings
      edge_index: [2, E] (src, dst)
      edge_feat: [E, D_e] (here we pass d^2 as D_e=1)
      ctx_vec: [B, C] per-graph context (broadcasted inside), or None
      node_batch: [N] graph id for each node
    """
    def __init__(self, h: int, edge_dim: int, ctx_dim: int):
        super().__init__()
        self.ctx_dim = ctx_dim

        self.phi_e = nn.Sequential(
            nn.Linear(2 * h + 1 + edge_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(h + h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.phi_x = nn.Sequential(nn.Linear(h, 1))

        self.film = nn.Linear(ctx_dim, 2 * h) if ctx_dim > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        ctx_vec: torch.Tensor | None,
        node_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        rij = x[src] - x[dst]                       # [E, 3]
        d2 = (rij * rij).sum(-1, keepdim=True)      # [E, 1]
        e_ij = torch.cat([h[src], h[dst], d2, edge_feat], dim=-1)  # [E, 2H+1+De]
        m_ij = self.phi_e(e_ij)                     # [E, H]

        # aggregate messages on dst
        N, H = h.shape
        m_i = torch.zeros(N, H, device=h.device, dtype=h.dtype)
        m_i.index_add_(0, dst, m_ij)

        h_new = self.phi_h(torch.cat([h, m_i], dim=-1))  # [N, H]

        # optional FiLM conditioning with per-graph context
        if (self.film is not None) and (ctx_vec is not None):
            # broadcast context to nodes by node_batch
            film_params = self.film(ctx_vec)  # [B, 2H]
            gamma, beta = film_params.chunk(2, dim=-1)    # [B, H], [B, H]
            h_new = (1 + gamma[node_batch]) * h_new + beta[node_batch]

        # coordinate update
        w_ij = self.phi_x(m_ij)                # [E, 1]
        coord_msg = w_ij * rij / (1.0 + d2)    # [E, 3]
        dx = torch.zeros_like(x)
        dx.index_add_(0, dst, coord_msg)
        x_new = x + dx
        return x_new, h_new


# ----------------- EGNN backbone -----------------
class EGNNBackbone(nn.Module):
    """
    Stack of EGNN layers.
    """
    def __init__(self, h: int, L: int, edge_dim: int, ctx_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([EGNNLayer(h, edge_dim, ctx_dim) for _ in range(L)])

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        ctx_vec: torch.Tensor | None,
        node_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x, h = layer(x, h, edge_index, edge_feat, ctx_vec, node_batch)
        return x, h


class ModeIndexPE(nn.Module):
    """Fourier features of normalized mode index; dim must be even."""
    def __init__(self, dim: int = 16):
        super().__init__()
        assert dim % 2 == 0, "pos_dim must be even"
        self.dim = dim
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(math.log(1.0), math.log(100.0), dim // 2))
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: [B,M] (1/0) — only used for shape; we encode 0..1 index anyway
        B, M = mask.shape
        idx = torch.arange(M, device=mask.device, dtype=mask.dtype) / max(1, M - 1)  # [M] in [0,1]
        idx = idx.unsqueeze(0).expand(B, M)  # [B,M]
        angles = idx.unsqueeze(-1) * self.freqs  # [B,M,dim/2]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B,M,dim]


class ModalX0HeadFiLM(nn.Module):
    """
    x0̂ per mode = Linear( FiLM( ModeMLP( [α·z_t, eig_stats, pos_PE] ) ) ) + Linear_shortcut([eig_stats, pos_PE])
    FiLM params (γ,β) come from [ctx, t_emb].
    """
    def __init__(self, ctx_dim: int, t_dim: int, pos_dim: int = 16, eig_dims: int = 2, hidden: int = 128):
        super().__init__()
        self.pos_enc = ModeIndexPE(dim=pos_dim) if pos_dim > 0 else None
        self.use_two_eig = (eig_dims >= 2)
        in_mode = 1 + (2 if self.use_two_eig else 1) + (pos_dim if self.pos_enc else 0)  # α·z_t + eig + pos

        # pure mode path (no ctx, no time)
        self.mode_mlp = nn.Sequential(
            nn.Linear(in_mode, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.out_lin = nn.Linear(hidden, 1)

        # FiLM from (ctx, t_emb) → γ,β (per sample, broadcast across modes)
        self.film = nn.Linear(ctx_dim + t_dim, 2 * hidden)
        with torch.no_grad():
            # init γ around 1, β around 0 so we start near identity
            self.film.weight.zero_()
            self.film.bias.zero_()

        # residual shortcut to guarantee per-mode identity reaches output
        self.shortcut = nn.Linear(in_mode - 1, 1, bias=False)  # (eig+pos) only
        nn.init.normal_(self.shortcut.weight, mean=0.0, std=1e-3)

    def forward(self, ctx, z_t, eigvals, mask, t_emb, t):
        B, M = z_t.shape
        # α-gate z_t so ε cannot leak at high-t
        alpha, _ = alpha_sigma_from_u(t)                         # [B]
        while alpha.dim() < z_t.dim():
            alpha = alpha.unsqueeze(-1)                          # [B,1]
        z_feat = (alpha * z_t).unsqueeze(-1)                     # [B,M,1]

        # eig features
        lam = eigvals.clamp_min(0.0)
        e1 = torch.log1p(lam).unsqueeze(-1)                      # [B,M,1]
        feats = [e1]
        if self.use_two_eig:
            e2 = (1.0 / torch.sqrt(1.0 + lam)).unsqueeze(-1)    # [B,M,1]
            feats.append(e2)
        eig_feat = torch.cat(feats, dim=-1)                      # [B,M,e]
        pos = self.pos_enc(mask) if self.pos_enc is not None else None  # [B,M,p] or None

        # pack per-mode input
        mode_in = [z_feat, eig_feat]
        if pos is not None:
            mode_in.append(pos)
        mode_in = torch.cat(mode_in, dim=-1)                     # [B,M,in_mode]

        # mode path
        h = self.mode_mlp(mode_in)                               # [B,M,H]

        # FiLM from (ctx, t_emb)
        g = torch.cat([ctx, t_emb], dim=-1)                      # [B, C+T]
        film = self.film(g)                                      # [B, 2H]
        H = h.shape[-1]
        gamma, beta = film.split(H, dim=-1)                      # [B,H], [B,H]
        # keep γ near 1, β near 0 to avoid blowing up
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta  = 0.1 * torch.tanh(beta)

        h = gamma.unsqueeze(1) * h + beta.unsqueeze(1)           # [B,M,H]
        x0_main = self.out_lin(h).squeeze(-1)                    # [B,M]

        # residual shortcut: only (eig+pos) — excludes z_feat to prevent ε leakage
        eigpos = eig_feat if pos is None else torch.cat([eig_feat, pos], dim=-1)  # [B,M, e(+p)]
        x0_sc = self.shortcut(eigpos).squeeze(-1)                # [B,M]

        return x0_main + x0_sc


# ----------------- Per-mode head -----------------
class ModalHeadWorkinWithLeak(nn.Module):
    def __init__(self, h: int, t_dim: int, p: float = 1.5):
        super().__init__()
        in_dim = h + 4 + 16 + t_dim   # [g | z_t | s | t_emb]
        self.z_skip = nn.Linear(1, 1, bias=False)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.SiLU(),
            nn.Linear(h, h),      nn.SiLU(),
            nn.Linear(h, 1)
        )
        self.gate_pow = p  # α^p

    def forward(self, g: torch.Tensor, z_t: torch.Tensor,
                eigvals: torch.Tensor, t_emb: torch.Tensor, u: torch.Tensor,mode_mask: torch.Tensor = None) -> torch.Tensor:
        B, M = z_t.shape
        H, T = g.shape[-1], t_emb.shape[-1]
        alpha_u, sigma_u = alpha_sigma_from_u(u)                # [B]
        while alpha_u.dim() < z_t.dim():
            alpha_u = alpha_u.unsqueeze(-1); sigma_u = sigma_u.unsqueeze(-1)


        # per-mode features
        g_tile = g.unsqueeze(1).expand(B, M, H)
        t_tile = t_emb.unsqueeze(1).expand(B, M, T)
        s = torch.log1p(eigvals)
        #s = torch.ones_like(eigvals)  # dummy constant feature
        mode_id = fourier_mode_id(M, K=8, device=z_t.device, dtype=z_t.dtype).expand(B, -1, -1)

        lam_z = _lam_features(eigvals, mode_mask)                 # [B, M]
        # ---- NEW rank/position feature ----
        rank = _rank_pos(eigvals, mode_mask)                      # [B, M]
        posPE  = fourier_mode_id(M, K=8, device=z_t.device, dtype=z_t.dtype).expand(B, -1, -1)
        z_over_sigma = (z_t / sigma_u.clamp_min(1e-6))
        #print(s[0,:50])
        #print(lam_z[0,:50])
        #feat = torch.cat([g_tile, z_t.unsqueeze(-1), s.unsqueeze(-1), t_tile], dim=-1)  # [B,M,H+1+1+T]

        feat = torch.cat([
            g_tile,
            z_over_sigma.unsqueeze(-1),      # helps learn ε at high-t *without* killing x0
            s.unsqueeze(-1),
            lam_z.unsqueeze(-1),
            rank.unsqueeze(-1),
            posPE,
            t_tile
        ], dim=-1)  # new in_dim = H + 1 + 1 + 1 + 1 + 16 + T


        eps = self.net(feat).squeeze(-1)  # [B,M]

        # time gate ~ α(t)^p
        #alpha_t, sigma_t = alpha_sigma_from_t(t)           # shapes [B]
        #gate = alpha_t.clamp(0, 1).pow(self.gate_pow).unsqueeze(-1)  # [B,1]
        #eps = eps + 0.0*self.z_skip(( z_t).unsqueeze(-1)).squeeze(-1)

        return eps


import torch
from torch import nn
from torch.nn import functional as F


class CartesianModalHead(nn.Module):
    """
    Head for RTB-DOF diffusion that explicitly parameterizes

        ε̂_v = ŝ * b + q̂   with  q̂ ⟂ b,

    where b = v_t / σ(u) is the analytic baseline direction in RTB-DOF space,
    ŝ is a learned scalar (per-graph), and q̂ is a learned residual vector field
    orthogonal to b.

    This works for any per-node DOF dimension `dof_dim` (3 for Cartesian blocks,
    6 for rigid-body RTB DOFs, etc.).
    """
    def __init__(self, node_dim: int, time_dim: int, ctx_dim: int, dof_dim: int = 6):
        super().__init__()
        self.dof_dim = dof_dim

        # project graph-level context into node space
        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_dim, node_dim),
            nn.SiLU(),
        )
        # time embedding → node space
        self.t_proj = nn.Sequential(
            nn.Linear(time_dim, node_dim),
            nn.SiLU(),
        )

        # per-node scalar pre-activations; we later pool to a single scalar per graph
        self.s_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1),
        )
        # residual branch: takes (h_eff, b_node) and outputs a raw DOF residual
        self.q_mlp = nn.Sequential(
            nn.Linear(node_dim + dof_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, dof_dim),
        )

    def forward(
        self,
        h_nodes: torch.Tensor,   # [N_tot, H]
        b_nodes: torch.Tensor,   # [N_tot, dof_dim] baseline direction v_t / σ per node
        t_emb: torch.Tensor,     # [B, T]
        ctx: torch.Tensor,       # [B, C]
        node_batch: torch.Tensor # [N_tot] with values in 0..B-1
    ) -> torch.Tensor:
        """
        Returns:
            eps_v: [N_tot, dof_dim]  (ε̂_v for each node)
        """
        device = h_nodes.device
        B = t_emb.shape[0]

        # Expand graph-level context + time to nodes
        ctx_node = self.ctx_proj(ctx)  # [B, H]
        t_node   = self.t_proj(t_emb)  # [B, H]

        ctx_per_node = ctx_node[node_batch]  # [N_tot, H]
        t_per_node   = t_node[node_batch]    # [N_tot, H]

        # Effective node features
        h_eff = h_nodes + ctx_per_node + t_per_node   # [N_tot, H]

        # ----- (1) Scalar along baseline: ŝ -----
        s_nodes = self.s_mlp(h_eff)                   # [N_tot, 1]

        sum_s  = torch.zeros(B, 1, device=device, dtype=s_nodes.dtype)
        count  = torch.zeros(B, 1, device=device, dtype=s_nodes.dtype)
        sum_s.index_add_(0, node_batch, s_nodes)
        count.index_add_(0, node_batch, torch.ones_like(s_nodes))

        s_graph = sum_s / count.clamp_min(1.0)        # [B, 1]
        s_per_node = s_graph[node_batch]              # [N_tot, 1]

        # ----- (2) Residual orthogonal to baseline: q̂ -----
        q_in = torch.cat([h_eff, b_nodes], dim=-1)    # [N_tot, H + dof_dim]
        q_raw = self.q_mlp(q_in)                      # [N_tot, dof_dim]

        b_norm2 = (b_nodes.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8))  # [N_tot,1]
        proj_coeff = (q_raw * b_nodes).sum(dim=1, keepdim=True) / b_norm2
        q_orth = q_raw - proj_coeff * b_nodes

        eps_v = s_per_node * b_nodes + q_orth
        return eps_v



class ModalHead(nn.Module):
    """
    ε̂ = c(u,g) · b  +  α(u)^p · r_⊥(b)
         ^scalar per item         ^per-mode residual kept small at high-noise

    b = (z_t / σ(u))  (DETACHED)
    Adds a learnable rank embedding to sharpen per-mode identity (less mixing).
    """
    def __init__(self, ctx_dim: int, t_dim: int,
                 max_modes: int = 4096,  # safe upper bound
                 Epos: int = 32, Kfourier: int = 0,
                 hidden: int | None = None, p_gate: float = 1.0):
        super().__init__()
        h = hidden or max(128, ctx_dim)
        self.p_gate = p_gate

        # per-item scalar parallel scale
        self.c_head = nn.Sequential(
            nn.Linear(ctx_dim + t_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1)
        )

        # learnable rank embedding (mode rank 0..M-1)
        self.pos_emb = nn.Embedding(max_modes, Epos)

        # optional Fourier pos (set Kfourier>0 if you still want it)
        self.Kfourier = Kfourier

        # residual head (per-mode)
        in_dim = ctx_dim + (1 + 1 + 1 + 1) + Epos + t_dim  # g, z/σ, log1p(λ), lam_feat, rank_feat, posEmb, t
        if self.Kfourier > 0:
            in_dim += 2*self.Kfourier

        self.res_head = nn.Sequential(
            nn.Linear(in_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1)
        )

    def forward(self, g: torch.Tensor, z_t: torch.Tensor,
                eigvals: torch.Tensor, t_emb: torch.Tensor,
                u: torch.Tensor, mode_mask: torch.Tensor) -> torch.Tensor:
        B, M = z_t.shape
        C, T = g.shape[-1], t_emb.shape[-1]
        m = mode_mask.to(z_t.dtype)

        # schedule + baseline
        alpha_u, sigma_u = alpha_sigma_from_u(u)
        while alpha_u.dim() < z_t.dim():
            alpha_u = alpha_u.unsqueeze(-1); sigma_u = sigma_u.unsqueeze(-1)
        b = (z_t / sigma_u.clamp_min(1e-8)).detach()  # [B,M]

        # parallel scale (per-item)
        c = 1.0 + 0.25*torch.tanh(self.c_head(torch.cat([g, t_emb], dim=-1)))  # ~[0.75,1.25]
        c = c.clamp(0.6, 1.4).view(B, 1)

        # features for residual
        g_tile = g.unsqueeze(1).expand(B, M, C)
        t_tile = t_emb.unsqueeze(1).expand(B, M, T)
        z_over_sigma = (z_t / sigma_u.clamp_min(1e-8))
        lam = eigvals.clamp_min(0)
        f_log = torch.log1p(lam)
        lam_feat = _lam_features(eigvals, mode_mask)
        rank_feat = _rank_pos(eigvals, mode_mask)

        # learnable rank embedding (use 0..M-1 over valid slice)
        idx = torch.arange(M, device=z_t.device).view(1, M).expand(B, M)
        pe_learn = self.pos_emb(idx)  # [B,M,Epos]

        feats = [g_tile,
                 z_over_sigma.unsqueeze(-1),
                 f_log.unsqueeze(-1),
                 lam_feat.unsqueeze(-1),
                 rank_feat.unsqueeze(-1),
                 pe_learn,
                 t_tile]

        if self.Kfourier > 0:
            posPE = fourier_mode_id(M, K=self.Kfourier, device=z_t.device, dtype=z_t.dtype).expand(B, -1, -1)
            feats.append(posPE)

        feat = torch.cat(feats, dim=-1)  # [B,M,in_dim]

        r = self.res_head(feat).squeeze(-1)  # [B,M]

        # orthogonalize residual to b
        num = (r * b * m).sum(dim=1, keepdim=True)
        den = (b.pow(2) * m).sum(dim=1, keepdim=True).clamp_min(1e-8)
        r_orth = (r - (num/den)*b) * m

        # gate residual by α(u)^p  (small at high-noise)
        gate = alpha_u.clamp(0, 1).pow(self.p_gate)
        eps_hat = (c * b + gate * r_orth) * m
        return eps_hat

    

# ----------------- Full model -----------------
class ModalEGNN(nn.Module):
    """
    Normal-mode diffusion denoiser with EGNN conditioning.

    forward signature matches usage in losses/trainer:
        eps_pred = model(
            z_t,           # [B, M]
            eigvals,       # [B, M]
            mask_modes,    # [B, M]  (1 for valid, 0 for pad)
            t,             # [B]
            x_blocks,      # [B, N, 3]  rigid-block centroids or reps
            nfeat,         # [B, N, F]  per-block scalar features
            node_mask,     # [B, N]     (1=valid)
            W=None,        # adjacency (optional, ignored if None)
            edge_radius=30 # float cutoff for dynamic edges
        )
    Returns:
        eps_pred [B, M]
    """
    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 6,
        time_emb_dim: int = 64,
        context_dim: int = 128,   # pooled graph -> FiLM context
        edge_feat_dim: int = 1,   # we only use d^2 as edge feature
    ):
        super().__init__()
        self.h = hidden_dim
        self.node_enc = nn.Linear(node_feat_dim, hidden_dim)
        self.backbone = EGNNBackbone(hidden_dim, n_layers, edge_feat_dim, context_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
            nn.SiLU(),
        )
        self.t_embed = TimeEmbedding(time_emb_dim)
        self.modal_head = ModalHead(context_dim, time_emb_dim)

        self.cartesian_head = CartesianModalHead(
            node_dim=hidden_dim,     # same H as your backbone hidden dim
            time_dim=time_emb_dim,
            ctx_dim=hidden_dim,
            dof_dim=6,
        )

        #self.modal_head = ModalX0HeadFiLM(
        #    ctx_dim=context_dim,          # <-- must match your readout output
        #    t_dim=time_emb_dim,           # <-- must match your time embedding dim
        #    pos_dim=16,
        #    eig_dims=2,
        #    hidden=max(128, hidden_dim)
        #)


    @staticmethod
    def _build_edges_radius(
        xb: torch.Tensor, radius: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edge_index and edge_feat (d^2) for a single graph by radius cutoff.
        xb: [n_i, 3]
        return: edge_index [2, E_i], edge_feat [E_i, 1]
        """
        n = xb.shape[0]
        if n <= 1:
            return (xb.new_zeros(2, 0, dtype=torch.long), xb.new_zeros(0, 1))

        # pairwise distances (n, n)
        d2 = torch.cdist(xb, xb, p=2.0) ** 2
        mask = (d2 <= (radius * radius)) & (~torch.eye(n, dtype=torch.bool, device=xb.device))
        ei, ej = mask.nonzero(as_tuple=True)
        edge_index = torch.stack([ei, ej], dim=0)
        edge_feat = d2[ei, ej].unsqueeze(-1)
        return edge_index, edge_feat

    def _concat_graphs(
        self,
        X_list: List[torch.Tensor],
        H_list: List[torch.Tensor],
        EI_list: List[torch.Tensor],
        EF_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Concatenate per-graph node/edge tensors into a single big graph.
        Returns node_batch to identify graph id per node.
        """
        node_batch = []
        x_all, h_all, ei_all, ef_all = [], [], [], []
        node_offset = 0
        for b, (x, h, ei, ef) in enumerate(zip(X_list, H_list, EI_list, EF_list)):
            x_all.append(x)
            h_all.append(h)
            if ei.numel() > 0:
                ei_all.append(ei + node_offset)
                ef_all.append(ef)
            node_batch.append(torch.full((x.shape[0],), b, device=x.device, dtype=torch.long))
            node_offset += x.shape[0]

        x_cat = torch.cat(x_all, dim=0) if x_all else torch.empty(0, 3, device=X_list[0].device)
        h_cat = torch.cat(h_all, dim=0) if h_all else torch.empty(0, self.h, device=H_list[0].device)
        if ei_all:
            ei_cat = torch.cat(ei_all, dim=1)
            ef_cat = torch.cat(ef_all, dim=0)
        else:
            ei_cat = torch.empty(2, 0, dtype=torch.long, device=x_cat.device)
            ef_cat = torch.empty(0, 1, device=x_cat.device)
        node_batch_cat = torch.cat(node_batch, dim=0) if node_batch else torch.empty(0, dtype=torch.long, device=x_cat.device)
        return x_cat, h_cat, ei_cat, ef_cat, node_batch_cat

    def forward(
        self,
        z_t: torch.Tensor,            # [B, M]
        eigvals: torch.Tensor,        # [B, M]
        mask_modes: torch.Tensor,     # [B, M]
        u: torch.Tensor,              # [B]
        x_blocks: torch.Tensor,       # [B, N, 3]
        nfeat: torch.Tensor,          # [B, N, F]
        node_mask: torch.Tensor,      # [B, N]
        W: torch.Tensor | None = None,
        edge_radius: float = 30.0,
    ) -> torch.Tensor:
        """
        Build per-graph EGNN, pool to a context vector, then predict ε per mode.
        """
        device = z_t.device
        B, N, _ = x_blocks.shape
        Fdim = nfeat.shape[-1]

        # Encode node features (masking padded rows)
        # We'll split the batch into per-graph lists (variable #blocks per item)
        X_list: List[torch.Tensor] = []
        H0_list: List[torch.Tensor] = []
        EI_list: List[torch.Tensor] = []
        EF_list: List[torch.Tensor] = []

        for b in range(B):
            valid = node_mask[b].bool()
            xb = x_blocks[b, valid]                 # [n_i, 3]
            hf = nfeat[b, valid]                    # [n_i, F]
            h0 = F.silu(self.node_enc(hf))      # [n_i, H]

            if W is not None:
                # If an adjacency is provided (same padding), use its valid subgraph.
                # Edge features = d^2 for those edges.
                Wb = W[b][valid][:, valid]          # [n_i, n_i], bool or 0/1
                ei, ej = (Wb > 0).nonzero(as_tuple=True)
                edge_index, edge_feat = torch.stack([ei, ej], dim=0), (torch.cdist(xb, xb) ** 2)[ei, ej].unsqueeze(-1)
            else:
                edge_index, edge_feat = self._build_edges_radius(xb, edge_radius)

            X_list.append(xb)
            H0_list.append(h0)
            EI_list.append(edge_index)
            EF_list.append(edge_feat)

        # Concatenate graphs and run EGNN backbone
        x_cat, h_cat, ei_cat, ef_cat, node_batch = self._concat_graphs(X_list, H0_list, EI_list, EF_list)
        if x_cat.numel() == 0:
            print("[WARNING] ModalEGNN: empty input graphs!!!!!!")
            # degenerate case: no nodes — return zeros
            return torch.zeros_like(z_t)

        x_cat, h_cat = self.backbone(x_cat, h_cat, ei_cat, ef_cat, None, node_batch)

        # Pool per-graph to get context
        g_list = []
        start = 0
        for b in range(B):
            n_i = X_list[b].shape[0]
            hb = h_cat[start:start + n_i]
            start += n_i
            g_list.append(hb.mean(dim=0, keepdim=True))  # mean pool
        H_pooled = torch.cat(g_list, dim=0)  # [B, H]
        ctx = H_pooled         # [B, H] (context_dim == H via readout chain)

        # Time embedding
        t_emb = self.t_embed(u)              # [B, T]

        #x0_hat = self.modal_head(ctx, z_t, eigvals, mask_modes, t_emb, t)
        #return x0_hat


        # Predict ε per mode (mask will be applied in the loss)
        eps_pred = self.modal_head(ctx, z_t, eigvals, t_emb,u,mask_modes)  # [B, M]
        return eps_pred

    def forward_cartesian(
        self,
        y_t: torch.Tensor,        # [B, N, dof_dim]  RTB DOFs at time t (3 or 6 etc.)
        x_apo: torch.Tensor,      # [B, N, 3]  apo block coords (geometry)
        nfeat: torch.Tensor,      # [B, N, F]  node scalar features (same F as original forward)
        node_mask: torch.Tensor,  # [B, N]
        u: torch.Tensor,          # [B] logSNR
        W: torch.Tensor | None = None,
        edge_radius: float = 30.0,
    ) -> torch.Tensor:
        """
        RTB-DOF path: use apo geometry (x_apo) for edges, and keep node features
        exactly as in the original modal forward (no change in dim).
        The baseline b = y_t / σ(u) is fed to the CartesianModalHead, which
        explicitly parameterizes ε̂_y = ŝ * b + q̂, q̂ ⟂ b.

        y_t is per-node DOFs (e.g. 3 for pure translations, 6 for full RTB DOFs).
        """
        device = x_apo.device
        B, N, dof_dim = y_t.shape

        X_list: list[torch.Tensor] = []
        Y_list: list[torch.Tensor] = []
        H0_list: list[torch.Tensor] = []
        EI_list: list[torch.Tensor] = []
        EF_list: list[torch.Tensor] = []
        node_batch_list: list[torch.Tensor] = []

        # Build per-graph subgraphs
        for b in range(B):
            valid = node_mask[b].bool()
            if not valid.any():
                continue

            xb = x_apo[b, valid]          # [n_i, 3] apo positions
            yb = y_t[b, valid]            # [n_i, dof_dim] RTB DOFs
            hf = nfeat[b, valid]          # [n_i, F]

            # same node encoder as main forward
            h0 = F.silu(self.node_enc(hf))   # [n_i, H]

            if W is not None:
                Wb = W[b][valid][:, valid]
                ei, ej = (Wb > 0).nonzero(as_tuple=True)
                edge_index = torch.stack([ei, ej], dim=0)
                d2 = (torch.cdist(xb, xb) ** 2)[ei, ej].unsqueeze(-1)
                edge_feat = d2
            else:
                edge_index, edge_feat = self._build_edges_radius(xb, edge_radius)

            X_list.append(xb)
            Y_list.append(yb)
            H0_list.append(h0)
            EI_list.append(edge_index)
            EF_list.append(edge_feat)
            node_batch_list.append(
                torch.full((xb.shape[0],), b, device=device, dtype=torch.long)
            )

        if len(X_list) == 0:
            # no valid nodes in batch
            return torch.zeros(B, N, dof_dim, device=device, dtype=y_t.dtype)

        # Concatenate graphs into a single big graph
        x_cat, h_cat, ei_cat, ef_cat, node_batch = self._concat_graphs(
            X_list, H0_list, EI_list, EF_list
        )
        y_cat = torch.cat(Y_list, dim=0)                   # [N_tot, dof_dim]
        N_tot = y_cat.shape[0]

        # ---- EGNN backbone on apo geometry ----
        # NOTE: use self.backbone, not self.egnn
        x_cat, h_cat = self.backbone(
            x_cat, h_cat, ei_cat, ef_cat, None, node_batch
        )

        # ---- graph pooling → context ----
        g_list = []
        start = 0
        for b in range(B):
            n_i = (node_batch == b).sum().item()
            if n_i == 0:
                g_list.append(torch.zeros(1, h_cat.shape[-1], device=device))
                continue
            hb = h_cat[start:start + n_i]
            start += n_i
            g_list.append(hb.mean(dim=0, keepdim=True))
        ctx = torch.cat(g_list, dim=0)                    # [B, H] or ctx_dim

        # time embedding
        t_emb = self.t_embed(u)                           # [B, time_dim]

        # ---- baseline b = y_t / σ(u) per node ----
        alpha_t, sigma_t = alpha_sigma_from_u(u)          # [B]
        sigma_nodes = sigma_t[node_batch].unsqueeze(-1)   # [N_tot, 1]
        b_cat = y_cat / sigma_nodes.clamp_min(1e-8)       # [N_tot, dof_dim]

        # ---- Cartesian head: ε̂_y = ŝ * b + q̂ (q̂ ⟂ b) ----
        eps_y_cat = self.cartesian_head(
            h_nodes=h_cat,
            b_nodes=b_cat,
            t_emb=t_emb,
            ctx=ctx,
            node_batch=node_batch,
        )  # [N_tot, dof_dim]

        # un-batch back to [B, N, dof_dim] with padding
        eps_y = x_apo.new_zeros(B, N, dof_dim)
        start = 0
        for b in range(B):
            valid = node_mask[b].bool()
            n_i = int(valid.sum().item())
            if n_i > 0:
                eps_y[b, valid] = eps_y_cat[start:start + n_i]
                start += n_i

        return eps_y
