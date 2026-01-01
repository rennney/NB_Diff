# nb_sim/adapters.py
from __future__ import annotations
from typing import Optional, Sequence
from pathlib import Path
import numpy as np
import torch

# === Your uploaded NOLB/ANM stack ============================================
from ..io.pdb_parser import Molecule, resolve_pdb_input
from ..core.anm import build_anm_hessian, mass_weight_hessian
from ..core.rtb import build_rtb_projection
from ..core.modes import compute_rtb_modes
from ..core.deform import deform_structure

# --------- Internal helpers ---------------------------------------------------

def _apply_rtb_vec(mol, blocks, rtb_vec, block_dofs, device):
    """
    Apply a FULL RTB vector (6-DoF per block concatenated) using your deform_structure().
    Returns a coords tensor [N,3]. Works with both signatures of deform_structure.
    """
    rtb_vec = rtb_vec.contiguous().to(device=device, dtype=mol.coords.dtype)
    try:
        out = deform_structure(mol, blocks, rtb_vec,
                               amplitude=1.0, mode_index=-1,
                               device=device, block_dofs=block_dofs)
    except TypeError:
        out = deform_structure(mol, blocks, rtb_vec,
                               device=device, block_dofs=block_dofs)
    return out if out is not None else mol.coords

def _cart_rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b).pow(2).sum(dim=-1).mean()).sqrt().item())

def _calibrate_decode_scales(mol, blocks, block_dofs, V: torch.Tensor,
                             target_rmsd: float = 1.0, eps: float = 1e-3) -> torch.Tensor:
    """
    Per-mode scale s_k such that a unit amplitude produces roughly target_rmsd Å.
    """
    device, dtype = V.device, V.dtype
    M = V.shape[1]
    s = torch.ones(M, device=device, dtype=dtype)
    base = mol.coords.clone()
    for k in range(M):
        z = torch.zeros(M, device=device, dtype=dtype); z[k] = eps
        rtb_vec = (V @ z).contiguous()
        mol.coords = base
        coords = _apply_rtb_vec(mol, blocks, rtb_vec, block_dofs, device)
        rmsd = _cart_rmsd(coords, base)
        s[k] = target_rmsd / max(rmsd / max(eps, 1e-12), 1e-12)
    mol.coords = base
    return s

# --------- Public API ---------------------------------------------------------

class ModeSet:
    __slots__ = ("eigvals", "eigvecs_rtb", "mol", "blocks", "block_dofs",
                 "device", "dtype", "decode_scales")

    def __init__(self, eigvals, eigvecs_rtb, mol, blocks, block_dofs,
                 device, dtype, decode_scales):
        self.eigvals = eigvals              # torch.Tensor [M]
        self.eigvecs_rtb = eigvecs_rtb      # torch.Tensor [R, M]
        self.mol = mol                      # Molecule (your parser class)
        self.blocks = blocks                # list/array of blocks
        self.block_dofs = block_dofs        # per-block DoFs structure
        self.device = device
        self.dtype = dtype
        self.decode_scales = decode_scales  # torch.Tensor [M]

    @property
    def n_modes(self) -> int:
        return int(self.eigvals.numel())

    def to_cartesian(self, z):
        """
        Decode modal amplitudes to Cartesian via a single RTB vector application.
        """
        z = z.to(device=self.device, dtype=self.dtype).reshape(-1)
        rtb_vec = (self.eigvecs_rtb @ (self.decode_scales * z)).contiguous()
        return _apply_rtb_vec(self.mol, self.blocks, rtb_vec, self.block_dofs, self.device)


def build_modes_from_pdb(pdb_path: str | Path,
                         k: int = 64,
                         calibrate: bool = True,
                         device: Optional[torch.device] = None,
                         dtype: torch.dtype = torch.float64):
    """
    Build RTB modes from an apo PDB:
      - ANM Hessian -> mass-weight -> RTB projection -> eigendecomp
      - Returns modeset with decode to Cartesian via NOLB-style block deformations.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    pdb_path = Path(resolve_pdb_input(str(pdb_path)))
    mol = Molecule(str(pdb_path), device=device)

    try:
        if not hasattr(mol, "path") or not getattr(mol, "path"):
            mol.path = str(pdb_path.resolve())
    except Exception:
        pass

    # 1) ANM Hessian
    K = build_anm_hessian(mol, mol.coords)
    # 2) mass-weighting (uses per-atom masses from your parser)
    masses_np = np.array([atom[2] for atom in mol.atoms], dtype=np.float64)
    K_w = mass_weight_hessian(K, masses_np)
    # 3) RTB projection (6-DoF per block)
    blocks = mol.blocks  # list of blocks from your Molecule
    P, block_dofs = build_rtb_projection(blocks, N_atoms=len(mol.atoms))
    # 4) eigendecomp in RTB space
    _, eigvals, eigvecs_rtb, _ = compute_rtb_modes(K_w, P, n_modes=k)
    # Drop the 6 trivial rigid-body modes if present
    if eigvals.shape[0] >= k + 6:
        eigvals = eigvals[6:6+k]
        eigvecs_rtb = eigvecs_rtb[:, 6:6+k]

    if hasattr(eigvecs_rtb, "toarray"):
        eigvecs_rtb = eigvecs_rtb.toarray()
    eigvals_t = torch.as_tensor(np.asarray(eigvals), dtype=dtype, device=device).reshape(-1)
    V = torch.as_tensor(np.asarray(eigvecs_rtb), dtype=dtype, device=device)  # [R, M]
    
    decode_scales = (_calibrate_decode_scales(mol, blocks, block_dofs, V)
                     if calibrate else torch.ones(V.shape[1], dtype=dtype, device=device))

    return ModeSet(
        eigvals=eigvals_t,
        eigvecs_rtb=V,
        mol=mol,
        blocks=blocks,
        block_dofs=block_dofs,
        device=device,
        dtype=dtype,
        decode_scales=decode_scales,
    )

def fit_modal_amplitudes(modeset,
                         target_coords: torch.Tensor,
                         iters: int = 200,
                         lr: float = 1.0) -> torch.Tensor:
    """
    Fit z* s.t. to_cartesian(z*) ≈ target_coords, with small stiffness regularizer.
    """
    device, dtype = modeset.device, modeset.dtype
    M = modeset.n_modes
    z = torch.zeros(M, device=device, dtype=dtype, requires_grad=True)
    V, s, lam = modeset.eigvecs_rtb, modeset.decode_scales, modeset.eigvals
    tgt = target_coords.to(device=device, dtype=dtype)
    opt = torch.optim.LBFGS([z], lr=lr, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        rtb_vec = (V @ (s * z)).contiguous()
        coords = _apply_rtb_vec(modeset.mol, modeset.blocks, rtb_vec, modeset.block_dofs, device)
        diff = coords - tgt
        loss = (diff.pow(2).sum(dim=-1).mean()) + 1e-4 * (lam * (z**2)).mean()
        #print("[LBFGS] loss: %.6f" % float(loss.item()))
        loss.backward()
        return loss

    opt.step(closure)
    return z.detach()

# --------- Robust PDB trajectory writer --------------------------------------

def _format_xyz(x, y, z):
    return f"{x:8.3f}{y:8.3f}{z:8.3f}"

def save_traj(frames: Sequence[torch.Tensor] | Sequence[np.ndarray],
              mol,
              out_path: str | Path,
              template_path: Optional[str | Path] = None) -> str:
    """
    Write a multi-MODEL PDB trajectory:
      - Uses apo PDB as template to preserve records.
      - Replaces ATOM/HETATM XYZ columns by line order (robust).
    """
    if not frames:
        raise ValueError("save_traj: no frames provided.")
    out_path = str(Path(out_path).resolve())

    frames_np = []
    for F in frames:
        if hasattr(F, "detach"):
            F = F.detach().cpu().numpy()
        frames_np.append(np.asarray(F, dtype=np.float64))

    tpl = template_path or getattr(mol, "path", None)
    if tpl is None:
        raise ValueError("save_traj: template_path is None and mol.path is missing.")
    tpl = str(Path(tpl).resolve())
    with open(tpl, "r") as fin:
        orig = fin.readlines()

    model_starts = [i for i, l in enumerate(orig) if l.startswith("MODEL")]
    if model_starts:
        s = model_starts[0]
        try:
            e = next(i for i in range(s + 1, len(orig)) if orig[i].startswith("ENDMDL"))
        except StopIteration:
            e = len(orig)
        body = [l for l in orig[s+1:e] if not l.startswith(("MODEL", "ENDMDL", "END"))]
    else:
        body = [l for l in orig if not l.startswith(("MODEL", "ENDMDL", "END"))]

    atom_lines = [i for i, l in enumerate(body) if l.startswith(("ATOM  ", "HETATM"))]
    max_replace = len(atom_lines)
    minN = min(min(F.shape[0] for F in frames_np), max_replace)
    frames_np = [F[:minN] for F in frames_np]

    with open(out_path, "w") as fout:
        for m_idx, F in enumerate(frames_np, start=1):
            fout.write(f"MODEL     {m_idx:4d}\n")
            a = 0
            for li, line in enumerate(body):
                if li in atom_lines and a < minN:
                    x, y, z = F[a, 0], F[a, 1], F[a, 2]
                    fout.write(line[:30] + _format_xyz(x, y, z) + line[54:])
                    a += 1
                else:
                    fout.write(line)
            fout.write("ENDMDL\n")
        fout.write("END\n")

    return out_path

def build_modes(pdb_path: str | Path,
                n_modes: int,
                device: torch.device,
                dtype: torch.dtype,
                calibrate: bool = False):
    """
    Thin wrapper so callers can import `build_modes` from adapters.
    Returns the same modeset object as `build_modes_from_pdb`:
      - .eigvals [M]
      - .eigvecs_rtb [R,M]
      - .mol, .blocks, .block_dofs
      - .to_cartesian(z)  (cartesian decode)
    """
    return build_modes_from_pdb(pdb_path, k=n_modes, calibrate=calibrate, device=device, dtype=dtype)



def build_block_graph(pdb_path: str | Path,
                      device: torch.device,
                      dtype: torch.dtype):
    """
    Build a rigid-block graph for the EGNN:
      x_blocks : [N,3] block centroids
      nfeat    : [N,2] scalar features = [block_size, block_mass_sum]
      node_mask: [N]   (all True; no padding here)
      W        : None  (model builds radius-edges at runtime)
    """
    mol = Molecule(str(resolve_pdb_input(str(pdb_path))), device=device)
    blocks = getattr(mol, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise RuntimeError("No rigid blocks found in molecule.")

    coords = mol.coords.to(device=device, dtype=dtype)  # [Natoms,3]

    # Per-atom masses (fallback to 12.0 if not available)
    masses_list = []
    for a in getattr(mol, "atoms", []):
        m = None
        if isinstance(a, dict):
            m = a.get("mass", None)
        elif isinstance(a, (list, tuple)) and len(a) >= 3:
            m = a[2]
        if m is None:
            m = 12.0
        masses_list.append(float(m))
    if not masses_list:
        masses_list = [12.0] * coords.shape[0]
    masses = torch.tensor(masses_list, device=device, dtype=dtype)  # [Natoms]

    x_blocks = []
    size_feat = []
    mass_feat = []

    for b in blocks:
        idx = getattr(b, "atom_indices", b)          # support list or object
        idx = torch.as_tensor(idx, device=device, dtype=torch.long)
        if idx.numel() == 0:
            continue
        pos = coords.index_select(0, idx)            # [n_i,3]
        x_blocks.append(pos.mean(dim=0, keepdim=True))                # [1,3]
        size_feat.append(idx.numel())                                   # scalar
        mass_feat.append(masses.index_select(0, idx).sum().item())      # scalar

    if not x_blocks:
        raise RuntimeError("All blocks were empty after filtering.")

    x_blocks = torch.cat(x_blocks, dim=0)                               # [N,3]
    size_feat = torch.tensor(size_feat, device=device, dtype=dtype).unsqueeze(1)  # [N,1]
    mass_feat = torch.tensor(mass_feat, device=device, dtype=dtype).unsqueeze(1)  # [N,1]
    nfeat = torch.cat([size_feat, mass_feat], dim=1)                    # [N,2]

    node_mask = torch.ones(x_blocks.shape[0], device=device, dtype=torch.bool)
    W = None  # edges built by model radius; no dense adjacency here
    return x_blocks, nfeat, node_mask, W
