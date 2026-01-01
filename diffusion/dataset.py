# nb_sim/diffusion/dataset.py
from __future__ import annotations
import os, json, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from .adapters import build_modes, fit_modal_amplitudes, build_block_graph



# ----------------------------- PDB parsing & matching -----------------------------

def _parse_pdb_atoms(path: str, keep_hydrogens: bool = False):
    """
    Minimal PDB/ENT parser returning:
      keys:  [(chain, resseq, icode, atomname), ...]
      coords:[N,3] float32
    Filters:
      - record in {"ATOM  ", "HETATM"}
      - altLoc in {" ", "A"} only
      - drop hydrogens unless keep_hydrogens=True
      - occupancy > 0.0
    """
    keys, coords = [], []
    with open(path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            name = line[12:16].strip()
            if (not keep_hydrogens) and (name.startswith("H") or name == "D"):
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                occ = float(line[54:60] or 0.0)
            except ValueError:
                continue
            if occ <= 0.0:
                continue
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            keys.append((chain, resseq, icode, name))
            coords.append((x, y, z))
    if not coords:
        raise ValueError(f"No atoms parsed from {path}")
    return keys, np.asarray(coords, dtype=np.float32)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute R,t s.t. Q @ R + t ~= P (P and Q are [N,3]).
    Returns (R[3,3], t[1,3]).
    """
    Pc = P - P.mean(0, keepdims=True)
    Qc = Q - Q.mean(0, keepdims=True)
    H = Qc.T @ Pc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = P.mean(0, keepdims=True) - Q.mean(0, keepdims=True) @ R
    return R, t


def _nn_map(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    For each point in src [N,3], return index of nearest neighbor in dst [M,3].
    Uses dense cdist; N,M are ~1e3 so this is fine and fast on CPU.
    """
    # cdist^2
    a2 = (src**2).sum(1, keepdims=True)     # [N,1]
    b2 = (dst**2).sum(1, keepdims=True).T   # [1,M]
    ab = src @ dst.T                         # [N,M]
    d2 = a2 - 2*ab + b2
    return d2.argmin(axis=1)                 # [N]


def _harmonize_and_align(apo_path: str, holo_path: str):
    """
    1) parse apo and holo atoms → keys & coords
    2) intersect keys to build a common atom set and ordered vectors (apoC, holoC)
    3) rigidly align holoC -> apoC (Kabsch) and return aligned holo coords
    Returns:
      keys_common    : List[tuple]  common keys in a stable order
      apoC, holoC_al : np.ndarray [K,3], [K,3]
    """
    apo_keys, apo_coords = _parse_pdb_atoms(apo_path, keep_hydrogens=False)
    holo_keys, holo_coords = _parse_pdb_atoms(holo_path, keep_hydrogens=False)

    apo_index: Dict[Tuple[str, str, str, str], int] = {k: i for i, k in enumerate(apo_keys)}
    holo_index: Dict[Tuple[str, str, str, str], int] = {k: i for i, k in enumerate(holo_keys)}
    common_keys = [k for k in apo_keys if k in holo_index]

    if len(common_keys) < max(100, int(0.5*min(len(apo_keys), len(holo_keys)))):
        # If the strict key match is too small, relax to C-alpha only for alignment seed
        apo_ca = [(k, i) for i, k in enumerate(apo_keys) if k[3] == "CA"]
        holo_ca = [(k, i) for i, k in enumerate(holo_keys) if k[3] == "CA"]
        ca_common = [k for k, _ in apo_ca if k in {k for k, _ in holo_ca}]
        if len(ca_common) >= 3:
            # build apoC/holoC using CA keys, then expand by nearest neighbor on full sets
            apoC = np.asarray([apo_coords[apo_index[k]] for k in ca_common], dtype=np.float32)
            holoC = np.asarray([holo_coords[holo_index[k]] for k in ca_common], dtype=np.float32)
            R, t = _kabsch(apoC, holoC)
            holo_all_al = holo_coords @ R + t
            # Now take intersection on residue (chain,resseq,icode), map atoms by NN inside each residue
            resid_index_apo: Dict[Tuple[str,str,str], List[int]] = {}
            resid_index_holo: Dict[Tuple[str,str,str], List[int]] = {}
            for i, (ch, rs, ic, an) in enumerate(apo_keys):
                resid_index_apo.setdefault((ch, rs, ic), []).append(i)
            for i, (ch, rs, ic, an) in enumerate(holo_keys):
                resid_index_holo.setdefault((ch, rs, ic), []).append(i)
            keys_common = []
            for resid, idxs_a in resid_index_apo.items():
                idxs_h = resid_index_holo.get(resid)
                if not idxs_h:
                    continue
                A = apo_coords[idxs_a]
                H = holo_all_al[idxs_h]
                nn = _nn_map(A, H)
                for j, ja in enumerate(idxs_a):
                    jh = idxs_h[nn[j]]
                    keys_common.append((resid[0], resid[1], resid[2], apo_keys[ja][3]))
            # deduplicate preserving apo order
            seen = set(); common_keys = []
            for k in apo_keys:
                if k in keys_common and k not in seen:
                    common_keys.append(k); seen.add(k)

    # Build aligned coordinate arrays for the common set
    apoC = np.asarray([apo_coords[apo_index[k]] for k in common_keys], dtype=np.float32)
    holoC = np.asarray([holo_coords[holo_index[k]] for k in common_keys], dtype=np.float32)
    # Rigid align holo->apo
    R, t = _kabsch(apoC, holoC)
    holoC_al = holoC @ R + t
    return common_keys, apoC, holoC_al


# ----------------------------- Dataset structures -----------------------------

@dataclass
class PairItem:
    apo: str
    holo: str




class ModesProxy:
    """
    Thin delegator around the underlying ModeSet so downstream code can use
    .to_cartesian(z), .mol, .device, .dtype, etc. Accepts optional device/dtype
    for backward compatibility with older call sites.
    """
    def __init__(self, modes_obj, device=None, dtype=None):
        self._m = modes_obj                 # the real ModeSet
        self._fallback_device = device      # only used if underlying lacks it
        self._fallback_dtype  = dtype

    # delegate decoding
    def to_cartesian(self, z):
        return self._m.to_cartesian(z)

    # commonly used attributes
    @property
    def mol(self):
        return getattr(self._m, "mol")

    @property
    def device(self):
        return getattr(self._m, "device", self._fallback_device)

    @property
    def dtype(self):
        return getattr(self._m, "dtype", self._fallback_dtype)

    # generic fallback for anything else
    def __getattr__(self, name):
        return getattr(self._m, name)

class ModalPairsDataset(Dataset):
    """
    Loads apo/holo pairs, harmonizes atoms, builds RTB modes on apo,
    fits z* to the matched holo, and prepares block-graph features.

    Each item provides:
      - z0:         [M] modal amplitudes fitted to holo (target x0 in modal space)
      - eigvals:    [M] modal eigenvalues (or frequencies)
      - mode_mask:  [M] (1 for valid modes, some may be <M if fewer modes exist)
      - x_blocks:   [N,3] rigid block centroids (or representatives)
      - nfeat:      [N,F] per-block scalar features
      - node_mask:  [N]   mask for variable block counts
      - W:          [N,N] optional adjacency (0/1), or None
      - edge_radius: float (for dynamic radius graph in the model)
      - idxs:       dataset index (int)
    """
    def __init__(
        self,
        pairs_json: str,
        n_modes: int,
        cache_dir: str,
        device: torch.device,
        dtype: torch.dtype,
        edge_radius: float = 30.0,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n_modes = n_modes
        self.edge_radius = float(edge_radius)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        with open(pairs_json, "r") as f:
            self.pairs: List[PairItem] = [PairItem(**d) for d in json.load(f)]

        self.items: List[Dict[str, Any]] = []
        self.modesets: List[ModesProxy] = []

        for idx, pair in enumerate(self.pairs):
            apo_path = pair.apo
            holo_path = pair.holo

            # 1) Harmonize atom sets and align holo→apo
            keys_common, apoC, holoC_al = _harmonize_and_align(apo_path, holo_path)

            # 2) Build RTB/ANM modes on the apo structure (full file),
            #    then find a stable mapping from the "modes order" to our common set.
            modes = build_modes(apo_path, n_modes=n_modes, device=device, dtype=dtype)

            # Get the coordinate order used by the modes object at z=0
            with torch.no_grad():
                z_zero = torch.zeros(n_modes, device=device, dtype=dtype)
                coords_modes0 = modes.to_cartesian(z_zero).detach().cpu().numpy()   # [Nm,3]

            # Map modes' apo atom order to the apo common set via nearest neighbors
            # (we rely on identical or near-identical coordinates for matched atoms)
            nn_idx = _nn_map(coords_modes0, apoC)  # [Nm], values in [0, K)
            tgt_holo_matched = holoC_al[nn_idx]    # [Nm,3] numpy

            # 3) Fit z* (modal amplitudes) so deform_structure(modes, z*) ~= matched holo coords
            tgt = torch.tensor(tgt_holo_matched, device=device, dtype=dtype)  # [Nm,3]
            z_fit = fit_modal_amplitudes(modes, tgt, iters=50, lr=1.0).to(device=device).to(dtype)

            # 4) Cache per-pair objects
            #    Eigenvalues / mode mask from the modes object
            eigvals = getattr(modes, "eigvals", None)
            if eigvals is None:
                raise RuntimeError("modes object must expose .eigvals")
            if isinstance(eigvals, torch.Tensor):
                eigvals = eigvals.to(device=device, dtype=dtype)
            else:
                eigvals = torch.as_tensor(eigvals, device=device, dtype=dtype)
            M = eigvals.shape[0]
            if M < n_modes:
                # pad eigvals and z_fit and build mask
                pad = n_modes - M
                eigvals = torch.cat([eigvals, torch.zeros(pad, device=device, dtype=dtype)], dim=0)
                z_fit   = torch.cat([z_fit,   torch.zeros(pad, device=device, dtype=dtype)], dim=0)
                mode_mask = torch.zeros(n_modes, device=device, dtype=dtype)
                mode_mask[:M] = 1.0
            else:
                eigvals = eigvals[:n_modes]
                z_fit   = z_fit[:n_modes]
                mode_mask = torch.ones(n_modes, device=device, dtype=dtype)

            # 5) Build block graph / features from adapters
            x_blocks, nfeat, node_mask, W = build_block_graph(apo_path, device=device, dtype=dtype)

            self.items.append({
                "z0": z_fit,                 # [M]
                "eigvals": eigvals,          # [M]
                "mode_mask": mode_mask,      # [M]
                "x_blocks": x_blocks,        # [N,3]
                "nfeat": nfeat,              # [N,F]
                "node_mask": node_mask,      # [N]
                "W": W,                      # [N,N] or None
                "edge_radius": torch.tensor(self.edge_radius, device=device, dtype=dtype),
                "idxs": torch.tensor(idx, device=device, dtype=torch.long),
            })

            self.modesets.append(ModesProxy(modes, device=device, dtype=dtype))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


@torch.no_grad()
def _compute_local_radius(x: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Per-node local radius from k-NN distances.
    x: [N,3] (device/dtype inherited from dataset items)
    Returns r: [N,1]
    """
    N = x.shape[0]
    if N <= 1:
        return torch.full((N, 1), 1.0, device=x.device, dtype=x.dtype)
    # Pairwise squared distances
    a2 = (x**2).sum(dim=1, keepdim=True)       # [N,1]
    d2 = a2 - 2.0 * (x @ x.t()) + a2.t()       # [N,N]
    d2 = d2.clamp_min_(0.0)
    d = torch.sqrt(d2 + 1e-12)
    # Exclude self (inf on diagonal), then take mean of k smallest
    d.fill_diagonal_(float("inf"))
    k_eff = min(k, N - 1)
    knn, _ = torch.topk(d, k_eff, dim=1, largest=False)
    r = knn.mean(dim=1, keepdim=True)          # [N,1]
    # Minimum floor to avoid zeros
    return r.clamp_min(1e-3)


def collate_batch(batch: List[Dict[str, Any]], edge_radius: float) -> Dict[str, torch.Tensor]:
    """
    Collate a list of dataset items into a padded batch.

    Inputs (per item):
      z0          : [M]
      eigvals     : [M]
      mode_mask   : [M]
      x_blocks    : [N,3]
      nfeat       : [N,2]  (block_size, block_mass_sum)
      node_mask   : [N]
      W           : [N,N] or None (ignored; edges built on-the-fly by the model)
      edge_radius : scalar tensor
      idxs        : int tensor

    Output (batched):
      z0          : [B, Mmax]
      eigvals     : [B, Mmax]
      mode_mask   : [B, Mmax]
      x           : [B, Nmax, 3]
      h           : [B, Nmax, 3]  (size, mass, local_radius)
      node_mask   : [B, Nmax] (bool)
      edge_radius : [B]  (float, usually constant)
      idxs        : [B]
    """
    device = batch[0]["z0"].device
    dtype  = batch[0]["z0"].dtype

    B = len(batch)
    Nmax = max(item["x_blocks"].shape[0] for item in batch)
    Mmax = max(item["z0"].shape[0] for item in batch)

    x  = torch.zeros(B, Nmax, 3, device=device, dtype=dtype)
    h  = torch.zeros(B, Nmax, 3, device=device, dtype=dtype)   # 3 features: size, mass, local_radius
    nm = torch.zeros(B, Nmax,    device=device, dtype=torch.bool)

    z0       = torch.zeros(B, Mmax, device=device, dtype=dtype)
    eigvals  = torch.zeros(B, Mmax, device=device, dtype=dtype)
    mmask    = torch.zeros(B, Mmax, device=device, dtype=dtype)
    idxs     = torch.zeros(B,       device=device, dtype=torch.long)
    er       = torch.zeros(B,       device=device, dtype=dtype)

    for b, item in enumerate(batch):
        Xi   = item["x_blocks"]        # [Ni,3]
        Fi   = item["nfeat"]           # [Ni,2] (size, mass)
        N    = Xi.shape[0]
        Zi   = item["z0"]              # [Mi]
        Ei   = item["eigvals"]         # [Mi]
        Mi   = Zi.shape[0]
        Mi_m = item["mode_mask"]       # [Mi]

        # Positions
        x[b, :N] = Xi
        nm[b, :N] = True

        # Local radius per node
        ri = _compute_local_radius(Xi)                 # [Ni,1]
        # Stack features: [size, mass, local_radius]
        hi = torch.cat([Fi, ri], dim=1)               # [Ni,3]
        h[b, :N] = hi

        # Modes
        z0[b, :Mi]      = Zi
        eigvals[b, :Mi] = Ei
        mmask[b, :Mi]   = Mi_m

        # Meta
        idxs[b] = item["idxs"]
        er[b]   = item["edge_radius"]

    return {
        "x": x,                 # [B,Nmax,3]
        "h": h,                 # [B,Nmax,3]
        "node_mask": nm,        # [B,Nmax]
        "z0": z0,               # [B,Mmax]
        "eigvals": eigvals,     # [B,Mmax]
        "mode_mask": mmask,     # [B,Mmax]
        "edge_radius": er,      # [B]
        "idxs": idxs,           # [B]
    }
