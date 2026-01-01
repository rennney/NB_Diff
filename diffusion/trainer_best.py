from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from .config import DiffusionConfig
from .dataset import ModalPairsDataset
from .losses import v_probe_metrics, probe_at_t, ddim_probe ,x0_loss ,eps_loss
from .model import ModalEGNN
from .utils import seed_all, save_ckpt, EMA
from .noise import t_from_logsnr, alpha_sigma_from_t

# ---------------- Collate wrapper ----------------
@dataclass
class CollateWithRadius:
    edge_radius: float
    def __call__(self, batch):
        from .dataset import collate_batch as _collate_batch
        return _collate_batch(batch, self.edge_radius)

def batch_to_device(batch, device, non_blocking=True):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out

def _u_bounds(epoch: int):
    """Start narrow in log-SNR and widen to full range by ~100 epochs."""
    widen = min(1.0, epoch / 100.0)
    a = -2.0 - (5.0 - 2.0) * widen
    b =  2.0 + (5.0 - 2.0) * widen
    return a, b

def train(cfg: DiffusionConfig):
    seed_all(cfg.seed)
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # (reserved for future use)

    # Dataset / Loader
    ds = ModalPairsDataset(cfg.pairs_json, cfg.n_modes, cfg.cache_dir, device, dtype,
                           edge_radius=cfg.edge_radius)
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
    decoder_bank = ds.modesets
    # Model / Opt
    model = ModalEGNN(
        node_feat_dim=3,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        time_emb_dim=cfg.time_dim
    ).to(device)
    ema    = EMA(model, decay=cfg.ema_decay)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                               betas=(0.9, 0.95))
    scaler = GradScaler(enabled=(device.type == "cuda") and bool(cfg.amp))

    # LR scheduler (per-epoch cosine)
    scheduler = CosineAnnealingLR(opt, T_max=max(2000, cfg.epochs), eta_min=cfg.lr * 0.1)

    # knobs
    log_interval  = max(1, int(getattr(cfg, "log_interval", 50)))   # epochs between logs
    ckpt_interval = max(1, int(getattr(cfg, "ckpt_interval", 500)))
    grad_clip     = float(getattr(cfg, "grad_clip", 1.0))
    K_t = int(cfg.t_repeats)

    last_batch = None  # keep the last batch of the epoch for diagnostics

    saved_t = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        u_min, u_max = -9.71, 7.8  # _u_bounds(epoch)
        epoch_loss_sum = 0.0
        n_iters = 0

        for it, batch in enumerate(dl, start=1):
            batch = batch_to_device(batch, device)
            last_batch = batch  # keep for diagnostics

            B, M = batch["z0"].shape
            opt.zero_grad(set_to_none=True)

            # ----- multi-t microbatch -----
            loss_sum_this_iter = 0.0
            last_dbg = None
            for _ in range(K_t):
                u = torch.empty(B, device=device).uniform_(u_min, u_max)
                t = t_from_logsnr(u).clamp(0.001, 0.995)  # match inference band

                if torch.rand((), device=device) < 0.5:
                    # deliberately hit the hardest regime
                    t = torch.empty(B, device=device).uniform_(0.98, 0.995)
                else:
                    t = t_from_logsnr(u).clamp(0.001, 0.995)
                #print("Epoch Time : ", t[0].item())
                #t = torch.empty(B, device=device).uniform_(0.001, 0.999)
                if epoch%500==0:
                    saved_t = t[0].item()
                #if torch.rand(()) < 0.5:
                #    t = torch.empty(B, device=device).uniform_(0.97, 0.999)
                #else:
                #    t = torch.empty(B, device=device).uniform_(0.001, 0.999)

                with autocast(enabled=(device.type == "cuda") and bool(cfg.amp)):
                    loss, dbg = eps_loss(model, batch, t,decoder_bank=decoder_bank)

                scaler.scale(loss / K_t).backward()   # average grads across K_t
                loss_sum_this_iter += float(loss)
                last_dbg = dbg

            # step opt
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
            ema.update(model)

            # bookkeeping
            epoch_loss_sum += (loss_sum_this_iter / K_t)  # per-step avg
            n_iters += 1

        # LR schedule: per-epoch
        scheduler.step()

        # epoch logging block
        if epoch%250 == 0: #(epoch % log_interval == 0) and (last_batch is not None):
            loss_avg = epoch_loss_sum / max(1, n_iters)
            print(f"[epoch {epoch:03d}] loss={loss_avg:.6f}")
        #if epoch ==25000:    
            B = last_batch["z0"].shape[0]
            t_mid = torch.full((B,), 0.5, device=device)

            with torch.no_grad():
                rep = v_probe_metrics(model, last_batch, t_mid)

            print(f"         z0_hat@t=0.5: MSE={rep['mse']:.4f} MAE={rep['mae']:.4f} corr={rep['corr']:.3f}")
            print(f"         check: E[α^2+σ^2]@t=0.5 = {rep['alpha_sigma_sq']:.3f}")
            print(f"         stds: eps={rep['eps_std']:.3f}  eps_hat={rep['eps_hat_std']:.3f}  "
                f"v_pred={rep['v_pred_std']:.3f}  v_tgt={rep['v_tgt_std']:.3f}")
            print(f"         [probe@t=0.5] MSE={rep['mse']:.4f} MAE={rep['mae']:.4f} "
                f"cos={rep['cos']:.3f} |z0_hat|={rep['z0_hat_norm']:.3f} |z*|={rep['z0_norm']:.3f}")
            t_mid = torch.full((B,), 0.99, device=device)

            with torch.no_grad():
                rep = v_probe_metrics(model, last_batch, t_mid)

            print(f"         z0_hat@t=0.99: MSE={rep['mse']:.4f} MAE={rep['mae']:.4f} cos={rep['corr']:.3f}")
            print(f"         check: E[α^2+σ^2]@t=0.99 = {rep['alpha_sigma_sq']:.3f}")
            print(f"         stds: eps={rep['eps_std']:.3f}  eps_hat={rep['eps_hat_std']:.3f}  "
                f"v_pred={rep['v_pred_std']:.3f}  v_tgt={rep['v_tgt_std']:.3f}")
            print(f"         [probe@t=0.99] MSE={rep['mse']:.4f} MAE={rep['mae']:.4f} "
                f"cos={rep['cos']:.3f} |z0_hat|={rep['z0_hat_norm']:.3f} |z*|={rep['z0_norm']:.3f}")
            print(f"         [hiT-probe] t=0.99  corr(zt/σ,ε)={rep['corr_base']:.3f}  corr(ε̂,ε)={rep['corr_hat']:.3f}  ")
            print(    f"σ/α≈{rep['sigma_t']:.3f}/{rep['alpha_t']:.3f}")

            t_mid = torch.full((B,), 0.01, device=device)

            with torch.no_grad():
                rep = v_probe_metrics(model, last_batch, t_mid)

            print(f"         z0_hat@t=0.01: MSE={rep['mse']:.4f} MAE={rep['mae']:.4f} corr={rep['corr']:.3f}")
            print(f"         check: E[α^2+σ^2]@t=0.01 = {rep['alpha_sigma_sq']:.3f}")
            print(f"         stds: eps={rep['eps_std']:.3f}  eps_hat={rep['eps_hat_std']:.3f}  "
                f"v_pred={rep['v_pred_std']:.3f}  v_tgt={rep['v_tgt_std']:.3f}")
            print(f"         [probe@t=0.01] MSE={rep['mse']:.4f} MAE={rep['mae']:.4f} "
                f"cos={rep['cos']:.3f} |z0_hat|={rep['z0_hat_norm']:.3f} |z*|={rep['z0_norm']:.3f}")
            

                        # 1) Key variables at t = 0.99, 0.50, 0.15 with (z_t, z*, x0̂) head dump
            #for t_scalar in (0.99, 0.50, 0.15):
            #    rep = probe_at_t(model, last_batch, t_scalar, seed=1234)
            #    m = last_batch["mode_mask"][0].bool()
            #    idx = m.nonzero(as_tuple=True)[0][:5]
            #    def fmt(x): return "[" + ", ".join(f"{float(v):.3f}" for v in x) + "]"
            #    print(f"         x0̂@t={t_scalar:.3f}: MSE={rep['metrics']['mse']:.4f} MAE={rep['metrics']['mae']:.4f} corr={rep['metrics']['corr']:.3f}")
            #    print(f"         [modes@t={t_scalar:.3f}] idx = {idx.tolist() if idx.numel()>0 else []}")
            #    if idx.numel() > 0:
            #        print("           z_t   =", fmt(rep["z_t"][0, idx].detach().cpu()))
            #        print("           z*    =", fmt(rep["z_star"][0, idx].detach().cpu()))
            #        print("           x0̂    =", fmt(rep["x0_hat"][0, idx].detach().cpu()))
            #    else:
            #        print("           (no valid modes)")

            # 3) DDIM probes (cold start) that mimic training, with finals dumped
            #for steps in (20, 40, 80):
            #    out = ddim_probe(model, last_batch, steps=steps, t_start=0.99, t_end=0.01,
            #                     seed=1234, start_mode="cold", return_all=True)
            #    m = last_batch["mode_mask"][0].bool()
            #    idx = m.nonzero(as_tuple=True)[0][:7]
            #    def fmt(x): return "[" + ", ".join(f"{float(v):.3f}" for v in x) + "]"
            #    print(f"         DDIM({steps}, cold): MSE={out['metrics']['mse']:.4f}  MAE={out['metrics']['mae']:.4f}  corr={out['metrics']['corr']:.3f}")
            #    if idx.numel() > 0:
            #        print("           (final) z_t  =", fmt(out["z_t"][0, idx].detach().cpu()))
            #        print("           (target) z*  =", fmt(out["z_star"][0, idx].detach().cpu()))
            #        print("           (final) x0̂  =", fmt(out["x0_hat"][0, idx].detach().cpu()))

        # ------------- checkpoints -------------
        if epoch % ckpt_interval == 0:
            save_ckpt(out_dir / f"ckpt_e{epoch:03d}.pt", model, opt, epoch, epoch, ema)

    save_ckpt(out_dir / "ckpt_final.pt", model, opt, cfg.epochs, cfg.epochs, ema)
    print("[train] done.")