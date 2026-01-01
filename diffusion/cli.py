# nb_sim/diffusion/cli.py
from __future__ import annotations
import argparse
from pathlib import Path
from .config import DiffusionConfig
from .trainer import train
from .inference import sample_trajectories

def main():
    p = argparse.ArgumentParser("Modal Diffusion with EGNN conditioning (NOLB decoder)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train diffusion in modal space")
    pt.add_argument("--pairs", type=str, required=True, help="JSON list of {'apo','holo'} pairs")
    pt.add_argument("--out", type=str, default="runs/diff_egnn", help="output directory")
    pt.add_argument("--epochs", type=int, default=200)
    pt.add_argument("--batch_size", type=int, default=4)
    pt.add_argument("--n_modes", type=int, default=64)
    pt.add_argument("--lr", type=float, default=2e-4)
    pt.add_argument("--amp", action="store_true")
    pt.add_argument("--num_workers", type=int, default=4)
    pt.add_argument("--edge_radius", type=float, default=30.0)

    ps = sub.add_parser("sample", help="Sample conformations")
    ps.add_argument("--pairs", type=str, required=True)
    ps.add_argument("--ckpt", type=str, required=True)
    ps.add_argument("--out", type=str, default="runs/samples_egnn")
    ps.add_argument("--n_modes", type=int, default=64)
    ps.add_argument("--steps", type=int, default=200)
    ps.add_argument("--n_per_pair", type=int, default=1)      # 1 by default, can increase
    ps.add_argument("--sampler", type=str, default="ddim", choices=["ddim","pf_ode"])
    ps.add_argument("--edge_radius", type=float, default=30.0)
    pt.add_argument("--t_repeats", type=int, default=8)

    args = p.parse_args()

    if args.cmd == "train":
        cfg = DiffusionConfig(
            pairs_json=args.pairs, out_dir=args.out, epochs=args.epochs,
            batch_size=args.batch_size, n_modes=args.n_modes, lr=args.lr,
            amp=args.amp, num_workers=args.num_workers, edge_radius=args.edge_radius,
            t_repeats=args.t_repeats,
        )
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        cfg.save(Path(cfg.out_dir) / "config.json")
        train(cfg)

    elif args.cmd == "sample":
        cfg = DiffusionConfig(
            pairs_json=args.pairs, n_modes=args.n_modes,
            n_samples_per_pair=args.n_per_pair, sample_steps=args.steps,
            sampler=args.sampler, edge_radius=args.edge_radius
        )
        sample_trajectories(cfg, args.ckpt, args.out)

if __name__ == "__main__":
    main()
