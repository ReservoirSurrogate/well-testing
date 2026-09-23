"""Train one neural-operator variant on one-step pairs (Milestone 5).

Pairs (u_n, log k) -> u_{n+1} are drawn at random from the stored trajectories: trajectory uniform,
n uniform in [0, n_steps), and a fraction `p_first` of each batch forced to n = 0 (the step from u = 1,
which every rollout starts with). Loss: relative L2 on the grid (ln r measure) plus
q_weight * |q_pred - q_true| / q_true, where q_D = F_{1/2} is the well rate from the first two nodes
(the grid L2 barely sees them). With --residual the model predicts u_{n+1} = u_n + v.

Fair comparison: every variant uses the same recipe (Adam, cosine learning-rate decay, batch, seed) and
about the same parameter count (MODES: dual and fno get 16 modes, see tasks.md).

The first run of a split extracts `u` from <split>.npz to <split>_u.npy, which is memory-mapped so that
parallel runs share one copy in the page cache.

Usage: python train.py --variant hankel [--residual --q-weight 0.1 --steps 5000 --threads 1 --out runs]
Writes: <out>/<variant>/{best.pt, last.pt, log.csv, config.json}
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import VARIANTS, build_model, n_params

HERE = Path(__file__).resolve().parent
MODES = {"hankel": 32, "hankel_local": 32, "logsine": 32, "dual": 16, "fno": 16}


def open_split(data_dir, split):
    """Return (u memmap (n_traj, n_states, N), logk (n_traj, N), r, r_e) for a split."""
    data_dir = Path(data_dir)
    npy = data_dir / f"{split}_u.npy"
    with np.load(data_dir / f"{split}.npz") as f:
        if not npy.exists():
            tmp = npy.with_suffix(".tmp.npy")
            np.save(tmp, f["u"])
            tmp.rename(npy)
        logk, r, r_e = f["logk"], f["r"], float(f["r_e"])
    return np.load(npy, mmap_mode="r"), logk, r, r_e


def sample_pairs(rng, n_traj, n_steps, size, p_first):
    """Indices (traj, n) of `size` pairs; about p_first of them have n = 0."""
    traj = rng.integers(0, n_traj, size)
    n = rng.integers(0, n_steps, size)
    n[rng.random(size) < p_first] = 0
    return traj, n


def rate_error(pred, target):
    """Per-sample relative error of the well rate q_D = trans * (u[1] - u[0]), shape (B,).

    The face transmissibility cancels, and u[0] = 0 in both, so this is |du[1]| / |u_true[1]|.
    """
    return (pred[:, 1] - target[:, 1]).abs() / target[:, 1].abs()


def batch(u, logk, traj, n):
    """Model input (B, 2, N) and target (B, N) as float32 tensors."""
    x = np.stack([u[traj, n], logk[traj]], axis=1)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), \
        torch.from_numpy(np.ascontiguousarray(u[traj, n + 1], dtype=np.float32))


def rel_l2(pred, target):
    """Per-sample relative L2 error on the grid (ln r measure), shape (B,)."""
    return torch.linalg.vector_norm(pred - target, dim=-1) / torch.linalg.vector_norm(target, dim=-1)


class ValSet:
    """Fixed validation pairs: one n = 0 pair per trajectory plus uniform random pairs."""

    def __init__(self, u, logk, n_pairs=2000, seed=1234):
        rng = np.random.default_rng(seed)
        n_traj, n_steps = u.shape[0], u.shape[1] - 1
        extra = max(n_pairs - n_traj, 0)
        traj = np.concatenate([np.arange(n_traj), rng.integers(0, n_traj, extra)])
        n = np.concatenate([np.zeros(n_traj, int), rng.integers(0, n_steps, extra)])
        order = np.lexsort((n, traj))                    # sorted access is faster on the memmap
        self.x, self.y = batch(u, logk, traj[order], n[order])
        self.first = torch.from_numpy(n[order] == 0)
        self.persistence = rel_l2(self.x[:, 0], self.y)  # error of the "copy the input" baseline

    @torch.no_grad()
    def evaluate(self, model, chunk=250):
        pred = torch.cat([model(self.x[i:i + chunk]) for i in range(0, len(self.y), chunk)])
        err = rel_l2(pred, self.y)
        return {"val": err.mean().item(), "val_first": err[self.first].mean().item(),
                "val_rest": err[~self.first].mean().item(),
                "val_q": rate_error(pred, self.y).mean().item()}


def train(variant, steps=5000, batch_size=32, lr=1e-3, p_first=0.05, width=32, n_layers=4,
          n_modes=None, residual=False, q_weight=0.0, seed=0, threads=1, eval_every=500,
          data_dir=HERE / "data", out=HERE / "runs", n_val=2000, log=print):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    u, logk, r, r_e = open_split(data_dir, "train")
    u_val, logk_val, _, _ = open_split(data_dir, "val")
    n_traj, n_steps = u.shape[0], u.shape[1] - 1

    n_modes = MODES[variant] if n_modes is None else n_modes
    model = build_model(variant, n_modes=n_modes, width=width, n_layers=n_layers, n_points=len(r), r_e=r_e,
                        residual=residual)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    val = ValSet(u_val, logk_val, n_val)

    run_dir = Path(out) / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    config = dict(variant=variant, steps=steps, batch_size=batch_size, lr=lr, p_first=p_first, width=width,
                  n_layers=n_layers, n_modes=n_modes, residual=residual, q_weight=q_weight, seed=seed,
                  threads=threads, n_params=n_params(model),
                  persistence_val=val.persistence.mean().item(),
                  persistence_val_rest=val.persistence[~val.first].mean().item())
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    log(f"{variant}: {config['n_params']} params, persistence baseline val {config['persistence_val']:.3e}")

    best, t0, running = np.inf, time.time(), []
    with open(run_dir / "log.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "train", "val", "val_first", "val_rest", "val_q", "lr", "seconds"])
        for step in range(1, steps + 1):
            traj, n = sample_pairs(rng, n_traj, n_steps, batch_size, p_first)
            x, y = batch(u, logk, traj, n)
            pred = model(x)
            loss = rel_l2(pred, y).mean()
            if q_weight:
                loss = loss + q_weight * rate_error(pred, y).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running.append(loss.item())

            if step % eval_every == 0 or step == steps:
                model.eval()
                v = val.evaluate(model)
                model.train()
                row = [step, np.mean(running), v["val"], v["val_first"], v["val_rest"], v["val_q"],
                       sched.get_last_lr()[0], time.time() - t0]
                writer.writerow(row)
                fh.flush()
                running = []
                objective = v["val"] + q_weight * v["val_q"]     # same combination as the training loss
                if objective < best:
                    best = objective
                    torch.save({"config": config, "state_dict": model.state_dict(), "step": step, **v},
                               run_dir / "best.pt")
                log(f"{variant} step {step}: train {row[1]:.3e} val {v['val']:.3e} "
                    f"(first {v['val_first']:.3e}, rest {v['val_rest']:.3e}, q {v['val_q']:.3e}) {row[-1]:.0f} s")
    torch.save({"config": config, "state_dict": model.state_dict(), "step": steps}, run_dir / "last.pt")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modes", type=int, default=None, help="default: MODES[variant]")
    ap.add_argument("--residual", action="store_true", help="predict u_{n+1} = u_n + v")
    ap.add_argument("--q-weight", type=float, default=0.0, help="weight of the well-rate loss term")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    args = ap.parse_args()
    train(args.variant, steps=args.steps, batch_size=args.batch_size, lr=args.lr, n_modes=args.modes,
          residual=args.residual, q_weight=args.q_weight, seed=args.seed, threads=args.threads, out=args.out, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
