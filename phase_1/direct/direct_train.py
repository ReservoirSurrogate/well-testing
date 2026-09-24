"""Train one direct-operator variant (Milestone 5b).

The PDE is autonomous, so any two states of a stored trajectory form a training example:
(u(t_i), log k, dt = t_j - t_i) -> u(t_j). The gap m = j - i is drawn log-uniformly in [1, n_steps] (so short and
long jumps are equally represented on a log scale), then i uniformly in [0, n_steps - m]; a fraction `p_start`
of each batch is forced to i = 0, the jump from u = 1 used at prediction time.
Loss: relative L2 on the grid (ln r measure) + q_weight * relative well-rate error (see phase_1/train.py).

Usage: python direct_train.py --variant logsine [--steps 5000 --q-weight 0.1 --threads 2 --out runs]
Writes: <out>/<variant>/{best.pt, last.pt, log.csv, config.json}
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from direct_model import build_direct, n_params  # noqa: E402
from train import MODES as BASE_MODES  # noqa: E402
from train import open_split, rate_error, rel_l2  # noqa: E402

MODES = {**BASE_MODES, "fftlog": 32, "fftlog_decay": 32}

DATA = HERE.parent / "data"
VAL_GAPS = (1, 3, 10, 30, 100, 300, 500)   # gaps (in stored steps) of the fixed jumps from u = 1


def sample_pairs(rng, n_traj, n_steps, size, p_start, gap=None):
    """Indices (traj, i, j) with 1 <= j - i <= n_steps, j - i log-uniform; about p_start of them have i = 0.

    With `gap`, every pair is the fixed jump (traj, 0, gap) from u = 1.
    """
    traj = rng.integers(0, n_traj, size)
    if gap is not None:
        return traj, np.zeros(size, int), np.full(size, gap)
    m = np.floor(np.exp(rng.uniform(0.0, np.log(n_steps + 1), size))).astype(int).clip(1, n_steps)
    i = rng.integers(0, n_steps - m + 1)
    i[rng.random(size) < p_start] = 0
    return traj, i, i + m


def batch(u, logk, t, traj, i, j):
    """Model input (B, 2, N), dt (B,) and target (B, N) as float32 tensors."""
    x = np.stack([u[traj, i], logk[traj]], axis=1)
    as_t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return as_t(x), as_t(t[j] - t[i]), as_t(u[traj, j])


class ValSet:
    """Fixed validation pairs: jumps from u = 1 by VAL_GAPS for every trajectory, plus random general pairs.

    With `gap`, only the jump (traj, 0, gap) of every trajectory.
    """

    def __init__(self, u, logk, t, n_general=1000, seed=1234, gap=None):
        rng = np.random.default_rng(seed)
        n_traj, n_steps = u.shape[0], u.shape[1] - 1
        gaps = [gap] if gap is not None else [g for g in VAL_GAPS if g <= n_steps]
        traj0 = np.repeat(np.arange(n_traj), len(gaps))
        j0 = np.tile(gaps, n_traj)
        n_general = 0 if gap is not None else n_general
        trajg, ig, jg = sample_pairs(rng, n_traj, n_steps, n_general, p_start=0.0)
        traj, i, j = np.r_[traj0, trajg], np.r_[np.zeros_like(j0), ig], np.r_[j0, jg]
        order = np.lexsort((j, i, traj))                   # sorted access is faster on the memmap
        traj, i, j = traj[order], i[order], j[order]
        self.x, self.dt, self.y = batch(u, logk, t, traj, i, j)
        self.start = torch.from_numpy(i == 0)

    @torch.no_grad()
    def evaluate(self, model, chunk=250):
        pred = torch.cat([model(self.x[k:k + chunk], self.dt[k:k + chunk]) for k in range(0, len(self.y), chunk)])
        err, q = rel_l2(pred, self.y), rate_error(pred, self.y)
        general = err[~self.start].mean().item() if (~self.start).any() else float("nan")
        return {"val": err.mean().item(), "val_start": err[self.start].mean().item(),
                "val_general": general, "val_q": q.mean().item(),
                "val_q_start": q[self.start].mean().item()}


def train(variant, steps=5000, batch_size=32, lr=1e-3, p_start=0.5, width=32, n_layers=4, n_modes=None,
          q_weight=0.1, seed=0, threads=1, eval_every=500, data_dir=DATA, out=HERE / "runs", n_val=1000,
          gap=None, log=print):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    u, logk, r, r_e = open_split(data_dir, "train")
    u_val, logk_val, _, _ = open_split(data_dir, "val")
    with np.load(Path(data_dir) / "train.npz") as f:
        t = f["t"].astype(float)
    n_traj, n_steps = u.shape[0], u.shape[1] - 1

    n_modes = MODES[variant] if n_modes is None else n_modes
    model = build_direct(variant, n_modes=n_modes, width=width, n_layers=n_layers, n_points=len(r), r_e=r_e)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    val = ValSet(u_val, logk_val, t, n_val, gap=gap)

    run_dir = Path(out) / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    config = dict(variant=variant, steps=steps, batch_size=batch_size, lr=lr, p_start=p_start, width=width,
                  n_layers=n_layers, n_modes=n_modes, q_weight=q_weight, seed=seed, threads=threads, gap=gap,
                  n_params=n_params(model))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    log(f"{variant}: {config['n_params']} params")

    best, t0, running = np.inf, time.time(), []
    with open(run_dir / "log.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "train", "val", "val_start", "val_general", "val_q", "val_q_start", "lr", "seconds"])
        for step in range(1, steps + 1):
            x, dt, y = batch(u, logk, t, *sample_pairs(rng, n_traj, n_steps, batch_size, p_start, gap))
            pred = model(x, dt)
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
                row = [step, np.mean(running), v["val"], v["val_start"], v["val_general"], v["val_q"],
                       v["val_q_start"], sched.get_last_lr()[0], time.time() - t0]
                writer.writerow(row)
                fh.flush()
                running = []
                objective = v["val"] + q_weight * v["val_q"]     # same combination as the training loss
                if objective < best:
                    best = objective
                    torch.save({"config": config, "state_dict": model.state_dict(), "step": step, **v},
                               run_dir / "best.pt")
                log(f"{variant} step {step}: train {row[1]:.3e} val {v['val']:.3e} (start {v['val_start']:.3e}, "
                    f"general {v['val_general']:.3e}, q {v['val_q']:.3e}) {row[-1]:.0f} s")
    torch.save({"config": config, "state_dict": model.state_dict(), "step": steps}, run_dir / "last.pt")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", required=True, choices=sorted(MODES))
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modes", type=int, default=None, help="default: MODES[variant]")
    ap.add_argument("--p-start", type=float, default=0.5, help="fraction of pairs starting from u = 1")
    ap.add_argument("--q-weight", type=float, default=0.1, help="weight of the well-rate loss term")
    ap.add_argument("--gap", type=int, default=None,
                    help="train only the fixed jump from u = 1 to stored state `gap` (1: t_D = 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    args = ap.parse_args()
    train(args.variant, steps=args.steps, batch_size=args.batch_size, lr=args.lr, n_modes=args.modes,
          p_start=args.p_start, q_weight=args.q_weight, seed=args.seed, threads=args.threads, out=args.out,
          gap=args.gap, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
