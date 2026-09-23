"""Training data for the one-step operator (u(t), log k) -> u(t + dt) (Milestone 3).

Random permeability (per sample):
    log k_D(r) = m + g(ln r) + skin(r),  clipped to [-3, 3]
    m ~ N(0, 0.3^2)                               mean shift
    g: zero-mean GRF in s = ln r, covariance sigma^2 exp(-(s - s')^2 / (2 l^2)),
       sigma ~ U(0.5, 1), l ~ U(0.3, 1.5)
    skin (probability 0.5): constant offset ln(k_s/k) ~ U(ln 0.1, ln 5) for r < r_s, r_s ~ logU[2, 20]

Trajectories start from u = 1 (u(1) = 0) at t = 0 and are stored at t = 0, dt, ..., n_steps dt.
The (u_n, log k) -> u_{n+1} pairs are formed at training time from consecutive states.

Usage: OMP_NUM_THREADS=1 python dataset.py [--n-train 800 --n-val 100 --n-test 100 --workers 8 --out data]
(one BLAS thread per worker: threaded eigh is slower here and oversubscribes the cores)
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from basis import make_grid
from solver import RadialSolver, time_grid

R_E = 1000.0
N_POINTS = 1024
DT = 1000.0
N_STEPS = 500
H_MAX = 20.0      # largest solver step; err 1.6e-5 at t = 1e3, < 1e-6 from t = 1e4 (k_D = 1)
LOGK_CLIP = 3.0
META_KEYS = ("mean", "sigma", "ell", "has_skin", "r_s", "skin")
SPLIT_SEEDS = {"train": 0, "val": 1, "test": 2}


def grf(rng, s, sigma, ell, size=None):
    """Zero-mean Gaussian field on points s with squared-exponential covariance.

    Returns one sample (len(s),), or `size` samples (size, len(s)) sharing one factorization.
    Uses a symmetric eigendecomposition: the SE covariance on a fine grid is numerically
    singular for large l, where Cholesky fails.
    """
    cov = sigma**2 * np.exp(-0.5 * ((s[:, None] - s[None, :]) / ell) ** 2)
    w, v = np.linalg.eigh(cov)
    z = rng.standard_normal(len(s) if size is None else (size, len(s)))
    return (np.sqrt(np.clip(w, 0.0, None)) * z) @ v.T


def sample_logk(rng, r, p_skin=0.5):
    """Random log k_D on the grid r; returns (logk, meta) with meta keyed by META_KEYS."""
    meta = {
        "mean": rng.normal(0.0, 0.3),
        "sigma": rng.uniform(0.5, 1.0),
        "ell": rng.uniform(0.3, 1.5),
        "has_skin": rng.random() < p_skin,
        "r_s": np.nan,
        "skin": 0.0,
    }
    logk = meta["mean"] + grf(rng, np.log(r), meta["sigma"], meta["ell"])
    if meta["has_skin"]:
        meta["r_s"] = np.exp(rng.uniform(np.log(2.0), np.log(20.0)))
        meta["skin"] = rng.uniform(np.log(0.1), np.log(5.0))
        logk = np.where(r < meta["r_s"], logk + meta["skin"], logk)
    return np.clip(logk, -LOGK_CLIP, LOGK_CLIP), meta


def trajectory(r, logk, dt=DT, n_steps=N_STEPS, h_max=H_MAX):
    """States u at t = 0, dt, ..., n_steps dt from u = 1; shape (n_steps + 1, len(r))."""
    t_grid, idx = time_grid(dt * np.arange(1, n_steps + 1), h_max=h_max)
    u0 = np.ones(len(r))
    u = RadialSolver(r, np.exp(logk)).integrate(u0, t_grid)
    return np.concatenate([u[:1], u[idx]])


def _sample(args):
    seed_seq, r, dt, n_steps, h_max = args
    rng = np.random.default_rng(seed_seq)
    logk, meta = sample_logk(rng, r)
    u = trajectory(r, logk, dt, n_steps, h_max)
    return u.astype(np.float32), logk.astype(np.float32), meta


def generate_split(n_traj, seed, n_points=N_POINTS, r_e=R_E, dt=DT, n_steps=N_STEPS,
                   h_max=H_MAX, workers=1):
    """Generate n_traj trajectories; sample i depends only on (seed, i), not on `workers`."""
    r, _ = make_grid(n_points, r_e)
    jobs = [(ss, r, dt, n_steps, h_max) for ss in np.random.SeedSequence(seed).spawn(n_traj)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_sample, jobs, chunksize=4))
    else:
        results = [_sample(j) for j in jobs]
    u, logk, metas = zip(*results)
    data = {
        "u": np.stack(u),                          # (n_traj, n_steps + 1, n_points)
        "logk": np.stack(logk),                    # (n_traj, n_points)
        "t": dt * np.arange(n_steps + 1),
        "r": r,
        "r_e": np.float64(r_e),
        "dt": np.float64(dt),
        "h_max": np.float64(h_max),
        "seed": np.int64(seed),
    }
    for key in META_KEYS:
        data[key] = np.array([m[key] for m in metas])
    return data


def save_split(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def load_split(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    args = ap.parse_args()

    for split, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        if n == 0:
            continue
        t0 = time.time()
        data = generate_split(n, SPLIT_SEEDS[split], workers=args.workers)
        save_split(args.out / f"{split}.npz", data)
        print(f"{split}: {n} trajectories, u {data['u'].shape}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
