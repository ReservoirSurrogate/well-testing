"""Synthetic build-up dataset (build-up project, Milestone 4).

Per sample (seeded independently, so results do not depend on the number of workers):
    r_eD ~ logU[300, 3000]; log grid 1..r_eD with spacing ~DLOG (N = 846..1187 nodes)
    ln k_D(r) from permeability.sample_logk (GRF in ln r + skin zone, clipped to [-3, 3])
    C_D = 0 with probability 0.2, otherwise logU[0.1, 1e3]
    one unit-rate drawdown p_wD(t) at the 200 log-spaced times T in [1e-2, 1e7] (solver.WellSolver)
    labels k_eff, S (permeability.labels)
Build-ups for any t_pD are formed later by superposition (wells.buildup).

Stored per split in <out>/<split>.npz:
    t (200,)                 output times
    p_w (n, 200) float64     drawdown well pressure (float64: build-ups are differences of these values)
    logk (n, N_max) float32  ln k_D on the sample's grid, NaN-padded to N_max; n_r (n,) its length
    r_e, c_d, k_eff, skin    per-sample parameters and labels (skin = the label S)
    mean, sigma, ell, has_skin, r_s, skin_log   generator parameters
    dlog (n,)                grid spacing ln(r_e) / (n_r - 1) (the grid is exp(dlog * arange(n_r)))

Usage: OMP_NUM_THREADS=1 python dataset.py [--workers 8] [--out data] [--splits train val test] [--n N]
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from permeability import labels, sample_logk
from solver import WellSolver, grid_for

HERE = Path(__file__).resolve().parent
DLOG = np.log(1000.0) / 1023
T = np.logspace(-2, 7, 200)
R_E_RANGE = (300.0, 3000.0)
P_NO_STORAGE = 0.2
C_D_RANGE = (0.1, 1e3)
N_MAX = int(round(np.log(R_E_RANGE[1]) / DLOG)) + 1
SPLITS = {"train": (18000, 0), "val": (1000, 1), "test": (1000, 2)}      # (size, seed)
META_KEYS = ("mean", "sigma", "ell", "has_skin", "r_s", "skin_log")


def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample(seed_seq):
    """One sample from its own seed sequence; returns a dict of arrays / scalars."""
    rng = np.random.default_rng(seed_seq)
    r_e = log_uniform(rng, *R_E_RANGE)
    r, dlog = grid_for(r_e, DLOG)
    logk, meta = sample_logk(rng, r)
    c_d = 0.0 if rng.random() < P_NO_STORAGE else log_uniform(rng, *C_D_RANGE)
    k = np.exp(logk)
    p_w = WellSolver(r, k, c_d=c_d).well_pressure(T)
    k_eff, skin = labels(r, k, meta["r_s"])
    return {"p_w": p_w, "logk": logk, "n_r": len(r), "r_e": r_e, "c_d": c_d, "k_eff": k_eff, "skin": skin,
            "dlog": dlog, **meta}


def generate_split(n, seed, workers=1):
    """n samples from SeedSequence(seed); sample i depends only on (seed, i)."""
    seeds = np.random.SeedSequence(seed).spawn(n)
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(sample, seeds, chunksize=16))
    else:
        results = [sample(s) for s in seeds]
    logk = np.full((n, N_MAX), np.nan, dtype=np.float32)
    for i, res in enumerate(results):
        logk[i, :res["n_r"]] = res["logk"]
    data = {"t": T, "p_w": np.stack([res["p_w"] for res in results]), "logk": logk}
    for key in ("n_r", "r_e", "c_d", "k_eff", "skin", "dlog") + META_KEYS:
        data[key] = np.array([res[key] for res in results])
    data["seed"] = np.int64(seed)
    return data


def grid(data, i):
    """The log grid r of sample i."""
    return np.exp(data["dlog"][i] * np.arange(data["n_r"][i]))


def save_split(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--n", type=int, default=None, help="override the split size (quick runs)")
    args = ap.parse_args()
    for split in args.splits:
        size, seed = SPLITS[split]
        n = size if args.n is None else args.n
        t0 = time.time()
        save_split(args.out / f"{split}.npz", generate_split(n, seed, args.workers))
        print(f"{split}: {n} samples in {time.time() - t0:.0f} s -> {args.out / f'{split}.npz'}", flush=True)


if __name__ == "__main__":
    main()
