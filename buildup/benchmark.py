"""Benchmark: neural inverse model vs model-based inversion on the same build-ups (build-up project, Milestone 7).

N_RES test reservoirs (seeded subset), one build-up each per noise level, with the realistic test settings of
examples.Bank.make (t_p ~ logU[1e4, 1e6], half the tests truncated at dt_max >= 1e6, 25% without p_i):
- network: runs/noise (noise-aware), means and predicted sigma;
- inversion: invert.invert on the valid shut-in times (with the drawdown at shut-in only when p_i is known), noise
  weight = the gauge noise (floored at 2e-4 for model error); ring uncertainty by sampling its Gauss-Newton
  covariance (N_SAMPLES draws).
Per method: ring ln k MAE and +/- 2 sigma coverage, ln C_D MAE (samples with storage) and coverage, r_eD median
relative error (all / p_i known / unknown) and coverage, run time per test, and for the inversion the share of poor
fits (data misfit chi2 / n > 4).

Usage: OMP_NUM_THREADS=1 python benchmark.py [--n 160] [--workers 8] [--noise 1e-2 1e-3]
Writes: runs/benchmark/results.json, runs/benchmark/cases_<noise>.npz
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from examples import DT, N_RING, denormalize
from invert import N_CTRL, N_NODES, invert, profile
from permeability import ring_labels
from solver import log_grid
from train import HERE, load_bank, predict

N_SAMPLES = 40
RUN = HERE / "runs" / "noise"
OUT = HERE / "runs" / "benchmark"


def rings_from_theta(theta):
    r, _ = log_grid(float(np.exp(theta[-1])), N_NODES)
    return ring_labels(r, np.exp(profile(theta[:N_CTRL], r)))


def invert_case(args):
    """One inversion; returns (means (N_RING + 2,), sigmas (N_RING + 2,), seconds, chi2, status)."""
    t_p, dt, dp, p_wf, noise, seed = args
    t0 = time.time()
    try:
        res = invert(t_p, dt, dp, p_wf, noise=max(noise, 2e-4))
    except Exception:                                            # a failed fit counts as a failure
        return np.full(N_RING + 2, np.nan), np.full(N_RING + 2, np.nan), time.time() - t0, np.inf, -99
    th, cov = res["theta"], res["cov"]
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(th, cov, size=N_SAMPLES, check_valid="ignore")
    draws[:, -1] = np.clip(draws[:, -1], np.log(100.0), np.log(1e4))
    ring_draws = np.stack([rings_from_theta(d) for d in draws])
    means = np.concatenate([rings_from_theta(th), [th[-2], th[-1]]])
    sig = np.concatenate([np.nanstd(ring_draws, axis=0), [res["std"][-2], res["std"][-1]]])
    return means, sig, time.time() - t0, res["chi2_data"], res["status"]


def metrics(rings_p, rings_sd, lnc_p, lnc_sd, lnre_p, lnre_sd, rings_t, lnc_t, lnre_t, has_c, known, secs):
    err_r = rings_p - rings_t
    valid = ~np.isnan(rings_t) & ~np.isnan(rings_p)
    z_r = np.abs(err_r) / rings_sd
    rel = np.abs(np.exp(lnre_p - lnre_t) - 1)
    z_re = np.abs(lnre_p - lnre_t) / lnre_sd
    z_c = np.abs(lnc_p - lnc_t) / lnc_sd
    ring_mae = [float(np.mean(np.abs(err_r[valid[:, j], j]))) for j in range(N_RING)]
    return {"ring_mae": ring_mae, "ring_mae_mean": float(np.mean(ring_mae)),
            "ring_cover2": float(np.mean(z_r[valid] < 2)),
            "lnC_mae": float(np.mean(np.abs(lnc_p - lnc_t)[has_c])), "C_cover2": float(np.mean(z_c[has_c] < 2)),
            "re_median": float(np.median(rel)), "re_median_known": float(np.median(rel[known])),
            "re_median_unknown": float(np.median(rel[~known])), "re_p90": float(np.percentile(rel, 90)),
            "re_cover2": float(np.mean(z_re < 2)), "seconds_per_test": float(np.mean(secs))}


def main():
    from evaluate import load_model

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--noise", type=float, nargs="+", default=[1e-2, 1e-3])
    args = ap.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)

    bank = load_bank("test")
    idx = np.sort(np.random.default_rng(7).choice(bank.n, args.n, replace=False))
    model, _ = load_model(RUN)
    results = {"n": args.n, "run": str(RUN.name)}
    for noise in args.noise:
        rng = np.random.default_rng(99)
        x, y, ns, meta = bank.make(idx, rng, noise=noise)
        rings_t, lnc_t, lnre_t = denormalize(np.nan_to_num(y.astype(float)))
        rings_t = np.where(np.isnan(y[:, :N_RING]), np.nan, rings_t)
        has_c, known = ns == 0, meta["pi_known"]

        t0 = time.time()
        mu, log_var, _ = (t.numpy().astype(float) for t in predict(model, torch.from_numpy(x)))
        net_secs = np.full(len(idx), (time.time() - t0) / len(idx))
        sd = np.exp(0.5 * log_var)
        rp, cp, rep = denormalize(mu)
        net = metrics(rp, sd[:, :N_RING], cp, sd[:, N_RING] * 2.7, rep, sd[:, N_RING + 1] * 0.67,
                      rings_t, lnc_t, lnre_t, has_c, known, net_secs)

        jobs = []
        for b in range(len(idx)):
            valid = x[b, 6] > 0
            dp = np.exp(5 * x[b, 0, valid].astype(float))
            p_wf = float(meta["p_wf"][b]) if known[b] else None
            jobs.append((float(meta["t_p"][b]), DT[valid], dp, p_wf, noise, int(b)))
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            out = list(ex.map(invert_case, jobs))
        wall = time.time() - t0
        means = np.stack([o[0] for o in out])
        sig = np.stack([o[1] for o in out])
        secs = np.array([o[2] for o in out])
        chi2 = np.array([o[3] for o in out])
        status = np.array([o[4] for o in out])
        inv = metrics(means[:, :N_RING], sig[:, :N_RING], means[:, N_RING], sig[:, N_RING], means[:, N_RING + 1],
                      sig[:, N_RING + 1], rings_t, lnc_t, lnre_t, has_c, known, secs)
        inv["poor_fit_share"] = float(np.mean((chi2 > 4) | (status <= 0)))
        inv["failed"] = int(np.sum(status == -99))
        inv["wall_seconds"] = wall
        results[f"{noise:g}"] = {"network": net, "inversion": inv}
        np.savez(OUT / f"cases_{noise:g}.npz", idx=idx, t_p=meta["t_p"], dt_max=meta["dt_max"], known=known,
                 has_c=has_c, rings_t=rings_t, lnc_t=lnc_t, lnre_t=lnre_t, net_mu=mu, net_log_var=log_var,
                 inv_means=means, inv_sig=sig, inv_secs=secs, inv_chi2=chi2, inv_status=status)
        print(f"== noise {noise:g}  ({args.n} tests; inversion wall time {wall:.0f} s)")
        print(f"{'':<11} {'ring MAE':>8} {'ring cov':>8} {'lnC MAE':>8} {'C cov':>6} {'r_e med':>8} "
              f"{'known':>7} {'unknown':>8} {'r_e p90':>8} {'r_e cov':>8} {'s/test':>8}")
        for name, m in (("network", net), ("inversion", inv)):
            print(f"{name:<11} {m['ring_mae_mean']:8.3f} {m['ring_cover2']:8.3f} {m['lnC_mae']:8.3f} "
                  f"{m['C_cover2']:6.3f} {m['re_median']:8.3f} {m['re_median_known']:7.3f} "
                  f"{m['re_median_unknown']:8.3f} {m['re_p90']:8.3f} {m['re_cover2']:8.3f} "
                  f"{m['seconds_per_test']:8.3g}")
        print(f"inversion poor fits (chi2/n > 4 or failed): {inv['poor_fit_share']:.1%}")
        print("ring MAE by ring  network  :", " ".join(f"{v:.3f}" for v in net["ring_mae"]))
        print("ring MAE by ring  inversion:", " ".join(f"{v:.3f}" for v in inv["ring_mae"]), flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
