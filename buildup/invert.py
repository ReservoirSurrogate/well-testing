"""Model-based inversion of one build-up test (build-up project): ln k(r), C_D and r_eD from the build-up curve.

Unknowns theta = [ln k at M control points, ln C_D, ln r_eD]:
- ln k(r) is piecewise linear in ln r between M control points evenly spaced on [0, ln X_MAX] (fixed physical radii,
  so the parametrization does not depend on r_eD; points beyond r_eD have no effect on the data);
- the forward model is solver.WellSolver on a log grid with a fixed number of nodes (so the misfit is smooth in
  r_eD), and the build-up follows by superposition: dp_bu(dt) = p_dd(t_p) - p_dd(t_p + dt) + p_dd(dt).

Objective (nonlinear least squares, scipy.optimize.least_squares):
    data:        (ln dp_sim - ln dp_obs) / noise                      (relative misfit, noise ~ gauge + model error)
                 (ln p_wf,sim - ln p_wf,obs) / noise                  (drawdown at shut-in: p_i measured before the
                                                                       test; carries the material balance -> r_eD)
    smoothness:  second differences of ln k at the control points / SMOOTH
    prior:       ln k / PRIOR_STD                                     (only matters where the data are silent)
Start from the classical estimates: C_D from the early unit slope, r_eD from material balance (stabilized build-up),
k from the Bourdet-derivative median. Uncertainty: Gauss-Newton covariance (J^T J)^-1 at the solution.

Usage: python invert.py [--sample 486] [--split test] [--t-p 1e6] [--noise 0.01] [--seed 0]
Writes: figs/fig_invert_<sample>.pdf (+ .png), prints the recovered parameters and ring averages
"""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from dataset import grid
from permeability import RINGS, ring_labels
from solver import WellSolver, log_grid
from wells import Drawdown, agarwal_time, bourdet_derivative, buildup

HERE = Path(__file__).resolve().parent
X_MAX = np.log(3000.0)
N_CTRL = 16
N_NODES = 900                   # grid nodes of the forward model (spacing ln r_e / 899 <= 0.0089)
SMOOTH = 0.3                    # allowed second difference of ln k between control points (~0.5 in ln r apart)
PRIOR_STD = 1.5                 # prior spread of ln k
X_CTRL = np.linspace(0.0, X_MAX, N_CTRL)


def profile(theta_k, r):
    """ln k on radii r from the control-point values."""
    return np.interp(np.log(r), X_CTRL, theta_k)


def forward(theta, t_p, dt):
    """(dp_bu(dt), p_wf) for parameters theta = [ln k controls, ln C_D, ln r_eD]; p_wf = drawdown at shut-in."""
    r, _ = log_grid(np.exp(theta[-1]), N_NODES)
    sol = WellSolver(r, np.exp(profile(theta[:N_CTRL], r)), c_d=np.exp(theta[-2]))
    p = sol.well_pressure(np.concatenate([dt, t_p + dt, [t_p]]))
    n = len(dt)
    return p[-1] - p[n:2 * n] + p[:n], p[-1]


def observe(data, i, t_p, dt, noise, rng):
    """Synthetic measured build-up of sample i (true heterogeneous forward, from the stored drawdown) and drawdown
    at shut-in p_wf, both with multiplicative gauge noise of relative size `noise`."""
    dd = Drawdown(data["t"], data["p_w"][i])
    _, dp = buildup(dd, t_p, dt)
    e = np.exp(noise * rng.standard_normal(len(dt) + 1))
    return dp * e[:-1], float(dd(t_p)) * e[-1]


def initial_guess(t_p, dt, dp):
    """Classical estimates: C_D (unit slope), the final pressure rise (for material balance), k (median derivative)."""
    c_d = max(float(np.median(dt[:3] / dp[:3])), 1e-3)
    der = bourdet_derivative(agarwal_time(t_p, dt), dp)
    k0 = 1.0 / (2 * np.median(der[(der > 0) & (agarwal_time(t_p, dt) > 10)]))
    return c_d, float(dp[-1]), float(np.clip(k0, 0.05, 20.0))


def invert(t_p, dt, dp_obs, p_wf, noise=0.01, verbose=0):
    """Fit theta to the observed build-up; p_wf = pressure drop at shut-in (known from the gauge).

    Returns a dict with theta, its standard deviations, the fitted curve and diagnostics.
    """
    c_d0, p_rise_end, k0 = initial_guess(t_p, dt, dp_obs)
    p_bar = p_wf - p_rise_end                    # average pressure drop after stabilization
    r_e0 = float(np.clip(np.sqrt(max(2 * (t_p / p_bar - c_d0) + 1, 1.0)), 150.0, 6000.0))
    theta0 = np.concatenate([np.full(N_CTRL, np.log(k0)), [np.log(c_d0), np.log(r_e0)]])
    d2 = np.diff(np.eye(N_CTRL), 2, axis=0)

    def residuals(theta):
        dp, p_wf_sim = forward(theta, t_p, dt)
        data_res = (np.log(np.maximum(dp, 1e-12)) - np.log(dp_obs)) / noise
        wf_res = (np.log(p_wf_sim) - np.log(p_wf)) / noise
        k = theta[:N_CTRL]
        return np.concatenate([data_res, [wf_res], d2 @ k / SMOOTH, k / PRIOR_STD])

    lo = np.concatenate([np.full(N_CTRL, -4.0), [np.log(1e-3), np.log(100.0)]])
    hi = np.concatenate([np.full(N_CTRL, 4.0), [np.log(1e4), np.log(1e4)]])
    t0 = time.time()
    fit = least_squares(residuals, np.clip(theta0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale=1.0,
                        diff_step=1e-4, verbose=verbose, max_nfev=400)
    j = fit.jac
    cov = np.linalg.pinv(j.T @ j)
    return {"theta": fit.x, "std": np.sqrt(np.clip(np.diag(cov), 0, None)), "theta0": theta0,
            "dp_fit": forward(fit.x, t_p, dt)[0], "cost": fit.cost, "nfev": fit.nfev, "njev": fit.njev,
            "status": fit.status, "seconds": time.time() - t0, "chi2_data": float(np.mean(fit.fun[:len(dt) + 1] ** 2))}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=int, default=486)
    ap.add_argument("--split", default="test")
    ap.add_argument("--t-p", type=float, default=1e6)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with np.load(HERE / "data" / f"{args.split}.npz") as f:
        data = {key: f[key] for key in f.files}
    i, t_p = args.sample, args.t_p
    dt = np.geomspace(data["t"][0], data["t"][-1] - t_p, 60)
    rng = np.random.default_rng(args.seed)
    dp_obs, p_wf = observe(data, i, t_p, dt, args.noise, rng)
    res = invert(t_p, dt, dp_obs, p_wf, noise=max(args.noise, 1e-3))

    th, sd = res["theta"], res["std"]
    r_true = grid(data, i)
    lnk_true = data["logk"][i, :data["n_r"][i]].astype(float)
    r_fit, _ = log_grid(np.exp(th[-1]), N_NODES)
    rings_true = ring_labels(r_true, np.exp(lnk_true))
    rings_fit = ring_labels(r_fit, np.exp(profile(th[:N_CTRL], r_fit)))
    print(f"sample {i}: {res['nfev']} iterations ({res['njev']} Jacobians), status {res['status']}, "
          f"{res['seconds']:.0f} s, data misfit chi2/n = {res['chi2_data']:.2f}")
    print(f"C_D : {np.exp(th[-2]):.3f} (x/ exp({sd[-2]:.2f}))   true {data['c_d'][i]:.3f}   start {np.exp(res['theta0'][-2]):.3f}")
    print(f"r_eD: {np.exp(th[-1]):.1f} (x/ exp({sd[-1]:.3f}))   true {data['r_e'][i]:.1f}   start {np.exp(res['theta0'][-1]):.1f}")
    print("ring      true ln k   recovered")
    for (a, b), t_, f_ in zip(RINGS, rings_true, rings_fit):
        print(f"[{a:4d},{b:4d}]   {t_:8.3f}   {f_:8.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.geomspace(1, 3000, 400)
    lnk_fit = profile(th[:N_CTRL], x)
    band = np.interp(np.log(x), X_CTRL, sd[:N_CTRL])
    ax[0].semilogx(r_true, lnk_true, color="black", lw=1.4, label="true ln k")
    ax[0].semilogx(x, lnk_fit, color="#2a78d6", lw=1.8, label="recovered (control points linear in ln r)")
    ax[0].fill_between(x, lnk_fit - 2 * band, lnk_fit + 2 * band, color="#2a78d6", alpha=0.18, lw=0, label="±2σ")
    ax[0].axvline(data["r_e"][i], color="black", lw=1, ls="--", label=f"true $r_{{eD}}$ = {data['r_e'][i]:.0f}")
    ax[0].axvline(np.exp(th[-1]), color="#2a78d6", lw=1, ls=":", label=f"recovered $r_{{eD}}$ = {np.exp(th[-1]):.0f}")
    if data["has_skin"][i]:
        ax[0].axvline(data["r_s"][i], color="#eb6834", lw=1, ls="--", label=f"skin zone $r_s$ = {data['r_s'][i]:.1f}")
    ax[0].set_ylim(-4, 4)
    ax[0].set_xlabel(r"$r_D$")
    ax[0].set_ylabel(r"$\ln k_D$")
    ax[0].set_title(f"(a) test sample {i}: permeability profile")
    ax[0].legend(fontsize=7, frameon=False, loc="lower left")

    te = agarwal_time(t_p, dt)
    for dp, color, lab in ((dp_obs, "black", "observed"), (res["dp_fit"], "#2a78d6", "fitted")):
        der = bourdet_derivative(te, dp)
        style = dict(ls="none", marker="o", ms=3) if lab == "observed" else dict(lw=1.8)
        ax[1].loglog(te, dp, color=color, label=f"{lab} $\\Delta p_{{bu}}$", **style)
        ok = der > 0
        ax[1].loglog(te[ok], der[ok], color=color, alpha=0.6, label=f"{lab} derivative",
                     **({**style, "marker": "^"} if lab == "observed" else {**style, "ls": "--"}))
    ax[1].set_xlabel(r"Agarwal time $\Delta t_e$")
    ax[1].set_title(f"(b) build-up after $t_p$ = {t_p:.0e}, {args.noise:.0%} noise;  "
                    f"$C_D$ = {np.exp(th[-2]):.2f} (true {data['c_d'][i]:.2f})")
    ax[1].set_ylim(1e-3, None)
    ax[1].legend(fontsize=7, frameon=False)
    for a in ax:
        a.grid(True, which="major", color="#e5e5e2", lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    out = HERE / "figs" / f"fig_invert_{i}.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
