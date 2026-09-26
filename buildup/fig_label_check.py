"""Label check (build-up project, Milestone 5): do the labels match what a build-up interpretation reads?

On the test split, each sample's build-up after t_pD = 1e6 (the longest production time) is interpreted classically:
- k from the Bourdet-derivative plateau, S from the semilog line (wells.interpret_radial_flow), in an oracle
  radial-flow window: after wellbore storage ends (10 (60 + 3.5 max(S, 0)) C_D / k_eff, at least 50) and before the
  boundary is felt (r_eD^2 / (4 k_eff) / 4); samples without such a window are skipped;
- r_eD by material balance from the stabilized build-up: the pressure settles at p_bar = t_p / ((r_eD^2 - 1)/2 + C_D)
  whatever k(r) is, so r_eD = sqrt(2 (t_p / p_bar - C_D) + 1); used where the shut-in lasts > 3 r_eD^2 / k_eff.

k is compared with the label k_eff (log-harmonic mean over [r_s, r_eD]) and with the log-harmonic mean over the
ring investigated during the window, r = 2 sqrt(k_eff dt) (what the plateau should average in a heterogeneous
reservoir).

Run with:  /home/daniel_88/py314/bin/python buildup/fig_label_check.py
Writes:    buildup/figs/fig_label_check.pdf (+ .png), prints summary statistics
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import grid
from permeability import resistance
from wells import Drawdown, agarwal_time, bourdet_derivative, buildup, buildup_times, interpret_radial_flow

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs" / "fig_label_check.pdf"
T_P = 1e6
COLORS = {False: "#2a78d6", True: "#eb6834"}          # no skin zone / skin zone
MUTED = "#52514e"


def ring_mean(r, k, a, b):
    """Log-harmonic mean of k over [a, b] (clipped to the grid)."""
    a, b = max(a, r[0]), min(b, r[-1])
    return np.log(b / a) / resistance(r, k, a, b)


def check(data):
    rows = []
    for i in range(len(data["r_e"])):
        k_eff, s, c_d, r_e = data["k_eff"][i], data["skin"][i], data["c_d"][i], data["r_e"][i]
        r = grid(data, i)
        k = np.exp(data["logk"][i, :data["n_r"][i]].astype(float))
        dd = Drawdown(data["t"], data["p_w"][i])
        dt = buildup_times(T_P, data["t"][0], data["t"][-1], n=300)
        p_w, dp = buildup(dd, T_P, dt)
        te = agarwal_time(T_P, dt)
        row = {"i": i, "k_eff": k_eff, "skin": s, "r_e": r_e, "has_skin": bool(data["has_skin"][i]),
               "k_est": np.nan, "s_est": np.nan, "k_ring": np.nan, "r_e_est": np.nan}
        lo = max(10 * (60 + 3.5 * max(s, 0)) * c_d / k_eff, 50.0)
        hi = min(r_e**2 / (4 * k_eff) / 4, T_P / 2)
        if hi > 3 * lo:
            row["k_est"], row["s_est"] = interpret_radial_flow(te, dp, bourdet_derivative(te, dp), (lo, hi))
            row["k_ring"] = ring_mean(r, k, 2 * np.sqrt(k_eff * lo), 2 * np.sqrt(k_eff * hi))
        if dt[-1] > 3 * r_e**2 / k_eff:
            p_bar = p_w[-1]
            row["r_e_est"] = np.sqrt(2 * (T_P / p_bar - c_d) + 1)
        rows.append(row)
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


def main():
    with np.load(HERE / "data" / "test.npz") as f:
        data = {key: f[key] for key in f.files}
    res = check(data)
    has_k, has_r = np.isfinite(res["k_est"]), np.isfinite(res["r_e_est"])
    err_k = np.log(res["k_est"][has_k] / res["k_eff"][has_k])
    err_ring = np.log(res["k_est"][has_k] / res["k_ring"][has_k])
    err_s = res["s_est"][has_k] - res["skin"][has_k]
    err_r = res["r_e_est"][has_r] / res["r_e"][has_r] - 1
    n = len(res["i"])
    print(f"radial-flow window found: {has_k.sum()} / {n}; stabilized build-up: {has_r.sum()} / {n}")
    print(f"k_est vs k_eff : median |ln ratio| {np.median(np.abs(err_k)):.3f}, 90th pct {np.percentile(np.abs(err_k), 90):.3f}, "
          f"within 10%: {np.mean(np.abs(err_k) < np.log(1.1)):.1%}")
    print(f"k_est vs k_ring: median |ln ratio| {np.median(np.abs(err_ring)):.3f}, 90th pct {np.percentile(np.abs(err_ring), 90):.3f}, "
          f"within 10%: {np.mean(np.abs(err_ring) < np.log(1.1)):.1%}")
    print(f"S_est - S      : median |diff| {np.median(np.abs(err_s)):.3f}, 90th pct {np.percentile(np.abs(err_s), 90):.3f}")
    print(f"r_e (mat. bal.): median |rel err| {np.median(np.abs(err_r)):.2e}, max {np.max(np.abs(err_r)):.2e}")

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for skin_zone in (False, True):
        m = has_k & (res["has_skin"] == skin_zone)
        lab = "skin zone" if skin_zone else "no skin zone"
        kw = dict(s=10, alpha=0.6, color=COLORS[skin_zone], label=lab, lw=0)
        ax[0].scatter(res["k_eff"][m], res["k_est"][m], **kw)
        ax[1].scatter(res["k_ring"][m], res["k_est"][m], **kw)
        ax[2].scatter(res["skin"][m], res["s_est"][m], **kw)
        mr = has_r & (res["has_skin"] == skin_zone)
        ax[3].scatter(res["r_e"][mr], res["r_e_est"][mr], **kw)
    for a, (x, y, title, log) in zip(ax, [
            ("label $k_{eff}$ (over $[r_s, r_{eD}]$)", "plateau $k$", "(a) permeability vs label", True),
            ("$k$ over the investigated ring", "plateau $k$", "(b) permeability vs investigated ring", True),
            ("label $S$", "semilog $S$", "(c) skin", False),
            ("true $r_{eD}$", "material-balance $r_{eD}$", "(d) boundary distance", True)]):
        lims = np.array([min(a.get_xlim()[0], a.get_ylim()[0]), max(a.get_xlim()[1], a.get_ylim()[1])])
        a.plot(lims, lims, color=MUTED, lw=0.8)
        if log:
            a.set_xscale("log")
            a.set_yscale("log")
        else:
            a.set_xscale("symlog", linthresh=3)
            a.set_yscale("symlog", linthresh=3)
        a.set_xlabel(x)
        a.set_ylabel(y)
        a.set_title(title)
        a.grid(True, which="major", color="#e5e5e2", lw=0.6)
        a.set_axisbelow(True)
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=110)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
