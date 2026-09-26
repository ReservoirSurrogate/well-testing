"""Gauge-noise summary (build-up report): test errors of the inverse model vs the relative gauge noise.

Two models: runs/long (trained at 1% noise) and runs/noise (noise level randomized in [0.01%, 1%] and given as an
input), evaluated on the same seeded test build-ups at noise 1e-4, 1e-3, 1e-2 (runs/<run>/eval_noise<level>.json).
(a) r_eD median relative error, p_i known / unknown, and classical material balance where it applies
(b) ln C_D MAE and the mean ring ln k MAE

Run with:  /home/daniel_88/py314/bin/python buildup/fig_noise_summary.py
Writes:    buildup/figs/fig_noise_summary.pdf (+ .png)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs" / "fig_noise_summary.pdf"
LEVELS = ("1e-4", "1e-3", "1e-2")
RUNS = {"noise-aware (runs/noise)": ("noise", "-"), "trained at 1% (runs/long)": ("long", "--")}
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def load(run):
    return [json.loads((HERE / "runs" / run / f"eval_noise{lv}.json").read_text()) for lv in LEVELS]


def main():
    noise = np.array([float(lv) for lv in LEVELS]) * 100            # percent
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for name, (run, ls) in RUNS.items():
        res = load(run)
        ax[0].loglog(noise, [100 * r["p_i known"]["re_median_rel"] for r in res], color=BLUE, ls=ls, marker="o",
                     lw=1.8, label=f"p_i known, {name}")
        ax[0].loglog(noise, [100 * r["p_i unknown"]["re_median_rel"] for r in res], color=ORANGE, ls=ls, marker="o",
                     lw=1.8, label=f"p_i unknown, {name}")
        ax[1].semilogx(noise, [r["all"]["lnC_mae"] for r in res], color=AQUA, ls=ls, marker="o", lw=1.8,
                       label=f"ln C_D, {name}")
        ax[1].semilogx(noise, [r["all"]["ring_mae_mean"] for r in res], color=BLUE, ls=ls, marker="o", lw=1.8,
                       label=f"ring ln k (mean), {name}")
    res = load("noise")
    ax[0].loglog(noise, [100 * r["classical"]["re_median_rel"] for r in res], color=MUTED, ls="none", marker="s",
                 label="classical material balance (where applicable)")
    for x, r in zip(noise, res):
        ax[0].annotate(f"{r['classical']['re_available']:.0%}", (x, 100 * r["classical"]["re_median_rel"]),
                       xytext=(5, -10), textcoords="offset points", fontsize=7, color=MUTED)
    ax[0].set_ylabel(r"median $|r_{eD}$ error$|$ [%]")
    ax[0].set_title(r"(a) boundary distance $r_{eD}$")
    ax[1].set_ylabel("mean absolute error (ln units)")
    ax[1].set_title("(b) storage and permeability rings")
    for a in ax:
        a.set_xlabel("relative gauge noise [%]")
        a.grid(True, which="major", color="#e5e5e2", lw=0.6)
        a.set_axisbelow(True)
        a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=110)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
