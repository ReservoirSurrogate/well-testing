"""Direct operator (stage 2, fno): predictions vs truth at t_D = 1000 on the test split.

(a) u(r) for the test trajectories at the median, 90th-percentile and worst u error: truth solid, prediction dashed
(b) ln k_D(r) of the same trajectories
(c) pointwise error u_pred - u_true of the same trajectories
(d) predicted vs true well rate q_D for all test trajectories

Run with:  OMP_NUM_THREADS=1 /home/daniel_88/py314/bin/python phase_1/direct/fig_direct_t1000.py [--png preview.png]
Writes:    tex/fig_direct_t1000.pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from direct_evaluate import load_model, predict  # noqa: E402
from direct_train import DATA  # noqa: E402
from evaluate import well_rate  # noqa: E402
from train import open_split  # noqa: E402

RUN = HERE / "runs" / "stage2" / "fno"
J = 1                                                   # stored state index: t_D = 1000
OUT = HERE.parent.parent / "tex" / "fig_direct_t1000.pdf"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")             # categorical slots 1-3
MUTED = "#52514e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=Path, default=None, help="also write a PNG preview here")
    args = ap.parse_args()
    torch.set_num_threads(1)

    u, logk, r, r_e = open_split(DATA, "test")
    with np.load(DATA / "test.npz") as f:
        t = f["t"]
    logk = logk.astype(float)
    model, _ = load_model(RUN, len(r), r_e)
    true = np.asarray(u[:, J], dtype=float)
    pred = predict(model, np.asarray(u[:, 0], dtype=float), logk, t[J])

    err = np.linalg.norm(pred - true, axis=1) / np.linalg.norm(true, axis=1)
    q_true, q_pred = well_rate(r, logk, true), well_rate(r, logk, pred)
    q_err = np.abs(q_pred - q_true) / q_true
    order = np.argsort(err)
    picks = {"median": order[len(order) // 2], "p90": order[int(0.9 * (len(order) - 1))], "worst": order[-1]}
    print(f"t_D = {t[J]:.0f}: u error median {np.median(err):.2%}, p90 {np.percentile(err, 90):.2%}, "
          f"max {err.max():.2%}; q_D error median {np.median(q_err):.2%}, p90 {np.percentile(q_err, 90):.2%}")

    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
    (a, b), (c, d) = ax
    for color, (name, i) in zip(COLORS, picks.items()):
        label = f"{name} (u err {err[i]:.2%}, q err {q_err[i]:.1%})"
        a.semilogx(r, true[i], color=color, lw=2, label=label)
        a.semilogx(r, pred[i], color="black", lw=1, ls="--")
        b.semilogx(r, logk[i], color=color, lw=1.5, label=name)
        c.semilogx(r, pred[i] - true[i], color=color, lw=1.5, label=name)

    a.set_xlabel(r"$r_D$")
    a.set_ylabel(r"$u$")
    a.set_title(r"(a) pressure at $t_D=1000$: truth (solid) vs fno (dashed)")
    handles = a.get_legend_handles_labels()[0] + [Line2D([], [], color="black", lw=1, ls="--", label="fno")]
    a.legend(handles=handles, fontsize=8, loc="lower right", frameon=False)

    b.set_xlabel(r"$r_D$")
    b.set_ylabel(r"$\ln k_D$")
    b.set_title("(b) permeability of the same trajectories")
    b.legend(fontsize=8, frameon=False)

    c.axhline(0, color=MUTED, lw=0.8)
    c.set_xlabel(r"$r_D$")
    c.set_ylabel(r"$u_{\mathrm{pred}}-u_{\mathrm{true}}$")
    c.set_title("(c) pointwise error")
    c.legend(fontsize=8, frameon=False)

    lim = [min(q_true.min(), q_pred.min()) * 0.9, max(q_true.max(), q_pred.max()) * 1.1]
    d.loglog(lim, lim, color=MUTED, lw=0.8)
    d.loglog(q_true, q_pred, "o", ms=4, color=COLORS[0], alpha=0.7)
    d.set_xlim(lim)
    d.set_ylim(lim)
    d.set_xlabel(r"true $q_D$")
    d.set_ylabel(r"predicted $q_D$")
    d.set_title(f"(d) well rate, {len(q_true)} test trajectories")
    d.text(0.04, 0.95, f"median error {np.median(q_err):.1%}\n90th pct. {np.percentile(q_err, 90):.1%}",
           transform=d.transAxes, va="top", fontsize=8, color=MUTED)

    for x in ax.flat:
        x.grid(True, which="major", color="#e5e5e2", lw=0.6)
        x.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    if args.png:
        fig.savefig(args.png, dpi=110)


if __name__ == "__main__":
    main()
