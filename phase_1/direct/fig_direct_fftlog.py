"""FFTLog-HNO (stage 4) vs truth and vs fno (stage 2) on the test split, one jump from u = 1 per time.

(a) u(r) of a representative trajectory (median time-averaged fftlog error) at t_D = 1e3, 1e4, 1e5, 5e5:
    truth solid, fftlog dashed
(b) pointwise error of the same trajectory: fftlog solid, fno dotted
(c) well rate q_D(t) of the median, 90th-percentile and worst trajectories: truth, fftlog, fno
(d) u error vs time over all test trajectories: median and 10-90% band, fftlog and fno

Run with:  OMP_NUM_THREADS=4 /home/daniel_88/py314/bin/python phase_1/direct/fig_direct_fftlog.py [--png preview.png]
Writes:    tex/fig_direct_fftlog.pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from direct_evaluate import load_model, predict  # noqa: E402
from direct_train import DATA  # noqa: E402
from evaluate import well_rate  # noqa: E402
from train import open_split  # noqa: E402

RUNS = {"fftlog": HERE / "runs" / "stage4" / "fftlog", "fno": HERE / "runs" / "stage2" / "fno"}
COLORS = {"fftlog": "#eda100", "fno": "#2a78d6"}        # same slots as fig_direct_per_time
SHOW_T = (1e3, 1e4, 1e5, 5e5)
TIME_COLORS = ("#9ec5f4", "#5a9be6", "#2a78d6", "#0f3f80")  # one hue, light -> dark with time
MUTED = "#52514e"
OUT = HERE.parent.parent / "tex" / "fig_direct_fftlog.pdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=Path, default=None, help="also write a PNG preview here")
    args = ap.parse_args()
    torch.set_num_threads(4)

    u, logk, r, r_e = open_split(DATA, "test")
    with np.load(DATA / "test.npz") as f:
        t = f["t"]
    logk = logk.astype(float)
    u0 = np.asarray(u[:, 0], dtype=float)
    # ~40 log-spaced stored times, always including the shown ones
    idx = np.unique(np.r_[np.round(np.geomspace(1, len(t) - 1, 40)).astype(int),
                          [int(np.argmin(np.abs(t - s))) for s in SHOW_T]])
    true = np.stack([np.asarray(u[:, j], dtype=float) for j in idx])            # (n_t, n_traj, N)

    pred, err = {}, {}
    for name, run in RUNS.items():
        model, _ = load_model(run, len(r), r_e)
        pred[name] = np.stack([predict(model, u0, logk, t[j]) for j in idx])
        err[name] = np.linalg.norm(pred[name] - true, axis=-1) / np.linalg.norm(true, axis=-1)
        print(f"{name}: median u error {np.median(err[name]):.2%}")
    q_true = well_rate(r, logk, true)
    q_pred = {n: well_rate(r, logk, p) for n, p in pred.items()}

    mean_err = err["fftlog"].mean(axis=0)
    order = np.argsort(mean_err)
    rep = order[len(order) // 2]
    picks = {"median": rep, "p90": order[int(0.9 * (len(order) - 1))], "worst": order[-1]}
    tt = t[idx]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    (a, b), (c, d) = ax
    for color, ts in zip(TIME_COLORS, SHOW_T):
        m = int(np.argmin(np.abs(tt - ts)))
        lab = rf"$t_D=10^{{{np.log10(ts):.0f}}}$" if ts != 5e5 else r"$t_D=5\cdot10^5$"
        a.semilogx(r, true[m, rep], color=color, lw=2, label=lab)
        a.semilogx(r, pred["fftlog"][m, rep], color="black", lw=1, ls="--")
        b.semilogx(r, pred["fftlog"][m, rep] - true[m, rep], color=color, lw=1.6, label=lab)
        b.semilogx(r, pred["fno"][m, rep] - true[m, rep], color=color, lw=1.2, ls=":")
    a.plot([], [], color="black", lw=1, ls="--", label="fftlog")
    a.set_title("(a) representative trajectory: truth (solid) vs fftlog (dashed)")
    a.set_xlabel(r"$r_D$")
    a.set_ylabel(r"$u$")
    a.legend(fontsize=8, frameon=False, loc="lower right")
    b.axhline(0, color=MUTED, lw=0.8)
    b.plot([], [], color=MUTED, lw=1.6, label="fftlog")
    b.plot([], [], color=MUTED, lw=1.2, ls=":", label="fno")
    b.set_title("(b) pointwise error, same trajectory")
    b.set_xlabel(r"$r_D$")
    b.set_ylabel(r"$u_{\mathrm{pred}}-u_{\mathrm{true}}$")
    b.legend(fontsize=8, frameon=False, ncol=2)

    for (label, i), lw in zip(picks.items(), (1.2, 1.2, 1.2)):
        c.loglog(tt, q_true[:, i], color="black", lw=2)
        c.loglog(tt, q_pred["fftlog"][:, i], color=COLORS["fftlog"], lw=lw, ls="--")
        c.loglog(tt, q_pred["fno"][:, i], color=COLORS["fno"], lw=lw, ls=":")
        c.annotate(f"{label} ({mean_err[i]:.1%})", (tt[0], q_true[0, i]), xytext=(4, 4),
                   textcoords="offset points", fontsize=8, color=MUTED)
    c.plot([], [], color="black", lw=2, label="truth")
    c.plot([], [], color=COLORS["fftlog"], ls="--", label="fftlog")
    c.plot([], [], color=COLORS["fno"], ls=":", label="fno")
    c.set_title(r"(c) well rate $q_D(t)$: median, p90, worst fftlog trajectories")
    c.set_xlabel(r"$t_D$")
    c.set_ylabel(r"$q_D$")
    c.legend(fontsize=8, frameon=False, loc="lower left")

    for name in ("fno", "fftlog"):
        lo, med, hi = np.percentile(err[name], [10, 50, 90], axis=1)
        d.fill_between(tt, lo, hi, color=COLORS[name], alpha=0.2, lw=0)
        d.loglog(tt, med, color=COLORS[name], lw=2, label=f"{name}: median, 10-90% band")
    d.axhline(0.02, color=MUTED, lw=1, ls=":")
    d.text(tt[0], 0.021, "bar 2%", color=MUTED, fontsize=8)
    d.set_title("(d) u error over all 100 test trajectories")
    d.set_xlabel(r"$t_D$")
    d.set_ylabel("relative L2 error of u")
    d.legend(fontsize=8, frameon=False, loc="upper left")

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
