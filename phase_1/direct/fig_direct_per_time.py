"""Direct operator: median test error vs time for the stage-2 and stage-4 variants (from eval.json).

(a) relative L2 error of u (grid / ln r measure), (b) relative well-rate error; both one jump from u = 1,
median over the 100 test trajectories, with the success bar (u < 2%, q_D < 5%) dashed.

Run with:  /home/daniel_88/py314/bin/python phase_1/direct/fig_direct_per_time.py [--png preview.png]
Writes:    tex/fig_direct_per_time.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "tex" / "fig_direct_per_time.pdf"
# (run dir, label, color, line style): stage 2 solid, stage 4 dashed; colors in fixed categorical order
RUNS = [("stage2/fno", "fno", "#2a78d6", "-"),
        ("stage2/logsine", "logsine", "#eb6834", "-"),
        ("stage2/dual", "dual", "#1baf7a", "-"),
        ("stage4/fftlog", "fftlog", "#eda100", "--"),
        ("stage4/fftlog_decay", "fftlog_decay", "#e87ba4", "--"),
        ("stage4/hankel", "hankel (dense)", "#008300", "--")]
MUTED = "#52514e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=Path, default=None, help="also write a PNG preview here")
    args = ap.parse_args()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for run, label, color, ls in RUNS:
        res = json.loads((HERE / "runs" / run / "eval.json").read_text())
        t = np.array(res["t"])
        for a, key in zip(ax, ("grid", "q")):
            a.loglog(t, res["per_time"][key]["median"], color=color, ls=ls, lw=1.8, label=label)

    for a, bar, name in zip(ax, (0.02, 0.05), (r"$u$", r"$q_D$")):
        a.axhline(bar, color=MUTED, lw=1, ls=":")
        a.text(1.05e3, bar * 1.08, f"bar {bar:.0%}", color=MUTED, fontsize=8)
        a.set_xlabel(r"$t_D$")
        a.set_ylabel(f"median relative error of {name}")
        a.grid(True, which="major", color="#e5e5e2", lw=0.6)
        a.set_axisbelow(True)
    ax[0].set_title(r"(a) pressure $u$, one jump from $u=1$")
    ax[1].set_title(r"(b) well rate $q_D$")
    ax[1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                 title="solid: stage 2\ndashed: stage 4", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    if args.png:
        fig.savefig(args.png, dpi=110)


if __name__ == "__main__":
    main()
