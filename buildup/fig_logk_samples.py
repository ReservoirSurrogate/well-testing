"""Random ln k_D(r) samples of the test split on a log and a linear radius axis (build-up project, Milestone 5).

(a) 12 samples without a skin zone vs r_D on a log axis, with the physical radius (r_w = 0.1 m) on top
(b) the same samples vs r_D on a linear axis, first 50 r_D
The field is stationary in ln r: equal statistics per unit of ln r everywhere, starting at the well (r_D = 1).

Run with:  /home/daniel_88/py314/bin/python buildup/fig_logk_samples.py
Writes:    buildup/figs/fig_logk_samples.pdf (+ .png)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import grid

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs" / "fig_logk_samples.pdf"
R_W = 0.1                                            # m, for the physical axis only


def main():
    with np.load(HERE / "data" / "test.npz") as f:
        data = {key: f[key] for key in f.files}
    idx = np.flatnonzero(~data["has_skin"])[:12]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for i in idx:
        r = grid(data, i)
        logk = data["logk"][i, :data["n_r"][i]]
        ax[0].semilogx(r, logk, lw=1)
        m = r <= 50
        ax[1].plot(r[m], logk[m], lw=1)
    ax[0].set_xlabel(r"$r_D = r / r_w$ (log axis; $r_D = 1$ is the wellbore face)")
    ax[0].set_title(r"(a) 12 samples without a skin zone")
    top = ax[0].secondary_xaxis("top", functions=(lambda x: x * R_W, lambda x: x / R_W))
    top.set_xlabel(r"physical radius $r$ [m] for $r_w = 0.1$ m")
    ax[1].set_xlabel(r"$r_D$ (linear axis, first 50)")
    ax[1].set_title("(b) same samples near the well")
    for a in ax:
        a.set_ylabel(r"$\ln k_D$")
        a.grid(True, which="major", color="#e5e5e2", lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=110)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
