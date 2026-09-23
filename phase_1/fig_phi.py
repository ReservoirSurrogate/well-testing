"""Figure 1 of tex/phase1.tex: first four annular eigenfunctions for r_eD = 1000.

Run with:  /home/daniel_88/py314/bin/python phase_1/fig_phi.py
Writes:    tex/fig_phi.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

R_E = 1000.0
N_MODES = 4
OUT = Path(__file__).resolve().parent.parent / "tex" / "fig_phi.pdf"


def eig_residual(lam):
    return j1(lam * R_E) * y0(lam) - y1(lam * R_E) * j0(lam)


def eigenvalues(n):
    grid = np.linspace(1e-5, 0.05, 50001)
    vals = eig_residual(grid)
    idx = np.nonzero(vals[:-1] * vals[1:] < 0)[0][:n]
    return [brentq(eig_residual, grid[i], grid[i + 1]) for i in idx]


def phi(r, lam):
    return j0(lam * r) * y0(lam) - y0(lam * r) * j0(lam)


def norm(lam):
    return R_E**2 / 2 * phi(R_E, lam) ** 2 - 2 / (np.pi**2 * lam**2)


def main():
    lams = eigenvalues(N_MODES)
    r = np.logspace(0, np.log10(R_E), 4000)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for n, lam in enumerate(lams, start=1):
        p = phi(r, lam)
        ax[0].semilogx(r, p, label=f"$n={n}$")
        ax[1].plot(r, np.sqrt(r) * p / np.sqrt(norm(lam)), label=f"$n={n}$")
    ax[0].semilogx(r, -2 / np.pi * np.log(r), "k--", lw=1, label=r"$-\frac{2}{\pi}\ln r$")

    ax[0].set_ylim(-5.5, 3)
    ax[0].set_xlabel(r"$r_D$")
    ax[0].set_ylabel(r"$\varphi_n(r)$")
    ax[0].set_title("(a) unnormalized, log axis")
    ax[1].set_xlabel(r"$r_D$")
    ax[1].set_ylabel(r"$\sqrt{r}\,\varphi_n/\sqrt{N_n}$")
    ax[1].set_title("(b) weighted & normalized, linear axis")
    for a in ax:
        a.axhline(0, color="gray", lw=0.5)
        a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
