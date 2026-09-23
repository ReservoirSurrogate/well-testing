"""Figure 2 of tex/phase1.tex: analytic pressure profiles and well rate for k_D = 1, r_eD = 1000.

u(r,t) = sum_n c_n phi_n(r) exp(-lambda_n^2 t),   c_n = -2 / (pi lambda_n^2 N_n)
q_D(t) = sum_n 4 / (pi^2 lambda_n^2 N_n) exp(-lambda_n^2 t)

Also prints the mode-convergence table (modes needed for 1e-3 accuracy at r = 2).

Run with:  /home/daniel_88/py314/bin/python phase_1/fig_pressure.py
Writes:    tex/fig_pressure.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

R_E = 1000.0
LAM_MAX = 6.0  # ~1900 modes; enough for t_D >= 10
TIMES = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
OUT = Path(__file__).resolve().parent.parent / "tex" / "fig_pressure.pdf"


def eig_residual(lam):
    return j1(lam * R_E) * y0(lam) - y1(lam * R_E) * j0(lam)


def eigenvalues(lam_max):
    # Scan step well below the root spacing ~ pi / (R_E - 1) ~ 3e-3.
    grid = np.linspace(1e-5, lam_max, int(lam_max / 2e-6))
    vals = eig_residual(grid)
    idx = np.nonzero(vals[:-1] * vals[1:] < 0)[0]
    return np.array([brentq(eig_residual, grid[i], grid[i + 1]) for i in idx])


def phi(r, lam):
    return j0(lam * r) * y0(lam) - y0(lam * r) * j0(lam)


def norm(lam):
    return R_E**2 / 2 * phi(R_E, lam) ** 2 - 2 / (np.pi**2 * lam**2)


def main():
    lam = eigenvalues(LAM_MAX)
    N = norm(lam)
    c = -2 / (np.pi * lam**2 * N)
    print(f"{len(lam)} modes, lambda_1 = {lam[0]:.4e}")

    r = np.logspace(0, np.log10(R_E), 1000)
    phi_rn = phi(r[:, None], lam[None, :])  # (n_r, n_modes)

    def u(t, modes=None):
        m = slice(None) if modes is None else slice(0, modes)
        return phi_rn[:, m] @ (c[m] * np.exp(-lam[m] ** 2 * t))

    t_q = np.logspace(1, 7, 400)
    q = (4 / (np.pi**2 * lam**2 * N)) @ np.exp(-np.outer(lam**2, t_q))
    q_late = 4 / (np.pi**2 * lam[0] ** 2 * N[0]) * np.exp(-lam[0] ** 2 * t_q)

    # Mode-convergence table at r = 2.
    phi_2 = phi(2.0, lam)
    print("t_D      u(r=2)   modes for 1e-3")
    for t in [1e1, 1e2, 1e4, 1e6]:
        terms = c * phi_2 * np.exp(-lam**2 * t)
        ref = terms.sum()
        err = np.abs(np.cumsum(terms) - ref)
        need = int(np.argmax(err < 1e-3)) + 1
        print(f"{t:<8.0e} {ref:.4f}   {need}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for t in TIMES:
        ax[0].semilogx(r, u(t), label=rf"$t_D=10^{{{int(np.log10(t))}}}$")
    ax[0].set_xlabel(r"$r_D$")
    ax[0].set_ylabel(r"$u=(p-p_{wf})/(p_i-p_{wf})$")
    ax[0].set_title(r"(a) pressure profiles, $k_D\equiv1$")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].legend(fontsize=8)

    ax[1].loglog(t_q, q, label="full series")
    ax[1].loglog(t_q, q_late, "k--", lw=1, label=r"first mode only")
    ax[1].set_ylim(q.min() * 0.5, q.max() * 2)
    ax[1].set_xlabel(r"$t_D$")
    ax[1].set_ylabel(r"$q_D$")
    ax[1].set_title("(b) well rate")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
