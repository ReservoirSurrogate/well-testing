"""Theory figure (build-up report): one homogeneous closed reservoir with storage and skin.

k_D = 1, C_D = 10, S = 5, r_eD = 2000, production t_pD = 5e5 at unit rate (infinite-acting: the boundary is felt
from ~r_eD^2 / 4 = 1e6), then shut-in.
(a) log-log diagnostic plot of the build-up: pressure change and Bourdet derivative vs Agarwal time, with the flow
    regimes (storage unit slope, skin hump, radial-flow plateau 1/2, closed-boundary drop)
(b) Horner plot: well pressure vs Horner time (t_p + dt) / dt, semilog slope 1/2
(c) pressure history of a longer-producing case (r_eD = 1000, t_pD = 1e6, the boundary is reached during production):
    drawdown then build-up, in p_D = drop below p_i; the build-up stabilizes at the average drop
    p_bar = t_p / ((r_eD^2 - 1)/2 + C_D) (material balance); the gauge measures only p_w - p_wf, not p_i - p_bar

Run with:  /home/daniel_88/py314/bin/python buildup/fig_theory.py
Writes:    buildup/figs/fig_theory.pdf (+ .png)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solver import WellSolver, grid_for
from wells import Drawdown, agarwal_time, bourdet_derivative, buildup, horner_time

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs" / "fig_theory.pdf"
C_D, SKIN, R_E, T_P = 10.0, 5.0, 2000.0, 5e5
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#52514e"


def main():
    r, _ = grid_for(R_E, np.log(1000.0) / 1023)
    t = np.logspace(-2, 7.3, 260)
    sol = WellSolver(r, np.ones_like(r), c_d=C_D, skin=SKIN)
    dd = Drawdown(t, sol.well_pressure(t))
    dt = np.geomspace(1e-2, t[-1] - T_P, 300)
    p_w, dp = buildup(dd, T_P, dt)
    te = agarwal_time(T_P, dt)
    der = bourdet_derivative(te, dp)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    a = ax[0]
    a.loglog(te, dp, color=BLUE, lw=2, label=r"$\Delta p_{bu}=p_{ws}-p_{wf}$")
    ok = der > 1e-3
    a.loglog(te[ok], der[ok], color=ORANGE, lw=1.8, label="Bourdet derivative")
    a.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    x = np.array([1e-2, 3e0])
    a.loglog(x, x / C_D, color=MUTED, lw=0.8, ls="--")
    a.text(2e-2, 1.2e-2, r"unit slope: $\Delta p = \Delta t / C_D$", color=MUTED, fontsize=8)
    a.text(1.5e4, 0.56, r"radial flow: derivative $= 1/(2k)$", color=MUTED, fontsize=8)
    a.text(8e1, 2.6, "skin hump", color=MUTED, fontsize=8)
    a.text(1.2e5, 0.08, "closed\nboundary", color=MUTED, fontsize=8)
    a.set_ylim(1e-3, 30)
    a.set_xlabel(r"Agarwal time $\Delta t_e = t_p\Delta t/(t_p+\Delta t)$")
    a.set_ylabel("dimensionless pressure")
    a.set_title("(a) log-log diagnostic plot")
    a.legend(fontsize=8, frameon=False, loc="upper left")

    a = ax[1]
    ht = horner_time(T_P, dt)
    a.semilogx(ht, p_w, color=BLUE, lw=2)
    m = (te > 1e4) & (te < 1e5)                               # after storage (~(60 + 3.5 S) C_D), before the boundary
    slope, icpt = np.polyfit(np.log(ht[m]), p_w[m], 1)
    xl = np.geomspace(1, ht.max(), 50)
    a.semilogx(xl, slope * np.log(xl) + icpt, color=MUTED, lw=0.8, ls="--",
               label=f"semilog line, slope {slope:.3f} per ln unit")
    a.axhline(icpt, color=ORANGE, lw=0.8, ls=":")
    a.text(1.3, icpt + 0.3, r"$p^*$ (extrapolated)", color=ORANGE, fontsize=8)
    a.invert_xaxis()
    a.set_xlabel(r"Horner time $(t_p+\Delta t)/\Delta t$")
    a.set_ylabel(r"$p_{wD}$ (drop below $p_i$)")
    a.set_title("(b) Horner plot")
    a.legend(fontsize=8, frameon=False, loc="upper right")

    a = ax[2]
    r2, _ = grid_for(1000.0, np.log(1000.0) / 1023)
    dd2 = Drawdown(t, WellSolver(r2, np.ones_like(r2), c_d=C_D, skin=SKIN).well_pressure(t))
    t_p2 = 1e6
    dt2 = np.geomspace(1e-2, t[-1] - t_p2, 300)
    p_w2, _ = buildup(dd2, t_p2, dt2)
    p_wf2 = float(dd2(t_p2))
    p_bar2 = t_p2 / ((1000.0**2 - 1) / 2 + C_D)
    t_draw = np.geomspace(1e-2, t_p2, 200)
    a.semilogx(np.concatenate([t_draw, t_p2 + dt2]), np.concatenate([dd2(t_draw), p_w2]), color=BLUE, lw=2)
    a.axhline(0, color="black", lw=1)
    a.text(2e-2, -0.5, r"$p_i$ (initial pressure, $p_D = 0$)", fontsize=8)
    a.axhline(p_bar2, color=ORANGE, lw=1, ls="--")
    a.text(2e-2, p_bar2 + 0.8, r"$\bar p$: $p_i-\bar p = q t_p / (c_t V_p)$ (material balance)", color=ORANGE,
           fontsize=8)
    a.axvline(t_p2, color=MUTED, lw=0.8, ls=":")
    a.text(t_p2 * 0.7, p_wf2 + 1.0, "shut-in", color=MUTED, fontsize=8, ha="right")
    a.annotate("", xy=(4e6, p_bar2), xytext=(4e6, p_wf2), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
    a.text(4.6e6, 0.5 * (p_bar2 + p_wf2), "measured by\nthe build-up\n($p_{wf}-\\bar p$)", color=MUTED, fontsize=7,
           va="center")
    a.annotate("", xy=(2e5, 0), xytext=(2e5, p_wf2), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=0.8))
    a.text(1.6e5, 0.55 * p_wf2, "$p_i-p_{wf}$\n(needs $p_i$)", color=ORANGE, fontsize=7, va="center", ha="right")
    a.set_ylim(p_wf2 + 2, -1.5)
    a.set_xlabel(r"time $t_D$")
    a.set_ylabel(r"$p_{wD}$ (drop below $p_i$)")
    a.set_title(r"(c) history and material balance ($r_{eD}$=1000, $t_p$=$10^6$)")
    for a in ax:
        a.grid(True, which="major", color="#e5e5e2", lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=110)
    print(f"wrote {OUT}; semilog slope {slope:.4f}, p* = {icpt:.3f}; case (c): p_wf = {p_wf2:.3f}, "
          f"p_bar = {p_bar2:.4f}, stabilized build-up {p_w2[-1]:.4f}")


if __name__ == "__main__":
    main()
