"""Example build-ups (build-up project, Milestone 5): four test samples after t_pD = 1e6.

(a) ln k_D(r) with the label k_eff (dotted) and the skin-zone radius
(b) log-log diagnostic plot: build-up pressure change (solid) and Bourdet derivative (dashed) vs Agarwal time;
    the dotted level 1 / (2 k_eff) is where the derivative should sit if k_eff were the radial-flow permeability
(c) Horner plot: well pressure vs Horner time (t_p + dt) / dt

Run with:  /home/daniel_88/py314/bin/python buildup/fig_examples.py
Writes:    buildup/figs/fig_examples.pdf (+ .png)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import grid
from wells import Drawdown, agarwal_time, bourdet_derivative, buildup, buildup_times, horner_time

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs" / "fig_examples.pdf"
T_P = 1e6
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MUTED = "#52514e"


def pick(data):
    """No skin zone and no storage / damaged zone / stimulated zone / large storage."""
    hs, s, c = data["has_skin"], data["skin"], data["c_d"]
    choices = {"no skin zone, $C_D=0$": np.flatnonzero(~hs & (c == 0)),
               "damaged zone": np.flatnonzero(hs & (s > 5) & (s < 20)),
               "stimulated zone": np.flatnonzero(hs & (s < -1)),
               "large storage": np.flatnonzero(c > 300)}
    return {name: int(idx[0]) for name, idx in choices.items()}


def main():
    with np.load(HERE / "data" / "test.npz") as f:
        data = {key: f[key] for key in f.files}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for color, (name, i) in zip(COLORS, pick(data).items()):
        r = grid(data, i)
        logk = data["logk"][i, :data["n_r"][i]]
        k_eff, s, c_d, r_e = data["k_eff"][i], data["skin"][i], data["c_d"][i], data["r_e"][i]
        label = f"{name}: $k_{{eff}}$={k_eff:.2f}, S={s:.1f}, $C_D$={c_d:.3g}, $r_{{eD}}$={r_e:.0f}"
        ax[0].semilogx(r, logk, color=color, lw=1.4, label=label)
        ax[0].axhline(np.log(k_eff), color=color, lw=1, ls=":")
        if data["has_skin"][i]:
            ax[0].axvline(data["r_s"][i], color=color, lw=0.8, ls="--")

        dd = Drawdown(data["t"], data["p_w"][i])
        dt = buildup_times(T_P, data["t"][0], data["t"][-1], n=300)
        p_w, dp = buildup(dd, T_P, dt)
        te = agarwal_time(T_P, dt)
        der = bourdet_derivative(te, dp)
        ok = der > 0
        ax[1].loglog(te, dp, color=color, lw=1.6)
        ax[1].loglog(te[ok], der[ok], color=color, lw=1.2, ls="--")
        ax[1].axhline(1 / (2 * k_eff), color=color, lw=1, ls=":")
        ax[2].semilogx(horner_time(T_P, dt), p_w, color=color, lw=1.6)

    ax[0].set_xlabel(r"$r_D$")
    ax[0].set_ylabel(r"$\ln k_D$")
    ax[0].set_title(r"(a) permeability (dotted: $\ln k_{eff}$; dashed: $r_s$)")
    ax[0].legend(fontsize=7, frameon=False, loc="lower left")
    ax[1].set_xlabel(r"Agarwal time $\Delta t_e$")
    ax[1].set_ylabel(r"$\Delta p_{bu}$ (solid), derivative (dashed)")
    ax[1].set_title(r"(b) log-log plot (dotted: $1/(2k_{eff})$)")
    ax[1].set_ylim(1e-3, None)
    ax[2].set_xlabel(r"Horner time $(t_p+\Delta t)/\Delta t$")
    ax[2].set_ylabel(r"$p_{wD}$ (pressure drop below $p_i$)")
    ax[2].set_title("(c) Horner plot")
    ax[2].invert_xaxis()
    for a in ax:
        a.grid(True, which="major", color="#e5e5e2", lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=110)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
