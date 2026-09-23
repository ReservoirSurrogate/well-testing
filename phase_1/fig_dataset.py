"""Milestone 3 figure: overview of the generated dataset (phase_1/data/{train,val}.npz).

(a) log k_D(r) for three validation samples: no skin, damaged well, stimulated well
(b) u(r) of the damaged-well sample at four stored times
(c) well rate q_D(t) of the three samples, 10-90% band over the training set, exact k_D = 1
(d) worst validation state at t_D = 5e4 versus its K = 32 Hankel projection
(e) relative L2 error of the K = 32 projection over the training set at t_D = 5e4
(f) where the projection residual sits in r (mean squared residual, training set)

Run with:  /home/daniel_88/py314/bin/python phase_1/fig_dataset.py
Writes:    tex/fig_dataset.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from basis import HankelBasis, eigenvalues, exact_rate
from dataset import load_split
from solver import transmissibility

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "tex" / "fig_dataset.pdf"
K = 32
N_PROJ = 50                      # stored state index for the projection panels (t_D = 5e4)
PROFILE_STEPS = [1, 10, 100, 500]
COLORS = {"A": "C0", "B": "C1", "C": "C2"}


def well_rate(r, logk, u):
    """q_D = F_{1/2} for states u (..., n_points) with log k (..., n_points)."""
    k = np.exp(logk[..., :2])
    trans = transmissibility(r[:2], np.ones(2))[0] * 2 * k[..., 0] * k[..., 1] / (k[..., 0] + k[..., 1])
    return trans[..., None] * (u[..., 1] - u[..., 0])


def main():
    val = load_split(HERE / "data" / "val.npz")
    train = load_split(HERE / "data" / "train.npz")
    r, t = val["r"], val["t"]
    R_E = float(val["r_e"])

    has, skin = val["has_skin"], val["skin"]
    picks = {
        "A": int(np.nonzero(~has)[0][0]),
        "B": int(np.nonzero(has & (skin < -1.2))[0][0]),
        "C": int(np.nonzero(has & (skin > 0.8))[0][0]),
    }
    labels = {}
    for key, i in picks.items():
        s = f"{key}: no skin" if not has[i] else \
            f"{key}: $k_s/k={np.exp(skin[i]):.2f}$, $r_s={val['r_s'][i]:.1f}$"
        labels[key] = s

    b = HankelBasis.build(K, len(r), R_E)
    P = b.B @ b.T

    fig, ax = plt.subplots(2, 3, figsize=(14, 7.6))
    ax = ax.ravel()

    # (a) permeability
    for key, i in picks.items():
        ax[0].semilogx(r, val["logk"][i], color=COLORS[key], label=labels[key])
    ax[0].set_xlabel(r"$r_D$")
    ax[0].set_ylabel(r"$\ln k_D$")
    ax[0].set_ylim(-3.1, 3.1)
    ax[0].set_title("(a) permeability samples (validation)")
    ax[0].legend(fontsize=8)

    # (b) pressure profiles of the damaged-well sample
    iB = picks["B"]
    shades = plt.cm.Blues(np.linspace(0.4, 1.0, len(PROFILE_STEPS)))
    for n, c in zip(PROFILE_STEPS, shades):
        ax[1].semilogx(r, val["u"][iB, n], color=c, label=rf"$t_D={t[n]:.0f}$")
    ax[1].set_xlabel(r"$r_D$")
    ax[1].set_ylabel(r"$u$")
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_title("(b) pressure profiles, sample B")
    ax[1].legend(fontsize=8)

    # (c) well rate
    tt = t[1:]
    q_train = well_rate(r, train["logk"].astype(float), train["u"][:, 1:, :2].astype(float))
    p10, p50, p90 = np.percentile(q_train, [10, 50, 90], axis=0)
    ax[2].fill_between(tt, p10, p90, color="0.85", label="train 10–90%")
    ax[2].loglog(tt, p50, color="0.55", lw=1, label="train median")
    for key, i in picks.items():
        q = well_rate(r, val["logk"][i].astype(float), val["u"][i, 1:, :2].astype(float))
        ax[2].loglog(tt, q, color=COLORS[key], label=key)
    lam = eigenvalues(4000, R_E)
    ax[2].loglog(tt, exact_rate(tt, lam, R_E), "k--", lw=1, label=r"exact, $k_D\equiv1$")
    ax[2].set_xlabel(r"$t_D$")
    ax[2].set_ylabel(r"$q_D$")
    ax[2].set_title("(c) well rate")
    ax[2].legend(fontsize=8, ncol=2)

    # (d) worst validation state versus its K-mode projection
    u_val = val["u"][:, N_PROJ].astype(float)
    err_val = np.linalg.norm(u_val - u_val @ P.T, axis=1) / np.linalg.norm(u_val, axis=1)
    w = int(np.argmax(err_val))
    ax[3].semilogx(r, u_val[w], "k", label="solver")
    ax[3].semilogx(r, P @ u_val[w], color="C4", label=f"{K}-mode projection")
    ax[3].set_xlabel(r"$r_D$")
    ax[3].set_ylabel(r"$u$")
    skin_txt = rf", $k_s/k={np.exp(skin[w]):.2f}$, $r_s={val['r_s'][w]:.1f}$" if has[w] else ", no skin"
    ax[3].set_title(f"(d) worst val. sample #{w}, $t_D={t[N_PROJ]:.0f}$\nerror {err_val[w]:.1%}{skin_txt}", fontsize=10)
    ax[3].legend(fontsize=8)

    # (e) projection error histogram over the training set
    u_tr = train["u"][:, N_PROJ].astype(float)
    res_tr = u_tr - u_tr @ P.T
    err_tr = np.linalg.norm(res_tr, axis=1) / np.linalg.norm(u_tr, axis=1)
    bins = np.logspace(-4, 0, 25)
    ts = train["has_skin"]
    ax[4].hist([err_tr[~ts], err_tr[ts]], bins=bins, stacked=True, color=["0.55", "C1"],
               label=[f"no skin (median {np.median(err_tr[~ts]):.1%})", f"skin (median {np.median(err_tr[ts]):.1%})"])
    ax[4].axvline(np.median(err_tr), color="k", lw=1)
    ax[4].set_xscale("log")
    ax[4].set_xlabel(f"relative $L_2$ error of {K}-mode projection")
    ax[4].set_ylabel("trajectories")
    ax[4].set_title(rf"(e) projection error, train, $t_D={t[N_PROJ]:.0f}$")
    ax[4].legend(fontsize=8)

    # (f) where the residual sits
    ms = np.mean(res_tr**2, axis=0)
    ax[5].loglog(r, np.sqrt(ms), "k")
    cum = np.cumsum(ms) / ms.sum()
    for rr in (10, 100):
        frac = cum[np.searchsorted(r, rr)]
        ax[5].axvline(rr, color="0.7", lw=1)
        ax[5].text(rr * 1.1, np.sqrt(ms).max() * 0.5, f"{frac:.0%} of\nresidual\nbelow {rr}", fontsize=8)
    ax[5].set_xlabel(r"$r_D$")
    ax[5].set_ylabel("rms residual over train")
    ax[5].set_title(f"(f) where the {K}-mode residual sits")

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    print(f"projection error at t={t[N_PROJ]:.0f}: median {np.median(err_tr):.4f}, p90 {np.percentile(err_tr, 90):.4f}")
    print(f"residual energy below r=10: {cum[np.searchsorted(r, 10)]:.3f}, below r=100: {cum[np.searchsorted(r, 100)]:.3f}")


if __name__ == "__main__":
    main()
