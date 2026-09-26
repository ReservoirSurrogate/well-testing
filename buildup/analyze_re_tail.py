"""What drives the large r_eD errors? (build-up project, Milestone 6)

On the test examples of evaluate.py (4 seeded build-ups per test reservoir), relates the r_eD error of a run to:
- p_i known or not;
- drainage depth t_p k_eff / r_eD^2 (how much of the reservoir the production drained: the material-balance signal
  p_bar = 2 t_p / r_eD^2 against the gauge noise on p_wf);
- boundary exposure dt_max k_eff / r_eD^2 (how far into the boundary regime the shut-in ran);
- the predicted uncertainty (does the model know when it is wrong?).

Usage: python analyze_re_tail.py [--run runs/long] [--noise 0.01] [--tag _x]
Writes: figs/fig_re_tail<tag>.pdf (+ .png, copies in <run>), prints tail statistics
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from evaluate import load_model
from examples import N_RING, denormalize
from train import HERE, fixed_set, load_bank, predict

COLORS = {True: "#2a78d6", False: "#eb6834"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=HERE / "runs" / "long")
    ap.add_argument("--noise", type=float, default=0.01, help="relative gauge noise of the test build-ups")
    ap.add_argument("--tag", default="", help="suffix of the output figure")
    args = ap.parse_args()
    torch.set_num_threads(4)
    bank = load_bank("test")
    with np.load(HERE / "data" / "test.npz") as f:
        k_eff, sigma, has_skin, skin = f["k_eff"], f["sigma"], f["has_skin"], f["skin"]
    model, _ = load_model(args.run)
    x, y, _, meta = fixed_set(bank, 4, seed=2024, noise=args.noise)
    idx = np.repeat(np.arange(bank.n), 4)
    mu, log_var, _ = (t.numpy().astype(float) for t in predict(model, x))
    _, _, lnre_p = denormalize(mu)
    lnre_t = np.log(bank.r_e[idx])
    err = lnre_p - lnre_t                                    # ln-ratio error
    rel = np.abs(np.exp(err) - 1)
    sd = np.exp(0.5 * log_var[:, N_RING + 1]) * 0.67
    r_e, k = bank.r_e[idx], k_eff[idx]
    drain = meta["t_p"] * k / r_e**2
    expose = meta["dt_max"] * k / r_e**2
    known = meta["pi_known"]
    tail = rel > np.percentile(rel, 90)

    def share(mask):
        return f"{mask[tail].mean():6.1%} of the tail vs {mask.mean():6.1%} overall"

    print(f"tail: |rel err| > {np.percentile(rel, 90):.2f} ({tail.sum()} examples)")
    print(f"p_i unknown          : {share(~known)}")
    for lo, hi in ((0, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, np.inf)):
        print(f"drainage in [{lo:g}, {hi:g}): {share((drain >= lo) & (drain < hi))}")
    for lo, hi in ((0, 1), (1, 10), (10, np.inf)):
        print(f"exposure in [{lo:g}, {hi:g}):  {share((expose >= lo) & (expose < hi))}")
    print(f"r_eD > 1500          : {share(r_e > 1500)}")
    print(f"skin zone            : {share(has_skin[idx])}")
    print(f"strong GRF (sigma>.8): {share(sigma[idx] > 0.8)}")
    print(f"tail: over-estimates {np.mean(err[tail] > 0):.0%}, under-estimates {np.mean(err[tail] < 0):.0%}")
    z = np.abs(err) / sd
    print(f"tail |error| / predicted sigma: median {np.median(z[tail]):.2f}, 90th pct {np.percentile(z[tail], 90):.2f} "
          f"(inside 2 sigma: {np.mean(z[tail] < 2):.0%})")
    print(f"median predicted sigma of ln r_eD: tail {np.median(sd[tail]):.3f}, rest {np.median(sd[~tail]):.3f}")
    for kn in (True, False):
        m = known == kn
        for lo, hi in ((0, 1e-2), (1e-2, np.inf)):
            mm = m & (drain >= lo) & (drain < hi)
            print(f"p_i {'known  ' if kn else 'unknown'} drainage [{lo:g}, {hi:g}): median rel err "
                  f"{np.median(rel[mm]):.3f}  (n = {mm.sum()})")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for kn in (True, False):
        m = known == kn
        lab = "p_i known" if kn else "p_i unknown"
        ax[0].scatter(drain[m], rel[m], s=4, alpha=0.4, color=COLORS[kn], lw=0, label=lab)
        ax[1].scatter(expose[m], rel[m], s=4, alpha=0.4, color=COLORS[kn], lw=0, label=lab)
        ax[2].scatter(sd[m], np.abs(err[m]), s=4, alpha=0.4, color=COLORS[kn], lw=0, label=lab)
    for a, xl, title in ((ax[0], r"drainage $t_p k_{eff} / r_{eD}^2$", "(a) error vs drainage before shut-in"),
                         (ax[1], r"boundary exposure $\Delta t_{max} k_{eff} / r_{eD}^2$", "(b) error vs test length"),
                         (ax[2], r"predicted $\sigma$ of $\ln r_{eD}$", "(c) actual vs predicted error")):
        a.set_xscale("log")
        a.set_xlabel(xl)
        a.set_title(title)
        a.grid(True, which="major", color="#e5e5e2", lw=0.6)
        a.set_axisbelow(True)
    for a in ax[:2]:
        a.set_yscale("log")
        a.set_ylabel(r"$|r_{eD,pred} / r_{eD} - 1|$")
        a.axhline(np.percentile(rel, 90), color="#52514e", lw=0.8, ls=":")
    s = np.geomspace(sd.min(), sd.max(), 10)
    ax[2].plot(s, 2 * s, color="#52514e", lw=0.8, ls="--", label=r"$2\sigma$")
    ax[2].set_yscale("log")
    ax[2].set_ylabel(r"$|\ln r_{eD,pred} - \ln r_{eD}|$")
    ax[2].legend(fontsize=8, frameon=False, markerscale=3)
    fig.tight_layout()
    out = HERE / "figs" / f"fig_re_tail{args.tag}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=110)
    for f in (out, out.with_suffix(".png")):
        shutil.copy(f, Path(args.run) / f.name)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
