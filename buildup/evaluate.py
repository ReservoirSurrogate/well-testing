"""Evaluate the inverse model on the test split (build-up project, Milestone 6).

Test examples: DRAWS seeded build-ups per test reservoir with the training realism of examples.Bank.make (t_p ~
logU(wells.T_P_RANGE), a fixed gauge noise (--noise, default 1%), 50% of tests truncated at dt_max ~ logU[DT_MAX_MIN, 9e6], 25% without p_i).
Reported, overall and per regime (p_i known / unknown; full / truncated test; boundary felt within the test,
dt_max > r_eD^2 / (4 k_eff)):
- ln k per ring: mean absolute error (MAE), and the coverage of the +/- 2 sigma interval;
- ln C_D (samples with storage): MAE and coverage; accuracy of the no-storage flag;
- r_eD: median absolute relative error and coverage;
- classical baseline on the same examples: C_D from the early unit slope (dt / dp_bu at the first 3 points),
  r_eD from material balance where p_i is known and the build-up has stabilized (log-derivative < 0.01 at the end).

Usage: python evaluate.py [--run runs/base] [--draws 4] [--tag _x] [--noise 0.01]
Writes: <run>/eval<tag>.json, figs/fig_inverse_eval<tag>.pdf (+ .png, copies in <run>)
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from examples import DT, N_RING, denormalize
from model import InverseFNO
from permeability import RINGS
from train import HERE, fixed_set, load_bank, predict

MUTED = "#52514e"


def load_model(run):
    ckpt = torch.load(Path(run) / "best.pt")
    c = ckpt["config"]
    model = InverseFNO(c["width"], c["modes"], c["n_layers"], c_in=c.get("c_in", 8))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def classical(x, meta):
    """Baseline estimates from the examples: ln C_D (unit slope) and r_eD (material balance, NaN if not usable)."""
    dp = np.exp(5 * x[:, 0].astype(float))
    c_d = np.median(DT[:3] / dp[:, :3], axis=1)
    last = np.array([np.flatnonzero(v > 0)[-1] for v in x[:, 6]])
    rows = np.arange(len(x))
    stabilized = np.abs(x[rows, 1, last]) < 0.01
    p_bar = meta["p_wf"] - dp[rows, last]
    arg = 2 * (meta["t_p"] / np.where(p_bar > 0, p_bar, np.nan) - c_d) + 1
    r_e = np.where(stabilized & meta["pi_known"] & (arg > 1), np.sqrt(np.abs(arg)), np.nan)
    return np.log(c_d), r_e


def summarize(mask, err_ring, z_ring, err_c, z_c, c_mask, re_rel, z_re, acc):
    ring_mae = [float(np.nanmean(np.abs(err_ring[mask, j]))) for j in range(N_RING)]
    zr = z_ring[mask]
    ring_cov = float(np.sum(zr < 2) / np.sum(~np.isnan(zr)))         # over defined rings only (NaN < 2 is False)
    cm = mask & c_mask
    return {"n": int(mask.sum()), "ring_mae": ring_mae, "ring_mae_mean": float(np.nanmean(ring_mae)),
            "ring_cover2": ring_cov, "lnC_mae": float(np.mean(np.abs(err_c[cm]))), "C_cover2": float(np.mean(z_c[cm] < 2)),
            "re_median_rel": float(np.median(np.abs(re_rel[mask]))), "re_p90_rel": float(np.percentile(np.abs(re_rel[mask]), 90)),
            "re_cover2": float(np.mean(z_re[mask] < 2)), "storage_acc": float(np.mean(acc[mask]))}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=HERE / "runs" / "base")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--tag", default="", help="suffix of the output files")
    ap.add_argument("--noise", type=float, default=0.01, help="relative gauge noise of the test build-ups")
    args = ap.parse_args()
    torch.set_num_threads(4)

    bank = load_bank("test")
    with np.load(HERE / "data" / "test.npz") as f:
        k_eff = f["k_eff"]
    model, ckpt = load_model(args.run)
    x, y, ns, meta = fixed_set(bank, args.draws, seed=2024, noise=args.noise)
    idx = np.repeat(np.arange(bank.n), args.draws)
    mu, log_var, logit = (t.numpy().astype(float) for t in predict(model, x))
    y, ns, x = y.numpy().astype(float), ns.numpy(), x.numpy()
    sd = np.exp(0.5 * log_var)

    rings_p, lnc_p, lnre_p = denormalize(mu)
    rings_t, lnc_t, lnre_t = denormalize(np.nan_to_num(y))
    sd_rings, sd_c, sd_re = sd[:, :N_RING], sd[:, N_RING] * 2.7, sd[:, N_RING + 1] * 0.67
    valid_ring = ~np.isnan(y[:, :N_RING])
    err_ring = np.where(valid_ring, rings_p - rings_t, np.nan)
    z_ring = np.where(valid_ring, np.abs(err_ring) / sd_rings, np.nan)
    c_mask = ns == 0
    err_c, z_c = lnc_p - lnc_t, np.abs(lnc_p - lnc_t) / sd_c
    re_rel = np.exp(lnre_p - lnre_t) - 1
    z_re = np.abs(lnre_p - lnre_t) / sd_re
    acc = ((logit > 0) == (ns == 1)).astype(float)

    felt = meta["dt_max"] > bank.r_e[idx] ** 2 / (4 * k_eff[idx])
    truncated = meta["dt_max"] < DT[-1]
    regimes = {"all": np.ones(len(x), bool), "p_i known": meta["pi_known"], "p_i unknown": ~meta["pi_known"],
               "full test": ~truncated, "truncated": truncated, "boundary felt": felt, "boundary not felt": ~felt}
    res = {name: summarize(m, err_ring, z_ring, err_c, z_c, c_mask, re_rel, z_re, acc) for name, m in regimes.items()}

    lnc_b, re_b = classical(x, meta)
    has_b = np.isfinite(re_b)
    res["classical"] = {"lnC_mae": float(np.mean(np.abs(lnc_b[c_mask] - lnc_t[c_mask]))),
                        "re_available": float(has_b.mean()),
                        "re_median_rel": float(np.median(np.abs(re_b[has_b] / bank.r_e[idx][has_b] - 1))),
                        "model_re_median_rel_same": float(np.median(np.abs(re_rel[has_b])))}
    res["best_step"] = ckpt["step"]
    res["noise"] = args.noise
    (Path(args.run) / f"eval{args.tag}.json").write_text(json.dumps(res, indent=2))

    print(f"{'regime':<18} {'n':>5} {'ring MAE':>9} {'ring cov':>9} {'lnC MAE':>8} {'C cov':>6} "
          f"{'r_e med':>8} {'r_e p90':>8} {'r_e cov':>8} {'no-C acc':>8}")
    for name in regimes:
        r = res[name]
        print(f"{name:<18} {r['n']:5d} {r['ring_mae_mean']:9.3f} {r['ring_cover2']:9.3f} {r['lnC_mae']:8.3f} "
              f"{r['C_cover2']:6.3f} {r['re_median_rel']:8.3f} {r['re_p90_rel']:8.3f} {r['re_cover2']:8.3f} "
              f"{r['storage_acc']:8.3f}")
    print("ring MAE by ring (all):", " ".join(f"{v:.3f}" for v in res["all"]["ring_mae"]))
    c = res["classical"]
    print(f"classical: ln C_D MAE {c['lnC_mae']:.3f} (model {res['all']['lnC_mae']:.3f}); r_eD available in "
          f"{c['re_available']:.0%} of examples, median rel err {c['re_median_rel']:.4f} "
          f"(model on the same examples {c['model_re_median_rel_same']:.4f})")

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.3))
    known = meta["pi_known"]
    for j, a in zip((2, 5), ax[:2]):
        m = valid_ring[:, j]
        for kk, color, lab in ((True, "#2a78d6", "p_i known"), (False, "#eb6834", "p_i unknown")):
            mm = m & (known == kk)
            a.scatter(rings_t[mm, j], rings_p[mm, j], s=4, alpha=0.4, color=color, lw=0, label=lab)
        a.set_title(f"(a{j}) ring ln k, $r_D$ in [{RINGS[j][0]}, {RINGS[j][1]}]")
        a.set_xlabel("true")
        a.set_ylabel("predicted")
    for kk, color, lab in ((True, "#2a78d6", "p_i known"), (False, "#eb6834", "p_i unknown")):
        mm = c_mask & (known == kk)
        ax[2].scatter(lnc_t[mm], lnc_p[mm], s=4, alpha=0.4, color=color, lw=0, label=lab)
        mm = known == kk
        ax[3].scatter(np.exp(lnre_t[mm]), np.exp(lnre_p[mm]), s=4, alpha=0.4, color=color, lw=0, label=lab)
    ax[2].set_title(r"(b) $\ln C_D$")
    ax[3].set_title(r"(c) $r_{eD}$")
    ax[3].set_xscale("log")
    ax[3].set_yscale("log")
    for a in ax:
        lo, hi = min(a.get_xlim()[0], a.get_ylim()[0]), max(a.get_xlim()[1], a.get_ylim()[1])
        a.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8)
        a.grid(True, which="major", color="#e5e5e2", lw=0.6)
        a.set_axisbelow(True)
    ax[3].legend(fontsize=8, frameon=False, markerscale=3)
    fig.tight_layout()
    out = HERE / "figs" / f"fig_inverse_eval{args.tag}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=110)
    for f in (out, out.with_suffix(".png")):
        shutil.copy(f, Path(args.run) / f.name)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
