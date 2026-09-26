"""Post-hoc calibration of the predicted uncertainties (build-up project, Milestone 6).

An apparent 85% coverage of the ring uncertainties turned out to be a bug in evaluate.py (masked rings counted as
misses); per target the raw coverage is already ~94-95%. This script checks the calibration and fits, for each target
j, a scale s_j on the validation set (independent of the test set):

    z = |mu - y| / sigma  over the validation examples where target j is defined
    s_j = quantile(z, 0.9545) / 2       -> exactly 95.45% of validation errors inside +/- 2 s_j sigma

(0.9545 is the Gaussian coverage of +/- 2 sigma). The calibrated uncertainty is s_j sigma_j; the means are unchanged.

Usage: python calibrate.py [--run runs/long]
Writes: <run>/calibration.json, figs/fig_calibration.pdf (+ .png, copies in <run>): reliability plot (empirical vs
nominal coverage of central Gaussian intervals) on the test set before and after calibration.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

from examples import N_RING, N_TARGET
from permeability import RINGS
from train import HERE, VAL_DRAWS, fixed_set, load_bank, predict

COVER = 0.9545
TARGET_NAMES = [f"ring {a}-{b}" for a, b in RINGS] + ["ln C_D", "ln r_eD"]


def z_scores(model, bank, draws, seed):
    """|mu - y| / sigma per example and target (NaN where the target is masked)."""
    x, y, _, _ = fixed_set(bank, draws, seed=seed)
    mu, log_var, _ = predict(model, x)
    return (mu - y).abs().numpy() * np.exp(-0.5 * log_var.numpy())


def fit_scales(z, cover=COVER):
    """s_j = quantile_cover(z_j) / 2 per target column (ignoring NaN)."""
    return np.array([np.nanquantile(z[:, j], cover) / 2 for j in range(z.shape[1])])


def inside(z, scales, level):
    """Mean over the defined (non-NaN) errors of the indicator |error| < half-width of the central interval of
    Gaussian probability `level` (sigma scaled); NaN < x would count as a miss, so masked entries are excluded."""
    half = norm.ppf(0.5 + level / 2)
    zs = z / scales
    defined = ~np.isnan(zs)
    return np.sum((zs < half) & defined, axis=0) / np.sum(defined, axis=0)


def coverage(z, scales, level):
    """Per-target fraction of defined errors inside the scaled central interval of probability `level`."""
    return inside(z, scales, level)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from evaluate import load_model

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=HERE / "runs" / "long")
    args = ap.parse_args()
    torch.set_num_threads(4)
    model, _ = load_model(args.run)
    z_val = z_scores(model, load_bank("val"), VAL_DRAWS, seed=1234)
    z_test = z_scores(model, load_bank("test"), 4, seed=2024)
    scales = fit_scales(z_val)
    (Path(args.run) / "calibration.json").write_text(json.dumps(
        {"cover": COVER, "fitted_on": "val", "scales": dict(zip(TARGET_NAMES, scales.tolist()))}, indent=2))

    ones = np.ones(N_TARGET)
    before, after = coverage(z_test, ones, COVER), coverage(z_test, scales, COVER)
    print(f"{'target':<16} {'scale':>6} {'test cov before':>16} {'after':>7}")
    for name, s, b, a in zip(TARGET_NAMES, scales, before, after):
        print(f"{name:<16} {s:6.2f} {b:16.3f} {a:7.3f}")
    rings = slice(0, N_RING)
    pooled = lambda sc: inside((z_test[:, rings] / sc[rings]).reshape(-1, 1), np.ones(1), COVER)[0]
    print(f"{'all rings':<16} {'':>6} {pooled(ones):16.3f} {pooled(scales):7.3f}")

    levels = np.linspace(0.05, 0.99, 40)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    groups = {"rings r_D < 100": slice(0, 5), "rings r_D > 100": slice(5, N_RING), "ln C_D": [N_RING],
              "ln r_eD": [N_RING + 1]}
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    for a, sc, title in ((ax[0], ones, "(a) before calibration"), (ax[1], scales, "(b) after calibration (fit on val)")):
        for (name, cols), color in zip(groups.items(), colors):
            zz = (z_test[:, cols] / sc[cols]).reshape(-1, 1)
            emp = [inside(zz, np.array([1.0]), lv)[0] for lv in levels]
            a.plot(levels, emp, color=color, lw=1.8, label=name)
        a.plot([0, 1], [0, 1], color="#52514e", lw=0.8)
        a.set_xlabel("nominal coverage of the central interval")
        a.set_title(title + ", test set")
        a.grid(True, color="#e5e5e2", lw=0.6)
        a.set_axisbelow(True)
    ax[0].set_ylabel("empirical coverage")
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = HERE / "figs" / "fig_calibration.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=110)
    for f in (out, out.with_suffix(".png")):
        shutil.copy(f, Path(args.run) / f.name)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
