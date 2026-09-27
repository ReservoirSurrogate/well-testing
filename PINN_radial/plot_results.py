"""Pressure profiles, histories, errors, and diagnostic well rate."""
import argparse
import os
from pathlib import Path
import tempfile

import numpy as np


def plot_results(directory):
    # Keep matplotlib's cache in a writable location on managed environments.
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="pinn-radial-mpl-")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    with np.load(directory / "reference.npz", allow_pickle=False) as ref:
        r, t, pressure, rate = ref["r"], ref["t"], ref["p"], ref["q_D"]
    with np.load(directory / "predictions.npz", allow_pickle=False) as pred:
        predicted, predicted_rate, error = pred["p"], pred["q_D"], pred["pressure_error_over_dp"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    indices = np.unique(np.linspace(0, len(t) - 1, 5, dtype=int))
    for index in indices:
        line, = axes[0, 0].semilogx(r, pressure[index], label=f"t={t[index]:.3g} s")
        axes[0, 0].semilogx(r, predicted[index], "--", color=line.get_color())
    axes[0, 0].set(xlabel="Radius r [m]", ylabel="Pressure p [Pa]",
                   title="Pressure profiles: reference solid, PINN dashed")
    axes[0, 0].legend(fontsize=8)
    for index in np.unique(np.linspace(1, len(r) - 1, 4, dtype=int)):
        line, = axes[0, 1].semilogx(t, pressure[:, index], label=f"r={r[index]:.3g} m")
        axes[0, 1].semilogx(t, predicted[:, index], "--", color=line.get_color())
    axes[0, 1].set(xlabel="Time t [s]", ylabel="Pressure p [Pa]", title="Pressure histories")
    axes[0, 1].legend(fontsize=8)
    mesh = axes[1, 0].pcolormesh(r, t, np.abs(error), shading="auto", cmap="magma")
    axes[1, 0].set(xscale="log", yscale="log", xlabel="Radius r [m]", ylabel="Time t [s]",
                   title="Absolute pressure error / (pi - pwf)")
    figure.colorbar(mesh, ax=axes[1, 0])
    # Keep negative predicted rates visible rather than silently dropping them on a log y-axis.
    axes[1, 1].semilogx(t, rate, label="Analytical")
    axes[1, 1].semilogx(t, predicted_rate, "--", label="PINN (autograd)")
    axes[1, 1].set(xlabel="Time t [s]", ylabel="Dimensionless well rate q_D", title="Well-gradient diagnostic")
    axes[1, 1].legend()
    figure.savefig(directory / "benchmark.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    plot_results(args.directory)


if __name__ == "__main__":
    main()
