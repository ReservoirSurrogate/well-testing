"""Compare an existing checkpoint with radial and Cartesian diffusion; no training."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .analytical import converged_reference, reference_grid
from .physics import pressure_derivatives
from .train import load_model


def cartesian_pressure(problem, r, t, modes=256):
    """Flat 1-D interval with the same alpha, IC, inner Dirichlet and outer Neumann BC.

    x=r-rw, L=re-rw, beta_n=(n+1/2)*pi/L, and u(x,0)=1:
    u=sum_n 2/((n+1/2)*pi) sin(beta_n*x) exp(-alpha*beta_n^2*t).
    """
    r, t = np.asarray(r), np.asarray(t)
    if np.any(t <= 0):
        raise ValueError("Geometry comparison requires strictly positive reference times.")
    p = problem
    index = np.arange(modes) + 0.5
    wave = index * np.pi / (p.re - p.rw)
    spatial = np.sin((r[:, None] - p.rw) * wave)
    temporal = np.exp(-p.alpha * t[:, None] * wave**2)
    return p.pwf + p.dp * ((temporal * (2 / (index * np.pi))) @ spatial.T)


def audit_geometry(checkpoint, output):
    model, config = load_model(checkpoint)
    torch.set_num_threads(config.training.threads)
    p = config.problem
    r, t = reference_grid(config)
    _, radial, _, convergence = converged_reference(config, r, t)
    modes = convergence["modes"]
    cartesian = cartesian_pressure(p, r, t, modes)
    cartesian_more = cartesian_pressure(p, r, t, 2 * modes)
    change = float(np.max(np.abs(cartesian_more - cartesian)) / p.dp)
    if change > config.reference.pressure_tolerance:
        raise RuntimeError("Cartesian comparison is not converged; increase reference.initial_modes.")
    cartesian = cartesian_more
    rr, tt = np.meshgrid(r, t)
    xy = np.column_stack((rr.ravel(), tt.ravel()))
    prediction, radial_residual, cartesian_residual = [], [], []
    for start in range(0, len(xy), 512):
        coords = torch.tensor(xy[start:start + 512], dtype=torch.float64)
        pressure, dr, dt, drr = pressure_derivatives(model, coords)
        scale = (p.t_scale + coords[:, 1:2]) / p.dp
        prediction.append(pressure.detach().numpy())
        radial_residual.append(((dt - p.alpha * (drr + dr / coords[:, :1])) * scale).detach().numpy())
        cartesian_residual.append(((dt - p.alpha * drr) * scale).detach().numpy())
    prediction = np.concatenate(prediction).reshape(rr.shape)
    radial_residual = np.concatenate(radial_residual).reshape(rr.shape)[:, 1:-1]
    cartesian_residual = np.concatenate(cartesian_residual).reshape(rr.shape)[:, 1:-1]
    metrics = {
        "checkpoint": str(checkpoint),
        "pressure_rmse_normalization": "pi - pwf, equal weighting of saved radius-time grid points",
        "residual_normalization": "(t_scale + t) / (pi - pwf), interior grid points",
        "pinn_pressure_rmse_vs_radial": float(np.sqrt(np.mean(((prediction-radial)/p.dp)**2))),
        "pinn_pressure_rmse_vs_cartesian": float(np.sqrt(np.mean(((prediction-cartesian)/p.dp)**2))),
        "pinn_radial_residual_rms": float(np.sqrt(np.mean(radial_residual**2))),
        "pinn_cartesian_residual_rms": float(np.sqrt(np.mean(cartesian_residual**2))),
        "radial_reference_convergence": convergence,
        "cartesian_modes": 2 * modes,
        "cartesian_pressure_change_over_dp": change,
        "interpretation": "The checkpoint is an inaccurate approximation. Closeness of pressure values "
                          "to one reference does not establish which PDE the model satisfies.",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "geometry_audit.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="pinn-radial-mpl-")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, index in zip(axes.flat, np.linspace(0, len(t)-1, 4, dtype=int)):
        ax.semilogx(r, radial[index] / 1e6, color="#1565c0", label="Radial analytical", linewidth=2)
        ax.semilogx(r, cartesian[index] / 1e6, color="#ef6c00", linestyle=":",
                    label="Cartesian analytical", linewidth=2)
        ax.semilogx(r, prediction[index] / 1e6, color="#c62828", linestyle="--",
                    label="Existing PINN", linewidth=2)
        ax.set(title=f"t = {t[index]:.3g} s", xlabel="Radius coordinate r [m]", ylabel="Pressure [MPa]")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    figure.suptitle("Geometry audit: identical diffusivity, initial pressure, and boundary values")
    figure.savefig(output / "geometry_comparison.png", dpi=160)
    plt.close(figure)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Existing evaluation directory; writes geometry audit files only")
    args = parser.parse_args()
    print(json.dumps(audit_geometry(args.checkpoint, args.output), indent=2))
