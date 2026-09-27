"""Evaluate a trained pressure PINN against a converged analytical series."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .analytical import converged_reference, reference_grid
from .physics import first_derivatives, pde_residual
from .train import load_model


def evaluate(checkpoint, output, device="cpu", plots=True):
    model, config = load_model(checkpoint, device)
    torch.set_num_threads(config.training.threads)
    p = config.problem
    output = Path(output)
    if (output / "metrics.json").exists():
        raise FileExistsError(f"Evaluation already exists at {output}; choose a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    r, t = reference_grid(config)
    _, reference, rate_reference, convergence = converged_reference(config, r, t)
    rr, tt = np.meshgrid(r, t)
    xy = np.column_stack((rr.ravel(), tt.ravel()))
    predictions = []
    for start in range(0, len(xy), 4096):
        chunk = xy[start:start + 4096]
        predictions.append(model.predict_pressure(chunk[:, 0], chunk[:, 1]).cpu().numpy())
    prediction = np.concatenate(predictions).reshape(reference.shape)

    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float64, device=device)

    well_xy = tensor(np.column_stack((np.full_like(t, p.rw), t)))
    outer_xy = tensor(np.column_stack((np.full_like(t, p.re), t)))
    _, _, well_gradient, _ = first_derivatives(model, well_xy)
    _, _, outer_gradient, _ = first_derivatives(model, outer_xy)
    rate = (p.rw * well_gradient / p.dp).detach().cpu().numpy().ravel()
    outer = (p.re * outer_gradient / p.dp).detach().cpu().numpy().ravel()
    initial = model.predict_pressure(r[1:], np.zeros(len(r) - 1)).cpu().numpy()

    # Independent PDE residuals and material balance from autograd, not finite differences in time.
    residuals, dp_dt = [], []
    for start in range(0, len(xy), 512):
        chunk = tensor(xy[start:start + 512])
        residuals.append((pde_residual(model, chunk, p) * p.t_scale / p.dp).detach().cpu().numpy())
        _, _, _, time_gradient = first_derivatives(model, chunk)
        dp_dt.append(time_gradient.detach().cpu().numpy())
    dp_dt = np.concatenate(dp_dt).reshape(reference.shape)
    mass_rate = np.trapezoid(r[None, :] * dp_dt, r, axis=1)
    balance = mass_rate / (p.alpha * p.dp) + rate

    normalized_error = (prediction - reference) / p.dp
    weighted_squared = np.trapezoid(r[None, :] * normalized_error**2, r, axis=1)
    reference_squared = np.trapezoid(r[None, :] * ((reference - p.pwf) / p.dp)**2, r, axis=1)
    annulus_area_factor = (p.re**2 - p.rw**2) / 2
    relative_l2 = np.sqrt(weighted_squared / np.maximum(reference_squared, annulus_area_factor * 1e-12))
    near = r <= min(10 * p.rw, p.rw + 0.1 * (p.re - p.rw))
    rate_floor = 1e-8
    relative_rate = np.abs(rate - rate_reference) / np.maximum(np.abs(rate_reference), rate_floor)
    interior_residual = np.concatenate(residuals).reshape(reference.shape)[:, 1:-1]
    time_weighted_residual = interior_residual * (1 + np.log1p(t[:, None] / p.t_scale))
    metrics = {
        "checkpoint": str(checkpoint), "description": config.description,
        "reference_convergence": convergence,
        "time_range_s": [float(t[0]), float(t[-1])],
        "units": {"r": "m", "t": "s", "p": "Pa"},
        "max_absolute_pressure_error_Pa": float(np.max(np.abs(prediction - reference))),
        "max_pressure_error_over_dp": float(np.max(np.abs(normalized_error))),
        "rmse_pressure_over_dp": float(np.sqrt(np.mean(normalized_error**2))),
        "max_area_weighted_relative_l2": float(np.max(relative_l2)),
        "max_near_well_error_over_dp": float(np.max(np.abs(normalized_error[:, near]))),
        "near_well_radius_limit_m": float(r[near][-1]),
        "max_rate_relative_error": float(np.max(relative_rate)),
        "rate_relative_denominator_floor": rate_floor,
        "relative_l2_reference_rms_floor_over_dp": 1e-6,
        "max_ic_error_over_dp": float(np.max(np.abs(initial - p.pi)) / p.dp),
        "max_bc1_error_over_dp": float(np.max(np.abs(prediction[:, 0] - p.pwf)) / p.dp),
        "max_bc2_scaled_gradient": float(np.max(np.abs(outer))),
        "rms_scaled_pde_residual": float(np.sqrt(np.mean(interior_residual**2))),
        "rms_time_weighted_pde_residual": float(np.sqrt(np.mean(time_weighted_residual**2))),
        "max_material_balance_error_in_qD_units": float(np.max(np.abs(balance))),
        "pressure_lower_bound_violation_over_dp": float(max(0, (p.pwf - prediction.min()) / p.dp)),
        "pressure_upper_bound_violation_over_dp": float(max(0, (prediction.max() - p.pi) / p.dp)),
        "criteria": {
            "pressure_l2_below_1_percent": bool(np.max(relative_l2) < 0.01),
            "near_well_error_below_0_01_dp": bool(np.max(np.abs(normalized_error[:, near])) < 0.01),
            "rate_error_below_5_percent": bool(np.max(relative_rate) < 0.05),
        },
    }
    np.savez_compressed(output / "reference.npz", r=r, t=t, X=xy, y=reference.reshape(-1, 1),
                        p=reference, q_D=rate_reference,
                        metadata=json.dumps({"config": config.to_dict(), "convergence": convergence,
                                             "purpose": "analytical evaluation only; not training labels"}))
    np.savez_compressed(output / "predictions.npz", r=r, t=t, p=prediction, q_D=rate,
                        pressure_error_over_dp=normalized_error, relative_l2=relative_l2,
                        material_balance_error=balance)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if plots:
        from .plot_results import plot_results
        plot_results(output)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.output, args.device, not args.no_plots), indent=2))


if __name__ == "__main__":
    main()
