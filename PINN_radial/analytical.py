"""Homogeneous series adapter for validation, derived in phase1.pdf Appendix A.

Reuse the existing annular Bessel implementation without changing phase_1.
"""
from dataclasses import dataclass

import numpy as np

from phase_1.basis import eigenvalues, exact_coeffs_ones, phi


@dataclass
class AnalyticalSolution:
    problem: object
    n_modes: int = 256

    def __post_init__(self):
        self.lam = eigenvalues(self.n_modes, self.problem.radius_ratio)
        self.coefficients = exact_coeffs_ones(self.lam, self.problem.radius_ratio)

    def pressure(self, coordinates, chunk_size=1024):
        """Pointwise (N, 2) physical coordinates -> (N, 1) pressure; initial corner excluded."""
        xy = np.asarray(coordinates, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("Expected finite coordinates with shape (N, 2).")
        p = self.problem
        r, t = xy.T
        if np.any((r < p.rw) | (r > p.re) | (t < 0)):
            raise ValueError("Analytical coordinates must lie in the annulus at nonnegative times.")
        if np.any((r == p.rw) & (t == 0)):
            raise ValueError("The initial well corner has incompatible data.")
        result = np.full(len(xy), p.pi, dtype=float)
        indices = np.flatnonzero(t > 0)
        for start in range(0, len(indices), chunk_size):
            idx = indices[start:start + chunk_size]
            spatial = phi(r[idx, None] / p.rw, self.lam)
            temporal = np.exp(-t[idx, None] / p.t_scale * self.lam**2)
            result[idx] = p.pwf + p.dp * np.sum(spatial * temporal * self.coefficients, axis=1)
        return result[:, None]

    def grid(self, radii, times):
        """Radial pressure on a radius-time product grid, shape (len(times), len(radii))."""
        p = self.problem
        r, t = np.asarray(radii, dtype=float), np.asarray(times, dtype=float)
        if r.ndim != 1 or t.ndim != 1 or not np.isfinite(r).all() or not np.isfinite(t).all():
            raise ValueError("Radii and times must be finite one-dimensional arrays.")
        if np.any((r < p.rw) | (r > p.re)) or np.any(t <= 0):
            raise ValueError("Reference mesh requires radii in the annulus and strictly positive times.")
        spatial = phi(r[:, None] / p.rw, self.lam)
        temporal = np.exp(-t[:, None] / p.t_scale * self.lam**2)
        return p.pwf + p.dp * ((temporal * self.coefficients) @ spatial.T)

    def rate(self, times):
        """Dimensionless production rate rw * p_r(rw, t) / (pi - pwf)."""
        t = np.asarray(times, dtype=float)
        if not np.isfinite(t).all() or np.any(t <= 0):
            raise ValueError("Well rate is evaluated only at finite, strictly positive times.")
        return np.exp(-t[..., None] / self.problem.t_scale * self.lam**2) @ (-2 / np.pi * self.coefficients)


def reference_grid(config):
    p, ref = config.problem, config.reference
    # Hybrid radial coverage. These evaluation coordinates are independent of training draws.
    fraction = np.linspace(0, 1, ref.n_r)
    r = p.rw + 0.8 * (p.rw * np.exp(fraction * np.log(p.radius_ratio)) - p.rw)
    r += 0.2 * fraction * (p.re - p.rw)
    r[0], r[-1] = p.rw, p.re
    t = np.geomspace(ref.min_tau * p.t_scale, p.t_max, ref.n_t)
    return r, t


def converged_reference(config, radii, times):
    """Double modes until pressure AND rate stabilize; fail rather than save unconverged labels."""
    ref, p = config.reference, config.problem
    modes = ref.initial_modes
    solution = AnalyticalSolution(p, modes)
    pressure, rate = solution.grid(radii, times), solution.rate(times)
    while 2 * modes <= ref.max_modes:
        modes *= 2
        candidate = AnalyticalSolution(p, modes)
        new_pressure, new_rate = candidate.grid(radii, times), candidate.rate(times)
        pressure_change = float(np.max(np.abs(new_pressure - pressure)) / p.dp)
        rate_change = float(np.max(np.abs(new_rate - rate) / np.maximum(np.abs(new_rate), 1e-12)))
        # Avoid a false plateau when the earliest time has barely damped the last included mode.
        tail_damping = float(np.exp(-candidate.lam[-1]**2 * np.min(times) / p.t_scale))
        if (pressure_change < ref.pressure_tolerance and rate_change < ref.rate_tolerance
                and tail_damping < min(ref.pressure_tolerance, ref.rate_tolerance)):
            return candidate, new_pressure, new_rate, {
                "modes": modes, "previous_modes": modes // 2,
                "max_pressure_change_over_dp": pressure_change,
                "max_relative_rate_change": rate_change,
                "last_mode_damping_at_first_time": tail_damping,
            }
        pressure, rate = new_pressure, new_rate
    raise RuntimeError(f"Reference did not converge before max_modes={ref.max_modes}; "
                       "increase max_modes or the minimum reference time.")
