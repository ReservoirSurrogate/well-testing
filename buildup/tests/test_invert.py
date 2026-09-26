"""Model-based inversion: forward model consistency and recovery on data the parametrization can represent."""
import numpy as np

from invert import N_CTRL, X_CTRL, forward, invert, profile
from solver import WellSolver, log_grid


def test_forward_matches_direct_solver():
    theta = np.concatenate([0.3 * np.sin(X_CTRL), [np.log(5.0), np.log(800.0)]])
    t_p, dt = 1e4, np.geomspace(1e-2, 1e6, 12)
    dp, p_wf = forward(theta, t_p, dt)
    r, _ = log_grid(800.0, 900)
    sol = WellSolver(r, np.exp(profile(theta[:N_CTRL], r)), c_d=5.0)
    p = sol.well_pressure(np.concatenate([dt, t_p + dt, [t_p]]))
    assert np.isclose(p_wf, p[-1])
    assert np.allclose(dp, p[-1] - p[12:24] + p[:12])


def test_recovers_representable_reservoir():
    """Noise-free data from a smooth profile on the control points: C_D, r_eD and the near-well ln k come back."""
    k_true = 0.4 * np.cos(0.8 * X_CTRL) - 0.2
    theta_true = np.concatenate([k_true, [np.log(20.0), np.log(700.0)]])
    t_p, dt = 1e6, np.geomspace(1e-2, 9e6, 40)
    dp, p_wf = forward(theta_true, t_p, dt)
    res = invert(t_p, dt, dp, p_wf, noise=0.01)
    th = res["theta"]
    assert abs(np.exp(th[-2]) / 20.0 - 1) < 0.03
    assert abs(np.exp(th[-1]) / 700.0 - 1) < 0.03
    near = X_CTRL <= np.log(300.0)                 # well resolved by the build-up
    assert np.max(np.abs(th[:N_CTRL][near] - k_true[near])) < 0.15
