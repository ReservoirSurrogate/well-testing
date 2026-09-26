"""Milestone 2: build-up tools on homogeneous reservoirs (known radial-flow answers) and exact functions."""
import numpy as np
import pytest

from solver import WellSolver, grid_for
from wells import (Drawdown, agarwal_time, bourdet_derivative, buildup, buildup_times, horner_time,
                   interpret_radial_flow)

DLOG = np.log(1000.0) / 1023
T = np.logspace(-2, 7, 200)                  # stored drawdown times (dataset decision)
R, _ = grid_for(3000.0, DLOG)                # boundary felt from t ~ r_e^2 / (4 k) >= 9e5 for k <= 2.5


def drawdown(k=1.0, c_d=0.0, skin=0.0):
    sol = WellSolver(R, np.full_like(R, k), c_d=c_d, skin=skin)
    return sol, Drawdown(T, sol.well_pressure(T))


def test_drawdown_spline_between_samples():
    sol, dd = drawdown(c_d=10.0, skin=2.0)
    tm = np.sqrt(T[:-1] * T[1:])
    assert np.max(np.abs(dd(tm) / sol.well_pressure(tm) - 1)) < 1e-5
    with pytest.raises(ValueError):
        dd([1e-3])
    with pytest.raises(ValueError):
        dd([2e7])


def test_buildup_matches_simulated_shut_in():
    sol, dd = drawdown(k=1.5, c_d=50.0, skin=1.0)
    t_p = 1e4
    dt = np.logspace(-1, 6, 20)
    p_w, dp = buildup(dd, t_p, dt)
    p_sim = sol.well_pressure(t_p + dt, rate=lambda t: 1.0 if t <= t_p * (1 + 1e-12) else 0.0, breaks=[t_p])
    assert np.max(np.abs(p_w - p_sim)) < 1e-3 * np.max(dp)
    assert np.allclose(dp, dd(t_p) - p_w)


def test_time_functions():
    dt = np.array([1e-2, 1.0, 1e4, 1e8])
    assert np.allclose(agarwal_time(1e4, dt), 1e4 * dt / (1e4 + dt))
    assert np.allclose(agarwal_time(1e4, dt[:2]), dt[:2], rtol=1e-4)      # dt << t_p: dt_e ~ dt
    assert np.allclose(horner_time(1e4, dt), (1e4 + dt) / dt)
    b = buildup_times(1e5, 1e-2, 1e7, n=50)
    assert np.isclose(b[0], 1e-2) and np.isclose(b[-1], 1e7 - 1e5)


def test_bourdet_derivative_on_exact_functions():
    x = np.logspace(-2, 4, 121)
    assert np.allclose(bourdet_derivative(x, 0.7 * np.log(x) + 3.0), 0.7, atol=1e-12)
    d = bourdet_derivative(x, x, window=0.1)                              # d x / d ln x = x
    assert np.max(np.abs(d[5:-5] / x[5:-5] - 1)) < 3e-3
    d_raw = bourdet_derivative(x, x, window=0.0)
    assert np.max(np.abs(d_raw[5:-5] / x[5:-5] - 1)) < 3e-3               # window 0: nearest neighbours


def test_homogeneous_radial_flow():
    """No storage or skin: derivative plateau 1/2, Horner slope 1/2, k = 1, S = 0."""
    _, dd = drawdown()
    t_p = 1e5
    dt = buildup_times(t_p, T[0], T[-1])
    p_w, dp = buildup(dd, t_p, dt)
    te = agarwal_time(t_p, dt)
    der = bourdet_derivative(te, dp)
    m = (te > 1e3) & (te < 5e4)
    assert np.max(np.abs(der[m] - 0.5)) < 5e-3
    assert abs(np.polyfit(np.log(horner_time(t_p, dt)[m]), p_w[m], 1)[0] - 0.5) < 3e-3
    k, s = interpret_radial_flow(te, dp, der, (1e3, 5e4))
    assert abs(k - 1) < 5e-3 and abs(s) < 0.02


@pytest.mark.parametrize("k, skin, c_d", [(1.0, 5.0, 10.0), (2.5, 3.0, 100.0), (0.4, 1.0, 1.0)])
def test_interpretation_with_storage_and_skin(k, skin, c_d):
    """Radial flow after storage ends (~(60 + 3.5 S_wt) C / k): recovers k and the well-test skin S_wt = k S."""
    _, dd = drawdown(k=k, c_d=c_d, skin=skin)
    t_p = 1e6
    dt = buildup_times(t_p, T[0], T[-1])
    _, dp = buildup(dd, t_p, dt)
    te = agarwal_time(t_p, dt)
    t_end = (60 + 3.5 * k * skin) * c_d / k
    lo = max(10 * t_end, 1e3)
    hi = min(1e5, 3000.0**2 / (4 * k) / 10)                               # well before the boundary is felt
    k_est, s_est = interpret_radial_flow(te, dp, bourdet_derivative(te, dp), (lo, hi))
    assert abs(k_est / k - 1) < 0.015
    assert abs(s_est - k * skin) < 0.15
