"""Milestone 1: finite-volume well solver against analytic solutions (k_D = 1) and exact discrete properties."""
import numpy as np
import pytest

from analytic import line_source, pss, well_pressure
from solver import WellSolver, grid_for, log_grid, time_grid

DLOG = np.log(1000.0) / 1023          # grid spacing used throughout the project
T = np.logspace(-1, 7, 17)


def homogeneous(r_e, **kw):
    r, _ = grid_for(r_e, DLOG)
    return WellSolver(r, np.ones_like(r), **kw)


def heterogeneous(r_e=300.0, seed=0, **kw):
    r, _ = grid_for(r_e, DLOG)
    rng = np.random.default_rng(seed)
    x = np.log(r)
    logk = 0.5 * np.sin(1.3 * x + rng.uniform(0, 6)) + 0.3 * np.cos(3.1 * x)
    logk[r < 5] += np.log(0.3)                                  # damaged zone
    return WellSolver(r, np.exp(logk), **kw)


@pytest.mark.parametrize("c_d, skin", [(0.0, 0.0), (100.0, 0.0), (0.0, 5.0), (100.0, 5.0)])
def test_matches_van_everdingen_hurst(c_d, skin):
    """Finite wellbore, closed circle r_e = 300, storage and skin: Stehfest-inverted Laplace solution."""
    p = homogeneous(300.0, c_d=c_d, skin=skin).well_pressure(T)
    ref = well_pressure(T, 300.0, c_d=c_d, skin=skin)
    assert np.max(np.abs(p / ref - 1)) < 1e-3


def test_line_source_before_the_boundary():
    """Infinite-acting period of r_e = 3000 (boundary felt from t ~ r_e^2 / 4 ~ 2e6)."""
    t = np.logspace(3, 5, 9)
    p = homogeneous(3000.0).well_pressure(t)
    assert np.max(np.abs(p / line_source(t) - 1)) < 1e-3


def test_pseudo_steady_state():
    t = np.logspace(6, 7, 5)                                     # t / r_e^2 >= 11 for r_e = 300
    for skin in (0.0, 3.0):
        p = homogeneous(300.0, skin=skin).well_pressure(t)
        assert np.max(np.abs(p / pss(t, 300.0, skin) - 1)) < 1e-4


def test_storage_unit_slope():
    t = np.logspace(-2, 0, 5)
    p = homogeneous(300.0, c_d=1e3).well_pressure(t)
    assert np.max(np.abs(p / (t / 1e3) - 1)) < 2e-3


@pytest.mark.parametrize("c_d, skin", [(0.0, 0.0), (50.0, 2.0)])
def test_conservation_with_rate_history(c_d, skin):
    """sum V p + C p_w = int q dt at every step, for heterogeneous k and a shut-in."""
    sol = heterogeneous(c_d=c_d, skin=skin)
    t_p = 1e3
    t_grid, brk = time_grid(np.logspace(-2, 6, 30), breaks=[t_p])
    x = sol.integrate(t_grid, lambda t: 1.0 if t <= t_p * (1 + 1e-12) else 0.0, brk)
    produced = np.minimum(t_grid, t_p)
    assert np.allclose(sol.cumulative_storage(x), produced, rtol=1e-9, atol=1e-12)


def test_shut_in_equals_superposed_drawdowns():
    """Linear problem: a simulated build-up equals p_dd(t_p + dt) - p_dd(dt), heterogeneous k with storage and skin."""
    sol = heterogeneous(c_d=50.0, skin=2.0)
    t_p = 1e4
    dt = np.logspace(-2, 6, 25)
    p_bu = sol.well_pressure(t_p + dt, rate=lambda t: 1.0 if t <= t_p * (1 + 1e-12) else 0.0, breaks=[t_p])
    p_dd = sol.well_pressure(np.concatenate([dt, t_p + dt]))
    superposed = p_dd[len(dt):] - p_dd[:len(dt)]
    p_wf = sol.well_pressure([t_p])[0]
    # compare the build-up pressure change p_wf - p_w(dt), relative to its final value
    change_sim, change_sup = p_wf - p_bu, p_wf - superposed
    assert np.max(np.abs(change_sim - change_sup)) < 1e-3 * np.max(change_sup)


def test_time_grid_hits_outputs_and_breaks():
    t_out = np.array([0.5, 3.0, 1e4])
    t, brk = time_grid(t_out, breaks=[2.0])
    assert t[0] == 0 and np.all(np.diff(t) > 0)
    assert set(t_out) <= set(t) and 2.0 in t
    assert brk[list(t).index(2.0)] and brk.sum() == 1


def test_invalid_inputs():
    r, _ = log_grid(300.0, 100)
    with pytest.raises(ValueError):
        WellSolver(r, np.ones_like(r), skin=-1.0)
    with pytest.raises(ValueError):
        WellSolver(np.linspace(1, 300, 100), np.ones(100))
