"""Milestone 2 tests: finite-volume reference solver (r_eD = 1000)."""
import numpy as np
import pytest

from basis import eigenvalues, exact_rate, exact_solution, make_grid
from solver import RadialSolver, storage, time_grid, transmissibility

R_E = 1000.0
N = 1024
N_EXACT = 4000  # modes in the reference series; converged for t >= 1


@pytest.fixture(scope="module")
def lam():
    return eigenvalues(N_EXACT, R_E)


def run_homogeneous(n_points, t_out, **grid_kw):
    r, _ = make_grid(n_points, R_E)
    s = RadialSolver(r, 1.0)
    t_grid, idx = time_grid(t_out, **grid_kw)
    return r, s, s.integrate(np.ones(n_points), t_grid)[idx]


def rel_l2(a, b):
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


# --- discretization ---------------------------------------------------------

def test_storage_sums_to_annulus():
    r, _ = make_grid(N, R_E)
    assert storage(r).sum() == pytest.approx((R_E**2 - 1) / 2, rel=1e-13)


def test_flux_exact_for_steady_profile_across_jump():
    # u = c ln r in zone 1 (k1), continued with the same flux k1 c in zone 2 (k2);
    # the jump sits at a face, so every face flux must equal k1 c exactly.
    r, _ = make_grid(64, R_E)
    j = 30
    f = np.sqrt(r[j] * r[j + 1])
    k1, k2, c = 1.0, 0.1, 0.7
    k = np.where(np.arange(len(r)) <= j, k1, k2)
    u = np.where(r <= f, c * np.log(r), c * np.log(f) + c * k1 / k2 * np.log(r / f))
    flux = RadialSolver(r, k).face_flux(u)
    assert np.allclose(flux, k1 * c, rtol=1e-12)


def test_transmissibility_uses_harmonic_mean():
    r = np.array([1.0, 2.0])
    k = np.array([1.0, 3.0])
    assert transmissibility(r, k)[0] == pytest.approx(1.5 / np.log(2.0), rel=1e-14)


# --- validation against the k_D = 1 eigen-series ----------------------------

T_LATE = np.array([1e3, 1e4, 1e5])


@pytest.fixture(scope="module")
def late_run():
    return run_homogeneous(N, T_LATE)


def test_matches_exact_solution_late(lam, late_run):
    r, _, u = late_run
    u_exact = np.array([exact_solution(r, t, lam, R_E) for t in T_LATE])
    u_exact[:, 0] = 0.0
    assert np.all(rel_l2(u, u_exact) < 2e-6)


def test_matches_exact_rate_late(lam, late_run):
    _, s, u = late_run
    q_exact = exact_rate(T_LATE, lam, R_E)
    assert np.allclose(s.rate(u), q_exact, rtol=5e-6)


def test_spatial_convergence(lam):
    t = np.array([1e5])
    errs = []
    for n in (256, 1024):
        r, _, u = run_homogeneous(n, t)
        errs.append(rel_l2(u[0], exact_solution(r, t[0], lam, R_E)))
    assert errs[0] / errs[1] > 8  # second order: ~16 for 4x points


def test_temporal_convergence(lam):
    t = np.array([1.0, 10.0])
    q_exact = exact_rate(t, lam, R_E)
    errs = []
    for h0, growth in [(1e-3, 1.1), (1e-5, 1.02)]:
        _, s, u = run_homogeneous(N, t, h0=h0, growth=growth)
        errs.append(np.max(np.abs(s.rate(u) / q_exact - 1)))
    assert errs[1] < 1e-4
    assert errs[0] / errs[1] > 10


# --- heterogeneous sanity ------------------------------------------------------

def test_heterogeneous_bounded_and_decaying():
    rng = np.random.default_rng(0)
    r, _ = make_grid(N, R_E)
    k = np.exp(np.convolve(rng.standard_normal(N), np.ones(50) / np.sqrt(50), mode="same"))
    s = RadialSolver(r, k)
    t_grid, idx = time_grid(np.array([10.0, 1e3, 1e5]))
    u = s.integrate(np.ones(N), t_grid)
    assert u.min() > -1e-6 and u.max() < 1 + 1e-6
    q = s.rate(u[idx])
    assert np.all(q > 0) and np.all(np.diff(q) < 0)


# --- time grid -----------------------------------------------------------------

def test_time_grid_hits_outputs():
    t_out = np.array([0.37, 1.0, 100.0, 1234.5])
    t_grid, idx = time_grid(t_out)
    assert t_grid[0] == 0.0
    assert np.array_equal(t_grid[idx], t_out)
    assert np.all(np.diff(t_grid) > 0)


def test_time_grid_step_ratios_bdf2_stable():
    t_grid, _ = time_grid(np.array([0.37, 1.0, 100.0, 100.01, 1234.5]))
    h = np.diff(t_grid)
    assert np.all(h[1:] / h[:-1] < 1 + np.sqrt(2))


def test_integrate_rejects_non_increasing_grid():
    r, _ = make_grid(16, R_E)
    with pytest.raises(ValueError):
        RadialSolver(r, 1.0).integrate(np.ones(16), np.array([0.0, 1.0, 1.0]))
