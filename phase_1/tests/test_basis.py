"""Milestone 1 tests: annular Hankel basis and transform (r_eD = 1000, K = 32)."""
import numpy as np
import pytest
from scipy.integrate import quad

from basis import (HankelBasis, dphi, eig_residual, eigenvalues, exact_coeffs_ones,
                   exact_rate, exact_solution, make_grid, norms, phi)

R_E = 1000.0
K = 32
N = 1024


@pytest.fixture(scope="module")
def lam():
    return eigenvalues(K, R_E)


@pytest.fixture(scope="module")
def basis():
    return HankelBasis.build(K, N, R_E)


# --- eigenvalues -----------------------------------------------------------

def test_eigenvalues_are_roots(lam):
    # Residual scale is ~ |J| |Y| ~ 1/lambda; compare against that.
    assert np.all(np.abs(eig_residual(lam, R_E)) * lam < 1e-12)


def test_first_eigenvalue(lam):
    assert lam[0] == pytest.approx(5.68798e-4, rel=1e-5)
    approx = 2 / (R_E**2 * (np.log(R_E) - 0.75))
    assert lam[0] ** 2 == pytest.approx(approx, rel=5e-3)


def test_no_missed_roots(lam):
    # A missed root would show up as a gap of ~2x the asymptotic spacing.
    gaps = np.diff(lam) / (np.pi / (R_E - 1))
    assert np.all((gaps > 0.9) & (gaps < 1.2))


# --- eigenfunctions --------------------------------------------------------

def test_boundary_conditions(lam):
    assert np.allclose(phi(1.0, lam), 0, atol=1e-14)
    assert np.allclose(dphi(R_E, lam), 0, atol=1e-12)


def test_dphi_matches_finite_difference(lam):
    r, h = 37.0, 1e-5
    fd = (phi(r + h, lam) - phi(r - h, lam)) / (2 * h)
    assert np.allclose(dphi(r, lam), fd, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("n", [0, 1, 4, K - 1])
def test_norm_closed_form(lam, n):
    numeric = quad(lambda r: r * phi(r, lam[n]) ** 2, 1, R_E, limit=2000)[0]
    assert norms(lam[n], R_E) == pytest.approx(numeric, rel=1e-8)


def test_orthogonality(lam):
    val = quad(lambda r: r * phi(r, lam[0]) * phi(r, lam[3]), 1, R_E, limit=2000)[0]
    assert abs(val) < 1e-8 * np.sqrt(norms(lam[0], R_E) * norms(lam[3], R_E))


@pytest.mark.parametrize("n", [0, 1, 2])
def test_interior_zero_count(lam, n):
    r = np.logspace(0, np.log10(R_E), 20000)[5:]
    p = phi(r, lam[n])
    assert np.count_nonzero(np.sign(p[1:]) != np.sign(p[:-1])) == n


# --- grid and quadrature ---------------------------------------------------

@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.0])
def test_grid_endpoints_and_weights(alpha):
    r, w = make_grid(N, R_E, alpha)
    assert r[0] == pytest.approx(1.0) and r[-1] == pytest.approx(R_E)
    assert np.all(np.diff(r) > 0)
    # int_1^{r_e} r dr = (r_e^2 - 1)/2
    assert w.sum() == pytest.approx((R_E**2 - 1) / 2, rel=1e-4)


# --- transform -------------------------------------------------------------

def test_round_trip_identity(basis):
    assert np.allclose(basis.T @ basis.B, np.eye(K), atol=1e-12)


def test_projection_idempotent(basis):
    P = basis.B @ basis.T
    assert np.allclose(P @ P, P, atol=1e-10)


def test_quadrature_method_close_to_lstsq():
    q = HankelBasis.build(K, N, R_E, method="quadrature")
    assert np.abs(q.T @ q.B - np.eye(K)).max() < 1e-3


def test_coefficients_of_ones(basis):
    exact = exact_coeffs_ones(basis.lam, R_E)
    got = basis.forward(np.ones(N))
    assert np.abs(got - exact).max() < 1e-6 * np.abs(exact).max()


def test_smooth_function_reconstruction(basis):
    # f satisfies both BCs: f(1) = 0, f'(r_e) = 0. Error is dominated by K-mode truncation.
    f = np.log(basis.r) - (basis.r - 1) / R_E
    rec = basis.inverse(basis.forward(f))
    assert np.abs(rec - f).max() < 2e-3 * np.abs(f).max()


def test_exact_homogeneous_step(basis):
    # k_D = 1: one step is diagonal in the basis, u_hat <- exp(-lambda^2 dt) u_hat.
    t, dt = 1e3, 100.0
    u0 = exact_solution(basis.r, t, basis.lam, R_E)
    u1 = exact_solution(basis.r, t + dt, basis.lam, R_E)
    stepped = basis.inverse(np.exp(-basis.lam**2 * dt) * basis.forward(u0))
    assert np.allclose(stepped, u1, atol=1e-12)


# --- analytic reference ----------------------------------------------------

def test_exact_solution_boundary_values(basis):
    u = exact_solution(basis.r, 1e4, basis.lam, R_E)
    assert abs(u[0]) < 1e-12
    assert 0 < u[-1] <= 1


def test_exact_rate_matches_derivative(basis):
    # q_D = r du/dr at r = 1 (k_D = 1).
    t = 1e4
    coef = exact_coeffs_ones(basis.lam, R_E) * np.exp(-basis.lam**2 * t)
    q_from_u = (dphi(1.0, basis.lam) @ coef) * 1.0
    assert exact_rate(t, basis.lam, R_E) == pytest.approx(q_from_u, rel=1e-10)


def test_exact_rate_infinite_acting_regime():
    # With enough modes, q_D(1e4) ~ 2 / (ln t + 0.809) (infinite-acting approximation).
    lam_many = eigenvalues(400, R_E)
    t = 1e4
    assert exact_rate(t, lam_many, R_E) == pytest.approx(2 / (np.log(t) + 0.809), rel=0.05)
