"""Y0-kernel FFTLog and the Weber transform outside a well of radius a = 1.

Test function f(r) = (r - 1)^5 exp(-(r - 1)^2) for r >= 1, 0 inside: f and its first four derivatives vanish at r = 1,
so the zero extensions of f and of its Laplacian are smooth and their transforms decay fast at large k. (A kink at
r = a makes the transform decay only algebraically, and the large-k end wraps around the periodic grid; with
f = (r - 1)^3 exp(-(r - 1)^2), whose Laplacian has a kink at r = 1, the diagonalization test reaches only ~7e-5.)

Bias q = 0.4 throughout: W(k) tends to a nonzero constant as k -> 0, so with q = 0 the transformed quantity k W(k)
is still ~6e-5 at the smallest k of the grid and wraps around the periodic ln k domain (round trip ~4e-5);
q = 0.4 suppresses that end (k^1.4 W ~ 2e-6) and gives a round trip to ~3e-7.
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma, j0, y0

from fftlog import FFTLog, Weber, _u_y, log_grid
from solver import RadialSolver, time_grid

N = 2048
R, D = log_grid(1e-4, 1e4, N)
A = 1.0
Q = 0.4


def f(r):
    x = np.clip(np.asarray(r, dtype=float) - A, 0.0, None)
    return x**5 * np.exp(-x**2)


def laplacian_f(r):
    """f'' + f'/r for r > a."""
    x = r - A
    e = np.exp(-x**2)
    d1 = (5 * x**4 - 2 * x**6) * e
    d2 = (20 * x**3 - 22 * x**5 + 4 * x**7) * e
    return np.where(r > A, d2 + d1 / r, 0.0)


def weber_quad(k):
    c = lambda r: j0(k * r) * y0(k * A) - y0(k * r) * j0(k * A)
    return quad(lambda r: f(r) * c(r) * r, A, 12.0, limit=800, epsabs=1e-14, epsrel=1e-12)[0]


def test_u_y_matches_gamma_formula():
    w = np.linspace(0.0, 20.0, 41)
    for q in (0.0, 0.4, -0.5):
        z = q + 1j * w
        ref = -(2**z / np.pi) * gamma((1 + z) / 2) ** 2 * np.cos((1 + z) * np.pi / 2)
        assert np.allclose(_u_y(0.0, q, w), ref, rtol=1e-10)
    assert np.all(np.isfinite(_u_y(0.0, 0.4, np.linspace(0, 500, 11))))   # no overflow at large w


def test_y_kernel_matches_quadrature():
    g = lambda r: np.exp(-(np.log(r) - 1.0) ** 2) / (1 + r**2)
    t = FFTLog(N, D, mu=0.0, q=Q, kernel="Y", low_ringing=False)
    k, G = t.forward(g(R), R[0])
    for kv in (0.05, 0.3, 1.0, 3.0):
        i = np.argmin(np.abs(np.log(k / kv)))
        ref = quad(lambda x: g(np.exp(x)) * y0(k[i] * np.exp(x)) * np.exp(2 * x), -12, 12,
                   limit=1000, epsabs=1e-14, epsrel=1e-12)[0]
        assert abs(G[i] - ref) < 1e-7 * max(abs(ref), 1e-3), (k[i], G[i], ref)


def test_weber_forward_matches_quadrature():
    k, w = Weber(N, D, a=A, q=Q).forward(f(R), R[0])
    for kv in (0.02, 0.2, 1.0, 3.0, 8.0):
        i = np.argmin(np.abs(np.log(k / kv)))
        ref = weber_quad(k[i])
        assert abs(w[i] - ref) < 1e-6 * max(abs(ref), 1e-2), (k[i], w[i], ref)


def test_weber_diagonalizes_the_laplacian_outside_the_well():
    """For f(a) = 0: W[f'' + f'/r] = -k^2 W[f]."""
    t = Weber(N, D, a=A, q=Q)
    k, w = t.forward(f(R), R[0])
    _, wl = t.forward(laplacian_f(R), R[0])
    m = (k > 1e-2) & (k < 10)
    assert np.max(np.abs(wl[m] + k[m] ** 2 * w[m])) < 1e-6 * np.max(np.abs(k[m] ** 2 * w[m]))


def test_weber_round_trip():
    t = Weber(N, D, a=A, q=Q)
    k, w = t.forward(f(R), R[0])
    r_back, f_back = t.inverse(w, k[0])
    assert np.allclose(r_back, R, rtol=1e-12)
    m = R >= A
    assert np.max(np.abs(f_back[m] - f(R[m]))) < 1e-6 * np.max(f(R))


@pytest.mark.parametrize("t_end", [0.5, 2.0])
def test_weber_heat_propagation_matches_fv_solver(t_end):
    """u_t = u'' + u'/r outside the well with u(1) = 0: W^{-1}[exp(-k^2 t) W[u0]] vs the finite-volume solver
    (outer boundary at r = 200, not reached for t <= 2)."""
    t = Weber(N, D, a=A, q=Q)
    k, w = t.forward(f(R), R[0])
    _, u_weber = t.inverse(np.exp(-k**2 * t_end) * w, k[0])

    r_fv = np.geomspace(1.0, 200.0, 4000)
    t_grid, idx = time_grid(np.array([t_end]), h0=1e-5, growth=1.02, h_max=2e-3)
    u_fv = RadialSolver(r_fv, np.ones_like(r_fv)).integrate(f(r_fv), t_grid)[idx[0]]
    m = (r_fv >= 1.0) & (r_fv <= 15.0)
    u_ref = np.interp(np.log(r_fv[m]), np.log(R), u_weber)
    assert np.max(np.abs(u_fv[m] - u_ref)) < 2e-4 * np.max(np.abs(u_fv[m]))
