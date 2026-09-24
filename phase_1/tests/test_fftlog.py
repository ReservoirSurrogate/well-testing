"""FFTLog tests: analytic Hankel pairs, round trip, low-ringing kappa, bias, and a quadrature check."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import jv

from fftlog import FFTLog, log_grid, low_ringing_kappa

N = 1024
R, D = log_grid(1e-5, 1e5, N)
# Analytic pairs whose r f(r) or k F(k) fall off only like r or k at the small end need a wide grid:
# the periodic-edge value (~ r_min) is divided by k when F is recovered, so it rings at small k.
NW = 2048
RW, DW = log_grid(1e-12, 1e12, NW)


def check(y, got, want, lo, hi, tol):
    """Max error in [lo, hi], relative to max |want| (edges of the periodic grid ring and are excluded)."""
    m = (y >= lo) & (y <= hi)
    assert m.sum() > 50
    err = np.max(np.abs(got[m] - want[m])) / np.max(np.abs(want))
    assert err < tol, err


@pytest.mark.parametrize("mu", [0.0, 1.0, 2.0])
def test_gaussian_pair(mu):
    """r^mu exp(-r^2/2) <-> k^mu exp(-k^2/2)."""
    t = FFTLog(NW, DW, mu=mu)
    k, F = t.forward(RW**mu * np.exp(-RW**2 / 2), RW[0])
    check(k, F, k**mu * np.exp(-k**2 / 2), 1e-2, 8.0, 1e-9)


def test_algebraic_pair():
    """(1 + r^2)^{-3/2} <-> exp(-k) for mu = 0: a slowly decaying (power-law) input."""
    t = FFTLog(NW, DW, mu=0.0)
    k, F = t.forward((1 + RW**2) ** -1.5, RW[0])
    check(k, F, np.exp(-k), 1e-2, 20.0, 1e-9)


def test_exponential_pair():
    """exp(-r) <-> (1 + k^2)^{-3/2} for mu = 0 (the reverse direction of the previous pair)."""
    t = FFTLog(NW, DW, mu=0.0)
    k, F = t.forward(np.exp(-RW), RW[0])
    check(k, F, (1 + k**2) ** -1.5, 1e-2, 100.0, 1e-9)


def test_bias_does_not_change_result():
    """A bias q inside (-mu-1, 1/2) only reshapes the periodic function; the transform is unchanged.

    mu = 1 so that q = -0.5 keeps both grid ends negligible (for mu = 0 it leaves k^{1/2} F ~ k^{1/2} at small k).
    """
    f = RW * np.exp(-RW**2 / 2)
    for q in (0.0, -0.5, -1.5, 0.3):                # low-ringing kappa depends on q, so compare to the exact F
        k, F = FFTLog(NW, DW, mu=1.0, q=q).forward(f, RW[0])
        check(k, F, k * np.exp(-k**2 / 2), 1e-2, 8.0, 1e-9)


def test_bias_out_of_range():
    with pytest.raises(ValueError):
        FFTLog(N, D, mu=0.0, q=0.6)


def test_round_trip_is_identity():
    """For q = 0 and low-ringing kappa, r -> k -> r reproduces any input to rounding error.

    The transform acts on r f, so the check is on r f: recovering f divides by r and amplifies rounding
    error by r_max / r_min at the small end.
    """
    rng = np.random.default_rng(0)
    f = rng.standard_normal((3, N))
    t = FFTLog(N, D, mu=0.0)
    k, F = t.forward(f, R[0])
    r_back, f_back = t.forward(F, k[0])
    assert np.allclose(r_back, R, rtol=1e-12)
    assert np.max(np.abs(R * (f_back - f))) < 1e-12 * np.max(np.abs(R * f))


def test_low_ringing_kappa_makes_nyquist_real():
    kappa = low_ringing_kappa(N, D, mu=0.0, q=0.0, kappa=1.0)
    t = FFTLog(N, D, mu=0.0, kappa=1.0)
    assert np.isclose(t.kappa, kappa)
    assert abs(t.multiplier[-1].imag) < 1e-10
    assert 0.9 < kappa < 1.1          # moved by at most half a grid step in ln kappa


def test_matches_quadrature_on_a_smooth_bump():
    """Against direct numerical integration for a function without a closed-form transform."""
    f = lambda r: np.exp(-(np.log(r) - 1.0) ** 2) / (1 + r**2)
    t = FFTLog(N, D, mu=0.0)
    k, F = t.forward(f(R), R[0])
    for kv in (0.05, 0.3, 1.0, 3.0):
        i = np.argmin(np.abs(np.log(k / kv)))           # compare on the grid, not interpolated
        # in x = ln r: int f(r) J0(k r) r dr = int f(e^x) J0(k e^x) e^{2x} dx
        ref = quad(lambda x: f(np.exp(x)) * jv(0, k[i] * np.exp(x)) * np.exp(2 * x), -12, 12,
                   limit=1000, epsabs=1e-14, epsrel=1e-12)[0]
        assert abs(F[i] - ref) < 1e-8 * max(abs(ref), 1e-3), (k[i], F[i], ref)


def test_batched_input():
    t = FFTLog(N, D)
    f = np.stack([np.exp(-R**2 / 2), 2 * np.exp(-R**2 / 2)])
    _, F = t.forward(f, R[0])
    assert F.shape == (2, N)
    assert np.allclose(F[1], 2 * F[0])


@pytest.mark.parametrize("q", [0.0, 0.4, -0.5])
def test_inverse_is_exact_for_any_bias(q):
    """inverse(forward(f)) = f to rounding error (checked on r^{1-q} f, the quantity the transform acts on)."""
    rng = np.random.default_rng(1)
    f = rng.standard_normal((2, N))
    t = FFTLog(N, D, mu=0.0, q=q)
    k, F = t.forward(f, R[0])
    r_back, f_back = t.inverse(F, k[0])
    assert np.allclose(r_back, R, rtol=1e-12)
    w = R ** (1 - q)
    assert np.max(np.abs(w * (f_back - f))) < 1e-12 * np.max(np.abs(w * f))


def test_inverse_equals_forward_for_zero_bias():
    f = np.exp(-RW**2 / 2)
    t = FFTLog(NW, DW, q=0.0)
    k, F = t.forward(f, RW[0])
    _, a = t.inverse(F, k[0])
    _, b = t.forward(F, k[0])
    assert np.max(np.abs(RW * (a - b))) < 1e-12
