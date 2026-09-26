"""Milestone 3: permeability generator (statistics, skin zone, reproducibility) and labels (k_eff, S)."""
import numpy as np
import pytest

from permeability import LOGK_CLIP, grf, grf_factor, labels, resistance, sample_logk
from solver import grid_for

DLOG = np.log(1000.0) / 1023
R, _ = grid_for(1000.0, DLOG)


def test_reproducible_per_seed():
    a, ma = sample_logk(np.random.default_rng(7), R)
    b, mb = sample_logk(np.random.default_rng(7), R)
    c, _ = sample_logk(np.random.default_rng(8), R)
    assert np.array_equal(a, b) and ma == mb
    assert not np.allclose(a, c)


@pytest.mark.parametrize("sampler", ["fft", "eigh"])
def test_grf_covariance(sampler):
    """Empirical covariance at lags 0, ell/2, ell and 2 ell matches sigma^2 exp(-lag^2 / (2 ell^2))."""
    rng = np.random.default_rng(0)
    x = np.log(R)
    sigma, ell = 0.8, 0.9
    if sampler == "fft":
        g = np.stack([grf(rng, x, sigma, ell) for _ in range(2000)])
    else:
        g = rng.standard_normal((2000, len(x))) @ grf_factor(x, sigma, ell).T
    for frac in (0.0, 0.5, 1.0, 2.0):
        lag = int(round(frac * ell / DLOG))
        cov = np.mean(g[:, :len(x) - lag] * g[:, lag:])
        assert abs(cov - sigma**2 * np.exp(-0.5 * frac**2)) < 0.05 * sigma**2, (sampler, frac, cov)


def test_skin_zone_and_clipping():
    rng = np.random.default_rng(1)
    metas = []
    for _ in range(300):
        logk, meta = sample_logk(rng, R)
        metas.append(meta)
        assert np.all(np.abs(logk) <= LOGK_CLIP)
        if meta["has_skin"]:
            assert 2.0 <= meta["r_s"] <= 20.0 and np.log(0.1) <= meta["skin_log"] <= np.log(5.0)
        else:
            assert np.isnan(meta["r_s"]) and meta["skin_log"] == 0.0
    assert 0.4 < np.mean([m["has_skin"] for m in metas]) < 0.6


def test_skin_zone_is_a_step_of_the_right_size():
    """Without the GRF (sigma -> 0 via a constant field) the step across r_s is ln(k_s / k)."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        logk, meta = sample_logk(rng, R, p_skin=1.0)
        j = np.searchsorted(R, meta["r_s"])
        jump = logk[j - 1] - logk[j]
        # the GRF changes by at most ~sigma * D / ell per grid step; clipping can shorten the step
        if np.all(np.abs(logk[j - 1:j + 1]) < LOGK_CLIP):
            assert abs(jump - meta["skin_log"]) < 0.05


def test_labels_homogeneous():
    k_eff, s = labels(R, np.full_like(R, 2.5))
    assert np.isclose(k_eff, 2.5) and abs(s) < 1e-12


@pytest.mark.parametrize("k_ratio, r_s", [(0.1, 10.0), (0.5, 3.0), (5.0, 20.0)])
def test_labels_match_hawkins(k_ratio, r_s):
    """Formation k with a skin zone k_s = k_ratio k for r < r_s: S = (k / k_s - 1) ln r_s (Hawkins)."""
    k = np.where(R < r_s, 2.0 * k_ratio, 2.0)
    k_eff, s = labels(R, k, r_s)
    assert np.isclose(k_eff, 2.0, rtol=1e-6)
    # the discrete step sits between two nodes: the jump location is uncertain by one grid spacing
    tol = abs(1 / k_ratio - 1) * DLOG
    assert abs(s - (1 / k_ratio - 1) * np.log(r_s)) < tol


def test_labels_scale_with_k_and_zero_skin_without_zone():
    rng = np.random.default_rng(5)
    logk, meta = sample_logk(rng, R, p_skin=0.0)
    k = np.exp(logk)
    k_eff, s = labels(R, k)
    k_eff3, s3 = labels(R, 3 * k)
    assert np.isclose(k_eff3, 3 * k_eff) and np.isclose(s3, s)
    assert abs(s) < 1e-10                           # k_eff is defined over the whole profile without a skin zone


def test_resistance_is_additive():
    k = np.exp(np.sin(np.log(R)))
    assert np.isclose(resistance(R, k, 1.0, 7.3) + resistance(R, k, 7.3, 1000.0), resistance(R, k, 1.0, 1000.0))
