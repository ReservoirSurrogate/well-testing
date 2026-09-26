"""Coverage helpers: masked (NaN) targets are excluded, not counted as misses."""
import numpy as np

from calibrate import fit_scales, inside


def test_inside_ignores_masked_entries():
    z = np.array([[0.5, np.nan], [3.0, 1.0], [1.0, np.nan], [np.nan, 0.2]])
    cov = inside(z, np.ones(2), 0.9545)                      # half-width 2
    assert np.allclose(cov, [2 / 3, 1.0])


def test_fit_scales_hits_the_target_coverage():
    rng = np.random.default_rng(0)
    z = np.abs(rng.standard_normal((20000, 2))) * np.array([1.0, 1.5])
    z[::3, 1] = np.nan
    s = fit_scales(z)
    assert np.allclose(s, [1.0, 1.5], rtol=0.03)
    assert np.allclose(inside(z, s, 0.9545), 0.9545, atol=0.005)
