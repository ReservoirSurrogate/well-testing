"""Milestone 4: dataset generation on a tiny split."""
import numpy as np
import pytest

from dataset import C_D_RANGE, N_MAX, R_E_RANGE, T, generate_split, grid, save_split
from permeability import labels
from wells import Drawdown, buildup

N = 12


@pytest.fixture(scope="module")
def data():
    return generate_split(N, seed=123, workers=1)


def test_independent_of_workers(data):
    other = generate_split(N, seed=123, workers=2)
    for key in ("p_w", "r_e", "c_d", "k_eff", "skin"):
        assert np.array_equal(data[key], other[key]), key
    assert np.array_equal(data["logk"], other["logk"], equal_nan=True)


def test_shapes_and_ranges(data):
    assert data["t"].shape == (200,) and np.allclose(data["t"], T)
    assert data["p_w"].shape == (N, 200) and data["p_w"].dtype == np.float64
    assert data["logk"].shape == (N, N_MAX) and data["logk"].dtype == np.float32
    assert np.all((data["r_e"] >= R_E_RANGE[0]) & (data["r_e"] <= R_E_RANGE[1]))
    c = data["c_d"]
    assert np.all((c == 0) | ((c >= C_D_RANGE[0]) & (c <= C_D_RANGE[1])))
    for i in range(N):
        n_r = data["n_r"][i]
        assert np.all(np.isfinite(data["logk"][i, :n_r])) and np.all(np.isnan(data["logk"][i, n_r:]))
        r = grid(data, i)
        assert np.isclose(r[0], 1.0) and np.isclose(r[-1], data["r_e"][i])


def test_drawdown_is_increasing_and_positive(data):
    assert np.all(data["p_w"] > 0)
    assert np.all(np.diff(data["p_w"], axis=1) > 0)


def test_labels_recomputed_from_stored_fields(data):
    for i in range(N):
        r = grid(data, i)
        k = np.exp(data["logk"][i, :data["n_r"][i]].astype(float))
        k_eff, s = labels(r, k, data["r_s"][i])
        # logk is stored as float32: the labels agree to float32 precision
        assert np.isclose(k_eff, data["k_eff"][i], rtol=1e-5) and abs(s - data["skin"][i]) < 1e-4


def test_buildups_form_from_the_stored_drawdown(data):
    dd = Drawdown(data["t"], data["p_w"][0])
    p_w, dp = buildup(dd, 1e4, np.logspace(-2, 6, 30))
    # the build-up rises monotonically; in a closed reservoir it levels off at the average pressure, where
    # successive values differ only by the spline's interpolation noise (~1e-7 relative)
    assert np.all(np.isfinite(p_w)) and np.all(np.diff(dp) >= -1e-6 * np.max(dp))


def test_save_and_load(tmp_path, data):
    save_split(tmp_path / "x.npz", data)
    with np.load(tmp_path / "x.npz") as f:
        assert np.array_equal(f["p_w"], data["p_w"]) and f["seed"] == 123
