"""Milestone 3 tests: random permeability and trajectory generation (small grids for speed)."""
import numpy as np
import pytest

from basis import eigenvalues, exact_solution, make_grid
from dataset import (LOGK_CLIP, META_KEYS, generate_split, grf, load_split, sample_logk,
                     save_split, trajectory)

R_E = 1000.0


# --- random log k ---------------------------------------------------------------

def test_grf_covariance():
    rng = np.random.default_rng(0)
    s = np.linspace(0.0, np.log(R_E), 128)
    sigma, ell = 0.8, 1.0
    g = grf(rng, s, sigma, ell, size=4000)
    lag = np.searchsorted(s, ell)                          # grid index nearest to s = l
    corr = np.mean(g[:, 0] * g[:, lag]) / sigma**2
    assert np.mean(g.var(axis=0)) == pytest.approx(sigma**2, rel=0.05)
    assert corr == pytest.approx(np.exp(-0.5 * (s[lag] / ell) ** 2), abs=0.05)


def test_sample_logk_ranges_and_meta():
    rng = np.random.default_rng(1)
    r, _ = make_grid(256, R_E)
    metas = []
    for _ in range(200):
        logk, meta = sample_logk(rng, r)
        assert logk.shape == r.shape
        assert np.all(np.abs(logk) <= LOGK_CLIP)
        assert set(meta) == set(META_KEYS)
        metas.append(meta)
    skin = [m for m in metas if m["has_skin"]]
    assert 0.35 < len(skin) / len(metas) < 0.65
    assert all(2.0 <= m["r_s"] <= 20.0 for m in skin)
    assert all(np.log(0.1) <= m["skin"] <= np.log(5.0) for m in skin)
    assert all(np.isnan(m["r_s"]) and m["skin"] == 0.0 for m in metas if not m["has_skin"])


def test_skin_offset_only_inside_r_s():
    r, _ = make_grid(256, R_E)

    def skin_sample(seed):
        logk, meta = sample_logk(np.random.default_rng(seed), r)
        j = np.searchsorted(r, meta["r_s"]) if meta["has_skin"] else 0
        # need a skin with a clear jump and no clipping next to r_s
        ok = meta["has_skin"] and abs(meta["skin"]) > 0.5 and np.abs(logk[j - 1:j + 1]).max() < LOGK_CLIP
        return ok, logk, meta, j

    ok, logk, meta, j = next(x for x in map(skin_sample, range(200)) if x[0])
    base = np.where(r < meta["r_s"], logk - meta["skin"], logk)
    # Removing the offset inside r_s must leave a smooth GRF: no jump at r_s.
    assert abs(base[j] - base[j - 1]) < 0.1
    assert logk[j - 1] - logk[j] == pytest.approx(meta["skin"], abs=0.1)


# --- trajectories -------------------------------------------------------------------

def test_trajectory_homogeneous_matches_exact():
    r, _ = make_grid(1024, R_E)
    u = trajectory(r, np.zeros_like(r), dt=1000.0, n_steps=3)
    assert u.shape == (4, len(r))
    assert u[0, 0] == 0.0 and np.all(u[0, 1:] == 1.0)
    lam = eigenvalues(4000, R_E)
    for n in (1, 2, 3):
        exact = exact_solution(r, 1000.0 * n, lam, R_E)
        exact[0] = 0.0
        assert np.linalg.norm(u[n] - exact) / np.linalg.norm(exact) < 5e-5


@pytest.fixture(scope="module")
def small_split():
    return generate_split(4, seed=7, n_points=64, n_steps=3)


def test_generate_split_shapes(small_split):
    d = small_split
    assert d["u"].shape == (4, 4, 64) and d["u"].dtype == np.float32
    assert d["logk"].shape == (4, 64) and d["logk"].dtype == np.float32
    assert np.array_equal(d["t"], [0.0, 1000.0, 2000.0, 3000.0])
    assert all(d[k].shape == (4,) for k in META_KEYS)
    assert np.all(d["u"][:, :, 0] == 0.0)
    assert d["u"].min() > -1e-6 and d["u"].max() <= 1.0


def test_generate_split_independent_of_workers(small_split):
    d2 = generate_split(4, seed=7, n_points=64, n_steps=3, workers=2)
    assert np.array_equal(small_split["u"], d2["u"])
    assert np.array_equal(small_split["logk"], d2["logk"])


def test_save_load_roundtrip(small_split, tmp_path):
    path = tmp_path / "split.npz"
    save_split(path, small_split)
    loaded = load_split(path)
    assert set(loaded) == set(small_split)
    for k, v in small_split.items():
        assert np.array_equal(loaded[k], v, equal_nan=np.issubdtype(np.asarray(v).dtype, np.floating))
