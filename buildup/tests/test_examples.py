"""Milestone 6: build-up examples formed on the fly (vectorized spline, superposition, features, masks, labels)."""
import numpy as np
import pytest

from dataset import generate_split, grid
from examples import C_IN, DT, G, N_RING, N_TARGET, Bank, denormalize
from permeability import ring_labels
from wells import Drawdown, buildup

N = 8


@pytest.fixture(scope="module")
def data():
    return generate_split(N, seed=321, workers=1)


@pytest.fixture(scope="module")
def bank(data):
    return Bank(data)


def test_vectorized_spline_matches_drawdown(data, bank):
    t = np.geomspace(1e-2, 1e7, 57)
    got = bank.drawdown(np.arange(N), np.broadcast_to(t, (N, 57)))
    for i in range(N):
        assert np.allclose(got[i], Drawdown(data["t"], data["p_w"][i])(t), rtol=1e-12, atol=1e-14)


def test_noise_free_buildup_matches_superposition(data, bank):
    rng = np.random.default_rng(0)
    x, _, _, meta = bank.make(np.arange(N), rng, t_p=3e4, noise=0.0, p_trunc=0.0, p_no_pi=0.0)
    for i in range(N):
        _, dp = buildup(Drawdown(data["t"], data["p_w"][i]), 3e4, DT)
        assert np.allclose(np.exp(5 * x[i, 0].astype(float)), dp, rtol=1e-5)
        assert np.isclose(meta["p_wf"][i], Drawdown(data["t"], data["p_w"][i])(3e4))


def test_shapes_masks_and_flags(bank):
    rng = np.random.default_rng(1)
    idx = np.array([0, 3, 5, 5, 7])
    x, y, ns, meta = bank.make(idx, rng, p_trunc=1.0, p_no_pi=0.5)
    assert x.shape == (5, C_IN, G) and x.dtype == np.float32
    assert y.shape == (5, N_TARGET) and ns.shape == (5,)
    for b in range(5):
        valid = DT <= meta["dt_max"][b]
        assert np.array_equal(x[b, 6] > 0, valid)
        assert np.all(x[b, [0, 1, 7]][:, ~valid] == 0) and np.all(x[b, 2] != 0)
        assert x[b, 5, 0] == float(meta["pi_known"][b])
        if not meta["pi_known"][b]:
            assert np.all(x[b, 4] == 0) and np.all(x[b, 7] == 0)
    assert np.all(np.isfinite(x))


def test_labels(data, bank):
    rings, ln_c, ln_re = denormalize(np.nan_to_num(bank.targets, nan=0.0))
    for i in range(N):
        ref = ring_labels(grid(data, i), np.exp(data["logk"][i, :data["n_r"][i]].astype(float)))
        assert np.allclose(np.isnan(bank.targets[i, :N_RING]), np.isnan(ref))
        assert np.allclose(rings[i][~np.isnan(ref)], ref[~np.isnan(ref)], atol=1e-5)
        assert np.isclose(np.exp(ln_re[i]), data["r_e"][i], rtol=1e-5)
        if data["c_d"][i] > 0:
            assert np.isclose(np.exp(ln_c[i]), data["c_d"][i], rtol=1e-5)
        else:
            assert np.isnan(bank.targets[i, N_RING]) and bank.no_storage[i] == 1


def test_reproducible(bank):
    a = bank.make(np.arange(N), np.random.default_rng(5))[0]
    b = bank.make(np.arange(N), np.random.default_rng(5))[0]
    assert np.array_equal(a, b)


def test_material_balance_channel_converges_to_r_e(data, bank):
    """Noise-free, full test, p_i known: the apparent radius rises and, once the build-up has stabilized,
    equals r_eD (up to the neglected C_D)."""
    from examples import RE_NORM
    rng = np.random.default_rng(2)
    x, _, _, _ = bank.make(np.arange(N), rng, t_p=1e6, noise=0.0, p_trunc=0.0, p_no_pi=0.0)
    for i in range(N):
        r_app = np.exp(x[i, 7].astype(float) * RE_NORM[1] + RE_NORM[0])
        stabilized = abs(x[i, 1, -1]) < 1e-3
        if stabilized:
            vol = (data["r_e"][i] ** 2 - 1) / 2
            expected = np.sqrt(2 * vol * (1 + data["c_d"][i] / vol) + 1)       # C_D neglected in the channel
            assert abs(r_app[-1] / expected - 1) < 0.01
        assert r_app[-1] <= data["r_e"][i] * 1.01


def test_noise_level_channel(bank):
    from examples import NOISE_RANGE
    rng = np.random.default_rng(4)
    x, _, _, meta = bank.make(np.arange(N), rng)                       # noise drawn per example
    assert np.all((meta["noise"] >= NOISE_RANGE[0]) & (meta["noise"] <= NOISE_RANGE[1]))
    assert np.allclose(x[:, 8, 0], np.log10(meta["noise"]) + 3, atol=1e-6)
    assert np.all(x[:, 8] == x[:, 8, :1])                              # constant along the curve
    x2, _, _, meta2 = bank.make(np.arange(N), rng, noise=1e-3)
    assert np.allclose(meta2["noise"], 1e-3) and np.allclose(x2[:, 8], 0.0, atol=1e-6)


def test_old_models_read_the_first_channels():
    import torch
    from model import InverseFNO
    torch.manual_seed(0)
    old = InverseFNO(width=8, modes=6, n_layers=2, head=16, c_in=8)
    x = torch.randn(2, C_IN, G)
    x[:, 6] = 1.0
    assert torch.allclose(old(x)[0], old(x[:, :8])[0])
