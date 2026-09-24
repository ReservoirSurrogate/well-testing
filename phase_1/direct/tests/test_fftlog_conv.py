"""FFTLogConv tests: agreement with the NumPy FFTLog, exact inverse, identity multiplier, 2-D heat kernel."""
import numpy as np
import pytest
import torch

from basis import make_grid
from direct_model import build_direct
from fftlog import FFTLog
from fftlog_conv import FFTLogConv

R_GRID, _ = make_grid(1024, 1000.0)                  # the solver grid: geometric on [1, 1000]
R_WIDE = np.geomspace(1e-4, 1e4, 1024)               # for fields defined on (0, inf)


def identity_conv(r, decay=False, width=1, n_modes=2, pad=256):
    conv = FFTLogConv(r, width, n_modes, pad=pad, q=0.4, decay=decay)
    with torch.no_grad():
        conv.weight.copy_(torch.eye(width).expand(n_modes, width, width))
    return conv


def test_rejects_non_geometric_grid():
    with pytest.raises(ValueError):
        FFTLogConv(np.linspace(1, 1000, 64), 4)


def test_transform_matches_numpy():
    conv = FFTLogConv(R_GRID, 1, 4, q=0.4)
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((2, 1, 1024)), dtype=torch.float32)
    x_pad = conv.pad_taper(x)
    y = conv.transform(x_pad).double().numpy()[:, 0]
    t = FFTLog(len(conv.r_pad), np.log(R_GRID[1] / R_GRID[0]), mu=0.0, q=0.4)
    k, F = t.forward(x_pad.double().numpy()[:, 0], conv.r_pad[0].item())
    assert np.allclose(k, conv.k.numpy(), rtol=1e-5)
    ref = F * k ** 1.4
    assert np.max(np.abs(y - ref)) < 1e-5 * np.max(np.abs(ref))


def test_inverse_transform_round_trip_float32():
    conv = FFTLogConv(R_GRID, 1, 4, q=0.4)
    s = torch.tensor(np.log(R_GRID) / np.log(1000.0), dtype=torch.float32)[None, None]
    x_pad = conv.pad_taper(s)
    back = conv.inverse_transform(conv.transform(x_pad))
    err = (back - x_pad).abs()[..., conv.pad:conv.pad + 1024][0, 0]
    # float32 rounding ~1e-7 of max |r^{0.6} h| (r^{0.6} ~ 180 at the padded outer end), divided by r^{0.6} ~ 1 at
    # the well: ~2e-5 absolute. Acceptable: hidden channels are O(1), and the model output u = s v has s ~ 1e-3
    # at the first node. (With q = 0 the factor would be r ~ 5600, i.e. ~30x worse.)
    assert err.max() < 3e-5


def test_identity_multiplier_returns_input():
    """R(k) = 1 and no decay: H^{-1} H = identity on the interior (the taper only acts in the pad)."""
    conv = identity_conv(R_GRID)
    x = torch.tensor(np.sin(3 * np.log(R_GRID)) + 0.5, dtype=torch.float32)[None, None]
    y = conv(x)
    assert y.shape == x.shape
    assert (y - x).abs().max() < 3e-5                 # float32 rounding, see the round-trip test


@pytest.mark.parametrize("d_dt", [0.1, 0.5, 2.0])
def test_decay_is_the_2d_heat_kernel(d_dt):
    """R = 1 with decay: a Gaussian spreads like the heat kernel on the plane,
    exp(-r^2 / 2 s^2) -> s^2 / (s^2 + 2 D dt) exp(-r^2 / 2 (s^2 + 2 D dt))."""
    conv = identity_conv(R_WIDE, decay=True)
    d, dt, s2 = 1.0, d_dt, 1.0
    with torch.no_grad():
        conv.log_d.fill_(np.log(d))
    x = torch.tensor(np.exp(-R_WIDE**2 / (2 * s2)), dtype=torch.float32)[None, None]
    y = conv(x, torch.tensor([dt]))[0, 0].detach().double().numpy()
    s2t = s2 + 2 * d * dt
    exact = s2 / s2t * np.exp(-R_WIDE**2 / (2 * s2t))
    m = (R_WIDE > 1e-2) & (R_WIDE < 20)
    assert np.max(np.abs(y[m] - exact[m])) < 1e-4


def test_decay_depends_on_dt_and_channel():
    conv = identity_conv(R_WIDE, decay=True, width=2)
    x = torch.tensor(np.exp(-R_WIDE**2 / 2), dtype=torch.float32)[None, None].repeat(1, 2, 1)
    y1 = conv(x, torch.tensor([0.1]))
    y2 = conv(x, torch.tensor([1.0]))
    assert y2[0, 0].max() < y1[0, 0].max()            # later time: lower peak
    assert not torch.allclose(y1[0, 0], y1[0, 1])     # different diffusivity per channel


@pytest.mark.parametrize("variant", ["fftlog", "fftlog_decay"])
def test_model_variants(variant):
    torch.manual_seed(0)
    model = build_direct(variant, n_modes=8, width=8, n_layers=2, n_points=1024)
    x = torch.randn(3, 2, 1024)
    y = model(x, torch.tensor([1e3, 1e4, 5e5]))
    assert y.shape == (3, 1024) and torch.all(y[:, 0] == 0) and torch.isfinite(y).all()
    y.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
