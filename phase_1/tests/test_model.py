"""Milestone 4 tests: spectral bases and neural-operator variants."""
import numpy as np
import pytest
import torch

from basis import eigenvalues, exact_solution, make_grid
from model import (VARIANTS, SpectralConv, build_model, hankel_pair, log_coordinate, logsine_pair,
                   n_params)

R_E = 1000.0
N = 1024
K = 32


@pytest.mark.parametrize("pair", [hankel_pair, logsine_pair])
def test_forward_inverse_identity(pair):
    T, B = pair(K)
    assert T.shape == (K, N) and B.shape == (N, K)
    assert np.allclose(T @ B, np.eye(K), atol=1e-10)


def test_logsine_boundary_conditions():
    _, B = logsine_pair(K)
    assert np.allclose(B[0], 0.0, atol=1e-14)                    # u(1) = 0
    r, _ = make_grid(N, R_E)
    s = log_coordinate(r)
    slope = (B[-1] - B[-2]) / (s[-1] - s[-2])                    # du/ds at r_e
    assert np.all(np.abs(slope) < 0.1 * np.pi * (np.arange(K) + 0.5))


def test_hankel_layer_reproduces_exact_step():
    # k_D = 1: one spectral layer with R_n = exp(-lambda_n^2 dt) is the exact step (tex eq. diag).
    torch.set_default_dtype(torch.float64)
    try:
        dt, t = 1000.0, 1000.0
        T, B = hankel_pair(K)
        conv = SpectralConv(T, B, width=1)
        lam = eigenvalues(K, R_E)
        with torch.no_grad():
            conv.weight.copy_(torch.as_tensor(np.exp(-lam**2 * dt))[:, None, None])
        r, _ = make_grid(N, R_E)
        lam_ref = eigenvalues(4000, R_E)
        u0, u1 = exact_solution(r, t, lam_ref, R_E), exact_solution(r, t + dt, lam_ref, R_E)
        pred = conv(torch.as_tensor(u0)[None, None])[0, 0].detach().numpy()
        assert np.linalg.norm(pred - u1) / np.linalg.norm(u1) < 1e-5
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.fixture(scope="module", params=VARIANTS)
def model(request):
    torch.manual_seed(0)
    return build_model(request.param, n_modes=K, width=16, n_layers=2)


def test_output_shape_and_hard_bc(model):
    x = torch.rand(3, 2, N)
    y = model(x)
    assert y.shape == (3, N)
    assert torch.all(y[:, 0] == 0.0)


def test_backward_reaches_all_parameters(model):
    model.zero_grad()
    model(torch.rand(2, 2, N)).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert n_params(model) > 0


def test_unknown_variant():
    with pytest.raises(ValueError):
        build_model("wavelet")


@pytest.mark.parametrize("variant", VARIANTS)
def test_residual_zero_update_copies_input(variant):
    torch.manual_seed(0)
    m = build_model(variant, n_modes=K, width=16, n_layers=2, residual=True)
    with torch.no_grad():
        m.proj[-1].weight.zero_()
        m.proj[-1].bias.zero_()
    x = torch.rand(2, 2, N)
    y = m(x)
    assert torch.all(y[:, 0] == 0.0)
    assert torch.equal(y[:, 1:], x[:, 0, 1:])
