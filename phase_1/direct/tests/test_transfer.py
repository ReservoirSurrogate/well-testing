"""Transfer to another r_eD: FFTLog control points in physical k, loading trained weights on a new grid."""
import numpy as np
import pytest
import torch

from direct_evaluate import load_model
from direct_model import build_direct
from fftlog_conv import FFTLogConv
from basis import make_grid

D = np.log(1000.0) / 1023
R_TRAIN, _ = make_grid(1024, 1000.0)
N_NEW = 845
RE_NEW = float(np.exp((N_NEW - 1) * D))


def mix(conv):
    """R(k) as (n_k, C, C) numpy array."""
    return torch.einsum("km,mcd->kcd", conv.interp, conv.weight).detach().numpy()


def test_default_k_range_reproduces_index_interpolation():
    """Placing control points by physical k over the grid's own range equals the original index-based placement."""
    conv = FFTLogConv(R_TRAIN, 2, n_modes=8)
    n_tot = len(conv.k)
    pos = np.linspace(0, 7, n_tot)
    lo = np.minimum(np.floor(pos).astype(int), 6)
    old = np.zeros((n_tot, 8))
    old[np.arange(n_tot), lo] = 1 - (pos - lo)
    old[np.arange(n_tot), lo + 1] = pos - lo
    assert np.allclose(conv.interp.numpy(), old, atol=1e-5)


def test_k_range_keeps_r_of_physical_k():
    """On another grid with the training k_range, R(k) is the same function of physical k (constant outside)."""
    torch.manual_seed(0)
    train = FFTLogConv(R_TRAIN, 2, n_modes=8)
    r_new, _ = make_grid(N_NEW, RE_NEW)
    new = FFTLogConv(r_new, 2, n_modes=8, k_range=train.k_range)
    with torch.no_grad():
        new.weight.copy_(train.weight)
    lk_train, lk_new = np.log(train.k.numpy()), np.log(new.k.numpy())
    r_train, r_new_mix = mix(train), mix(new)
    for c, d in [(0, 0), (0, 1), (1, 0)]:
        ref = np.interp(lk_new, lk_train, r_train[:, c, d])        # constant beyond the ends, like the clip
        assert np.allclose(r_new_mix[:, c, d], ref, atol=1e-5)


@pytest.mark.parametrize("variant", ["fno", "fftlog", "logsine"])
def test_load_model_on_another_grid(tmp_path, variant):
    torch.manual_seed(0)
    model = build_direct(variant, n_modes=8, width=8, n_layers=2)
    config = dict(variant=variant, n_modes=8, width=8, n_layers=2, n_params=0)
    torch.save({"config": config, "state_dict": model.state_dict(), "step": 1}, tmp_path / "best.pt")

    new, _ = load_model(tmp_path, N_NEW, RE_NEW)
    for name, p in new.named_parameters():
        assert torch.equal(p, dict(model.named_parameters())[name])
    r_new, _ = make_grid(N_NEW, RE_NEW)
    assert np.allclose(new.s.numpy(), np.log(r_new) / np.log(1000.0), atol=1e-6)   # physical s, not ln r / ln r_e
    y = new(torch.randn(2, 2, N_NEW), torch.tensor([1e3, 1e5]))
    assert y.shape == (2, N_NEW) and torch.all(y[:, 0] == 0) and torch.isfinite(y).all()


def test_load_model_on_training_grid_is_unchanged(tmp_path):
    torch.manual_seed(0)
    model = build_direct("fftlog", n_modes=8, width=8, n_layers=2)
    config = dict(variant="fftlog", n_modes=8, width=8, n_layers=2, n_params=0)
    torch.save({"config": config, "state_dict": model.state_dict(), "step": 1}, tmp_path / "best.pt")
    same, _ = load_model(tmp_path, 1024, 1000.0)
    x, dt = torch.randn(2, 2, 1024), torch.tensor([1e3, 1e5])
    assert torch.allclose(same(x, dt), model(x, dt))
