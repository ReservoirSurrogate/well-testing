"""Milestone 6: inverse model shapes, masking, loss and a short training run on a tiny dataset."""
import numpy as np
import pytest
import torch

from dataset import generate_split, save_split
from examples import C_IN, G, N_TARGET
from model import InverseFNO, loss_terms
from train import train


def test_shapes_and_masked_pooling():
    torch.manual_seed(0)
    model = InverseFNO(width=8, modes=6, n_layers=2, head=16)
    x = torch.randn(3, C_IN, G)
    x[:, 6] = 1.0
    x[1, 6, 60:] = 0.0                                   # truncated test
    mu, log_var, logit = model(x)
    assert mu.shape == (3, N_TARGET) and log_var.shape == (3, N_TARGET) and logit.shape == (3,)
    # the measured curves (channels 0, 1, 7) in the masked tail of example 1 do not affect the output
    x2 = x.clone()
    x2[1, [0, 1, 7], 60:] = torch.randn(3, G - 60)
    assert torch.allclose(model(x2)[0][1], mu[1], atol=1e-5)


def test_loss_ignores_masked_targets():
    mu = torch.zeros(2, N_TARGET)
    log_var = torch.zeros(2, N_TARGET)
    y = torch.zeros(2, N_TARGET)
    y[0, 3] = float("nan")
    nll, bce = loss_terms(mu, log_var, torch.zeros(2), y, torch.tensor([0.0, 1.0]), beta=0.0)
    assert torch.isfinite(nll) and abs(nll.item()) < 1e-6
    assert np.isclose(bce.item(), np.log(2), atol=1e-6)


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    for split, seed in [("train", 0), ("val", 1)]:
        save_split(d / f"{split}.npz", generate_split(6, seed=seed))
    return d


def test_train_runs(tiny_data, tmp_path):
    best = train(steps=6, batch=4, width=8, modes=6, n_layers=2, threads=1, eval_every=3,
                 data_dir=tiny_data, out=tmp_path, log=lambda s: None)
    assert np.isfinite(best)
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
