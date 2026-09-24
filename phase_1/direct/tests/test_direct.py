"""Milestone 5b tests: direct model, pair sampling, training and evaluation smoke tests."""
import json

import numpy as np
import pytest
import torch

from dataset import generate_split, save_split
from direct_evaluate import evaluate
from direct_model import DT_MAX, DT_MIN, build_direct, time_feature
from direct_train import sample_pairs, train

VARIANTS = ("hankel", "hankel_local", "logsine", "dual", "fno")


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    for split, seed in [("train", 0), ("val", 1), ("test", 2)]:
        save_split(d / f"{split}.npz", generate_split(4, seed=seed, n_points=64, n_steps=3))
    return d


def test_time_feature_endpoints():
    tau = time_feature(torch.tensor([DT_MIN, DT_MAX]))
    assert torch.allclose(tau, torch.tensor([0.0, 1.0]), atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_model_shape_and_boundary(variant):
    torch.manual_seed(0)
    model = build_direct(variant, n_modes=4, width=8, n_layers=2, n_points=64)
    x = torch.randn(3, 2, 64)
    y = model(x, torch.tensor([1e3, 1e4, 5e5]))
    assert y.shape == (3, 64)
    assert torch.all(y[:, 0] == 0)


def test_output_depends_on_dt():
    torch.manual_seed(0)
    model = build_direct("logsine", n_modes=4, width=8, n_layers=2, n_points=64)
    x = torch.randn(1, 2, 64).expand(2, -1, -1)
    y = model(x, torch.tensor([1e3, 5e5]))
    assert not torch.allclose(y[0], y[1])


def test_film_starts_as_identity():
    model = build_direct("logsine", n_modes=4, width=8, n_layers=2, n_points=64)
    film = model.time(torch.rand(5, 1))
    assert torch.all(film == 0)


def test_sample_pairs():
    rng = np.random.default_rng(0)
    traj, i, j = sample_pairs(rng, n_traj=10, n_steps=500, size=40000, p_start=0.5)
    m = j - i
    assert traj.min() >= 0 and traj.max() < 10
    assert i.min() >= 0 and j.max() <= 500 and m.min() >= 1
    assert 0.48 < np.mean(i == 0) < 0.53
    # log-uniform gap: about half the gaps below sqrt(501) ~ 22
    assert 0.45 < np.mean(m <= 22) < 0.55


def test_sample_pairs_fixed_gap():
    traj, i, j = sample_pairs(np.random.default_rng(0), n_traj=10, n_steps=500, size=100, p_start=0.5, gap=1)
    assert np.all(i == 0) and np.all(j == 1)


def test_train_and_evaluate_fixed_gap(tiny_data, tmp_path):
    train("logsine", steps=4, batch_size=4, n_modes=4, width=8, n_layers=2, eval_every=2,
          data_dir=tiny_data, out=tmp_path, gap=1, log=lambda s: None)
    res = evaluate(tmp_path / "logsine", data_dir=tiny_data)
    assert res["t"] == [1000.0]
    assert len(res["per_time"]["grid"]["median"]) == 1 and len(res["homogeneous"]["per_time"]) == 1
    assert res["semigroup"] == {}


@pytest.mark.parametrize("variant", ("logsine", "dual", "fno", "fftlog_decay"))
def test_train_and_evaluate_run(tiny_data, tmp_path, variant):
    best = train(variant, steps=4, batch_size=4, n_modes=4, width=8, n_layers=2, eval_every=2,
                 data_dir=tiny_data, out=tmp_path, n_val=8, log=lambda s: None)
    run = tmp_path / variant
    assert np.isfinite(best)
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    assert json.loads((run / "config.json").read_text())["variant"] == variant
    res = evaluate(run, data_dir=tiny_data)
    assert len(res["t"]) == 3
    assert np.isfinite(res["all_times"]["grid"]["median"])
    assert "1+1" in res["semigroup"]
    assert (run / "eval.json").exists()
