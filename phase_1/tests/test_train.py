"""Milestone 5 smoke tests: pair sampling and a few training steps on a tiny dataset."""
import json

import numpy as np
import pytest
import torch

from dataset import generate_split, save_split
from evaluate import evaluate
from train import MODES, open_split, rate_error, sample_pairs, train


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    for split, seed in [("train", 0), ("val", 1), ("test", 2)]:
        save_split(d / f"{split}.npz", generate_split(4, seed=seed, n_points=64, n_steps=3))
    return d


def test_sample_pairs_ranges_and_first_fraction():
    rng = np.random.default_rng(0)
    traj, n = sample_pairs(rng, n_traj=10, n_steps=500, size=20000, p_first=0.05)
    assert traj.min() >= 0 and traj.max() < 10
    assert n.min() >= 0 and n.max() < 500
    assert 0.04 < np.mean(n == 0) < 0.06


def test_open_split_memmap(tiny_data):
    u, logk, r, r_e = open_split(tiny_data, "train")
    assert isinstance(u, np.memmap) and u.shape == (4, 4, 64)
    assert logk.shape == (4, 64) and len(r) == 64 and r_e == 1000.0


@pytest.mark.parametrize("variant", sorted(MODES))
def test_train_runs(tiny_data, tmp_path, variant):
    best = train(variant, steps=4, batch_size=4, n_modes=4, width=8, n_layers=2, eval_every=2,
                 data_dir=tiny_data, out=tmp_path, n_val=8, log=lambda s: None)
    run = tmp_path / variant
    assert np.isfinite(best)
    assert {"best.pt", "last.pt", "log.csv", "config.json"} <= {p.name for p in run.iterdir()}
    assert json.loads((run / "config.json").read_text())["variant"] == variant
    ckpt = torch.load(run / "best.pt")
    assert ckpt["val"] == pytest.approx(best)


def test_train_residual_with_rate_loss(tiny_data, tmp_path):
    best = train("logsine", steps=4, batch_size=4, n_modes=4, width=8, n_layers=2, eval_every=2,
                 residual=True, q_weight=0.1, data_dir=tiny_data, out=tmp_path, n_val=8, log=lambda s: None)
    ckpt = torch.load(tmp_path / "logsine" / "best.pt")
    assert ckpt["config"]["residual"] and ckpt["config"]["q_weight"] == 0.1
    assert best == pytest.approx(ckpt["val"] + 0.1 * ckpt["val_q"])
    res = evaluate(tmp_path / "logsine", data_dir=tiny_data)      # rebuilds the residual model
    assert np.isfinite(res["one_step"]["grid"]["median"])


def test_rate_error_matches_well_rate():
    from evaluate import well_rate
    rng = np.random.default_rng(0)
    r = np.linspace(1.0, 2.0, 8)
    logk = rng.normal(size=(3, 8))
    y = rng.random((3, 8)); y[:, 0] = 0.0
    p = y + 0.01 * rng.random((3, 8)); p[:, 0] = 0.0
    q_true, q_pred = well_rate(r, logk, y), well_rate(r, logk, p)
    expected = np.abs(q_pred - q_true) / q_true
    assert np.allclose(rate_error(torch.from_numpy(p), torch.from_numpy(y)).numpy(), expected)


def test_evaluate_runs(tiny_data, tmp_path):
    train("logsine", steps=2, batch_size=4, n_modes=4, width=8, n_layers=2, eval_every=2,
          data_dir=tiny_data, out=tmp_path, n_val=8, log=lambda s: None)
    res = evaluate(tmp_path / "logsine", data_dir=tiny_data)
    assert (tmp_path / "logsine" / "eval.json").exists()
    assert len(res["one_step_per_n"]["grid"]) == 3
    assert len(res["rollout_per_n"]["q"]["median"]) == 3
    assert np.isfinite(res["homogeneous_rollout"]["final"])
    # the copy-the-input baseline is poor on the first step: u = 1 is far from u(1000)
    assert res["one_step_first"]["persist_grid"] > 0.1
