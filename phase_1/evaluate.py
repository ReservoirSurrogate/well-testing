"""Evaluate trained variants on the test split (Milestone 5).

For each run in <runs>/<variant>/best.pt:
- one-step errors on every test pair (u_n, log k) -> u_{n+1}, per n:
    rel. L2 on the grid (ln r measure), rel. L2 with weight r dr, the "copy the input" baseline,
    and the error relative to the true step change |u_{n+1} - u_n|;
- rollouts from u = 1 over all stored steps: rel. L2 of u and relative error of the well rate q_D, per n;
- the k_D = 1 rollout against the exact eigen-series;
- summaries split by skin / no skin.

Usage: python evaluate.py [--variants hankel logsine ...] [--threads 4]
Writes: <runs>/<variant>/eval.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from basis import eigenvalues, exact_solution, make_grid
from model import build_model
from solver import transmissibility
from train import HERE, MODES, open_split


def load_model(run_dir, n_points, r_e):
    ckpt = torch.load(Path(run_dir) / "best.pt")
    c = ckpt["config"]
    model = build_model(c["variant"], n_modes=c["n_modes"], width=c["width"], n_layers=c["n_layers"],
                        n_points=n_points, r_e=r_e, residual=c.get("residual", False))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def norms(e, w):
    """Grid (ln r measure) and r-weighted L2 norms over the last axis."""
    return np.linalg.norm(e, axis=-1), np.sqrt((e**2 * w).sum(-1))


def well_rate(r, logk, u):
    """q_D = F_{1/2} from the first two nodes; logk (B, N), u (..., B, N)."""
    k = np.exp(logk[:, :2])
    trans = transmissibility(r[:2], np.ones(2))[0] * 2 * k[:, 0] * k[:, 1] / (k[:, 0] + k[:, 1])
    return trans * (u[..., 1] - u[..., 0])


@torch.no_grad()
def predict(model, u, logk, chunk=200):
    x = torch.from_numpy(np.stack([u, logk], axis=1).astype(np.float32))
    return torch.cat([model(x[i:i + chunk]) for i in range(0, len(x), chunk)]).double().numpy()


def rollout(model, u0, logk, n_steps):
    """States from u0 by repeated application; shape (n_steps + 1, B, N)."""
    out = [u0.astype(float)]
    for _ in range(n_steps):
        out.append(predict(model, out[-1], logk))
    return np.stack(out)


def evaluate(run_dir, data_dir=HERE / "data"):
    u, logk, r, r_e = open_split(data_dir, "test")
    with np.load(Path(data_dir) / "test.npz") as f:
        has_skin = f["has_skin"]
    _, w = make_grid(len(r), r_e)
    logk = logk.astype(float)
    n_traj, n_steps = u.shape[0], u.shape[1] - 1
    model, ckpt = load_model(run_dir, len(r), r_e)

    # one-step errors, per n
    one = {k: np.zeros((n_steps, n_traj)) for k in ("grid", "rw", "persist_grid", "persist_rw", "skill")}
    for n in range(n_steps):
        un, un1 = u[:, n].astype(float), u[:, n + 1].astype(float)
        pred = predict(model, un, logk)
        eg, er = norms(pred - un1, w)
        tg, tr = norms(un1, w)
        pg, pr = norms(un - un1, w)
        one["grid"][n], one["rw"][n] = eg / tg, er / tr
        one["persist_grid"][n], one["persist_rw"][n] = pg / tg, pr / tr
        one["skill"][n] = eg / np.maximum(pg, 1e-30)

    # rollouts from u = 1
    true = np.moveaxis(np.asarray(u, dtype=float), 1, 0)            # (n_states, n_traj, N)
    roll = rollout(model, true[0], logk, n_steps)
    eg, er = norms(roll - true, w)
    tg, tr = norms(true, w)
    q_true, q_pred = well_rate(r, logk, true), well_rate(r, logk, roll)
    roll_err = {"grid": (eg / np.maximum(tg, 1e-30))[1:], "rw": (er / np.maximum(tr, 1e-30))[1:],
                "q": (np.abs(q_pred - q_true) / q_true)[1:]}

    # k_D = 1 rollout against the exact series
    lam = eigenvalues(4000, r_e)
    t = 1000.0 * np.arange(1, n_steps + 1)
    exact = np.stack([exact_solution(r, ti, lam, r_e) for ti in t])
    exact[:, 0] = 0.0
    zero = np.zeros((1, len(r)))
    roll1 = rollout(model, np.r_[0.0, np.ones(len(r) - 1)][None], zero, n_steps)[1:, 0]
    homog = np.linalg.norm(roll1 - exact, axis=-1) / np.linalg.norm(exact, axis=-1)

    def summary(a, mask=slice(None)):
        return {"mean": float(np.mean(a[:, mask])), "median": float(np.median(a[:, mask])),
                "p90": float(np.percentile(a[:, mask], 90))}

    res = {
        "variant": ckpt["config"]["variant"], "best_step": ckpt["step"], "n_params": ckpt["config"]["n_params"],
        "one_step": {k: summary(v) for k, v in one.items()},
        "one_step_first": {k: float(np.mean(v[0])) for k, v in one.items()},
        "one_step_per_n": {k: np.median(v, axis=1).tolist() for k, v in one.items()},
        "rollout": {k: summary(v) for k, v in roll_err.items()},
        "rollout_final": {k: summary(v[-1:]) for k, v in roll_err.items()},
        "rollout_skin": {k: summary(v, has_skin) for k, v in roll_err.items()},
        "rollout_noskin": {k: summary(v, ~has_skin) for k, v in roll_err.items()},
        "rollout_per_n": {k: {"median": np.median(v, axis=1).tolist(), "p90": np.percentile(v, 90, axis=1).tolist()}
                          for k, v in roll_err.items()},
        "homogeneous_rollout": {"per_n": homog.tolist(), "final": float(homog[-1]), "max": float(homog.max())},
    }
    (Path(run_dir) / "eval.json").write_text(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", nargs="+", default=sorted(MODES))
    ap.add_argument("--runs", type=Path, default=HERE / "runs")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    print(f"{'variant':<13} {'1-step':>9} {'copy':>9} {'skill':>7} {'roll u':>9} {'roll q':>9} "
          f"{'final u':>9} {'final q':>9} {'k=1 final':>9}")
    for v in args.variants:
        res = evaluate(args.runs / v)
        o, ro, rf = res["one_step"], res["rollout"], res["rollout_final"]
        print(f"{v:<13} {o['grid']['median']:9.2e} {o['persist_grid']['median']:9.2e} {o['skill']['median']:7.2f} "
              f"{ro['grid']['median']:9.2e} {ro['q']['median']:9.2e} {rf['grid']['median']:9.2e} "
              f"{rf['q']['median']:9.2e} {res['homogeneous_rollout']['final']:9.2e}", flush=True)


if __name__ == "__main__":
    main()
