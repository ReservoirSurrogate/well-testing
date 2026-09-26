"""Evaluate trained direct-operator variants on the test split (Milestone 5b).

For each run in <runs>/<variant>/best.pt:
- one jump from u = 1 to every stored time t_j (only to t_gap for a model trained with --gap):
  rel. L2 of u (grid / ln r measure and weight r dr) and relative well-rate error, per j and summarized
  (all times, final time, skin / no skin);
- success bar at every time: u error < 2% and q_D error < 5% (median over test trajectories);
- k_D = 1 jumps against the exact eigen-series;
- semigroup consistency: G(G(1, dt_1), dt_2) against G(1, dt_1 + dt_2) and against the truth (skipped with --gap).

Usage: python direct_evaluate.py [--variants logsine dual fno] [--runs runs] [--threads 4]
Writes: <runs>/<variant>/eval.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from basis import eigenvalues, exact_solution, make_grid  # noqa: E402
from dataset import N_POINTS as TRAIN_POINTS  # noqa: E402
from dataset import R_E as TRAIN_RE  # noqa: E402
from direct_model import build_direct  # noqa: E402
from direct_train import DATA  # noqa: E402
from evaluate import norms, well_rate  # noqa: E402
from train import open_split  # noqa: E402

U_BAR, Q_BAR = 0.02, 0.05
SEMIGROUP_SPLITS = ((1, 1), (10, 10), (50, 450), (250, 250))   # (m_1, m_2) in stored steps


def load_model(run_dir, n_points, r_e):
    """Trained model on the grid (n_points, r_e).

    On a grid other than the training grid (transfer to another r_eD, same ln-r spacing), the model is rebuilt
    with the training grid's physical settings - s = ln r / ln TRAIN_RE, the fno pad length, the FFTLog control
    points at the training grid's physical k - and only the learned parameters are loaded (grid-dependent
    buffers are recomputed for the new grid).
    """
    ckpt = torch.load(Path(run_dir) / "best.pt")
    c = ckpt["config"]
    kw = dict(n_modes=c["n_modes"], width=c["width"], n_layers=c["n_layers"])
    if n_points == TRAIN_POINTS and np.isclose(r_e, TRAIN_RE):
        model = build_direct(c["variant"], n_points=n_points, r_e=r_e, **kw)
        model.load_state_dict(ckpt["state_dict"])
        return model.eval(), ckpt
    k_range = None
    if c["variant"].startswith("fftlog"):
        ref = build_direct(c["variant"], n_points=TRAIN_POINTS, r_e=TRAIN_RE, **kw)
        k_range = ref.layers[0].spectral[0].k_range
    model = build_direct(c["variant"], n_points=n_points, r_e=r_e, fft_pad=TRAIN_POINTS // 4, s_ref_re=TRAIN_RE,
                         fftlog_k_range=k_range, **kw)
    names = dict(model.named_parameters())
    params = {k: v for k, v in ckpt["state_dict"].items() if k in names}
    if set(params) != set(names):
        raise RuntimeError(f"parameters missing from checkpoint: {sorted(set(names) - set(params))}")
    model.load_state_dict(params, strict=False)
    return model.eval(), ckpt


@torch.no_grad()
def predict(model, u0, logk, dt, chunk=200):
    """u(dt) from u0; u0, logk (B, N), dt scalar or (B,)."""
    x = torch.from_numpy(np.stack([u0, logk], axis=1).astype(np.float32))
    dt = torch.from_numpy(np.broadcast_to(np.asarray(dt, dtype=np.float32), (len(x),)).copy())
    return torch.cat([model(x[i:i + chunk], dt[i:i + chunk]) for i in range(0, len(x), chunk)]).double().numpy()


def evaluate(run_dir, data_dir=DATA, out_name="eval.json"):
    u, logk, r, r_e = open_split(data_dir, "test")
    with np.load(Path(data_dir) / "test.npz") as f:
        has_skin, t = f["has_skin"], f["t"].astype(float)
    _, w = make_grid(len(r), r_e)
    logk = logk.astype(float)
    n_steps = u.shape[1] - 1
    model, ckpt = load_model(run_dir, len(r), r_e)
    gap = ckpt["config"].get("gap")
    steps = [gap] if gap is not None else list(range(1, n_steps + 1))   # evaluated stored states

    # one jump from u = 1 to every evaluated time
    u0 = np.asarray(u[:, 0], dtype=float)
    true = np.stack([np.asarray(u[:, j], dtype=float) for j in steps])  # (n_times, n_traj, N)
    pred = np.stack([predict(model, u0, logk, t[j]) for j in steps])
    eg, er = norms(pred - true, w)
    tg, tr = norms(true, w)
    q_true, q_pred = well_rate(r, logk, true), well_rate(r, logk, pred)
    err = {"grid": eg / tg, "rw": er / tr, "q": np.abs(q_pred - q_true) / q_true}

    # success bar per time (median over trajectories)
    med_u, med_q = np.median(err["grid"], axis=1), np.median(err["q"], axis=1)
    bar = {"u_max_median": float(med_u.max()), "q_max_median": float(med_q.max()),
           "u_final_median": float(med_u[-1]), "q_final_median": float(med_q[-1]),
           "passes_final": bool(med_u[-1] < U_BAR and med_q[-1] < Q_BAR),
           "passes_all_times": bool(med_u.max() < U_BAR and med_q.max() < Q_BAR)}

    # k_D = 1 against the exact series
    lam = eigenvalues(4000, r_e)
    exact = np.stack([exact_solution(r, t[j], lam, r_e) for j in steps])
    exact[:, 0] = 0.0
    u1, zero = u0[:1], np.zeros((1, len(r)))
    homog_pred = np.stack([predict(model, u1, zero, t[j])[0] for j in steps])
    homog = np.linalg.norm(homog_pred - exact, axis=-1) / np.linalg.norm(exact, axis=-1)

    # semigroup consistency (not meaningful for a fixed-gap model)
    semigroup = {}
    for m1, m2 in SEMIGROUP_SPLITS if gap is None else ():
        if m1 + m2 > n_steps:
            continue
        mid = predict(model, u0, logk, t[m1])
        two = predict(model, mid, logk, t[m1 + m2] - t[m1])
        one = pred[m1 + m2 - 1]
        ref = np.linalg.norm(true[m1 + m2 - 1], axis=-1)
        semigroup[f"{m1}+{m2}"] = {
            "two_vs_one": float(np.median(np.linalg.norm(two - one, axis=-1) / ref)),
            "two_vs_true": float(np.median(np.linalg.norm(two - true[m1 + m2 - 1], axis=-1) / ref)),
            "one_vs_true": float(np.median(np.linalg.norm(one - true[m1 + m2 - 1], axis=-1) / ref))}

    def summary(a, mask=slice(None)):
        return {"mean": float(np.mean(a[:, mask])), "median": float(np.median(a[:, mask])),
                "p90": float(np.percentile(a[:, mask], 90))}

    res = {
        "variant": ckpt["config"]["variant"], "best_step": ckpt["step"], "n_params": ckpt["config"]["n_params"],
        "t": t[steps].tolist(),
        "all_times": {k: summary(v) for k, v in err.items()},
        "final": {k: summary(v[-1:]) for k, v in err.items()},
        "skin": {k: summary(v, has_skin) for k, v in err.items()},
        "noskin": {k: summary(v, ~has_skin) for k, v in err.items()},
        "per_time": {k: {"median": np.median(v, axis=1).tolist(), "p90": np.percentile(v, 90, axis=1).tolist()}
                     for k, v in err.items()},
        "bar": bar,
        "homogeneous": {"per_time": homog.tolist(), "final": float(homog[-1]), "max": float(homog.max())},
        "semigroup": semigroup,
    }
    res["r_e"] = float(r_e)
    (Path(run_dir) / out_name).write_text(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", nargs="+", default=["logsine", "dual", "fno"])
    ap.add_argument("--runs", type=Path, default=HERE / "runs")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    print(f"{'variant':<13} {'u med':>9} {'u p90':>9} {'q med':>9} {'u max-t':>9} {'q max-t':>9} "
          f"{'u final':>9} {'q final':>9} {'k=1 max':>9} {'250+250':>9} {'pass':>5}")
    for v in args.variants:
        res = evaluate(args.runs / v)
        a, b = res["all_times"], res["bar"]
        sg = res["semigroup"].get("250+250", {}).get("two_vs_one", float("nan"))
        print(f"{v:<13} {a['grid']['median']:9.2e} {a['grid']['p90']:9.2e} {a['q']['median']:9.2e} "
              f"{b['u_max_median']:9.2e} {b['q_max_median']:9.2e} {b['u_final_median']:9.2e} "
              f"{b['q_final_median']:9.2e} {res['homogeneous']['max']:9.2e} {sg:9.2e} "
              f"{str(b['passes_all_times']):>5}", flush=True)


if __name__ == "__main__":
    main()
