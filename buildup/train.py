"""Train the inverse model (build-up project, Milestone 6).

Every step draws a fresh batch of build-ups from the training reservoirs (examples.Bank.make: new t_p, gauge noise,
test length, p_i masking and gauge noise level), so the model never sees the same example twice. Validation uses a fixed, seeded set
of VAL_DRAWS build-ups per validation reservoir. Loss: beta-NLL (beta = 0.5) + BCE (model.loss_terms); the
validation NLL is the plain one (beta = 0). AdamW, cosine decay.

Usage: OMP_NUM_THREADS=4 python train.py [--steps 20000 --batch 64 --lr 1e-3 --threads 4 --out runs/base]
Writes: <out>/{best.pt, last.pt, log.csv, config.json}
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples import C_IN, N_RING, Bank
from model import InverseFNO, loss_terms, n_params

HERE = Path(__file__).resolve().parent
VAL_DRAWS = 4


def load_bank(split, data_dir=HERE / "data"):
    with np.load(Path(data_dir) / f"{split}.npz") as f:
        return Bank({key: f[key] for key in f.files})


def fixed_set(bank, draws, seed, **make_kw):
    """Seeded examples: `draws` build-ups per reservoir."""
    rng = np.random.default_rng(seed)
    idx = np.repeat(np.arange(bank.n), draws)
    x, y, ns, meta = bank.make(idx, rng, **make_kw)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(ns), meta


@torch.no_grad()
def predict(model, x, chunk=1000):
    outs = [model(x[i:i + chunk]) for i in range(0, len(x), chunk)]
    return tuple(torch.cat([o[j] for o in outs]) for j in range(3))


@torch.no_grad()
def val_metrics(model, x, y, ns):
    mu, log_var, logit = predict(model, x)
    nll, bce = loss_terms(mu, log_var, logit, y, ns, beta=0.0)          # report the plain NLL
    valid = ~torch.isnan(y)
    err = torch.where(valid, (mu - torch.nan_to_num(y)) ** 2, torch.zeros_like(y))
    rmse = torch.sqrt(err.sum(0) / valid.sum(0).clamp(min=1))
    z = torch.where(valid, (mu - torch.nan_to_num(y)).abs() * torch.exp(-0.5 * log_var), torch.zeros_like(y))
    cover = ((z < 2) & valid).sum() / valid.sum()
    acc = ((logit > 0).float() == ns).float().mean()
    return {"nll": nll.item(), "bce": bce.item(), "rmse_ring": rmse[:N_RING].mean().item(),
            "rmse_c": rmse[N_RING].item(), "rmse_re": rmse[N_RING + 1].item(), "cover2": cover.item(),
            "acc_storage": acc.item()}


def train(steps=20000, batch=64, lr=1e-3, width=48, modes=24, n_layers=4, seed=0, threads=4, eval_every=1000,
          data_dir=HERE / "data", out=HERE / "runs" / "base", log=print):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    bank, val = load_bank("train", data_dir), load_bank("val", data_dir)
    xv, yv, nsv, _ = fixed_set(val, VAL_DRAWS, seed=1234)
    model = InverseFNO(width, modes, n_layers, c_in=C_IN)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(steps=steps, batch=batch, lr=lr, width=width, modes=modes, n_layers=n_layers, seed=seed, beta=0.5,
                  c_in=C_IN, n_params=n_params(model), n_train=bank.n)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    log(f"{config['n_params']} params, {bank.n} training reservoirs")

    best, t0, running = np.inf, time.time(), []
    with open(out / "log.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        keys = ["nll", "bce", "rmse_ring", "rmse_c", "rmse_re", "cover2", "acc_storage"]
        writer.writerow(["step", "train"] + keys + ["lr", "seconds"])
        for step in range(1, steps + 1):
            x, y, ns, _ = bank.make(rng.integers(0, bank.n, batch), rng)
            mu, log_var, logit = model(torch.from_numpy(x))
            nll, bce = loss_terms(mu, log_var, logit, torch.from_numpy(y), torch.from_numpy(ns))
            loss = nll + bce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            running.append(loss.item())
            if step % eval_every == 0 or step == steps:
                model.eval()
                v = val_metrics(model, xv, yv, nsv)
                model.train()
                writer.writerow([step, np.mean(running)] + [v[k] for k in keys] + [sched.get_last_lr()[0],
                                                                                    time.time() - t0])
                fh.flush()
                running = []
                if v["nll"] + v["bce"] < best:
                    best = v["nll"] + v["bce"]
                    torch.save({"config": config, "state_dict": model.state_dict(), "step": step, **v}, out / "best.pt")
                log(f"step {step}: val nll {v['nll']:.3f} bce {v['bce']:.3f} rmse ring {v['rmse_ring']:.3f} "
                    f"C {v['rmse_c']:.3f} r_e {v['rmse_re']:.3f} cover2 {v['cover2']:.3f} "
                    f"acc {v['acc_storage']:.3f} ({time.time() - t0:.0f} s)")
    torch.save({"config": config, "state_dict": model.state_dict(), "step": steps}, out / "last.pt")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--modes", type=int, default=24)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "base")
    args = ap.parse_args()
    train(args.steps, args.batch, args.lr, args.width, args.modes, args.layers, args.seed, args.threads,
          out=args.out, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
