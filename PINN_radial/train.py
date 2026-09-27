"""Train using only domain, IC, BC1, and BC2 collocation constraints."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from .config import Config
from .dataset import load_collocation, sample_collocation, save_collocation
from .model import PressurePINN
from .physics import physics_losses, weighted_loss


def tensor_points(points, device):
    return {name: torch.as_tensor(xy, dtype=torch.float64, device=device) for name, xy in points.items()}


def loss_values(losses, weights):
    values = {name: float(value.detach().cpu()) for name, value in losses.items()}
    values["total"] = sum(weights[name] * value for name, value in values.items())
    return values


def train(config, output, device="cpu", collocation=None, warm_start=None):
    output = Path(output)
    if (output / "config.json").exists():
        raise FileExistsError(f"Run directory already contains a configuration: {output}. Choose a new output.")
    tr, seed = config.training, config.sampling.seed
    torch.set_num_threads(tr.threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PressurePINN(config.problem, config.model).to(device)
    if warm_start:
        previous, previous_config = load_model(warm_start, device)
        if previous_config.problem != config.problem or previous_config.model != config.model:
            raise ValueError("Warm-start checkpoint must have the same physical problem and network architecture.")
        model.load_state_dict(previous.state_dict())
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.json")
    points = load_collocation(collocation, config) if collocation else sample_collocation(config)
    save_collocation(output / "collocation_initial.npz", points, config)
    pool = tensor_points(points, device)
    validation_config = replace(config, sampling=replace(
        config.sampling, seed=seed + 1_000_000, n_domain=min(512, config.sampling.n_domain),
        n_ic=min(128, config.sampling.n_ic), n_bc1=min(128, config.sampling.n_bc1),
        n_bc2=min(128, config.sampling.n_bc2),
    ))
    validation_points = sample_collocation(validation_config)
    save_collocation(output / "collocation_validation.npz", validation_points, validation_config)
    validation = tensor_points(validation_points, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr.learning_rate)
    started = time.perf_counter()
    best = float("inf")
    history = []

    def compute_losses(current_points):
        return physics_losses(model, current_points, config.problem, time_weighting=tr.pde_time_weighting)

    def checkpoint(name, step, validation_loss):
        torch.save({"config": config.to_dict(), "model": model.state_dict(), "step": step,
                    "validation_loss": validation_loss, "training_kind": "physics_only",
                    "warm_start": str(warm_start) if warm_start else None}, output / name)

    def record(step, phase, train_losses):
        nonlocal best
        # Autograd stays enabled: validation includes spatial and time derivatives.
        val = loss_values(compute_losses(validation), tr.weights)
        training = loss_values(train_losses, tr.weights)
        if not all(np.isfinite(list(v.values())).all() for v in (val, training)):
            raise FloatingPointError("Nonfinite physics loss; inspect scaling, parameters, and optimizer settings.")
        row = {"step": step, "phase": phase, "seconds": time.perf_counter() - started}
        row.update({f"train_{key}": value for key, value in training.items()})
        row.update({f"validation_{key}": value for key, value in val.items()})
        history.append(row)
        with (output / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if val["total"] < best:
            best = val["total"]
            checkpoint("best.pt", step, best)
        print(f"step={step:6d} {phase:7s} loss={training['total']:.4e} "
              f"validation={val['total']:.4e} pde={val['pde']:.3e} ic={val['ic']:.3e} "
              f"bc1={val['bc1']:.3e} bc2={val['bc2']:.3e}", flush=True)
        return val["total"]

    def batch():
        result = {}
        for name, xy in pool.items():
            size = tr.domain_batch if name == "domain" else tr.condition_batch
            index = rng.choice(len(xy), size=min(size, len(xy)), replace=False)
            result[name] = xy[torch.as_tensor(index, device=device)]
        return result

    last_validation = record(0, "initial", compute_losses(validation))
    step = 0
    for step in range(1, tr.adam_steps + 1):
        if tr.resample_every and step > 1 and (step - 1) % tr.resample_every == 0:
            points = sample_collocation(config, seed=seed + step)
            pool = tensor_points(points, device)
        optimizer.zero_grad(set_to_none=True)
        current_batch = batch()
        losses = compute_losses(current_batch)
        total = weighted_loss(losses, tr.weights)
        if not torch.isfinite(total):
            raise FloatingPointError(f"Nonfinite training loss at step {step}.")
        total.backward()
        optimizer.step()
        if step % tr.log_every == 0 or step == tr.adam_steps:
            last_validation = record(step, "adam", compute_losses(current_batch))

    if tr.lbfgs_steps:
        # The pool remains fixed throughout all line searches and L-BFGS steps.
        # max_iter=1 otherwise defaults max_eval to 1, leaving no line-search budget
        # in PyTorch versions that enforce max_eval inside strong_wolfe.
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, max_eval=25,
                                      history_size=30, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            total = weighted_loss(compute_losses(pool), tr.weights)
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite L-BFGS loss.")
            total.backward()
            return total

        for index in range(1, tr.lbfgs_steps + 1):
            step = tr.adam_steps + index
            optimizer.step(closure)
            if index % tr.log_every == 0 or index == tr.lbfgs_steps:
                last_validation = record(step, "lbfgs", compute_losses(pool))

    checkpoint("last.pt", step, last_validation)
    save_collocation(output / "collocation_final.npz", points, config)
    summary = {"steps": step, "best_validation_loss": best,
               "seconds": time.perf_counter() - started, "device": str(device),
               "training_kind": "physics_only", "torch_version": str(torch.__version__),
               "numpy_version": np.__version__, "warm_start": str(warm_start) if warm_start else None}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_model(checkpoint, device="cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    config = Config.from_dict(saved["config"])
    model = PressurePINN(config.problem, config.model).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collocation", help="Optional NPZ generated by PINN_radial.dataset")
    parser.add_argument("--steps", type=int, help="Override Adam steps")
    parser.add_argument("--lbfgs-steps", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warm-start", help="Load compatible weights; start fresh optimizers in a new run directory")
    args = parser.parse_args()
    config = Config.load(args.config)
    overrides = {name: value for name, value in (("adam_steps", args.steps), ("lbfgs_steps", args.lbfgs_steps))
                 if value is not None}
    config = replace(config, training=replace(config.training, **overrides))
    result = train(config, args.output, args.device, args.collocation, args.warm_start)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
