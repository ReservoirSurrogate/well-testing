"""Four collocation sets in physical (r [m], t [s]) coordinates; no interior pressure labels."""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import Config


SETS = ("domain", "ic", "bc1", "bc2")


def sample_collocation(config, seed=None):
    p, s = config.problem, config.sampling
    rng = np.random.default_rng(s.seed if seed is None else seed)

    def times(n):
        lo = s.min_tau * p.t_scale
        fraction = rng.random(n)
        uniform = lo + fraction * (p.t_max - lo)
        logarithmic = np.exp(np.log(lo) + fraction * np.log(p.t_max / lo))
        return np.where(rng.random(n) < 0.5, uniform, logarithmic)

    def radii(n):
        fraction = rng.random(n)
        # Strictly logarithmic in radius: equal probability per decade of r.
        r = p.rw * np.exp(fraction * np.log(p.radius_ratio))
        return np.clip(r, np.nextafter(p.rw, p.re), np.nextafter(p.re, p.rw))

    t, r = times(s.n_domain), radii(s.n_domain)
    # Resolve the moving early-time pressure layer, in addition to broad radial coverage.
    front = rng.random(s.n_domain) < 0.2
    distance = np.sqrt(p.alpha * t[front]) * np.exp(rng.uniform(np.log(0.02), np.log(3), front.sum()))
    width = p.re - p.rw
    r[front] = p.rw + width * distance / (width + distance)
    r = np.clip(r, np.nextafter(p.rw, p.re), np.nextafter(p.re, p.rw))
    result = {
        "domain": np.column_stack((r, t)),
        "ic": np.column_stack((radii(s.n_ic), np.zeros(s.n_ic))),
        "bc1": np.column_stack((np.full(s.n_bc1, p.rw), times(s.n_bc1))),
        "bc2": np.column_stack((np.full(s.n_bc2, p.re), times(s.n_bc2))),
    }
    validate_collocation(result, p)
    return result


def validate_collocation(points, problem):
    if set(points) != set(SETS):
        raise ValueError(f"Expected exactly these coordinate sets: {SETS}")
    p = problem
    for name, xy in points.items():
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or not np.isfinite(xy).all():
            raise ValueError(f"{name}: expected a nonempty finite (N, 2) array.")
        r, t = xy.T
        valid = (r >= p.rw) & (r <= p.re) & (t >= 0) & (t <= p.t_max)
        if name == "domain":
            valid &= (r > p.rw) & (r < p.re) & (t > 0)
        elif name == "ic":
            valid &= (r > p.rw) & (t == 0)
        else:
            valid &= (r == (p.rw if name == "bc1" else p.re)) & (t > 0)
        if not valid.all():
            raise ValueError(f"{name}: coordinates violate the condition/domain or include the initial well corner.")


def save_collocation(path, points, config):
    validate_collocation(points, config.problem)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = config.problem
    np.savez_compressed(
        path, **points,
        ic_p=np.full((len(points["ic"]), 1), p.pi),
        bc1_p=np.full((len(points["bc1"]), 1), p.pwf),
        bc2_r_dpdr=np.zeros((len(points["bc2"]), 1)),
        metadata=json.dumps({"config": config.to_dict(), "units": {"r": "m", "t": "s", "p": "Pa"},
                             "purpose": "physics-only collocation; domain has no pressure labels"}),
    )


def load_collocation(path, config):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["config"]["problem"] != config.to_dict()["problem"]:
            raise ValueError("Collocation file and requested physical configuration differ.")
        points = {name: data[name].copy() for name in SETS}
    validate_collocation(points, config.problem)
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = Config.load(args.config)
    points = sample_collocation(config)
    save_collocation(args.output, points, config)
    print(f"Saved {args.output}: " + ", ".join(f"{key}={len(value)}" for key, value in points.items()))


if __name__ == "__main__":
    main()
