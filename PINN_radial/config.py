"""Validated configuration. All dimensional quantities use SI units."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Problem:
    k: float
    phi: float
    mu: float
    c_t: float
    rw: float
    re: float
    pi: float
    pwf: float
    t_max: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Physical parameters must be finite.")
        if any(getattr(self, key) <= 0 for key in ("k", "phi", "mu", "c_t", "rw", "t_max")):
            raise ValueError("k, phi, mu, c_t, rw, and t_max must be positive.")
        if self.phi > 1 or self.re <= self.rw or self.pi <= self.pwf:
            raise ValueError("Require phi <= 1, re > rw, and pi > pwf for this production model.")

    @property
    def alpha(self):
        return self.k / (self.phi * self.mu * self.c_t)

    @property
    def t_scale(self):
        return self.rw**2 / self.alpha

    @property
    def dp(self):
        return self.pi - self.pwf

    @property
    def radius_ratio(self):
        return self.re / self.rw


@dataclass(frozen=True)
class Sampling:
    n_domain: int = 4096
    n_ic: int = 512
    n_bc1: int = 512
    n_bc2: int = 512
    # Smallest positive sampled time, as a fraction of rw^2 / alpha.
    min_tau: float = 1e-3
    seed: int = 42


@dataclass(frozen=True)
class Network:
    width: int = 64
    depth: int = 4


@dataclass(frozen=True)
class Training:
    adam_steps: int = 5000
    learning_rate: float = 1e-3
    domain_batch: int = 1024
    condition_batch: int = 256
    resample_every: int = 100
    log_every: int = 100
    lbfgs_steps: int = 0
    threads: int = 2
    pde_time_weighting: bool = True
    weights: dict = field(default_factory=lambda: {"pde": 1.0, "ic": 10.0, "bc1": 10.0, "bc2": 1.0})


@dataclass(frozen=True)
class Reference:
    n_r: int = 257
    n_t: int = 81
    min_tau: float = 1.0
    initial_modes: int = 128
    max_modes: int = 8192
    pressure_tolerance: float = 1e-7
    rate_tolerance: float = 1e-6


@dataclass(frozen=True)
class Config:
    problem: Problem
    description: str = ""
    sampling: Sampling = field(default_factory=Sampling)
    model: Network = field(default_factory=Network)
    training: Training = field(default_factory=Training)
    reference: Reference = field(default_factory=Reference)

    def __post_init__(self):
        s, m, tr, ref = self.sampling, self.model, self.training, self.reference
        counts = (s.n_domain, s.n_ic, s.n_bc1, s.n_bc2, m.width, m.depth,
                  tr.domain_batch, tr.condition_batch, tr.log_every, tr.threads,
                  ref.n_r, ref.n_t, ref.initial_modes, ref.max_modes)
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
            raise ValueError("Point counts, network dimensions, batches, and intervals must be positive integers.")
        if ref.n_r < 3 or ref.n_t < 2 or ref.max_modes < 2 * ref.initial_modes:
            raise ValueError("Reference requires n_r >= 3, n_t >= 2, and room to double the initial mode count.")
        for tau in (s.min_tau, ref.min_tau):
            if not math.isfinite(tau) or not 0 < tau * self.problem.t_scale < self.problem.t_max:
                raise ValueError("Minimum positive sample/reference times must lie below t_max.")
        for n in (tr.adam_steps, tr.lbfgs_steps, tr.resample_every, s.seed):
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ValueError("Step counts, resampling interval, and seed must be nonnegative integers.")
        if not isinstance(tr.pde_time_weighting, bool):
            raise ValueError("pde_time_weighting must be a boolean.")
        for value in (tr.learning_rate, ref.pressure_tolerance, ref.rate_tolerance):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Learning rate and reference tolerances must be positive and finite.")
        if set(tr.weights) != {"pde", "ic", "bc1", "bc2"}:
            raise ValueError("Loss weights must contain pde, ic, bc1, and bc2.")
        if any(not math.isfinite(w) or w <= 0 for w in tr.weights.values()):
            raise ValueError("All four physics loss weights must be positive and finite.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for name, kind in (("problem", Problem), ("sampling", Sampling), ("model", Network),
                           ("training", Training), ("reference", Reference)):
            if name in data:
                data[name] = kind(**data[name])
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")
