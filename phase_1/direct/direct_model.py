"""Direct operator G: (u_0, log k, dt) -> u(dt) (Milestone 5b).

Replaces the one-step rollout: a trajectory is predicted by one jump per output time from u = 1, so errors
do not accumulate. Same backbone as phase_1/model.py (lift -> L spectral layers -> project, u = s * v), with
dt entering in two places:

    tau = (log10 dt - log10 DT_MIN) / (log10 DT_MAX - log10 DT_MIN)    (0 at dt = 1e3, 1 at dt = 5e5)
    - as a constant input channel: inputs [u_0, log k, s, tau]
    - FiLM: an MLP maps tau to a per-layer, per-channel scale and shift, y <- y * (1 + gamma) + beta,
      applied before each activation. The last MLP layer starts at zero, so FiLM starts as the identity.

Variants and spectral bases as in phase_1/model.py (hankel, hankel_local, logsine, dual, fno), plus
    fftlog        FFTLog Hankel spectral convolution (fftlog_conv.py): radial Fourier multiplier R(k)
    fftlog_decay  same, with a learned per-channel heat-kernel factor exp(-D_c k^2 dt)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basis import make_grid  # noqa: E402
from fftlog_conv import FFTLogConv  # noqa: E402
from model import (N_POINTS, R_E, FourierConv, SpectralConv, hankel_pair,  # noqa: E402
                   log_coordinate, logsine_pair, n_params)
from model import VARIANTS as BASE_VARIANTS  # noqa: E402

VARIANTS = BASE_VARIANTS + ("fftlog", "fftlog_decay")

DT_MIN, DT_MAX = 1e3, 5e5

__all__ = ["DT_MIN", "DT_MAX", "VARIANTS", "DirectOperator", "build_direct", "n_params", "time_feature"]


def time_feature(dt):
    """tau = log10(dt) mapped linearly to [0, 1] on [DT_MIN, DT_MAX]; dt a tensor of shape (B,)."""
    lo, hi = np.log10(DT_MIN), np.log10(DT_MAX)
    return (torch.log10(dt) - lo) / (hi - lo)


class FiLMLayer(nn.Module):
    def __init__(self, spectral, width, local_kernel=0, act=True):
        super().__init__()
        self.spectral = nn.ModuleList(spectral)
        self.w = nn.Conv1d(width, width, 1)
        self.local = nn.Conv1d(width, width, local_kernel, padding="same") if local_kernel else None
        self.act = act

    def forward(self, x, gamma, beta, dt):
        y = self.w(x) + sum(sc(x, dt) if getattr(sc, "uses_dt", False) else sc(x) for sc in self.spectral)
        if self.local is not None:
            y = y + self.local(x)
        y = y * (1 + gamma[..., None]) + beta[..., None]
        return nn.functional.gelu(y) if self.act else y


class DirectOperator(nn.Module):
    """forward(x, dt): x (batch, 2, N) = [u_0, log k], dt (batch,) -> u(dt) (batch, N) with u(1) = 0 exactly."""

    def __init__(self, make_spectral, s, width=32, n_layers=4, local_kernel=0, proj_width=128, time_width=64):
        super().__init__()
        self.width, self.n_layers = width, n_layers
        self.register_buffer("s", torch.as_tensor(s, dtype=torch.get_default_dtype()))
        self.lift = nn.Conv1d(4, width, 1)
        self.time = nn.Sequential(nn.Linear(1, time_width), nn.GELU(), nn.Linear(time_width, 2 * width * n_layers))
        nn.init.zeros_(self.time[-1].weight)
        nn.init.zeros_(self.time[-1].bias)
        self.layers = nn.ModuleList(
            FiLMLayer(make_spectral(), width, local_kernel, act=i < n_layers - 1) for i in range(n_layers))
        self.proj = nn.Sequential(nn.Conv1d(width, proj_width, 1), nn.GELU(), nn.Conv1d(proj_width, 1, 1))

    def forward(self, x, dt):
        b, _, n = x.shape
        tau = time_feature(dt).to(x.dtype)
        film = self.time(tau[:, None]).view(b, self.n_layers, 2, self.width)
        h = self.lift(torch.cat([x, self.s.expand(b, 1, n), tau[:, None, None].expand(b, 1, n)], dim=1))
        for i, layer in enumerate(self.layers):
            h = layer(h, film[:, i, 0], film[:, i, 1], dt)
        return self.s * self.proj(h)[:, 0]


def build_direct(variant, n_modes=32, width=32, n_layers=4, n_points=N_POINTS, r_e=R_E, fft_pad=None,
                 fftlog_pad=256, fftlog_q=0.4, s_ref_re=None, fftlog_k_range=None):
    """Factory for the direct operator; spectral bases as in phase_1/model.build_model.

    For transfer to another r_eD (see transfer_evaluate.py): s_ref_re keeps s = ln r / ln s_ref_re as a fixed
    physical coordinate (default: ln r / ln r_e), and fftlog_k_range pins the FFTLog control points to physical k.
    """
    r, _ = make_grid(n_points, r_e)
    s = log_coordinate(r, r_e if s_ref_re is None else s_ref_re)
    local = 0
    if variant in ("hankel", "hankel_local"):
        T, B = hankel_pair(n_modes, n_points, r_e)
        make = lambda: [SpectralConv(T, B, width)]
        local = 5 if variant == "hankel_local" else 0
    elif variant == "logsine":
        T, B = logsine_pair(n_modes, n_points, r_e)
        make = lambda: [SpectralConv(T, B, width)]
    elif variant == "dual":
        Th, Bh = hankel_pair(n_modes, n_points, r_e)
        Ts, Bs = logsine_pair(n_modes, n_points, r_e)
        make = lambda: [SpectralConv(Th, Bh, width), SpectralConv(Ts, Bs, width)]
    elif variant == "fno":
        pad = n_points // 4 if fft_pad is None else fft_pad
        make = lambda: [FourierConv(n_modes, width, pad)]
    elif variant in ("fftlog", "fftlog_decay"):
        decay = variant == "fftlog_decay"
        make = lambda: [FFTLogConv(r, width, n_modes, pad=fftlog_pad, q=fftlog_q, decay=decay,
                                   k_range=fftlog_k_range)]
    else:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    return DirectOperator(make, s, width, n_layers, local)
