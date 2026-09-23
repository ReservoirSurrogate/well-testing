"""Neural operators for the one-step map (u_n, log k) -> u_{n+1} on the log grid (Milestone 4).

All variants share one architecture:

    inputs [u, log k, s]  (s = ln r / ln r_e)
    -> lift (pointwise, width C)
    -> L x [ sum of spectral convolutions + pointwise W (+ optional local conv) ; GELU except after the last ]
    -> project (pointwise C -> 128 -> 1) = v
    -> u_{n+1} = s * v                       (hard BC: u(1) = 0)

A spectral convolution transforms each channel to K modes (u_hat = T u), mixes channels per mode with
learned real weights R_k (C x C), and maps back (u = B u_hat). Only the basis (T, B) differs:

    hankel   annular J0/Y0 eigenfunctions (basis.py), T by r-weighted least squares
    logsine  sin((n - 1/2) pi s): evenly spread in ln r, u(1) = 0 and u'(r_e) = 0 built in
    fft      standard FNO-1D: rFFT of the zero-padded signal, complex weights on the lowest K modes

Variants (see tasks.md, Milestone 4):
    A "hankel"        Hankel K
    B "hankel_local"  Hankel K + local conv (kernel 5) beside W
    C "logsine"       log-sine K
    D "dual"          Hankel K + log-sine K in every layer
    E "fno"           FFT K (baseline)
"""
import numpy as np
import torch
from torch import nn

from basis import HankelBasis, make_grid

R_E = 1000.0
N_POINTS = 1024


def log_coordinate(r, r_e=R_E):
    return np.log(r) / np.log(r_e)


def hankel_pair(n_modes, n_points=N_POINTS, r_e=R_E):
    """(T, B) of the annular Hankel basis on the log grid; T @ B = I."""
    b = HankelBasis.build(n_modes, n_points, r_e)
    return b.T, b.B


def logsine_pair(n_modes, n_points=N_POINTS, r_e=R_E):
    """(T, B) of sin((n - 1/2) pi s), T by least squares with trapezoid weights in s; T @ B = I."""
    r, _ = make_grid(n_points, r_e)
    s = log_coordinate(r, r_e)
    B = np.sin((np.arange(n_modes) + 0.5) * np.pi * s[:, None])
    w = np.full(n_points, 1.0 / (n_points - 1))
    w[[0, -1]] *= 0.5
    BtW = B.T * w
    return np.linalg.solve(BtW @ B, BtW), B


class SpectralConv(nn.Module):
    """x (batch, C, N) -> B (R_k mixing of T x): channel mixing per mode in a fixed real basis."""

    def __init__(self, T, B, width):
        super().__init__()
        self.register_buffer("T", torch.as_tensor(T, dtype=torch.get_default_dtype()))
        self.register_buffer("B", torch.as_tensor(B, dtype=torch.get_default_dtype()))
        n_modes = self.T.shape[0]
        self.weight = nn.Parameter(torch.randn(n_modes, width, width) / width)

    def forward(self, x):
        x_hat = torch.einsum("kn,bcn->bck", self.T, x)
        y_hat = torch.einsum("bck,kcd->bdk", x_hat, self.weight)
        return torch.einsum("nk,bdk->bdn", self.B, y_hat)


class FourierConv(nn.Module):
    """FNO-1D spectral convolution: zero-pad by `pad` points (non-periodic domain), keep the lowest K modes."""

    def __init__(self, n_modes, width, pad):
        super().__init__()
        self.n_modes, self.pad = n_modes, pad
        scale = 1.0 / width
        self.weight = nn.Parameter(scale * torch.randn(n_modes, width, width, dtype=torch.cfloat))

    def forward(self, x):
        n = x.shape[-1]
        x_hat = torch.fft.rfft(nn.functional.pad(x, (0, self.pad)))[..., :self.n_modes]
        y_hat = torch.einsum("bck,kcd->bdk", x_hat, self.weight)
        return torch.fft.irfft(y_hat, n=n + self.pad)[..., :n]


class Layer(nn.Module):
    def __init__(self, spectral, width, local_kernel=0, act=True):
        super().__init__()
        self.spectral = nn.ModuleList(spectral)
        self.w = nn.Conv1d(width, width, 1)
        self.local = nn.Conv1d(width, width, local_kernel, padding="same") if local_kernel else None
        self.act = act

    def forward(self, x):
        y = self.w(x) + sum(sc(x) for sc in self.spectral)
        if self.local is not None:
            y = y + self.local(x)
        return nn.functional.gelu(y) if self.act else y


class NeuralOperator(nn.Module):
    """Input (batch, 2, N) = [u_n, log k]; output (batch, N) = u_{n+1} with u(1) = 0 exactly."""

    def __init__(self, make_spectral, s, width=32, n_layers=4, local_kernel=0, proj_width=128):
        super().__init__()
        self.register_buffer("s", torch.as_tensor(s, dtype=torch.get_default_dtype()))
        self.lift = nn.Conv1d(3, width, 1)
        self.layers = nn.ModuleList(
            Layer(make_spectral(), width, local_kernel, act=i < n_layers - 1) for i in range(n_layers))
        self.proj = nn.Sequential(nn.Conv1d(width, proj_width, 1), nn.GELU(), nn.Conv1d(proj_width, 1, 1))

    def forward(self, x):
        s = self.s.expand(x.shape[0], 1, -1)
        h = self.lift(torch.cat([x, s], dim=1))
        for layer in self.layers:
            h = layer(h)
        return self.s * self.proj(h)[:, 0]


VARIANTS = ("hankel", "hankel_local", "logsine", "dual", "fno")


def build_model(variant, n_modes=32, width=32, n_layers=4, n_points=N_POINTS, r_e=R_E, fft_pad=None):
    """Factory for variants A-E (see module docstring)."""
    r, _ = make_grid(n_points, r_e)
    s = log_coordinate(r, r_e)
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
    else:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    return NeuralOperator(make, s, width, n_layers, local)


def n_params(model):
    """Number of real trainable parameters (a complex weight counts twice)."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)
