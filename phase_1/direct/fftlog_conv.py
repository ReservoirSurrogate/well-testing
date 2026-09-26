"""FFTLog Hankel spectral convolution (Milestone 5b, step 2).

For each channel h(r) on a geometric grid, computes  H^{-1}[ R(k) H[h] ]  with H the order-mu Hankel transform
F(k) = int_0^inf h(r) J_mu(k r) r dr, evaluated by FFTLog (phase_1/fftlog.py):

1. pad in ln r by `pad` points on each side (same spacing), holding the edge value and tapering it to zero
   with a cos^2 window, so r^{1-q} h is negligible at both ends of the periodic ln r domain;
2. forward FFTLog: y(k) = k^{q+1} F(k) on the log grid k in [kappa / r_max, kappa / r_min];
3. channel mixing y(k) <- R(k) y(k), R(k) (C x C, real) linearly interpolated in ln k between K control points,
   optionally with a per-channel heat-kernel factor exp(-D_c k^2 dt) (decay=True, learned D_c > 0);
4. exact inverse FFTLog and crop to the original grid.

The k^{q+1} factors of the forward and inverse transforms cancel around a per-k multiplier, so the layer works
with y directly. For mu = 0, R(k) = exp(-D k^2 dt) is exactly the propagator of dh/dt = D (1/r)(r h')' on the
plane: step 3 turns H into a radial Fourier multiplier, the axisymmetric analogue of an FNO layer.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fftlog import FFTLog  # noqa: E402

D_INIT = (1e-4, 20.0)      # log-uniform initial range of the per-channel diffusivities (decay=True)


class FFTLogConv(nn.Module):
    """x (batch, C, N) [, dt (batch,)] -> (batch, C, N); r must be geometric (constant ln-spacing)."""

    def __init__(self, r, width, n_modes=32, pad=256, mu=0.0, q=0.4, decay=False, k_range=None):
        """k_range = (k_lo, k_hi): physical wavenumbers of the first and last control point of R(k).
        Default: this grid's own k range. Pass the training grid's range to reuse trained weights on another
        grid (e.g. another r_eD): R(k) then stays the same function of physical k, held constant outside."""
        super().__init__()
        r = np.asarray(r, dtype=float)
        dlog = np.log(r[1] / r[0])
        if not np.allclose(np.diff(np.log(r)), dlog, rtol=1e-8, atol=1e-12):
            raise ValueError("FFTLogConv needs a geometric grid")
        self.n, self.pad, self.decay = len(r), pad, decay
        n_tot = self.n + 2 * pad
        r_pad = r[0] * np.exp(dlog * (np.arange(n_tot) - pad))
        t = FFTLog(n_tot, dlog, mu=mu, q=q)
        k = t.output_grid(r_pad[0])
        self.kappa = t.kappa

        taper = np.ones(n_tot)
        ramp = np.sin(0.5 * np.pi * (np.arange(pad) + 0.5) / pad) ** 2          # 0 -> 1 across the pad
        taper[:pad], taper[n_tot - pad:] = ramp, ramp[::-1]

        # linear interpolation in ln k from n_modes control points, log-spaced over k_range
        k_lo, k_hi = (k[0], k[-1]) if k_range is None else k_range
        self.k_range = (float(k_lo), float(k_hi))
        pos = np.clip((np.log(k) - np.log(k_lo)) / (np.log(k_hi) - np.log(k_lo)), 0.0, 1.0) * (n_modes - 1)
        lo = np.minimum(np.floor(pos).astype(int), n_modes - 2)
        frac = pos - lo
        interp = np.zeros((n_tot, n_modes))
        interp[np.arange(n_tot), lo] = 1 - frac
        interp[np.arange(n_tot), lo + 1] = frac

        f32 = torch.get_default_dtype()
        self.register_buffer("r_pad", torch.as_tensor(r_pad, dtype=f32))
        self.register_buffer("k", torch.as_tensor(k, dtype=f32))
        self.register_buffer("taper", torch.as_tensor(taper, dtype=f32))
        self.register_buffer("weight_r", torch.as_tensor(r_pad ** (1 - q), dtype=f32))
        self.register_buffer("multiplier", torch.as_tensor(t.multiplier, dtype=torch.complex64))
        self.register_buffer("interp", torch.as_tensor(interp, dtype=f32))
        self.weight = nn.Parameter(torch.randn(n_modes, width, width) / width)
        if decay:
            lo_d, hi_d = np.log(D_INIT[0]), np.log(D_INIT[1])
            self.log_d = nn.Parameter(torch.linspace(lo_d, hi_d, width))
        self.uses_dt = decay

    def pad_taper(self, x):
        x = nn.functional.pad(x, (self.pad, self.pad), mode="replicate")
        return x * self.taper

    def transform(self, x_pad):
        """Padded field (..., n_tot) -> y = k^{q+1} F(k) (..., n_tot)."""
        c = torch.fft.rfft(x_pad * self.weight_r, dim=-1)
        return torch.fft.irfft(torch.conj_physical(c * self.multiplier), n=x_pad.shape[-1], dim=-1)

    def inverse_transform(self, y):
        """Exact inverse of `transform`."""
        d = torch.conj_physical(torch.fft.rfft(y, dim=-1))
        return torch.fft.irfft(d / self.multiplier, n=y.shape[-1], dim=-1) / self.weight_r

    def forward(self, x, dt=None):
        y = self.transform(self.pad_taper(x))
        if self.decay:
            y = y * torch.exp(-torch.exp(self.log_d)[None, :, None] * self.k**2 * dt[:, None, None])
        mix = torch.einsum("km,mcd->kcd", self.interp, self.weight)
        y = torch.einsum("bck,kcd->bdk", y, mix)
        return self.inverse_transform(y)[..., self.pad:self.pad + self.n]
