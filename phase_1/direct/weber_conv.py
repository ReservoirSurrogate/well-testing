"""Weber-transform spectral convolution (Milestone 5b, step B): the transform that fits a well.

STATUS: DOES NOT WORK - kept as a documented dead end, not wired into the model and not tested. The Weber inverse is
evaluated with the continuous formula (separate J0 and Y0 FFTLog passes), which needs an alias-free transform. The
network's hidden channels are O(1) out to r_eD, so W(k) -> W(0) ~ 1e8 at small k and wraps around the periodic
ln k grid: the round trip fails by O(1)-O(100) even in float64 (grids up to 5000 points only reach ~2e-2). It works for
compact fields (round trip ~1e-7, see phase_1/tests/test_weber.py for the transform itself).

For each channel h(r) on a geometric grid starting at the well r = a = 1, computes  W^{-1}[ R(k) W[m h] ]  with the
Weber transform of order 0 outside the well (phase_1/fftlog.py),

    W[f](k) = int_a^inf f(r) C(k, r) r dr,   C(k, r) = J0(k r) Y0(k a) - Y0(k r) J0(k a),

which diagonalizes radial diffusion with u(a) = 0: W[f'' + f'/r] = -k^2 W[f] for f(a) = 0. Steps:

1. m(r) = 1 - exp(-(ln(r/a) / ramp)^2) makes every channel vanish smoothly at the well (hidden channels are not zero
   there; a jump or kink at r = a would ring on the periodic grid). The output is not divided by m: it vanishes at the
   well like the pressure, and the pointwise path carries the values next to the well.
2. pad: `pad` zero points below the well (the Weber domain is r >= a; the pad extends the k range), `pad` points
   above r_e holding the edge value with a cos^2 taper;
3. forward: J0 and Y0 FFTLog passes with bias q, combined into y = k^{q+1} W(k);
4. channel mixing y <- R(k) y, R(k) (C x C, real) linear in ln k between K control points (as in fftlog_conv.py);
5. inverse: f(r) = H_J[W Y0(ka) / D](r) - H_Y[W J0(ka) / D](r), D = J0(ka)^2 + Y0(ka)^2, as two FFTLog passes
   k -> r with their own bias q_inv (the inverse passes return r^{q_inv+1} f; q_inv = q would give r^1.4 f, whose
   ~1e5 dynamic range costs ~1e-2 near the well in float32).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import j0, y0
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fftlog import FFTLog  # noqa: E402


def _pass(x, weight, multiplier):
    """One FFTLog pass: y = irfft(conj(rfft(x * weight) * multiplier)) along the last axis (N factors cancel)."""
    c = torch.fft.rfft(x * weight, dim=-1)
    return torch.fft.irfft(torch.conj_physical(c * multiplier), n=x.shape[-1], dim=-1)


class WeberConv(nn.Module):
    """x (batch, C, N) -> (batch, C, N); r geometric with r[0] = a (the well)."""

    def __init__(self, r, width, n_modes=32, pad=256, q=0.4, q_inv=-0.6, ramp=0.1, k_range=None):
        super().__init__()
        r = np.asarray(r, dtype=float)
        dlog = np.log(r[1] / r[0])
        if not np.allclose(np.diff(np.log(r)), dlog, rtol=1e-8, atol=1e-12):
            raise ValueError("WeberConv needs a geometric grid")
        a = r[0]
        self.n, self.pad = len(r), pad
        n_tot = self.n + 2 * pad
        r_pad = r[0] * np.exp(dlog * (np.arange(n_tot) - pad))

        fj = FFTLog(n_tot, dlog, q=q, kernel="J")
        fy = FFTLog(n_tot, dlog, q=q, kappa=fj.kappa, low_ringing=False, kernel="Y")
        k = fj.output_grid(r_pad[0])
        ij = FFTLog(n_tot, dlog, q=q_inv, kappa=fj.kappa, low_ringing=False, kernel="J")
        iy = FFTLog(n_tot, dlog, q=q_inv, kappa=fj.kappa, low_ringing=False, kernel="Y")
        r_back = ij.output_grid(k[0])
        if not np.allclose(r_back, r_pad, rtol=1e-10):
            raise RuntimeError("inverse grid does not match the input grid")
        ja, ya = j0(k * a), y0(k * a)
        dnm = ja**2 + ya**2

        window = np.where(r_pad >= a * (1 - 1e-12), 1 - np.exp(-(np.log(np.maximum(r_pad, a) / a) / ramp) ** 2), 0.0)
        taper = np.ones(n_tot)
        taper[n_tot - pad:] = np.sin(0.5 * np.pi * (np.arange(pad) + 0.5) / pad)[::-1] ** 2

        k_lo, k_hi = (k[0], k[-1]) if k_range is None else k_range
        self.k_range = (float(k_lo), float(k_hi))
        pos = np.clip((np.log(k) - np.log(k_lo)) / (np.log(k_hi) - np.log(k_lo)), 0.0, 1.0) * (n_modes - 1)
        lo = np.minimum(np.floor(pos).astype(int), n_modes - 2)
        interp = np.zeros((n_tot, n_modes))
        interp[np.arange(n_tot), lo] = 1 - (pos - lo)
        interp[np.arange(n_tot), lo + 1] = pos - lo

        f32, c64 = torch.get_default_dtype(), torch.complex64
        buf = lambda name, v, dt=f32: self.register_buffer(name, torch.as_tensor(v, dtype=dt))
        buf("r_pad", r_pad)
        buf("k", k)
        buf("window", window * taper)                           # m(r) and the outer taper, zero below the well
        buf("w_fwd", r_pad ** (1 - q))
        buf("mj", fj.multiplier, c64)
        buf("my", fy.multiplier, c64)
        buf("ya", ya)
        buf("ja", ja)
        # inverse inputs: G(k) k^{1 - q_inv} with G = W * (Y0 or J0)(ka) / D and W = y / k^{q+1}
        buf("c_ij", ya / dnm * k ** (1 - q_inv - q - 1))
        buf("c_iy", ja / dnm * k ** (1 - q_inv - q - 1))
        buf("mij", ij.multiplier, c64)
        buf("miy", iy.multiplier, c64)
        buf("w_inv", r_pad ** -(q_inv + 1))
        buf("interp", interp)
        self.weight = nn.Parameter(torch.randn(n_modes, width, width) / width)

    def pad_window(self, x):
        """Pad (zeros below the well, edge value above r_e) and apply the well ramp m(r) and outer taper."""
        x = nn.functional.pad(x, (0, self.pad), mode="replicate")
        x = nn.functional.pad(x, (self.pad, 0))
        return x * self.window

    def transform(self, x_pad):
        """Padded field -> y = k^{q+1} W(k)."""
        return self.ya * _pass(x_pad, self.w_fwd, self.mj) - self.ja * _pass(x_pad, self.w_fwd, self.my)

    def inverse_transform(self, y):
        """y = k^{q+1} W(k) -> f(r) on the padded grid (meaningful for r >= a)."""
        return (_pass(y * self.c_ij, 1.0, self.mij) - _pass(y * self.c_iy, 1.0, self.miy)) * self.w_inv

    def forward(self, x):
        y = self.transform(self.pad_window(x))
        mix = torch.einsum("km,mcd->kcd", self.interp, self.weight)
        y = torch.einsum("bck,kcd->bdk", y, mix)
        return self.inverse_transform(y)[..., self.pad:self.pad + self.n]
