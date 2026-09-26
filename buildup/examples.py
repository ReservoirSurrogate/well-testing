"""Training examples for the inverse model (build-up project, Milestone 6): build-ups formed on the fly.

For each stored reservoir (one unit-rate drawdown p_dd at 200 log-spaced times), an example is a build-up after a
production time t_p ~ logU(wells.T_P_RANGE) = logU[1e4, 1e6] on a fixed grid of G = 128 shut-in times dt, log-spaced
in [1e-2, 9e6]:

    dp_bu(dt) = p_dd(t_p) - p_dd(t_p + dt) + p_dd(dt)        (superposition; p_dd from a cubic spline in ln t)

with gauge noise (multiplicative, relative size `noise`; by default drawn per example, log-uniform in NOISE_RANGE =
[1e-4, 1e-2], i.e. 0.01% to 1% of the pressure), a random test length and p_i known or not (the drawdown at
shut-in p_wD(t_p) = p_i - p_wf is an input only if p_i is known). Test length: full (dt_max = 9e6) or, with
probability p_trunc, dt_max ~ logU[DT_MAX_MIN, 9e6] with DT_MAX_MIN = 1e6 (days of shut-in, long enough to feel the
boundary in nearly all reservoirs). t_p and dt_max are drawn independently of the reservoir: a length tied to
r_eD^2 / k would leak r_eD through the validity mask.

Input channels (C_IN = 9) on the dt grid; the measured curves (0, 1, 7) are zero after the end of the test:
    0 ln dp_bu / 5          1 d ln dp_bu / d ln dt (log-derivative; vs Agarwal time it diverges once dt >> t_p)
    2 (ln dt - mid) / half  3 (ln t_p - 10.36) / 1.73 (constant)
    4 ln p_wD(t_p) / 3 (constant; 0 if p_i unknown)   5 p_i-known flag   6 validity mask
    7 material-balance radius (0 if p_i unknown): the pressure drop still below p_i, p_drop = p_wf - dp_bu(dt), gives
      the apparent size r_app = sqrt(2 t_p / p_drop + 1) the reservoir would have if the pressure had stabilized at
      dt (C_D << (r_eD^2 - 1)/2 is neglected); a lower bound of r_eD that converges to it as the build-up
      stabilizes. Normalized like the r_eD target, clipped to [-3, 3] (noise-dominated p_drop).
    8 gauge noise level (constant): log10(noise) + 3, i.e. -1 at 0.01%, 0 at 0.1%, +1 at 1% (known from the gauge
      specification). Added last, so models trained with 8 channels still read the first 8 unchanged.
Targets (normalized; NaN where masked): 8 ring ln k (permeability.RINGS), (ln C_D - 2.3) / 2.7, (ln r_eD - 6.85) / 0.67,
and the no-storage flag.

The drawdown splines of all samples are precomputed as one coefficient array (4, 199, n), so a batch with a
different t_p per example is evaluated vectorized.
"""
import numpy as np
from scipy.interpolate import CubicSpline

from dataset import grid
from permeability import RINGS, ring_labels
from wells import T_P_RANGE

G = 128
DT = np.geomspace(1e-2, 9e6, G)
LN_DT = np.log(DT)
C_IN = 9
MEASURED = (0, 1, 7)                        # channels zeroed after the end of the test
N_RING = len(RINGS)
N_TARGET = N_RING + 2                       # rings, ln C_D, ln r_eD
LN_TP_MID, LN_TP_HALF = 10.36, 1.73          # fixed input normalization of ln t_p (independent of T_P_RANGE)
DT_MAX_MIN = 1e6                             # shortest (truncated) test
NOISE_RANGE = (1e-4, 1e-2)                   # relative gauge noise, log-uniform per example
C_NORM = (2.3, 2.7)
RE_NORM = (6.85, 0.67)


class Bank:
    """All reservoirs of a split, ready to form build-up examples."""

    def __init__(self, data):
        self.lnt = np.log(data["t"])
        self.n = len(data["r_e"])
        spline = CubicSpline(self.lnt, data["p_w"], axis=1)
        self.coef = spline.c                                   # (4, 199, n)
        rings = np.stack([ring_labels(grid(data, i), np.exp(data["logk"][i, :data["n_r"][i]].astype(float)))
                          for i in range(self.n)])
        c_d = data["c_d"]
        ln_c = np.where(c_d > 0, (np.log(np.where(c_d > 0, c_d, 1.0)) - C_NORM[0]) / C_NORM[1], np.nan)
        ln_re = (np.log(data["r_e"]) - RE_NORM[0]) / RE_NORM[1]
        self.targets = np.column_stack([rings, ln_c, ln_re]).astype(np.float32)   # NaN = masked
        self.no_storage = (c_d == 0).astype(np.float32)
        self.r_e, self.c_d = data["r_e"], c_d

    def drawdown(self, idx, t):
        """p_dd of samples idx (B,) at times t (B, m), cubic spline in ln t (t within the stored range)."""
        x = np.log(t)
        j = np.clip(np.searchsorted(self.lnt, x) - 1, 0, len(self.lnt) - 2)
        h = x - self.lnt[j]
        c = self.coef[:, j, idx[:, None]]                       # (4, B, m)
        return ((c[0] * h + c[1]) * h + c[2]) * h + c[3]

    def make(self, idx, rng, t_p=None, noise=None, p_trunc=0.5, p_no_pi=0.25):
        """Examples for samples idx: (x (B, C_IN, G) float32, targets (B, N_TARGET), no_storage (B,), meta).

        noise: None draws a level per example from NOISE_RANGE; a number fixes it for all examples.
        """
        idx = np.asarray(idx)
        b = len(idx)
        t_p = np.exp(rng.uniform(*np.log(T_P_RANGE), b)) if t_p is None else np.broadcast_to(t_p, (b,)).astype(float)
        p_dt = self.drawdown(idx, np.broadcast_to(DT, (b, G)))
        p_tail = self.drawdown(idx, t_p[:, None] + DT[None, :])
        p_wf = self.drawdown(idx, t_p[:, None])[:, 0]
        dp = p_wf[:, None] - p_tail + p_dt
        sig = np.exp(rng.uniform(*np.log(NOISE_RANGE), b)) if noise is None else np.full(b, float(noise))
        dp = dp * np.exp(sig[:, None] * rng.standard_normal(dp.shape))
        p_wf = p_wf * np.exp(sig * rng.standard_normal(b))
        dp = np.maximum(dp, 1e-12)

        ln_dp = np.log(dp)
        der = np.gradient(ln_dp, LN_DT, axis=1)

        dt_max = np.where(rng.random(b) < p_trunc, np.exp(rng.uniform(np.log(DT_MAX_MIN), np.log(DT[-1]), b)), DT[-1])
        valid = (DT[None, :] <= dt_max[:, None]).astype(float)
        pi_known = (rng.random(b) >= p_no_pi).astype(float)

        x = np.zeros((b, C_IN, G))
        x[:, 0] = ln_dp / 5
        x[:, 1] = np.clip(der, -5, 5)
        x[:, 2] = (LN_DT - LN_DT.mean()) / (LN_DT[-1] - LN_DT.mean())
        x[:, 3] = ((np.log(t_p) - LN_TP_MID) / LN_TP_HALF)[:, None]
        x[:, 4] = (pi_known * np.log(p_wf) / 3)[:, None]
        x[:, 5] = pi_known[:, None]
        p_drop = p_wf[:, None] - dp
        ln_r_app = 0.5 * np.log(2 * t_p[:, None] / np.maximum(p_drop, 1e-12) + 1)
        x[:, 7] = pi_known[:, None] * np.clip((ln_r_app - RE_NORM[0]) / RE_NORM[1], -3, 3)
        x[:, 8] = (np.log10(np.maximum(sig, 1e-5)) + 3)[:, None]
        x[:, MEASURED] *= valid[:, None, :]
        x[:, 6] = valid
        meta = {"t_p": t_p, "dt_max": dt_max, "pi_known": pi_known.astype(bool), "p_wf": p_wf, "noise": sig}
        return x.astype(np.float32), self.targets[idx], self.no_storage[idx], meta


def denormalize(y):
    """Normalized targets (..., N_TARGET) -> (ring ln k, ln C_D, ln r_eD)."""
    y = np.asarray(y, float)
    return y[..., :N_RING], y[..., N_RING] * C_NORM[1] + C_NORM[0], y[..., N_RING + 1] * RE_NORM[1] + RE_NORM[0]
