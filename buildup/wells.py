"""Build-up tools (build-up project, Milestone 2): superposition, Agarwal / Horner time, Bourdet derivative,
and a classical radial-flow interpretation.

All in dimensionless variables of tasks.md; p_dd(t) is the unit-rate drawdown well pressure of one sample.

Build-up after producing for t_p (rate 1) and shutting in for dt:
    p_w(t_p + dt) = p_dd(t_p + dt) - p_dd(dt)                       (superposition, exact for this linear problem)
    dp_bu(dt)     = p_w(t_p) - p_w(t_p + dt) = p_dd(t_p) - p_dd(t_p + dt) + p_dd(dt)   (pressure rise since shut-in)
    Agarwal equivalent time dt_e = t_p dt / (t_p + dt); Horner time (t_p + dt) / dt.

Radial flow in a homogeneous reservoir of permeability k with skin S:
    p_dd(t) = (ln(k t) + 0.80907 + 2 S) / (2 k)   =>   dp_bu = (ln(k dt_e) + 0.80907 + 2 S) / (2 k)
so the Bourdet derivative d dp_bu / d ln dt_e is 1 / (2 k) and the Horner plot p_w vs ln Horner time has slope
1 / (2 k).
"""
import numpy as np
from scipy.interpolate import CubicSpline

EULER_TERM = 0.80907                   # ln(4) - Euler's gamma
T_P_RANGE = (1e4, 1e6)                 # production time before shut-in, log-uniform (decision 2026-09-25;
                                       # was [1e3, 1e6]: too short a production barely drains to the boundary)


class Drawdown:
    """Unit-rate drawdown p_dd(t) from samples on log-spaced times; cubic spline in ln t (exact at the samples).

    Evaluation outside [t_min, t_max] raises: the build-up needs p_dd(dt) for dt >= t_min and p_dd(t_p + dt) for
    t_p + dt <= t_max.
    """

    def __init__(self, t, p):
        self.t = np.asarray(t, float)
        self._spline = CubicSpline(np.log(self.t), np.asarray(p, float))

    def __call__(self, t):
        t = np.asarray(t, float)
        if np.any(t < self.t[0] * (1 - 1e-12)) or np.any(t > self.t[-1] * (1 + 1e-12)):
            raise ValueError(f"times outside the stored range [{self.t[0]:g}, {self.t[-1]:g}]")
        return self._spline(np.log(np.clip(t, self.t[0], self.t[-1])))


def buildup_times(t_p, t_min, t_max, n=100):
    """n log-spaced shut-in times dt in [t_min, t_max - t_p]."""
    return np.geomspace(t_min, t_max - t_p, n)


def buildup(p_dd, t_p, dt):
    """(p_w(t_p + dt), dp_bu(dt)) for a drawdown callable p_dd, production time t_p and shut-in times dt."""
    dt = np.asarray(dt, float)
    p_w = p_dd(t_p + dt) - p_dd(dt)
    return p_w, p_dd(t_p) - p_w


def agarwal_time(t_p, dt):
    dt = np.asarray(dt, float)
    return t_p * dt / (t_p + dt)


def horner_time(t_p, dt):
    dt = np.asarray(dt, float)
    return (t_p + dt) / dt


def bourdet_derivative(x, y, window=0.1):
    """Bourdet derivative dy / d ln x with smoothing window L = `window` in ln x.

    For each point i the left and right neighbours are the first points at least L away in ln x; the derivative is
    the distance-weighted average of the left and right slopes (Bourdet et al. 1989). Where no neighbour is L
    away (the ends), the nearest available point on that side is used; at the very ends, the one-sided slope.
    """
    lx, y = np.log(np.asarray(x, float)), np.asarray(y, float)
    n = len(lx)
    d = np.full(n, np.nan)
    for i in range(n):
        left = np.flatnonzero(lx[:i] <= lx[i] - window)
        right = np.flatnonzero(lx[i + 1:] >= lx[i] + window) + i + 1
        j = left[-1] if len(left) else (i - 1 if i > 0 else None)
        k = right[0] if len(right) else (i + 1 if i < n - 1 else None)
        if j is None and k is None:
            continue
        if j is None:
            d[i] = (y[k] - y[i]) / (lx[k] - lx[i])
        elif k is None:
            d[i] = (y[i] - y[j]) / (lx[i] - lx[j])
        else:
            dl, dr = lx[i] - lx[j], lx[k] - lx[i]
            sl, sr = (y[i] - y[j]) / dl, (y[k] - y[i]) / dr
            d[i] = (sl * dr + sr * dl) / (dl + dr)
    return d


def interpret_radial_flow(dt_e, dp_bu, derivative, window):
    """Classical radial-flow interpretation over the shut-in times with dt_e in `window` = (lo, hi).

    Permeability from the derivative plateau m (median in the window): k = 1 / (2 m).
    Skin from the semilog straight line: S = (2 k dp_bu - ln(k dt_e) - 0.80907) / 2, median in the window.
    Returns (k, S).
    """
    dt_e = np.asarray(dt_e, float)
    m = (dt_e >= window[0]) & (dt_e <= window[1])
    if not m.any():
        raise ValueError("no shut-in times in the radial-flow window")
    k = 1.0 / (2 * np.median(derivative[m]))
    s = np.median((2 * k * np.asarray(dp_bu)[m] - np.log(k * dt_e[m]) - EULER_TERM) / 2)
    return k, s
