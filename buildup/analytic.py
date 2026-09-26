"""Analytic references for a constant-rate well with k_D = 1 (validation of solver.py; build-up project).

- line source, infinite reservoir: p_wD = E1(1 / (4 t)) / 2 ~ (ln t + 0.80907) / 2
- finite wellbore, closed circle r_e (van Everdingen & Hurst 1949), in Laplace space:
      f(s) = s p_wD0(s) = [K0(a) I1(b) + I0(a) K1(b)] / (a [I1(b) K1(a) - K1(b) I1(a)]),  a = sqrt(s), b = a r_e
  with wellbore storage C and skin S (Agarwal et al. 1970):
      p_wD(s) = (f + S) / (s [1 + C s (f + S)])
  inverted numerically with the Gaver-Stehfest algorithm;
- pseudo-steady state (late time, closed circle): p_wD ~ 2 t / (r_e^2 - 1) + ln r_e - 3/4 + S.
"""
from math import factorial

import numpy as np
from scipy.special import exp1, i0e, i1e, k0e, k1e


def line_source(t):
    return 0.5 * exp1(1.0 / (4.0 * np.asarray(t, float)))


def pss(t, r_e, skin=0.0):
    return 2 * np.asarray(t, float) / (r_e**2 - 1) + np.log(r_e) - 0.75 + skin


def _f_closed(s, r_e):
    """s times the Laplace transform of p_wD without storage and skin (exponentially scaled Bessel functions)."""
    a = np.sqrt(s)
    b = a * r_e
    e = np.exp(2 * (a - b))                                   # ratio exp(a - b) / exp(b - a) of the scalings
    num = k0e(a) * i1e(b) + i0e(a) * k1e(b) * e
    den = i1e(b) * k1e(a) - k1e(b) * i1e(a) * e
    return num / (a * den)


def laplace_well(s, r_e, c_d=0.0, skin=0.0):
    f = _f_closed(s, r_e) + skin
    return f / (s * (1 + c_d * s * f))


def stehfest_weights(n=12):
    """Gaver-Stehfest weights V_i, i = 1..n (n even)."""
    h = n // 2
    v = np.zeros(n)
    for i in range(1, n + 1):
        acc = 0.0
        for k in range((i + 1) // 2, min(i, h) + 1):
            acc += (k**h * factorial(2 * k)
                    / (factorial(h - k) * factorial(k) * factorial(k - 1) * factorial(i - k) * factorial(2 * k - i)))
        v[i - 1] = (-1) ** (i + h) * acc
    return v


def stehfest(fs, t, n=12):
    """Inverse Laplace transform of fs(s) at times t (array)."""
    t = np.asarray(t, float)
    v = stehfest_weights(n)
    ln2 = np.log(2.0)
    s = np.outer(ln2 / t, np.arange(1, n + 1))
    return ln2 / t * (fs(s) @ v)


def well_pressure(t, r_e, c_d=0.0, skin=0.0, n=12):
    """Constant-rate p_wD(t) for k_D = 1, finite wellbore, closed circle, storage and skin."""
    return stehfest(lambda s: laplace_well(s, r_e, c_d, skin), t, n)
