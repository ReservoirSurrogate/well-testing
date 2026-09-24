"""FFTLog: fast Hankel transform on logarithmic grids (Hamilton 2000, MNRAS 312, 257).

Transform pair (self-reciprocal):

    F(k) = int_0^inf f(r) J_mu(k r) r dr,        f(r) = int_0^inf F(k) J_mu(k r) k dk

On r_j = r_0 exp(j D), k_n = k_0 exp(n D) (j, n = 0..N-1), expand the biased function r f(r) r^{-q} in a
Fourier series in ln r,

    r f(r) r^{-q} = sum_m c_m (r / r_0)^{i w_m},     w_m = 2 pi m / (N D),

and integrate term by term with  int_0^inf r^s J_mu(k r) dr = 2^s Gamma((mu+1+s)/2) / Gamma((mu+1-s)/2) k^{-s-1}
(valid for -mu-1 < Re s < 1/2, so -mu-1 < q < 1/2):

    k_n^{q+1} F(k_n) = sum_m c_m U_m (k_0 r_0)^{-i w_m} exp(-2 pi i m n / N),
    U_m = 2^{q + i w_m} Gamma((mu+1+q+i w_m)/2) / Gamma((mu+1-q-i w_m)/2).

The result is periodic in n, so k_0 is a choice. We use the centered convention kappa = k_c r_c (grid centers),
i.e. k_0 r_0 = kappa exp(-(N-1) D): the output grid is k in [kappa / r_max, kappa / r_min].

Cost O(N log N). The discrete transform treats r f(r) r^{-q} as periodic in ln r over [r_0, r_0 e^{N D}), so the
input must be small at both ends of the grid (pad in ln r otherwise). For q = 0, |U_m| = 1 and applying the
transform twice (r -> k -> r) returns the input exactly, provided U at the Nyquist mode is real ("low-ringing"
kappa, see `low_ringing_kappa`); otherwise the Nyquist term is only approximately inverted.
"""
import numpy as np
from scipy.special import loggamma


def log_grid(r_min, r_max, n):
    """n log-spaced points from r_min to r_max inclusive; returns (r, D)."""
    r = np.geomspace(r_min, r_max, n)
    return r, np.log(r[1] / r[0])


def _u(mu, q, w):
    """U(w) = 2^{q+iw} Gamma((mu+1+q+iw)/2) / Gamma((mu+1-q-iw)/2) for real array w."""
    z = q + 1j * w
    return np.exp(z * np.log(2.0) + loggamma((mu + 1 + z) / 2) - loggamma((mu + 1 - z) / 2))


def low_ringing_kappa(n, dlog, mu=0.0, q=0.0, kappa=1.0):
    """kappa closest (in ln) to the given one for which U_{N/2} kappa^{-i w_{N/2}} is real (Hamilton 2000, eq. 186)."""
    w_nyq = np.pi / dlog
    theta = np.angle(_u(mu, q, np.array([w_nyq]))[0]) - w_nyq * np.log(kappa)
    return float(np.exp(np.log(kappa) + (theta - np.pi * np.round(theta / np.pi)) / w_nyq))


class FFTLog:
    """Hankel transform of order mu on a log grid of n points with spacing dlog in ln r.

    forward(f, r_0) returns (k, F) on k_n = (kappa / r_{N-1}) exp(n dlog), i.e. k in [kappa / r_max, kappa / r_min].
    Because the pair is self-reciprocal, the inverse is forward(F, k_0), which returns the original r grid.
    """

    def __init__(self, n, dlog, mu=0.0, q=0.0, kappa=1.0, low_ringing=True):
        if not -mu - 1 < q < 0.5:
            raise ValueError(f"bias q = {q} outside (-mu-1, 1/2) = ({-mu - 1}, 0.5)")
        self.n, self.dlog, self.mu, self.q = n, dlog, mu, q
        self.kappa = low_ringing_kappa(n, dlog, mu, q, kappa) if low_ringing else kappa
        w = 2 * np.pi * np.arange(n // 2 + 1) / (n * dlog)
        log_k0r0 = np.log(self.kappa) - (n - 1) * dlog                            # k_0 r_0 from centered kappa
        self.multiplier = _u(mu, q, w) * np.exp(-1j * w * log_k0r0)              # U_m (k_0 r_0)^{-i w_m}, m >= 0

    def output_grid(self, x_0):
        """Output grid for an input grid starting at x_0: kappa / x_{N-1} * exp(n dlog)."""
        return self.kappa / (x_0 * np.exp(self.dlog * (self.n - 1))) * np.exp(self.dlog * np.arange(self.n))

    def forward(self, f, x_0):
        """Transform samples f (..., n) given on x_j = x_0 exp(j dlog); returns (y, F) with F on y = output_grid(x_0)."""
        f = np.asarray(f, dtype=float)
        x = x_0 * np.exp(self.dlog * np.arange(self.n))
        y = self.output_grid(x_0)
        c = np.fft.rfft(f * x ** (1 - self.q), axis=-1) / self.n
        # sum_m d_m exp(-2 pi i m n / N) for Hermitian d = n * irfft(conj(d)); irfft keeps Re(d) at Nyquist
        out = self.n * np.fft.irfft(np.conj(c * self.multiplier), n=self.n, axis=-1)
        return y, out / y ** (self.q + 1)

    def inverse(self, F, y_0):
        """Exact discrete inverse of forward for any q: F given on y_j = y_0 exp(j dlog) (the output grid of
        forward) -> (x, f) on the original input grid. For q = 0 this equals forward(F, y_0) (self-reciprocal)."""
        F = np.asarray(F, dtype=float)
        y = y_0 * np.exp(self.dlog * np.arange(self.n))
        x = self.output_grid(y_0)
        d = np.conj(np.fft.rfft(F * y ** (self.q + 1), axis=-1)) / self.n
        a = self.n * np.fft.irfft(d / self.multiplier, n=self.n, axis=-1)
        return x, a / x ** (1 - self.q)
