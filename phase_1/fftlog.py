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

kernel="Y" computes G(k) = int_0^inf f(r) Y_mu(k r) r dr the same way, with the Mellin transform of Y_mu,

    int_0^inf r^s Y_mu(k r) dr = -(2^s / pi) Gamma((1+s+mu)/2) Gamma((1+s-mu)/2) cos((1+s-mu) pi / 2) k^{-s-1}

(valid for |mu| < 1 + Re s < 3/2, so |mu| - 1 < q < 1/2). It is not self-reciprocal.

Weber transform (domain r >= a with f(a) = 0, e.g. outside a well of radius a):

    W(k) = int_a^inf f(r) C(k, r) r dr,   C(k, r) = J0(k r) Y0(k a) - Y0(k r) J0(k a),
    f(r) = int_0^inf W(k) C(k, r) k dk / (J0(k a)^2 + Y0(k a)^2).

C(k, a) = 0, and for f(a) = 0 the transform diagonalizes the radial Laplacian: W[f'' + f'/r] = -k^2 W[f].
`Weber` computes it from one J0 and one Y0 FFTLog on a common k grid:
W = Y0(k a) H_J[f] - J0(k a) H_Y[f], and the inverse likewise with the roles of r and k swapped.
"""
import numpy as np
from scipy.special import j0, loggamma, y0


def log_grid(r_min, r_max, n):
    """n log-spaced points from r_min to r_max inclusive; returns (r, D)."""
    r = np.geomspace(r_min, r_max, n)
    return r, np.log(r[1] / r[0])


def _u(mu, q, w):
    """U(w) = 2^{q+iw} Gamma((mu+1+q+iw)/2) / Gamma((mu+1-q-iw)/2) for real array w."""
    z = q + 1j * w
    return np.exp(z * np.log(2.0) + loggamma((mu + 1 + z) / 2) - loggamma((mu + 1 - z) / 2))


def _log_cos(z):
    """log cos(z) for complex z, stable for large |Im z| (cos grows like exp(|Im z|) / 2)."""
    z = np.asarray(z, dtype=complex)
    sgn = np.where(z.imag >= 0, 1.0, -1.0)
    # cos z = exp(-i sgn z) / 2 * (1 + exp(2 i sgn z)); the second factor is O(1) since |exp(2 i sgn z)| <= 1
    return -1j * sgn * z - np.log(2.0) + np.log1p(np.exp(2j * sgn * z))


def _u_y(mu, q, w):
    """U_Y(w) = -(2^z / pi) Gamma((1+z+mu)/2) Gamma((1+z-mu)/2) cos((1+z-mu) pi / 2), z = q + iw."""
    z = q + 1j * w
    log_u = (z * np.log(2.0) - np.log(np.pi) + loggamma((1 + z + mu) / 2) + loggamma((1 + z - mu) / 2)
             + _log_cos((1 + z - mu) * np.pi / 2))
    return -np.exp(log_u)


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

    def __init__(self, n, dlog, mu=0.0, q=0.0, kappa=1.0, low_ringing=True, kernel="J"):
        """kernel "J" (Hankel transform) or "Y" (Y_mu kernel). low_ringing adjusts kappa for the J multiplier
        in both cases, so J and Y transforms built with the same arguments share their output grid."""
        lo = -mu - 1 if kernel == "J" else abs(mu) - 1
        if not lo < q < 0.5:
            raise ValueError(f"bias q = {q} outside ({lo}, 0.5) for kernel {kernel}")
        self.n, self.dlog, self.mu, self.q, self.kernel = n, dlog, mu, q, kernel
        self.kappa = low_ringing_kappa(n, dlog, mu, q, kappa) if low_ringing else kappa
        w = 2 * np.pi * np.arange(n // 2 + 1) / (n * dlog)
        log_k0r0 = np.log(self.kappa) - (n - 1) * dlog                            # k_0 r_0 from centered kappa
        u = {"J": _u, "Y": _u_y}[kernel](mu, q, w)
        self.multiplier = u * np.exp(-1j * w * log_k0r0)                           # U_m (k_0 r_0)^{-i w_m}, m >= 0

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


class Weber:
    """Weber transform of order 0 outside r = a (see module docstring) on a log grid of n points, spacing dlog.

    forward(f, r_0): f on r_j = r_0 exp(j dlog) (values at r < a are ignored) -> (k, W) on the J-FFTLog output grid.
    inverse(W, k_0): back to the r grid; the result is only meaningful for r >= a.
    Both halves use the same bias q, so the r grid and k grid are shared by the J0 and Y0 passes.
    """

    def __init__(self, n, dlog, a=1.0, q=0.0, kappa=1.0):
        self.a = a
        self.hj = FFTLog(n, dlog, mu=0.0, q=q, kappa=kappa, kernel="J")
        self.hy = FFTLog(n, dlog, mu=0.0, q=q, kappa=self.hj.kappa, low_ringing=False, kernel="Y")

    def forward(self, f, r_0):
        r = r_0 * np.exp(self.hj.dlog * np.arange(self.hj.n))
        f = np.where(r >= self.a * (1 - 1e-12), np.asarray(f, dtype=float), 0.0)
        k, fj = self.hj.forward(f, r_0)
        _, fy = self.hy.forward(f, r_0)
        return k, y0(k * self.a) * fj - j0(k * self.a) * fy

    def inverse(self, w, k_0):
        k = k_0 * np.exp(self.hj.dlog * np.arange(self.hj.n))
        ja, ya = j0(k * self.a), y0(k * self.a)
        g = np.asarray(w, dtype=float) / (ja**2 + ya**2)
        r, fj = self.hj.forward(g * ya, k_0)
        _, fy = self.hy.forward(g * ja, k_0)
        return r, fj - fy
