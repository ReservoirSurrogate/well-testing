"""Random radial permeability fields and their well-test labels (build-up project, Milestone 3).

Field (per sample, on the log grid r of the solver):
    ln k_D(r) = m + g(ln r) + skin_zone(r),  clipped to [-3, 3]
    m ~ N(0, 0.3^2)                                  mean shift
    g: zero-mean Gaussian random field in x = ln r, covariance sigma^2 exp(-(x - x')^2 / (2 ell^2)),
       sigma ~ U(0.5, 1), ell ~ U(0.3, 1.5)
    skin zone (probability 0.5): ln(k_s / k) ~ U(ln 0.1, ln 5) added for r < r_s, r_s ~ logU[2, 20]

Labels (tasks.md), from the steady radial-flow resistance R(a, b) = int_a^b dr / (r k) = int d(ln r) / k,
evaluated with the solver's discrete face permeabilities (harmonic means, R = sum D / k_face over the faces):
    k_eff = ln(r_e / r_1) / R(r_1, r_e)     radial-flow permeability outside the skin zone: r_1 = first grid node at
                                            or beyond r_s (the first node without the skin offset; r_1 = 1 without
                                            a zone), so the face straddling the step is not counted
    S     = k_eff R(1, r_e) - ln r_e          equivalent skin relative to k_eff (0 without a skin zone)
For a homogeneous formation k with a skin zone k_s this is Hawkins' formula S = (k / k_s - 1) ln r_s.
"""
import numpy as np

LOGK_CLIP = 3.0
P_SKIN = 0.5


def grf_factor(x, sigma, ell):
    """Matrix F with F F^T = the squared-exponential covariance on the points x (eigendecomposition: robust for the
    numerically singular covariance of smooth fields on fine grids)."""
    cov = sigma**2 * np.exp(-0.5 * ((x[:, None] - x[None, :]) / ell) ** 2)
    w, v = np.linalg.eigh(cov)
    return v * np.sqrt(np.clip(w, 0.0, None))


def grf(rng, x, sigma, ell):
    """One zero-mean Gaussian field on points x with squared-exponential covariance.

    Uniform x (the log grid): circulant embedding - the covariance is extended periodically over m >= 2 (n + 10 ell/dx)
    points, where the kernel has decayed to ~1e-22, and diagonalized by the FFT; the few tiny negative eigenvalues
    are clipped. Milliseconds instead of an O(n^3) eigendecomposition. Non-uniform x: eigendecomposition.
    """
    dx = np.diff(x)
    if not np.allclose(dx, dx[0], rtol=1e-8):
        return grf_factor(x, sigma, ell) @ rng.standard_normal(len(x))
    n = len(x)
    m = 1 << int(np.ceil(np.log2(2 * (n + int(np.ceil(10 * ell / dx[0]))))))
    lag = dx[0] * np.minimum(np.arange(m), m - np.arange(m))
    lam = np.clip(np.fft.fft(sigma**2 * np.exp(-0.5 * (lag / ell) ** 2)).real, 0.0, None)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    return np.fft.fft(np.sqrt(lam / m) * z).real[:n]


def sample_logk(rng, r, p_skin=P_SKIN):
    """Random ln k_D on the grid r; returns (logk, meta)."""
    meta = {"mean": rng.normal(0.0, 0.3), "sigma": rng.uniform(0.5, 1.0), "ell": rng.uniform(0.3, 1.5),
            "has_skin": bool(rng.random() < p_skin), "r_s": np.nan, "skin_log": 0.0}
    logk = meta["mean"] + grf(rng, np.log(r), meta["sigma"], meta["ell"])
    if meta["has_skin"]:
        meta["r_s"] = float(np.exp(rng.uniform(np.log(2.0), np.log(20.0))))
        meta["skin_log"] = float(rng.uniform(np.log(0.1), np.log(5.0)))
        logk = np.where(r < meta["r_s"], logk + meta["skin_log"], logk)
    return np.clip(logk, -LOGK_CLIP, LOGK_CLIP), meta


def resistance(r, k, a, b):
    """R(a, b) = int_a^b d(ln r) / k with the solver's harmonic face permeabilities, linear in ln r within a face."""
    lx = np.log(r)
    k_face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])
    cum = np.concatenate([[0.0], np.cumsum(np.diff(lx) / k_face)])
    return np.interp(np.log(b), lx, cum) - np.interp(np.log(a), lx, cum)


def labels(r, k, r_s=np.nan):
    """(k_eff, S) of a permeability profile k on the grid r (r[0] = 1, r[-1] = r_e); r_s = NaN without a skin zone."""
    r_e = r[-1]
    r_1 = 1.0 if not np.isfinite(r_s) else r[min(np.searchsorted(r, r_s), len(r) - 2)]
    k_eff = np.log(r_e / r_1) / resistance(r, k, r_1, r_e)
    return float(k_eff), float(k_eff * resistance(r, k, 1.0, r_e) - np.log(r_e))


RINGS = ((1, 2), (2, 5), (5, 10), (10, 30), (30, 100), (100, 300), (300, 1000), (1000, 3000))


MIN_RING_RATIO = 1.5        # a ring clipped at r_e counts if its part inside spans at least this radius ratio


def ring_labels(r, k, rings=RINGS):
    """ln of the log-harmonic mean of k over each ring [a, min(b, r_e)] in r_D; NaN where the part inside the
    reservoir spans less than MIN_RING_RATIO (r_e = r[-1])."""
    lx = np.log(r)
    k_face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])
    cum = np.concatenate([[0.0], np.cumsum(np.diff(lx) / k_face)])
    a, b = np.array(rings, float).T
    b = np.minimum(b, r[-1])
    valid = b / a >= MIN_RING_RATIO
    a, b = a[valid], b[valid]
    out = np.full(len(rings), np.nan)
    out[valid] = np.log(np.log(b / a) / (np.interp(np.log(b), lx, cum) - np.interp(np.log(a), lx, cum)))
    return out
