"""Annular finite Hankel basis on [1, r_e] (Milestone 1).

Eigenfunctions of (r phi')' + lambda^2 r phi = 0 with phi(1) = 0, phi'(r_e) = 0:

    phi_n(r) = J0(lambda_n r) Y0(lambda_n) - Y0(lambda_n r) J0(lambda_n)
    J1(lambda_n r_e) Y0(lambda_n) - Y1(lambda_n r_e) J0(lambda_n) = 0

Transform pair (weight r):

    u_hat_n = (1/N_n) int_1^{r_e} r u phi_n dr,    u = sum_n u_hat_n phi_n

See tex/phase1.tex, Section 2.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1


def eig_residual(lam, r_e):
    """Left-hand side of the eigenvalue equation; zero at lambda_n."""
    return j1(lam * r_e) * y0(lam) - y1(lam * r_e) * j0(lam)


def eigenvalues(n_modes, r_e):
    """First n_modes roots lambda_n of the eigenvalue equation, ascending.

    Brackets sign changes on a scan grid much finer than the asymptotic root
    spacing pi/(r_e - 1), then refines each root with Brent's method.
    """
    spacing = np.pi / (r_e - 1)
    step = spacing / 50
    lam_hi = (n_modes + 2) * spacing  # lambda_1 < spacing, so n_modes roots lie below this
    grid = np.arange(step / 10, lam_hi, step)
    vals = eig_residual(grid, r_e)
    idx = np.nonzero(vals[:-1] * vals[1:] < 0)[0]
    if len(idx) < n_modes:
        raise RuntimeError(f"found {len(idx)} roots below {lam_hi:.3e}, need {n_modes}")
    return np.array([brentq(eig_residual, grid[i], grid[i + 1], args=(r_e,), xtol=1e-15)
                     for i in idx[:n_modes]])


def phi(r, lam):
    """phi_n(r); broadcasts over r and lam."""
    return j0(lam * r) * y0(lam) - y0(lam * r) * j0(lam)


def dphi(r, lam):
    """d phi_n / dr = -lambda [J1(lambda r) Y0(lambda) - Y1(lambda r) J0(lambda)]."""
    return -lam * (j1(lam * r) * y0(lam) - y1(lam * r) * j0(lam))


def norms(lam, r_e):
    """N_n = int_1^{r_e} r phi_n^2 dr = (r_e^2/2) phi_n(r_e)^2 - 2/(pi^2 lambda_n^2)."""
    return r_e**2 / 2 * phi(r_e, lam) ** 2 - 2 / (np.pi**2 * lam**2)


def make_grid(n_points, r_e, alpha=1.0):
    """Grid r_j = r(s_j) on uniform s_j in [0, 1] with trapezoidal weights for int r f dr.

    r(s) = 1 + (r_e - 1) [alpha (r_e^s - 1)/(r_e - 1) + (1 - alpha) s]
    alpha = 1: log-spaced; alpha = 0: uniform; in between: hybrid.

    Returns (r, w) with int_1^{r_e} r f(r) dr ~= sum_j w_j f(r_j).
    """
    s = np.linspace(0.0, 1.0, n_points)
    log_re = np.log(r_e)
    r = 1 + alpha * (np.exp(s * log_re) - 1) + (1 - alpha) * (r_e - 1) * s
    dr_ds = alpha * log_re * np.exp(s * log_re) + (1 - alpha) * (r_e - 1)
    c = np.full(n_points, 1.0 / (n_points - 1))
    c[[0, -1]] *= 0.5
    return r, c * r * dr_ds


@dataclass
class HankelBasis:
    """Discrete annular Hankel transform on a fixed grid.

    T (K x N): forward, u_hat = T @ u.   B (N x K): inverse, u = B @ u_hat.

    method="lstsq" (default): T = (B^T W B)^{-1} B^T W, the W-weighted least-squares
        inverse of B. T @ B = I exactly and B @ T is a projector onto the K modes.
    method="quadrature": T_nj = w_j phi_n(r_j) / N_n, the direct discretization of the
        continuous transform; T @ B = I only up to O(h^2) quadrature error.
    """
    r_e: float
    r: np.ndarray
    w: np.ndarray
    lam: np.ndarray
    norm: np.ndarray
    T: np.ndarray
    B: np.ndarray

    @classmethod
    def build(cls, n_modes, n_points, r_e, alpha=1.0, method="lstsq"):
        lam = eigenvalues(n_modes, r_e)
        norm = norms(lam, r_e)
        r, w = make_grid(n_points, r_e, alpha)
        B = phi(r[:, None], lam[None, :])
        BtW = B.T * w
        if method == "lstsq":
            T = np.linalg.solve(BtW @ B, BtW)
        elif method == "quadrature":
            T = BtW / norm[:, None]
        else:
            raise ValueError(f"unknown method {method!r}")
        return cls(r_e, r, w, lam, norm, T, B)

    def forward(self, u):
        return self.T @ u

    def inverse(self, u_hat):
        return self.B @ u_hat


def exact_coeffs_ones(lam, r_e):
    """Exact Hankel coefficients of u = 1: -2 / (pi lambda_n^2 N_n)."""
    return -2 / (np.pi * lam**2 * norms(lam, r_e))


def exact_solution(r, t, lam, r_e):
    """Homogeneous (k_D = 1) solution u(r, t) from u(r, 0) = 1, truncated to len(lam) modes."""
    coef = exact_coeffs_ones(lam, r_e) * np.exp(-lam**2 * t)
    return phi(np.asarray(r)[..., None], lam) @ coef


def exact_rate(t, lam, r_e):
    """Homogeneous well rate q_D(t) = sum_n 4/(pi^2 lambda_n^2 N_n) exp(-lambda_n^2 t)."""
    amp = 4 / (np.pi**2 * lam**2 * norms(lam, r_e))
    return np.exp(-np.multiply.outer(np.asarray(t), lam**2)) @ amp
