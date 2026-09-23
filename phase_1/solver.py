"""Finite-volume reference solver for the radial diffusivity equation (Milestone 2).

    du/dt = (1/r) d/dr ( r k(r) du/dr ),   u(1) = 0,   r du/dr (r_e) = 0

Vertex-centered finite volumes on the same grid as the Hankel basis (basis.make_grid):
- control volume of node j spans the faces r_{j-1/2}, r_{j+1/2} = sqrt(r_j r_{j+1});
  its storage is V_j = int r dr = (r_{j+1/2}^2 - r_{j-1/2}^2) / 2 (half cells at the ends);
- face flux F_{j+1/2} = r k du/dr = k_{j+1/2} (u_{j+1} - u_j) / ln(r_{j+1}/r_j), exact for steady
  radial flow (u linear in ln r); k_{j+1/2} is the harmonic mean of k_j, k_{j+1}, exact for a
  permeability jump at the face;
- node 0 is Dirichlet (u = 0); the outer face flux is zero;
- well rate q_D = F_{1/2} (flux into the fixed-pressure node 0).

Time integration: variable-step BDF2, first step backward Euler, on a user-supplied time grid.
"""
import numpy as np
from scipy.linalg import solve_banded


def face_radii(r):
    """Geometric-midpoint faces between nodes, length N-1."""
    return np.sqrt(r[:-1] * r[1:])


def storage(r):
    """Control-volume storage V_j = int r dr over each node's cell, length N."""
    f = face_radii(r)
    edges = np.concatenate([[r[0]], f, [r[-1]]])
    return (edges[1:] ** 2 - edges[:-1] ** 2) / 2


def transmissibility(r, k):
    """Face transmissibility k_{j+1/2} / ln(r_{j+1}/r_j), length N-1."""
    k_face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])
    return k_face / np.log(r[1:] / r[:-1])


class RadialSolver:
    """Implicit FV solver for one permeability field k (values at the grid nodes)."""

    def __init__(self, r, k):
        r = np.asarray(r, dtype=float)
        k = np.broadcast_to(np.asarray(k, dtype=float), r.shape).copy()
        self.r, self.k = r, k
        self.V = storage(r)
        self.trans = transmissibility(r, k)

    def face_flux(self, u):
        """F_{j+1/2} = r k du/dr on the N-1 interior faces."""
        return self.trans * np.diff(u, axis=-1)

    def rate(self, u):
        """Well rate q_D = F_{1/2}."""
        return self.face_flux(u)[..., 0]

    def _banded(self, a):
        """Banded form of (a * M - A) on the unknown nodes 1..N-1, where M u' = A u."""
        tr = self.trans
        n = len(self.r) - 1
        ab = np.zeros((3, n))
        diag = a * self.V[1:] + tr               # every unknown node has an inner face
        diag[:-1] += tr[1:]                      # outer face of all but the last node
        ab[1] = diag
        ab[0, 1:] = -tr[1:]                      # super-diagonal
        ab[2, :-1] = -tr[1:]                     # sub-diagonal
        return ab

    def integrate(self, u0, t_grid):
        """Integrate from t_grid[0] to t_grid[-1]; returns u at every t_grid point, shape (len(t_grid), N).

        Variable-step BDF2 (backward Euler for the first step). With omega = h_n / h_{n-1}:
            (1+2w)/(1+w) u^{n+1} - (1+w) u^n + w^2/(1+w) u^{n-1} = h_n M^{-1} A u^{n+1}.
        u0[0] is overwritten by the boundary value 0.
        """
        t_grid = np.asarray(t_grid, dtype=float)
        h = np.diff(t_grid)
        if np.any(h <= 0):
            raise ValueError("t_grid must be strictly increasing")

        out = np.empty((len(t_grid), len(self.r)))
        out[0] = u0
        out[:, 0] = 0.0
        V = self.V[1:]
        cache = {}

        for n, hn in enumerate(h):
            if n == 0:
                a, rhs = 1.0 / hn, V * out[0, 1:] / hn
            else:
                w = hn / h[n - 1]
                a = (1 + 2 * w) / ((1 + w) * hn)
                rhs = V * ((1 + w) * out[n, 1:] - w**2 / (1 + w) * out[n - 1, 1:]) / hn
            key = round(a, 12)
            if key not in cache:
                cache[key] = self._banded(a)
            out[n + 1, 1:] = solve_banded((1, 1), cache[key], rhs)
        return out


def time_grid(t_out, h0=1e-3, growth=1.1, h_max=5.0):
    """Time grid from 0 that contains every t_out exactly.

    Steps start at h0 and grow geometrically by `growth` up to h_max, which resolves the
    early-time front at the well. Steps are shortened to land exactly on each output time
    (a remainder under half a step is merged into the previous step), so step ratios stay
    well inside the BDF2 zero-stability limit 1 + sqrt(2).

    Returns (t_grid, idx) with t_grid[idx] == t_out.
    """
    t_out = np.asarray(t_out, dtype=float)
    ts, idx, t, h = [0.0], [], 0.0, h0
    for target in t_out:
        while t < target:
            step = min(h, target - t)
            if target - (t + step) < 0.5 * step:  # absorb a tiny remainder into this step
                step = target - t
            t += step
            ts.append(t)
            h = min(step * growth, h_max)
        idx.append(len(ts) - 1)
    return np.array(ts), np.array(idx)
