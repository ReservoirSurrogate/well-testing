"""Finite-volume solver for a rate-controlled well in a closed radial reservoir (build-up project, Milestone 1).

Dimensionless problem (see tasks.md), p = p_D (pressure drop, 0 initially):

    dp/dt = (1/r) d/dr ( r k(r) dp/dr ),   1 <= r <= r_e,    r dp/dr = 0 at r = r_e

Well (radius 1) with wellbore storage C and thin skin S >= 0, rate history q(t):

    C dp_w/dt + q_sf = q(t)          q_sf = -k(1) r dp/dr |_{r=1}   (sandface rate)
    p_w = p(1) + S q_sf               (thin skin: extra drop proportional to the sandface rate)

S is scaled with k_ref (like p_D). The well-test skin, defined relative to the formation permeability k, is
S_wt = k S: e.g. for homogeneous k, radial flow gives p_w = (ln(k t) + 0.80907 + 2 k S) / (2 k).

Discretization:
- log grid r_j = exp(j D), j = 0..N-1 (r_0 = 1, r_{N-1} = r_e), vertex-centered cells between the faces
  r_{j+1/2} = sqrt(r_j r_{j+1}); cell storage V_j = int r dr over the cell (half cells at both ends);
- face flux F_{j+1/2} = r k dp/dr = k_{j+1/2} (p_{j+1} - p_j) / D: exact for steady radial flow; k_{j+1/2} is the
  harmonic mean of the node values (exact for a permeability jump at the face);
- unknowns x = [p_w, p_0, ..., p_{N-1}]: the system stays tridiagonal. Equations
      C dp_w/dt = q - (p_w - p_0) / S        V_0 dp_0/dt = F_{1/2} + (p_w - p_0) / S
  and for S = 0 the constraint p_w = p_0 replaces the skin conductance (well and node 0 merge:
  (C + V_0) dp_0/dt = F_{1/2} + q). With C = 0 the well equation is algebraic (index-1 DAE), fine for BDF.
- variable-step BDF2, restarted with a backward-Euler step after every rate change (q is piecewise constant).

Conservation: sum_j V_j p_j + C p_w = int q dt holds to rounding error at every step (tested).
"""
import numpy as np
from scipy.linalg import solve_banded


def log_grid(r_e, n):
    """n log-spaced nodes from 1 to r_e; returns (r, D)."""
    r = np.exp(np.linspace(0.0, np.log(r_e), n))
    return r, np.log(r_e) / (n - 1)


def grid_for(r_e, dlog):
    """Log grid from 1 to r_e with spacing as close as possible to dlog (at least 3 nodes)."""
    n = max(int(round(np.log(r_e) / dlog)) + 1, 3)
    return log_grid(r_e, n)


def cell_storage(r):
    """V_j = int r dr over each node's cell."""
    faces = np.sqrt(r[:-1] * r[1:])
    edges = np.concatenate([[r[0]], faces, [r[-1]]])
    return (edges[1:] ** 2 - edges[:-1] ** 2) / 2


def time_grid(t_out, h0=1e-4, growth=1.05, breaks=()):
    """Time steps from 0 through all output times and rate-change times.

    Steps start at h0 and grow geometrically by `growth` (relative to the time since the last rate change),
    capped at `growth - 1` times the elapsed time since that change; output and break times are hit exactly.
    Returns (t, is_break) with t[0] = 0; is_break[i] marks t[i] as a rate-change time (restart BDF after it).
    """
    marks = np.unique(np.concatenate([np.asarray(t_out, float), np.asarray(breaks, float)]))
    brk = set(float(b) for b in breaks)
    t, flags, last_break, h = [0.0], [False], 0.0, h0
    for m in marks:
        while t[-1] < m * (1 - 1e-12):
            since = t[-1] - last_break
            h = max(h0, min(h * growth, (growth - 1) * max(since, h0 / (growth - 1))))
            t.append(min(t[-1] + h, m))
            flags.append(False)
        flags[-1] = float(m) in brk
        if flags[-1]:
            last_break, h = t[-1], h0
    return np.array(t), np.array(flags)


class WellSolver:
    """Implicit solver for one permeability field k (node values on the log grid r)."""

    def __init__(self, r, k, c_d=0.0, skin=0.0):
        r, k = np.asarray(r, float), np.asarray(k, float)
        if skin < 0:
            raise ValueError("thin skin must be >= 0 (use the permeability profile for negative skin)")
        dlog = np.diff(np.log(r))
        if not np.allclose(dlog, dlog[0], rtol=1e-8):
            raise ValueError("the grid must be log-spaced")
        self.r, self.k, self.c_d, self.skin = r, k, float(c_d), float(skin)
        self.n = len(r)
        k_face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])
        self.trans = k_face / dlog                                   # F_{j+1/2} = trans_j (p_{j+1} - p_j)

        # system M dx/dt = A x + b q over x = [p_w, p_0, ..., p_{N-1}] (merged well/node 0 if S = 0)
        m = self.n + 1
        mass = np.concatenate([[self.c_d], cell_storage(r)])
        lower, diag, upper = np.zeros(m), np.zeros(m), np.zeros(m)  # A as three diagonals (row-indexed)
        diag[2:] -= self.trans
        diag[1:-1] -= self.trans
        upper[1:-1] = self.trans                                     # row i, column i+1
        lower[2:] = self.trans                                       # row i, column i-1
        b = np.zeros(m)
        if self.skin > 0:
            g = 1.0 / self.skin
            diag[0] -= g
            upper[0] = g
            diag[1] -= g
            lower[1] = g
            b[0] = 1.0
        else:
            mass[1] += mass[0]                                       # (C + V_0) dp_0/dt = F_{1/2} + q
            mass[0] = 0.0
            diag[0], upper[0] = -1.0, 1.0                            # algebraic row: p_0 - p_w = 0
            b[1] = 1.0
        self.mass, self.lower, self.diag, self.upper, self.b = mass, lower, diag, upper, b

    def _solve(self, a0, rhs):
        """Solve (a0 M - A) x = rhs."""
        ab = np.zeros((3, self.n + 1))
        ab[0, 1:] = -self.upper[:-1]
        ab[1] = a0 * self.mass - self.diag
        ab[2, :-1] = -self.lower[1:]
        return solve_banded((1, 1), ab, rhs)

    def integrate(self, t_grid, rate, is_break=None):
        """States x(t) on t_grid (t_grid[0] = 0, x = 0 there); rate(t) is the rate on (t_prev, t].

        `rate` is a callable returning the (piecewise-constant) rate for the step ending at t; is_break[i] marks
        t_grid[i] as a rate change, after which BDF2 restarts with a backward-Euler step.
        Returns x of shape (len(t_grid), N + 1): column 0 = p_w, columns 1.. = p on the grid.
        """
        t_grid = np.asarray(t_grid, float)
        is_break = np.zeros(len(t_grid), bool) if is_break is None else np.asarray(is_break, bool)
        x = np.zeros((len(t_grid), self.n + 1))
        restart = True
        for i in range(1, len(t_grid)):
            h = t_grid[i] - t_grid[i - 1]
            q = rate(t_grid[i])
            if restart:                                              # backward Euler
                rhs = self.mass * x[i - 1] / h + self.b * q
                x[i] = self._solve(1.0 / h, rhs)
            else:                                                    # variable-step BDF2
                w = h / (t_grid[i - 1] - t_grid[i - 2])
                a0 = (1 + 2 * w) / (1 + w)
                hist = (1 + w) * x[i - 1] - w**2 / (1 + w) * x[i - 2]
                rhs = self.mass * hist / h + self.b * q
                x[i] = self._solve(a0 / h, rhs)
            if self.skin == 0:
                x[i, 0] = x[i, 1]
            restart = bool(is_break[i])
        return x

    def well_pressure(self, t_out, rate=lambda _t: 1.0, breaks=(), **grid_kw):
        """p_w at the output times t_out for a rate history with the given break times."""
        t_grid, is_break = time_grid(t_out, breaks=breaks, **grid_kw)
        x = self.integrate(t_grid, rate, is_break)
        idx = np.searchsorted(t_grid, np.asarray(t_out, float))
        return x[idx, 0]

    def cumulative_storage(self, x):
        """sum_j V_j p_j + C p_w for each state (equals the produced volume int q dt)."""
        v = cell_storage(self.r)
        return x[:, 1:] @ v + self.c_d * x[:, 0]
