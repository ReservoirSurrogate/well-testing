# Project
Build a Hankel Neural Operator similar in principle to Fourier Neural Operator.

## Problem: radial single-phase diffusivity equation (dimensionless)

Variables: r_D = r/r_w, k_D(r) = k(r)/k_ref, u = (p - p_wf)/(p_i - p_wf), t_D defined with k_ref.

    du/dt_D = (1/r_D) d/dr_D ( r_D * k_D(r_D) * du/dr_D ),   1 <= r_D <= r_eD

- Inner BC (constant bottomhole pressure): u(1, t) = 0
- Outer BC (no flow): r_D * du/dr_D = 0 at r_D = r_eD
- IC: u(r_D, 0) = 1
- k_D stays inside the derivative (heterogeneous form).

### Operator to learn

    G : ( u(., t), log k_D(.) )  ->  u(., t + dt)

- 2 input channels, 1 output channel; full trajectories by autoregressive rollout.
- For fixed k, G is linear in u, nonlinear in k.
- Sanity check: for k_D = 1 the exact step is diagonal in the annular basis:
  u_hat_n <- u_hat_n * exp(-lambda_n^2 * dt).

## Decisions

| Item | Decision |
|---|---|
| Geometry | r_eD = 1000 (fixed) |
| Varying input | k_D(r) only: log k_D ~ GRF in ln r (sigma ~ 0.5-1) + optional near-well skin-zone steps |
| Output | u at a single time t + dt |
| Time step | Fixed dt_D = 100 (tunable); later: dt as an input channel |
| Spectral modes | K = 32 (tunable) |
| Grid | Log-spaced in r_D, N = 1024 (revisit after solver) |
| Forward transform | Least squares T = (B^T W B)^-1 B^T W (T B = I exactly); quadrature kept as an option |
| Initial state | Start trajectories at t_D = t0 > 0 (from solver) to avoid the t=0 discontinuity |
| Stack | Plain PyTorch + SciPy |

## Annular Hankel basis

Eigenfunctions satisfying phi(1) = 0 and phi'(r_eD) = 0:

    phi_n(r) = J0(lambda_n r) Y0(lambda_n) - Y0(lambda_n r) J0(lambda_n)

Eigenvalue equation:

    J1(lambda_n r_eD) Y0(lambda_n) - Y1(lambda_n r_eD) J0(lambda_n) = 0

Orthogonal with weight r on [1, r_eD]. Roots found numerically (e.g. brentq);
spacing ~ pi/(r_eD - 1) for large n; lambda_1^2 ~ 2/(r_eD^2 (ln r_eD - 3/4)).

## Milestones

1. [x] **Basis & transform** - compute lambda_n for r_eD; build annular Bessel basis on log grid;
       forward (quadrature, weight r dr) and inverse transform matrices; test orthogonality and round-trip error.
       Done: `phase_1/basis.py`, `phase_1/tests/test_basis.py` (25 tests), `phase_1/grid_study.py`.
       Least-squares T chosen over quadrature (T B - I: 2e-15 vs 4e-4; coeff. error of u=1: 2e-7 vs 4e-4 at N=1024).
2. [x] **Reference solver** - implicit finite-volume solver on log grid with heterogeneous k;
       validate for k_D = 1 against the analytic eigen-series.
       Done: `phase_1/solver.py` (vertex-centered FV, log-exact face fluxes, harmonic-mean k, variable-step BDF2),
       `phase_1/tests/test_solver.py` (11 tests). k_D = 1 at N = 1024: u rel. L2 ~4e-7 and q_D ~4e-7 for t >= 1e4
       (second order in space); early times (t <= 100) limited by time step (q_D err 1.4e-3 at t = 1 with default grid).
3. [ ] **Dataset** - random log k_D(r) (GRF + skin-zone steps); generate trajectories;
       store (u_n, log k) -> u_{n+1} pairs.
4. [ ] **HNO model** - lift -> L x [annular Hankel spectral conv + pointwise W + GELU] -> project;
       real-valued spectral weights; hard constraint u(1) = 0.
5. [ ] **Training & evaluation** - one-step relative L2; rollout error over a limited window (~100-500 steps);
       well rate q_D ~ r du/dr at r_D = 1; baseline FNO-1D on the same data.

## Layout
- `phase_1/` - all phase-1 Python (code, figure scripts, `tests/`); run with `/home/daniel_88/py314/bin/python`.
- `tex/` - only .tex and .pdf (`phase1.tex` summarizes the math).

## Open topics
- Network architecture details: number of layers, width, hard-BC enforcement method.
- Rollout stability over long horizons; move to dt-as-input (log-spaced times) to reach boundary-dominated flow (t_D ~ 1e6).
