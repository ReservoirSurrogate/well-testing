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
| Time step | Fixed dt_D = 1000, 500 steps (t_D up to 5e5: infinite-acting -> boundary arrival -> early decline); later: dt as an input channel |
| Spectral modes | K = 32 (tunable) |
| Grid | Log-spaced in r_D, N = 1024 (revisit after solver) |
| Forward transform | Least squares T = (B^T W B)^-1 B^T W (T B = I exactly); quadrature kept as an option |
| Initial state | Start at t_D = 0 from u = 1 (surrogate runs without the solver). With dt_D = 1000 and K = 32 the exact first step is representable to 3e-7 (vs 1.5e-2 at dt_D = 100) |
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
3. [x] **Dataset** - random log k_D(r) (GRF + skin-zone steps); generate trajectories;
       store (u_n, log k) -> u_{n+1} pairs.
       Plan: `phase_1/dataset.py`. log k = mean N(0, 0.3^2) + SE-covariance GRF in ln r (sigma ~ U(0.5, 1),
       l ~ U(0.3, 1.5)) + skin offset (p = 0.5, r_s log-U[2, 20], ln(k_s/k) ~ U(ln 0.1, ln 5)), clipped to [-3, 3].
       800 / 100 / 100 trajectories of 501 states (t_D = 0, 1000, ..., 5e5) on the N = 1024 grid, float32 .npz
       per split in `phase_1/data/` (git-ignored); pairs formed at training time. Solver time grid h_max = 20.
       Done: data generated (~28 min, 8 workers, 2 GB); `phase_1/tests/test_dataset.py` (7 tests); figure
       `phase_1/fig_dataset.py` -> `tex/fig_dataset.pdf`, Data section of `tex/phase1.tex` updated.
       Finding: K = 32 k_D = 1 modes miss near-well structure of heterogeneous states (projection error median 4.5%,
       p90 11% at t_D = 5e4; 94% of residual at r < 10) -> near-well detail must go through the pointwise path.
4. [x] **HNO model** - lift -> L x [annular Hankel spectral conv + pointwise W + GELU] -> project;
       real-valued spectral weights; hard constraint u(1) = 0.
       Basis study (200 train trajectories, ln r measure): K = 32 Hankel misses 3e-2 of the states (median) and
       128 Hankel modes 1e-2, but K = 32 log-sine modes sin((n - 1/2) pi s), s = ln r / ln r_eD, only 1e-4
       (also better in the r-weighted norm: 3e-6 vs 9e-4). The first step's change u(1000) - 1 is not
       representable in any basis with phi(1) = 0 -> predict u_{n+1} directly.
       Done: `phase_1/model.py` - one architecture, swappable spectral basis (T, B):
       A `hankel`, B `hankel_local` (+ kernel-5 conv), C `logsine`, D `dual` (Hankel + log-sine), E `fno` (FFT, padded).
       Inputs [u, log k, s]; output u_{n+1} = s * v (hard BC); 4 layers, width 32, K = 32 (140k-270k params).
       `phase_1/tests/test_model.py` (15 tests, incl. exact k_D = 1 step via one Hankel layer).
       CPU cost: 0.3-0.6 s per training step of 64 pairs -> 35-65 min per full pass over 400k pairs.
5. [ ] **Training & evaluation** - loss: relative L2 in the ln r measure (grid-uniform); also report r-weighted L2; one-step relative L2; rollout error over a limited window (~100-500 steps);
       well rate q_D ~ r du/dr at r_D = 1; baseline FNO-1D on the same data.
       In progress: `phase_1/train.py`, `phase_1/evaluate.py`. Stage 1 (direct) and 1b (residual + q_D loss)
       trained: one-step errors reach the copy-input level, but rollouts are unstable (errors amplify
       ~1.3-1.5x per step). Pending decision: switch to a direct model (log k, t) -> u(r, t). See `STATUS.md`.

## Layout
- `phase_1/` - all phase-1 Python (code, figure scripts, `tests/`); run with `/home/daniel_88/py314/bin/python`.
- `tex/` - only .tex and .pdf (`phase1.tex` summarizes the math).

## Open topics
- Network architecture details: number of layers, width, hard-BC enforcement method.
- Rollout stability over long horizons; move to dt-as-input (log-spaced times) to resolve early times (t_D < 1000) and reach late boundary-dominated flow (t_D ~ 1e6+).
