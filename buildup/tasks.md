# Build-up test dataset (independent project)

Synthetic pressure build-up tests for a vertical well in a radial reservoir with heterogeneous permeability.
Purpose: the inverse problem a build-up test solves — from the well pressure after shut-in, recover the reservoir
permeability, the skin, the wellbore storage and the distance to the (closed) outer boundary.
Self-contained: every Python file lives in `buildup/`; nothing is imported from `phase_1/` or elsewhere in the repo.
Environment: `/home/daniel_88/py314` venv, CPU only.

## Problem (dimensionless)

Variables: r_D = r / r_w, k_D(r) = k(r) / k_ref, t_D = k_ref t / (phi mu c_t r_w^2),
p_D = 2 pi k_ref h (p_i - p) / (q_ref B mu)   (q_ref: the production rate before shut-in).

    dp_D/dt_D = (1/r_D) d/dr_D ( r_D k_D(r_D) dp_D/dr_D ),   1 <= r_D <= r_eD

- IC: p_D(r_D, 0) = 0 (reservoir at p_i)
- Outer BC (closed): r_D dp_D/dr_D = 0 at r_D = r_eD
- Inner BC (rate-controlled well, with wellbore storage C_D):

      C_D dp_wD/dt_D - k_D(1) r_D dp_D/dr_D |_{r_D=1} = q_D(t_D)

  q_D = 1 while producing (0 < t_D <= t_pD), 0 after shut-in. C_D = 0: no storage.
- Thin skin S (optional): p_wD = p_D(1, t_D) - S r_D dp_D/dr_D |_{r_D=1}. A finite skin zone is part of k_D(r).
- Observed: well pressure p_wD(t_D); physical p_w = p_i - q_ref B mu p_wD / (2 pi k_ref h).

### Build-up by superposition

The PDE and boundary conditions are linear in p_D (for any k_D(r), constant C_D and S). One unit-rate drawdown
p_dd(t_D) per permeability field gives every build-up exactly:

    p_wD,bu(dt_D) = p_dd(t_pD + dt_D) - p_dd(dt_D)          (drop below p_i after shut-in time dt_D)
    p_ws(dt_D) - p_wf(t_pD) = p_dd(t_pD) - p_dd(t_pD + dt_D) + p_dd(dt_D)   (build-up pressure change)

so t_pD (and multi-rate histories) can be chosen after the simulation.

### Diagnostics

- Bourdet derivative d(dp_bu)/d ln(dt_equivalent), with Agarwal equivalent time dt_e = t_p dt / (t_p + dt)
- Horner time (t_p + dt) / dt
- Expected regimes: wellbore storage (unit slope), skin hump, radial flow (derivative plateau 1/(2 k_D)
  for homogeneous k_D), heterogeneity signatures, boundary effects.

## Inversion targets (labels per sample) — decided 2026-09-25 (option A)

The Milestone 5 check showed that a single k_eff and S are not what a build-up reads in these heterogeneous
reservoirs; the build-up constrains a smoothed radial permeability profile. Targets:

| Target | Definition |
|---|---|
| ln k over 8 rings | log-harmonic mean of k_D over r_D in [1,2], [2,5], [5,10], [10,30], [30,100], [100,300], [300,1000], [1000,3000] (solver face permeabilities), each ring clipped at r_eD; masked where the part inside spans less than a radius ratio of 1.5 |
| C_D | ln C_D (masked when C_D = 0) + a no-storage flag (C_D = 0 in 20% of samples) |
| r_eD | ln r_eD |
| p_bar (derived) | average pressure after stabilization, p_bar_D = t_pD / ((r_eD^2 - 1)/2 + C_D), from the predicted r_eD, C_D |

k_eff and S stay in the dataset as generator-side summaries (any skin definition can be computed from the profile).

## Measured inputs

- The build-up record: p_ws(dt) - p_wf over the shut-in times (the gauge), and t_pD (production time).
- The drawdown at shut-in p_i - p_wf (dimensionless p_wD(t_pD)): requires the initial pressure p_i, usually known
  for a new or exploration well (formation tester, DST, initial static survey). It carries the material balance
  and pins r_eD (without it the Milestone 5 inversion drifted from r_eD 1645 to 1246). It is an input that is
  masked in part of the training examples, so the model also covers tests without a reliable p_i (with larger
  uncertainty on r_eD). The average pressure p_bar is an output, not an input.
- Scope: a single well, one constant-rate production period from uniform p_i, closed circle (a new or exploration
  well); not a mature field with a long rate history or interference.

## Decisions (confirmed 2026-09-24)

| Item | Decision |
|---|---|
| Geometry | r_eD sampled per trajectory, log-uniform in [300, 3000] |
| Grid | log-spaced in r_D with a fixed spacing (N ~ 850-1190 depending on r_eD); exact log-flux faces |
| Permeability | ln k_D = mean + GRF in ln r_D (SE covariance) + optional skin-zone step; own implementation |
| Wellbore storage | C_D = 0 in 20% of samples, otherwise log-uniform in [0.1, 1e3] (proposed default) |
| Thin skin S | 0 at first (skin via the k_D zone); solver supports S >= 0 |
| Times | log-spaced t_D from 1e-2 to 1e7, ~200 stored times |
| Stored per sample | k_D(r) (padded), r_eD, C_D, labels, unit-rate drawdown p_wD(t); build-ups by superposition |
| Samples | 20 000 reservoirs: train / val / test = 18 000 / 1 000 / 1 000 (decision 2026-09-25) |
| Time integration | implicit (BDF2, variable step, restarted at rate changes) |
| Production time | t_pD log-uniform in [1e3, 1e6], drawn when forming build-ups (decision 2026-09-25) |
| Stack | NumPy + SciPy; PyTorch for the inverse model (Milestone 6) |

## Milestones

1. [x] **Solver** (`buildup/solver.py`): vertex-centered finite volumes on the log grid, harmonic-mean face
       permeability, rate BC with wellbore storage and thin skin, closed outer boundary, arbitrary rate history q_D(t).
       Tests (`buildup/tests/test_solver.py`), all with k_D = 1:
       - infinite-acting line source: p_wD = -Ei(-1/(4 t_D)) / 2 ~ (ln t_D + 0.80907) / 2 for t_D > 25 (before
         the boundary is felt);
       - finite wellbore radius, closed boundary: van Everdingen-Hurst solution by numerical Laplace inversion
         (Stehfest), including C_D and S;
       - pseudo-steady state: p_wD ~ 2 t_D / r_eD^2 + ln r_eD - 3/4 for late times;
       - storage: p_wD ~ t_D / C_D at early times (unit slope);
       - superposition: a simulated shut-in equals the superposed drawdowns.
       Done (2026-09-24): `solver.py` (unknowns [p_w, p_0..p_N-1], tridiagonal; S = 0 merges well and node 0; C = 0
       is an algebraic well equation), `analytic.py` (line source, van Everdingen-Hurst + Agarwal storage/skin in
       Laplace space, Gaver-Stehfest n = 12, PSS), `tests/test_solver.py` (12 tests). Grid spacing D = ln 1000 / 1023.
       Accuracy vs the exact solution (r_e = 300, t = 0.1..1e7, C in {0, 100}, S in {0, 5}): <= 3e-4 relative;
       line source (t = 1e3..1e5) < 1e-3; PSS < 1e-4; conservation exact to 1e-9 (heterogeneous k, shut-in);
       simulated shut-in = superposed drawdowns to < 1e-3 of the build-up change.
2. [x] **Build-up tools** (`buildup/wells.py`): superposition, Agarwal time, Horner time, Bourdet derivative
       (log-smoothed). Tests on the analytic homogeneous case (Horner slope, derivative plateau 0.5).
       Done (2026-09-25): `Drawdown` (cubic spline in ln t through the 200 stored times; 3e-6 between samples),
       `buildup`, `agarwal_time`, `horner_time`, `bourdet_derivative` (window L in ln t), `interpret_radial_flow`
       (k from the plateau, S from the semilog line), `T_P_RANGE`; `tests/test_wells.py` (8 tests).
       Homogeneous, no storage/skin: plateau 0.5 within 5e-3, Horner slope within 3e-3, k within 0.5%, S within
       0.02. With storage and skin (window after storage ends, ~(60 + 3.5 S) C / k): k within 1.5%, S within 0.15.
       Convention: the solver's thin skin is scaled with k_ref; the well-test skin is S_wt = k S.
3. [x] **Permeability generator** (`buildup/permeability.py`): GRF in ln r_D + skin zone, clipping, seeds per sample.
       Tests: statistics, reproducibility.
       Done (2026-09-25): `sample_logk` (same model as phase 1, own code), GRF by circulant embedding (FFT on the
       uniform ln r grid; eigendecomposition fallback for non-uniform grids), `resistance` and `labels` (k_eff, S)
       from the solver's harmonic face permeabilities; k_eff starts at the first node beyond r_s so the face straddling
       the step is excluded (exact for a composite reservoir). `tests/test_permeability.py` (11 tests): covariance at
       lags 0 .. 2 ell within 5% of sigma^2, skin fraction and step size, clipping, reproducibility, Hawkins
       S = (k / k_s - 1) ln r_s within one grid spacing, scaling, additivity.
4. [x] **Dataset** (`buildup/dataset.py`): parallel generation of 20 000 samples (18 000 / 1 000 / 1 000), one
       unit-rate drawdown per sample at 200 log-spaced t_D in [1e-2, 1e7]; per sample r_eD, C_D, ln k_D(r) padded to
       the largest grid (+ length), generator parameters and labels (k_eff, S); `.npz` in `buildup/data/`
       (git-ignored, ~110 MB). Tests on a tiny dataset.
       Done (2026-09-25): generated in 6 min on 8 workers (0.045 s/sample), 128 MB; `tests/test_dataset.py` (6 tests:
       worker-independent seeding, shapes/padding, monotone drawdown, labels recomputed from stored fields, build-ups
       from the stored drawdown, save/load). p_w stored in float64 (build-ups are differences of drawdown values).
       Train split: k_eff 0.07 .. 7.2 (median 0.84); S median 0, 95th pct 19, max 133 (heavy tail: damaged zone on
       top of a GRF low clipped at ln k = -3); C_D = 0 in 20.2%; skin zone in 50.3%; boundary felt (r_eD^2/(4 k_eff))
       before dt = 1e6 in 78%, before 9e6 in 99.9%; storage visible in ~80%.
       Note for Milestone 6: learn S in a compressed scale (e.g. asinh S) because of the heavy tail.
5. [ ] **Figures and label check** (`buildup/fig_*.py` -> PDFs in `buildup/figs/`): example k_D fields, drawdown and
       build-up log-log plots with Bourdet derivatives, Horner plots; check that the derivative plateau matches k_eff
       and the Horner extrapolation matches S on examples.
       Status (2026-09-25): figures done (`fig_examples.py`, `fig_label_check.py` -> `figs/`). Label check on the test
       split (t_pD = 1e6, oracle radial-flow window after storage and before the boundary; 684 / 1000 samples have one):
       - r_eD by material balance from the stabilized build-up (p_bar = t_p / ((r_eD^2 - 1)/2 + C_D)): exact to 3e-7
         (695 / 1000 stabilized) -> r_eD is a clean, identifiable label;
       - plateau k vs label k_eff (log-harmonic mean over [r_s, r_eD]): median error 23%, 90th pct ~97%;
         vs the log-harmonic mean over the investigated ring r = 2 sqrt(k dt): median 17% — the plateau measures a
         local average, and GRF heterogeneity makes the derivative wander (no single plateau);
       - semilog S vs label S: median |diff| 1.1, 90th pct 6; samples without a skin zone (label S = 0) read S from
         -1 to ~100, because near-well GRF variations act as skin relative to the local plateau k.
       => the labels k_eff and S as defined are not what a build-up interpretation reads in these heterogeneous
       reservoirs. Decided: option A, ring-averaged ln k profile (see Inversion targets).
       Model-based inversion (`invert.py`, `tests/test_invert.py`, 2 tests): unknowns ln k at 16 control points
       (linear in ln r on [1, 3000]), ln C_D, ln r_eD; forward = solver on 900 nodes + superposition; misfit on
       ln dp_bu and ln p_wf (p_i known: carries the material balance) with 1% noise, smoothness and weak prior on
       ln k; start from classical estimates; Gauss-Newton uncertainty. Demo on test sample 486 (1% noise,
       t_p = 1e6; `figs/fig_invert_486`): C_D 0.582 (true 0.584), r_eD 1658 (true 1645, +/- 10%), ring ln k
       within 0.04 for r < 100 and within ~0.25 beyond; the skin step is smoothed over one control spacing.
       ~10-40 s per inversion. Without p_wf in the misfit r_eD drifted to 1246: the build-up change alone
       constrains the boundary poorly.
6. [ ] **Inverse model** (`buildup/model.py`, `buildup/train.py`, `buildup/evaluate.py`; PyTorch):
       build-up response -> (8 ring ln k, C_D, r_eD) with uncertainties.
       - Input: the build-up on a fixed grid of 128 shut-in times, log-spaced in [1e-2, 9e6] (valid for every
         t_pD <= 1e6); channels ln dp_bu(dt), log-derivative d ln dp_bu / d ln dt_e, ln dt, ln t_pD (constant),
         ln p_wD(t_pD) (constant, 0 when masked) + p_i-known flag, validity mask (truncated tests).
       - Architecture: FNO encoder along ln dt (lift -> 4 spectral layers: FFT in ln dt, low modes, channel mixing,
         pointwise path) -> pooling over time (mean + max) -> MLP head with 8 outputs (mean and log-variance of 4
         targets).
       - Targets (normalized): 8 ring ln k (masked beyond r_eD), ln C_D (masked if 0) + no-storage flag (logit, BCE),
         ln r_eD.
       - Loss: Gaussian negative log-likelihood (heteroscedastic), so unidentifiable cases report large uncertainty.
       - Training data built on the fly from the stored drawdowns: fresh t_pD ~ logU[1e3, 1e6] per example
         (superposition), gauge noise on dp, random truncation of the test length (mask), optionally dropping the
         earliest points.
       - Identifiability: the boundary is felt only after ~r_eD^2 / (4 k_eff) (~1e3 .. 5e7 over the ranges), tiny C_D
         ends before dt = 1e-2, k_eff and S trade off under strong near-well heterogeneity. Evaluate per regime
         (boundary reached or not, storage visible or not) and check the calibration of the predicted uncertainty.
       - Baselines: classical estimates (C_D from the unit slope, r_eD from material balance when p_i is known) and
         the model-based inversion `invert.py` on a subset (slow, ~30 s per test, but accurate).
       - Metrics: errors of the ring ln k, ln C_D, ln r_eD per regime (p_i known or not, boundary felt, storage
         visible); storage-flag accuracy; coverage of the predicted intervals.
       Status (2026-09-25): `examples.py` (on-the-fly build-ups, 5 ms / batch of 64), `model.py` (InverseFNO, 484k
       params, masked pooling), `train.py`, `evaluate.py`; tests 47 pass.
       - runs/base (plain Gaussian NLL): r_eD stuck at its prior spread (RMSE 0.99 normalized) through step 8000;
         stopped. Cause: the NLL mean gradient (y - mu) / var vanishes for a target given a large variance.
       - runs/beta (beta-NLL, beta = 0.5, 20k steps, 16 min): test (4000 examples, 1% noise, 50% truncated, 25% without
         p_i): ln C_D MAE 0.025 (classical unit slope 0.119), no-storage flag 100%; ring ln k MAE 0.09-0.13 for
         r_D < 100, 0.27 / 0.46 / 0.56 for the outer rings; r_eD median rel. error 35% (29% boundary felt, 45% not;
         34% with p_i, 39% without). Coverage of +/- 2 sigma: C_D 96%, r_eD 98%, rings 85% (overconfident).
         The model does not exploit the material balance: on stabilized p_i-known examples classical material
         balance gives 18%, the model 27%. Next: an input channel with the material-balance r_eD estimate along the
         curve (p_i known), then recalibrate ring uncertainties if still needed.
       - runs/mb (+ channel 7: material-balance radius r_app = sqrt(2 t_p / (p_wf - dp_bu) + 1), p_i known; C_IN = 8):
         r_eD median rel. error 31% overall (was 35%): p_i known 27% (34%), boundary felt 21% (29%), full test 21%
         (29%), p_i unknown 41% (39%, unchanged as expected); now better than classical material balance on the
         stabilized p_i-known examples (17% vs 18%). C_D and rings unchanged (ln C_D MAE 0.024; ring MAE 0.09-0.14
         for r_D < 100). Ring +/- 2 sigma coverage still 85% (overconfident).
       - Realistic test lengths (decision 2026-09-25): t_p ~ logU[1e4, 1e6] (was [1e3, 1e6]); truncated tests only for
         dt_max in [1e6, 9e6] (days of shut-in; was down to 1e2); drawn independently of the reservoir (a length tied to
         r_eD^2 / k would leak r_eD through the validity mask). ln t_p input normalization fixed at (10.36, 1.73).
         runs/long (retrained, 20k steps), test on the new distribution (boundary felt in 97% of tests): r_eD median
         rel. error 13.6% (p_i known 9.5%, unknown 28.5%; 90th pct 53%); ln C_D MAE 0.024; ring MAE 0.07-0.13 for
         r_D < 300, 0.31 / 0.48 for the outer rings; better than classical material balance where it applies (6.2% vs
         8.0%). runs/mb evaluated on the same tests (eval_newdist): r_eD 13.8% (p_i known 9.9%, unknown 31%) — the gain
         comes almost entirely from the longer tests (r_eD becomes identifiable), not from retraining.
         Open: r_eD tail (90th pct 53%).
       - Calibration (`calibrate.py`, 2026-09-25): the "85%" ring coverage was a bug in evaluate.py (NaN < 2 counted
         masked rings as misses; fixed, `tests/test_calibrate.py`). Corrected test coverage of +/- 2 sigma for
         runs/long: every ring 94-95%, C_D 95%, r_eD 96%; the reliability plot (`figs/fig_calibration`) follows the
         diagonal at all levels. Val-fitted scales are 0.98-1.12 and slightly over-cover mid levels, so the raw
         uncertainties are kept (scales stored in runs/long/calibration.json for reference).
       - r_eD tail (`analyze_re_tail.py`, `figs/fig_re_tail`; 90th pct error 53%, 400 test examples): an information
         limit, not a model failure. Driver 1: drainage before shut-in t_p k_eff / r_eD^2 — with p_i known the median
         error is 33% below 0.01 and 6.9% above (the material-balance signal ~2 t_p / r_eD^2 is buried in the 1% gauge
         noise when drainage is small). Driver 2: p_i unknown (47% of the tail vs 24% overall; 26-39% error). Test
         length matters little once tests are days long; strong heterogeneity slightly (51% vs 41%); skin zones not.
         The model flags these cases: predicted sigma of ln r_eD 0.47 in the tail vs 0.24 elsewhere, 82% of the tail
         inside 2 sigma; it over-estimates r_eD in 76% of tail cases (weak data only bound the size from below).
         Design rule: t_p >~ 0.01 r_eD^2 / k for r_eD within ~7% with p_i known. The 1% gauge noise is pessimistic
         (modern gauges ~0.01-0.1% of the drawdown).
       - 0.1% gauge noise, same model (trained at 1%; `evaluate.py --noise 0.001`, tag _noise1e-3): r_eD median 9.8%
         (p_i known 5.8%, unknown 27.6%; 90th pct 44%), ln C_D MAE 0.021, ring MAE 0.05-0.11 for r_D < 300; coverage
         over-conservative (r_eD 99%, rings 97%). With p_i known: drainage < 0.01 -> 27%, >= 0.01 -> 3.7%.
         Classical material balance becomes usable in 63% of tests with 1.3% median error, the model only 5.3% on the
         same tests: a model trained at 1% noise does not exploit a more precise gauge. Next option: train with the
         noise level randomized (e.g. logU[1e-4, 1e-2]) and given as an input (gauge precision is known in practice).
       - Noise-aware model (runs/noise, 2026-09-25): noise level per example logU[1e-4, 1e-2], input channel 8 =
         log10(noise) + 3 (C_IN = 9; older 8-channel models read the first 8). Test (`runs/noise/compare.txt`),
         median r_eD error at noise 0.01% / 0.1% / 1%: all 1.7 / 3.4 / 14.1% (runs/long 10.0 / 9.8 / 13.6%); p_i known
         1.1 / 2.0 / 10.1% (5.7 / 5.8 / 9.5%); p_i unknown 19.6 / 20.0 / 29.6% (27.8 / 27.6 / 28.5%). Ring MAE 0.10 /
         0.11 / 0.19 (0.15 / 0.15 / 0.17); ln C_D MAE 0.026 / 0.023 / 0.042 (0.021 / 0.021 / 0.024). Coverage 92-97% at
         low noise, 90-94% at 1% (slightly overconfident). Classical material balance where it applies: 0.15% / 1.3% /
         8.0% vs model 1.1% / 1.9% / 7.6% (usable in 75 / 63 / 9% of tests).
         Takeaways: the model now exploits precise gauges (p_i unknown also improves: the derivative pins the far-field
         k better); small loss at 1% noise (capacity spread over a 100x noise range); exact material balance still
         wins on stabilized, precise, p_i-known tests. Options: longer / wider training; hybrid output (classical
         material balance where p_i known and stabilized); flowing-pressure record before shut-in for p_i unknown.

## Layout

    buildup/
      tasks.md
      solver.py  analytic.py  wells.py  permeability.py  dataset.py  fig_*.py
      model.py  train.py  evaluate.py   (Milestone 6)
      tests/     (pytest; run: /home/daniel_88/py314/bin/python -m pytest -q buildup/tests)
      data/      (generated, git-ignored)
      figs/      (PDF figures)

## Open questions

(none)
