# Session status — resume here

Last updated: 2026-09-23. Plan and decisions: `tasks.md`. Math: `tex/phase1.tex` (incl. Appendix A: derivation of the
homogeneous series coefficients A_m).

## Where we are

- Milestones 1-4 done. Milestone 5 (one-step operator + rollout) committed in `b94397c` and **abandoned**:
  rollouts amplify errors in every variant (see "Milestone 5: one-step rollouts").
- **Milestone 5b (direct operator) in progress**, committed in `c503ed7`. Code in `phase_1/direct/`, runs in
  `phase_1/direct/runs/` (git-ignored). All tests pass (113):
  `/home/daniel_88/py314/bin/python -m pytest -q phase_1/tests phase_1/direct/tests`
- Stage 4 (`hankel`, `fftlog`, `fftlog_decay`) finished and evaluated (`runs/stage4/eval.txt`).
- Untracked user files (do not commit unless asked): `phase_1/test.ipynb`, `phase_1/dt=*.png`.

## Next steps

1. Transfer test done (both fail, fftlog worse). Options: train on mixed r_eD (with r_eD as input), locality
   constraints, hybrid layer, or accept fno for phase 1.
2. Open question from the user: why FNO beats the HNO. Answer so far: our "fno" is an FFT in ln r on the log grid
   (already radial-aware), while the dense Hankel basis is sine-like in r and cannot resolve r < 10; the FFTLog
   variant is the proper log-grid Hankel operator. Stage 4: fftlog still 2.5-4x behind fno on u.
3. Possible follow-ups: faster FFTLogConv (mixing on a coarser k grid); fix `dual` q_D; revisit the fftlog_decay
   D initialization if it lags.

## Milestone 5b: direct operator G: (u_0, log k, dt) -> u(dt)

User's formulation (rollout error growth was expected). The PDE is autonomous, so any two stored states of a
trajectory form a pair: (u(t_i), log k, t_j - t_i) -> u(t_j). Prediction: one jump from u = 1 per output time.

- `direct_model.py`: phase-1 backbone (lift -> 4 spectral layers -> project, u = s·v); dt enters as
  tau = normalized log10 dt (input channel + FiLM scale/shift per layer, FiLM initialized to identity).
- `direct_train.py`: gap j - i log-uniform in [1, 500], half of each batch from u = 1; loss rel. L2 (ln r measure)
  + 0.1 · q_D error; `--gap 1` trains only the jump 0 -> 1000.
- `direct_evaluate.py`: jump to all 500 times on the test set; bar (median u < 2%, q_D < 5% at every time);
  k_D = 1 vs exact series; semigroup check G(G(1, dt1), dt2) vs G(1, dt1 + dt2).
- `fig_direct_t1000.py` -> `tex/fig_direct_t1000.pdf` (+ copies in `runs/stage2/`).

**Stage 2 — all gaps, 5000 steps** (`runs/stage2/`), test set, medians over trajectories:

| variant | u all times | u worst time | q_D worst time | k_D = 1 max | 250+250 vs 500 | bar |
|---|---|---|---|---|---|---|
| **fno** | 0.42% | 0.59% | 1.1% | 0.20% | 0.66% | pass |
| logsine | 0.80% | 1.08% | 2.1% | 0.33% | 0.96% | pass |
| dual | 0.96% | 1.25% | 11% | 0.45% | 0.81% | fail (q_D) |

fno at t_D = 1000: u 0.33% (p90 0.98%, max 2.3%), q_D 0.82% (p90 3.9%). Worst case = strong high-k zone near
r ~ 4 (plateau smoothed); ~±0.005 ripples beyond the front (r > 100).

**Stage 3 — fixed dt = 1000 only (`--gap 1`)** (`runs/stage3_dt1000/`), test at t_D = 1000:
fno u 0.25% / q_D 0.54%, logsine 0.34% / 1.0%, dual 0.44% / 1.9%. Only ~25-35% better than stage-2 fno at its
single time -> keep the general (all-gaps) model.

**Stage 4 — Hankel variants, all gaps, 5000 steps** (`runs/stage4/`), test set, medians over trajectories:

| variant | u all times | u worst time | q_D worst time | k_D = 1 max | 250+250 vs 500 | bar |
|---|---|---|---|---|---|---|
| fftlog | 1.73% | 2.53% (at t = 5e5) | 2.5% | 1.06% | 1.79% | fail (u, late times only) |
| fftlog_decay | 3.86% | 5.32% | 2.9% | 2.76% | 3.63% | fail (u) |
| hankel (dense) | 7.22% | 8.75% | 35% | 3.22% | 3.51% | fail |

u error at t_D = 1e3 / 1e4 / 1e5 / 2.5e5 / 5e5: fno 0.33 / 0.29 / 0.33 / 0.41 / 0.59%;
fftlog 0.87 / 0.97 / 1.25 / 1.62 / 2.53% (grows with time; q_D flat ~2%); fftlog_decay 1.5 -> 5.3%.
Ranking: fno > logsine > dual (u) > fftlog > fftlog_decay >> dense hankel. fftlog >> dense Hankel (q_D 2% vs 33%),
so the log-grid Hankel transform fixes the dense basis' near-well problem, but it stays 2.5-4x behind fno on u.
Worst fftlog case at t = 1000: sharp skin step in k near r ~ 5 (u error up to 20% at r ~ 20).
Interpretation (to test): the plane-Hankel multiplier assumes 2-D translation invariance, broken by the well, the
heterogeneous k and the no-flow boundary (error growth at late times = boundary-dominated regime); the ln-r FFT
matches the scale invariance of well flow (r^2/t similarity). Proposed next test: generalization to other r_eD
(where the Hankel k-space should transfer), and a hybrid ln-r FFT + FFTLog layer.
Figures: `tex/fig_direct_per_time.pdf` (all variants vs time), `tex/fig_direct_t1000_fftlog.pdf`
(+ PNG/PDF copies in `runs/stage4/`).

**Transfer to other r_eD (zero-shot)** (`transfer_evaluate.py`, `fig_transfer.py` -> `tex/fig_transfer.pdf`,
copies in `runs/transfer/`). Test sets with the same ln-r spacing and k statistics: `data/re299` (N = 845),
`data/re2986` (N = 1186); results in `runs/<run>/eval_re*.json`. Models rebuilt on the new grid with physical
settings (s = ln r / ln 1000, fno pad 256, FFTLog control points at the training k via `k_range`).

| model | r_eD 299: u / q_D (t = 1e3) | r_eD 2986: u / q_D (t = 1e3) | r_eD 2986 all times u | trained r_eD 1000 u |
|---|---|---|---|---|
| fno | 10.6% / 13.9% | 8.3% / 10.2% | 5.8% | 0.42% |
| fftlog | 18.1% / 36.7% | 12.0% / 39.2% | 13.7% | 1.73% |

Both fail to transfer; fftlog worse. r_eD 299 at late times also extrapolates physics (deep depletion,
errors up to 300-400%). Diagnostic: at r_eD 2986 and t <= 5e4 (physics identical to training), feeding only
r <= 1000 restores in-distribution accuracy (fno 0.3%, fftlog 0.8-1.0%): the models are non-local and have
learned the domain end at r = 1000; the extra domain contaminates them. fftlog's plane-convolution kernel is
more global, hence more sensitive. Physical-k R(k) alone does not make the operator transferable.

## FFTLog Hankel layer (step 2 of the user's FFTLog request)

- `phase_1/fftlog.py`: FFTLog (Hamilton 2000), F(k) = ∫ f J_mu(kr) r dr on log grids; centered kappa
  (k in [kappa/r_max, kappa/r_min]), low-ringing kappa, bias q in (-mu-1, 1/2), exact `inverse` for any q.
  Tests: Gaussian / algebraic / exponential pairs to 1e-9, quadrature, round trip exact. Input must be negligible
  at both grid ends (periodic in ln r); round trip is exact in r^{1-q} f, not f.
- `phase_1/direct/fftlog_conv.py`: per channel H^{-1}[R(k) H[h]], mu = 0 (radial Fourier multiplier on the plane):
  pad 256 points each side in ln r (r in [0.18, 5620], N = 1536), replicate + cos² taper; **q = 0.4** (user's
  choice; float32 rounding ~2e-5 at the well vs ~30x worse for q = 0); R(k) = real C x C from 32 control points
  linear in ln k; `fftlog_decay` adds exp(-D_c k² dt) with learned D_c, init log-uniform [1e-4, 20].
  2-D heat-kernel test matches to 6e-7. No-flow at r_eD is not built in. Cost 0.56-0.68 s/step alone
  (fno 0.12) — mixing at all 1536 k points dominates.

## Milestone 5: one-step rollouts (abandoned)

One-step operator (u_n, log k) -> u_{n+1}, dt_D = 1000, rolled out 500 steps (`phase_1/train.py`, `evaluate.py`,
`runs/stage1_direct/`, `runs/stage1b/`). Best final rollout errors: stage 1 (u = s·v) fno u 0.14 / q_D 0.31;
stage 1b (residual) fno 0.31 / 120, others blow up (1e9-1e14 or NaN). Rollouts started from the true u(1000)
reach 6-14% after 10 steps: the learned step amplifies errors ~1.3-1.5x per step (the exact step contracts);
one-step training never penalizes this.

## Key findings and explanations (so they need not be re-derived)

- **Basis study** (Milestone 4): for heterogeneous k, 32 log-sine modes sin((n-½)πs) represent states to
  1e-4 vs 3e-2 for 32 Hankel modes (ln r measure). Near-well kinks (∂u/∂r ∝ 1/k) are what Hankel misses;
  94% of the Hankel residual is at r < 10. The dense Hankel modes are sine-like in r (wavelength ~2·1000/n).
- **Our "fno"** is FFT in ln r on the log grid (zero-padded), i.e. already log-radial — this is why it does well.
- **Norms**: the r-weighted norm nearly ignores r < 10; use the grid (ln r) measure plus q_D.
- **q_D sensitivity**: q_D = T·u₁ with u₁ ≈ 1e-3 at the first interior node (r = 1.0068); an absolute error
  of 1e-4 there is a 10% q_D error but invisible in the L2 of u. Hence u = s·v and the q_D loss term.
- **s·v**: s = ln r / ln r_e; s = 0 at the well (hard BC), u ≈ c·s near the well, so v is O(1).
  The residual form of stage 1b lost this scaling → q_D errors.
- **Series**: A_m = π J1(λ_m r_e)² / (J1(λ_m r_e)² − J0(λ_m)²); N_m = (2/(π²λ_m²))(J0(λ_m)²/J1(λ_m r_e)² − 1)
  (Appendix A). Mode n decays as exp(−λ_n² t); λ_1² t_D ≤ 0.16 over the dataset (mode 1 at ~85%).
- **Time scale**: t = t_D φ μ c_t r_w² / k_ref. With k = 100 md, φ = 0.2, μ = 1 cp, c_t = 1e-9 1/Pa,
  r_w = 0.1 m: Δt_D = 1000 ≈ 20 s, t_D = 5e5 ≈ 2.8 h (1 md: 34 min / 12 days).
- **Data**: 800/100/100 trajectories, 501 states (t_D = 0..5e5 every 1000) on the geometric N = 1024 grid;
  only 800 examples per fixed dt, ~125k pairs per trajectory for the direct operator.

## How to run

```bash
cd /home/daniel_88/hno/phase_1/direct
PY=/home/daniel_88/py314/bin/python
# train one direct variant (writes runs/<out>/<variant>/{best.pt,last.pt,log.csv,config.json})
OMP_NUM_THREADS=2 $PY direct_train.py --variant fno --steps 5000 --threads 2 --out runs/stage2
#   variants: hankel hankel_local logsine dual fno fftlog fftlog_decay; --gap 1 for dt = 1000 only
# evaluate (writes eval.json per variant; ~15 min)
OMP_NUM_THREADS=6 $PY direct_evaluate.py --runs runs/stage2 --variants logsine dual fno --threads 6
# t_D = 1000 figure (tex/fig_direct_t1000.pdf, optional PNG)
OMP_NUM_THREADS=1 $PY fig_direct_t1000.py --png runs/stage2/fig_direct_t1000.png
# regenerate the dataset (~28 min, 8 workers, 2 GB)
cd .. && OMP_NUM_THREADS=1 $PY dataset.py --workers 8
```

Machine: 8 cores, no GPU (WSL), 7 GB RAM — run training in parallel at 1-2 threads each; data is memory-mapped.

## Working conventions

- Ask before creating or editing files; write code into files, not into chat; commit only when asked.
- Results as local PDFs (`fig_*.py` -> `tex/fig_*.pdf`); for training runs also PNG + PDF in `runs/<stage>/`.
- `tex/` holds only .tex/.pdf.
