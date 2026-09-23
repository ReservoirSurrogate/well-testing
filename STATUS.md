# Session status — resume here

Last updated: 2026-09-23. Plan and decisions: `tasks.md`. Math: `tex/phase1.tex`.

## Where we are

- Milestones 1-4 done and committed (last commit `f7a7c63`).
- **Milestone 5 (training & evaluation) in progress; a design decision is pending (see below).**
- Uncommitted (all tests pass, 73 total: `/home/daniel_88/py314/bin/python -m pytest -q phase_1/tests`):
  - `phase_1/train.py` — trainer (random pairs, ~5% first steps, `--residual`, `--q-weight`)
  - `phase_1/evaluate.py` — test metrics, 500-step rollouts, k_D = 1 check → `eval.json`
  - `phase_1/tests/test_train.py` — 10 smoke tests
  - `phase_1/model.py`, `phase_1/tests/test_model.py` — `residual=True` option (+5 tests)
  - `.gitignore` — `phase_1/runs/`
- Git-ignored artifacts: `phase_1/data/` (dataset .npz + memory-mapped `*_u.npy`), `phase_1/runs/`.

## Pending decision (first thing to settle on resume)

The one-step operator G: (u_n, log k) -> u_{n+1} (Δt_D = 1000) gives unusable rollouts in all variants.
Claude's recommendation: **switch to a direct model (log k, t) -> u(r, t)** (no rollout; every trajectory
starts from u = 1 anyway), keeping u = s·v and the q_D loss term; keep the one-step operator as future work
(idea: build damping into the architecture, e.g. per-mode decay factors <= 1 like exp(-λ²Δt)).
Estimated odds of meeting the bar: direct model ~70-80%; one-step stabilization (input noise +
2-4-step unrolled training + direct first step) ~20-30%.
Next step if agreed: propose the direct-model design (inputs, how t enters, loss, variants) before coding.
Also decide whether to commit the uncommitted Milestone 5 code as is.

Success bar agreed for rollouts: q_D error < ~5% and u error < ~2% at t_D = 5e5, no blow-ups.

## Results so far

Variants (all ~140k params; `hankel_local` 160k): `hankel` (A), `hankel_local` (B), `logsine` (C),
`dual` (D, 16 Hankel + 16 log-sine), `fno` (E, 16 complex modes). 5000 steps, batch 32, Adam 1e-3 cosine,
5 runs in parallel on 8 CPU cores (~70 min).

**Stage 1 — direct output u_{n+1} = s·v** (`runs/stage1_direct/`), test set:

| variant | 1-step (copy-input: 3.8e-4) | rollout u @10/100/500 | rollout q_D @10/100/500 |
|---|---|---|---|
| fno | 2.1e-3 | 1.6e-2 / 2.7e-2 / 0.14 | 0.26 / 0.23 / 0.31 |
| logsine | 2.7e-3 | 3.2e-2 / 3.4e-2 / 0.14 | 0.59 / 0.51 / 0.70 |
| dual | 4.1e-3 | 1.8e-2 / 4.0e-2 / 0.16 | 0.59 / 0.53 / 0.48 |
| hankel_local | 9.5e-4 | 0.13 / 0.41 / NaN | 0.48 / 0.49 / NaN |
| hankel | 2.4e-3 | 0.15 / NaN / NaN | 0.38 / NaN / NaN |

**Stage 1b — residual u_{n+1} = u_n + v (node 0 set to 0), q_weight 0.1** (`runs/stage1b/`), test set:

| variant | 1-step | first step (u) | rollout u @10/100/500 | rollout q_D @1/10/500 |
|---|---|---|---|---|
| hankel_local | **2.1e-4** | 0.10 | 0.16 / 0.23 / 0.80 | 0.24 / 0.25 / 3.7 |
| fno | 5.4e-4 | 0.26 | 0.18 / 0.19 / 0.31 | 0.66 / 35 / 120 |
| logsine | 4.8e-4 | 0.055 | 0.13 / 1.3 / 1e13 | 0.43 / 0.83 / 1e13 |
| dual | 5.4e-4 | 0.038 | 0.11 / 0.61 / 1e9 | 0.29 / 0.39 / 1e9 |
| hankel | 8.1e-4 | 0.11 | 0.16 / 1e14 / NaN | 0.37 / 0.22 / NaN |

Diagnosis: rollouts started from the *true* u(1000) still reach 6-14% u error after 10 steps
(linear accumulation would give ~5e-3) → the learned step amplifies errors ~1.3-1.5x per step
(the exact step is contracting). One-step training never penalizes this.

## Key findings and explanations (so they need not be re-derived)

- **Basis study** (Milestone 4): for heterogeneous k, 32 log-sine modes sin((n-½)πs) represent states to
  1e-4 vs 3e-2 for 32 Hankel modes (ln r measure). Near-well kinks (∂u/∂r ∝ 1/k) are what Hankel misses;
  94% of the Hankel residual is at r < 10.
- **Norms**: the r-weighted norm nearly ignores r < 10; use the grid (ln r) measure plus q_D.
- **q_D sensitivity**: q_D = T·u₁ with u₁ ≈ 1e-3 at the first interior node (r = 1.0068); an absolute error
  of 1e-4 there is a 10% q_D error but invisible in the L2 of u. Hence u = s·v and the q_D loss term.
- **s·v**: s = ln r / ln r_e (= ln(r/r_w)/ln(r_e/r_w)); s = 0 at the well (hard BC), u ≈ c·s near the well,
  so v is O(1) and q_D ≈ k·v₁/ln r_e. The residual form of stage 1b lost this scaling → q_D errors.
  s = ln(r/r_e) would be wrong (zero at the outer boundary, not the well).
- **Time scale**: t = t_D φ μ c_t r_w² / k_ref. With k = 100 md, φ = 0.2, μ = 1 cp, c_t = 1e-9 1/Pa,
  r_w = 0.1 m: Δt_D = 1000 ≈ 20 s, t_D = 5e5 ≈ 2.8 h (1 md: 34 min / 12 days).
- **Rollout** lives only in `evaluate.py` (`predict`, `rollout`); training uses single pairs.

## How to run

```bash
cd /home/daniel_88/hno/phase_1
PY=/home/daniel_88/py314/bin/python
# train one variant (writes runs/<out>/<variant>/{best.pt,last.pt,log.csv,config.json})
OMP_NUM_THREADS=2 $PY train.py --variant logsine --residual --q-weight 0.1 --steps 5000 --threads 2 --out runs/stage1b
# evaluate all variants of a stage (writes eval.json per variant, ~15 min with 8 threads)
OMP_NUM_THREADS=8 $PY evaluate.py --runs runs/stage1b --threads 8
# regenerate the dataset (~28 min, 8 workers, 2 GB)
OMP_NUM_THREADS=1 $PY dataset.py --workers 8
```

Machine: 8 cores, no GPU (WSL), 7 GB RAM — run training in parallel at 1-2 threads each; data is memory-mapped.

## Working conventions

- Ask before creating or editing files; write code into files, not into chat; results as local PDFs
  (`phase_1/fig_*.py` -> `tex/fig_*.pdf`); `tex/` holds only .tex/.pdf; commit only when asked.
