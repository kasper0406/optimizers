# Pre-registration — Program #22: BBP Alignment Frontier, Phase A-empirical

Registered 2026-07-24, before any gradient-probe run. Source:
`reports/ideation-geometry-theory-2026-07-24.md` §1 program #2 (human GO for
the portfolio, "Please proceed" after #20 closed). This document registers the
**empirical stage only**: measuring the msign alignment curve a(B) at frozen
checkpoints. The zero-free-parameter RMT prediction (free-probability curve
from measured spectra + noise covariance) is Phase A-theory, registered
separately before its computation touches these curves' held-out use.

## 1. Objects

At a frozen parameter point W (no optimizer steps), per Muon matrix m:
G_B = mean gradient over B disjoint 49,152-token record chunks (the record's
own BOS-aligned chunking); msign(·) = the record's 5-step Newton-Schulz
orthogonalization in bf16 (what Muon actually computes, not exact msign).

**Estimator (split-half doubling tree):** stream M = 512 chunk gradients;
merge pairwise (binary-counter streaming). At each merge of two independent
half-sums A, B each aggregating b chunks, record
c_m(b) = ⟨msign(A_m), msign(B_m)⟩_F / (‖msign(A_m)‖_F · ‖msign(B_m)‖_F).
The alignment estimate is â_m(b) = sqrt(max(mean_merges c_m(b), 0)) — the
standard split-half debiasing under independent fluctuation halves.
b grid: 1, 2, 4, …, 256 chunks (49K – 12.6M tokens; the record batch is 8
chunks, so the grid spans 1/8× to 32× the training batch, ≥ 16× as the
ideation spec requires). Estimates per level: 512/2b merges; propagate
split-half scatter as the error band.

**Momentum correction:** e_m(b) = â_m(b_eff) with b_eff = b·(1+β)/(1−β),
β = 0.95 (EMA variance-reduction factor), read off the measured curve by
log-b interpolation; b_eff = 39b. Registered before measurement.

## 2. Runs

Checkpoints (3 per seed, seeds **1511 and 1512**):
- **early**: step 321 (new truncated hot runs, max_steps 321, checkpoint kept);
- **mid**: step 963 (the stored Wave-1 prefix checkpoints);
- **annealed**: step 1750 (the stored arm-A final-weight artifacts).

Data stream: train shards, fixed registered offset (file index 3, position 0),
identical for every checkpoint/seed — gradient statistics at matched data.
Attention window set to the checkpoint's record-equivalent step. 512 chunks
per probe; 6 probes total; ~1 GPU-h expected all-in.

## 3. Registered quantities and criteria

Primary per probe: the per-matrix curves â_m(b) with bands; the aggregate
median curve â(b); the **saturation statistic** s = â(8)/â(64) aggregated as
the median over matrices (record batch vs 8× record batch).

- **(V) Vacuity guard:** the measurement is informative iff ≥ 50% of Muon
  matrices show dynamic range max_b â_m − min_b â_m ≥ 0.2 across the grid on
  every probe. If violated at the low end, extend the grid downward (b < 1
  chunk is impossible → declare the low-b regime unmeasurable at this token
  granularity and report as such).
- **(S) Saturation consistency (the falsifiable claim of this stage):** the
  lab measured Muon's useful-LR band batch-INVARIANT on nanogpt across 8×.
  Under the BBP account this requires the momentum-corrected alignment to be
  saturated at the training batch: **PASS iff median_m e_m(8)/e_m(64) ≥ 0.9
  at the mid checkpoint** (the training-relevant one) on both seeds.
  **FAIL** (e(b) still rising steeply at the record batch while the measured
  frontier is invariant) kills the noise-side account of the nanogpt
  frontier — registered as an informative negative for program #22, exactly
  the "non-saturated a(B) with measured invariance" branch of the ideation
  spec.
- Descriptive (no criteria): early-vs-mid-vs-annealed curve evolution;
  per-matrix-class (attn vs MLP) breakdown; raw c distributions.

## 4. What this stage does NOT claim

No RMT comparison, no airbench exponent prediction, no lr*(b) claim — those
require the Phase A-theory computation and airbench probes, each with its own
pre-registration. This stage only measures a(B) and adjudicates (S).

Cost: ~1.5 GPU-h (incl. two 321-step truncated runs) + half an analyst-day.
