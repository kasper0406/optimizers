# WP1.2 — Phase-1 Measurement Report (airbench, descriptive)

Date: 2026-07-19 · Git: `b9aab63` (pre-registration commit; predates all runs)
· 20 instrumented seeds (dev seeds 1000–1019) · RTX A6000 spot · sweep ~9 min,
~$0.06 attributed · **This report is descriptive only. It makes no pass/fail
claim on the pre-registered criteria; Gate 1 is a human decision.**

## Setup

Instrumented-but-stock Muon on the airbench94 record recipe (`airbench`
experiment + read-only `InstrumentationHub`): per filter matrix (6), top-k₁=16
subspace-iterated singular pairs of momentum + k₂=16 bulk probes, raw
pre-momentum projections every step, EMAs at β ∈ {0.9, 0.99}
(WP0.5-validated `src/stats`, array mode), T_refresh=50, snapshots every 5
steps, HVP disabled (validation-only feature; see gaps). Sidecars + results in
`results/airbench_instrumented_seed10??_*.{json,instrumentation.json}`.

**Non-perturbation check:** instrumented tta_val_acc mean 0.94036, std 0.00152
(n=20, dev seeds) vs the WP0.1 baseline 0.94003, std 0.00141 (n=100, eval
seeds) — indistinguishable; instrumentation left training behavior unchanged.
Overhead at these settings: 7.7% median step-time
(`results/bench_overhead_airbench_v2.json`).

## The three plots (plan §1.2) — `reports/figures/wp12/`

1. **Regime scatter** (`regime_scatter.png`): the (SNR, ρ) cloud is broad
   (SNR spanning ~1e-4–1) with its ρ mass centered clearly below zero and a
   long tail to ρ ≈ −1. Visually the cloud reads as one connected mass with
   asymmetric structure rather than three separated islands; the pre-registered
   test is the GMM/BIC statistic below, not visual separation. Points are
   uncolored (no HVP records).
2. **Regime occupancy** (`regime_occupancy.png`): all directions start in the
   signal prior; after the first refresh (step 50) occupancy settles at roughly
   65% signal / 25–30% noise / 5–10% oscillating (β=0.9; β=0.99 similar with a
   thinner oscillating band that shrinks over training). The oscillating
   fraction is largest right after step 50 (the highest-LR stretch tracked
   post-refresh) and thins toward the anneal tail while noise grows — the
   direction of DynMuon's "positive p early, negative p late" story is visible
   in these labels. **Artifact note:** at each refresh (steps 50/100/150)
   occupancy transiently flips almost entirely to "noise" for ~2–5 steps before
   recovering — post-reset low-ESS classification, not a training event.
   Labels here depend on the dev-placeholder classifier thresholds in the
   config (the pre-registered quantities below do not).
3. **ηλ calibration** (`eta_lambda_calibration.png`): intentionally empty —
   no HVP records (HVP is Phase-1-validation-only and was off). See gaps.

## Pre-registered quantities (criteria/phase1_preregistration.md; computed by
`scripts/analyze_phase1.py`, full tables in `wp12-phase1-preregistered-stats.md`)

- **Criterion-1 statistic:** GMM/BIC (k ∈ {1,2,3}, random_state=0, points =
  per-direction snapshots in (log₁₀ SNR, ρ), pooled over 20 runs) prefers
  **≥2 components in 100% of the 24 (matrix, phase) cells**, for both β,
  with and without a 10-step post-reset burn-in. (Pre-registered bar: ≥70%.)
- **Criterion-2 statistic:** the ρ < −0.2 population during phase 1 (highest
  LR; airbench LR decays linearly) is **59.8–68.2% of snapshots** (β=0.99 /
  β=0.9), i.e., emphatically non-empty; it peaks in phase 2 (80–89%) and
  falls to 37–46% in the anneal tail. (Pre-registered bar: non-empty.)

## Caveats the human should weigh at Gate 1

1. **BIC at large n:** each cell pools ~38k snapshots; at that sample size BIC
   prefers ≥2 components for nearly any non-Gaussianity. The criterion is met
   in form, but the statistic is weakly discriminating as pre-registered — the
   scatter's visual "connected mass with structure" is the honest companion
   fact.
2. **Breadth of negative ρ:** 60–89% of snapshots below −0.2 (including bulk
   probes) is broader than an edge-of-stability-only story predicts. Candidate
   mundane contributor: airbench samples batches without replacement within an
   epoch, which can induce negative lag-1 correlation in the noise component
   independent of curvature. An HVP-enabled run (gap below) and/or a
   with-replacement ablation would separate these.
3. **Timescale sensitivity** (itself a finding, per plan §1.1): the β=0.9 and
   β=0.99 tables differ by up to ~9 points in phase-wise ρ<−0.2 fractions but
   agree qualitatively everywhere.

## Gaps / prepared next steps (no launches without the respective gate)

- **ηλ ↔ HVP calibration (plan §1.2 plot 3):** needs one small HVP-enabled
  instrumented run (hub supports it; config flag) — cheap on the same A6000
  class.
- **nanogpt leg of WP1.2:** 2–3 instrumented seeds on the pinned BosAlign
  record — requires the WP0.2 port first (world_size assert relaxation +
  grad-accum), H100-class VM.
- Classifier-threshold sync (occupancy labels only) if Phase 2 proceeds.

## Cost

VM `rm-wp12-phase1` total $3.23 (98% of it environment setup, driver upgrade,
image builds, dataset download, and the overhead benchmark; the 20-seed
instrumented sweep itself was ~9 min ≈ $0.06). Cumulative project cloud spend:
$4.47.
