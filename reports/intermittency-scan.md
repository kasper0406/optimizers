# Intermittency scan — per-direction kurtosis / spike-rate (offline)

2026-07-22. Prompted by the user's hypothesis: *sporadic-but-real signal is
heavy-tailed in the per-direction projections s_i(t); excess kurtosis or a
spike-rate counter separates it from Gaussian noise where mean-based tests
(the t-ceiling result, program #4) are structurally null.* Zero new GPU
runs: 218 instrumentation sidecars from the frontier programs (seeds
1400–1414, lr × batch grid) carry raw 800-step per-direction series.
Script `scripts/analyze_intermittency.py` (synthetic-validated,
`tests/test_analyze_intermittency.py`); AR(1)-Gaussian empirical null
through the identical pipeline (serial correlation does not confound the
kurtosis null — stationary Gaussian AR(1) is marginally Gaussian).

## 1. Naive scan: a spectacular false positive

25,920 directions: median excess kurtosis **+4.6** (null q99 +1.44), 56%
of directions above the null 99th percentile, split-half spike-count
stability **+0.80**, and the effect concentrated in top-anchored
directions (median g2 +47, 100% above null) vs bulk probes (+0.55, 13%).

**Artifact.** The spike-offset diagnostic shows **93.4% of top-direction
spikes occur in the first 5 steps after subspace re-anchoring** (81% in
the first 2). At refresh, the tracked direction is aligned to the current
top of momentum; the projection starts large and decays — MAD-z flags
those steps as spikes. Bulk probes (no anchoring) show a flat/rising
offset profile. **Methodological rule for any future 4th-moment
instrumentation: burn in ≥5 steps after every subspace refresh.**

## 2. Burn-in-corrected scan (5-step post-refresh burn-in)

Series-length cutoff limits this pass to B ∈ {500, 1000} (16,128
directions, 84 runs); higher-B runs have too few post-burn-in samples.

- Median g2 **+0.61** vs null mean +0.42, q99 +1.68 — the bulk of the
  population is Gaussian-compatible.
- Directions above null q99: **g2 9.8%, p4 12.2%** (vs 1% expected) — a
  real but modest residual, now roughly equal in top (11.3%) and bulk
  (8.3%) directions.
- Split-half stability drops to **+0.29** — residual spikes have some
  per-direction persistence, but weak.
- **The surviving structure is strongly LR-dependent** (B=1000):

| lr band | frac g2 > null99 | frac p4 > null99 |
|---|---|---|
| lr ≤ 0.32 (≈ record) | 4.3% | 8.3% |
| 0.32 < lr ≤ 0.55 | 6.7% | 12.0% |
| lr > 0.55 | **18.9%** | **25.5%** |

## 3. Verdict on the hypothesis

- **"Sporadic but real signal that mean-tests miss": not supported at
  healthy LR.** Near the record LR the population is close to the
  Gaussian null (4.3% vs 1%). The heavy-tail excess concentrates where
  training is already degrading (above the useful-LR shoulder) — it reads
  as transient instability events, not intermittent persistent features.
- The proposed gate rule ("high kurtosis + near-zero mean → leave at Muon
  default") would therefore protect almost nothing at healthy LR on this
  testbed — and the per-direction gain lever it feeds is placebo-null
  anyway (Gate 2).
- **What the scan did yield:** (a) the anchoring-artifact methodology
  note (§1) — a trap any kurtosis-style instrumentation must document;
  (b) an LR-monotone spike-rate population — a *new candidate observable*
  for the open frontier-invariant question (all four previously
  instrumented scalars failed the pre-registered tracking signature;
  spike-rate was never tested); (c) motivation for a spike-gate on the
  program-#8 tempo controller at LM scale (a spike step pushes
  cos(G_t,G_{t-1}) toward 0, which the controller reads as "too hot" —
  harmless on airbench where spikes are rare at healthy LR, but nanogpt
  training has documented loss spikes at normal LR).

## 4. Caveats

- Frontier sidecars only (seeds 1400–1414); the Phase-1 cloud sidecars
  (incl. with-replacement and momentum=0 ablations) were never synced, so
  the batch-composition control is indirect (B=500 vs B=1000 differ
  little post-burn-in, weakly against a rare-example origin).
- B > 1000 cells underpowered (short series); the batch-fade test of the
  residual is open pending longer high-B instrumented runs.
- Burn-in 5 was chosen from the offset histogram, post hoc. Re-run at
  burn-in 10: the residual attenuates but persists (g2 6.6% / p4 10.1%
  above null vs 9.8% / 12.2% at burn-in 5; split-half +0.23; the LR
  monotonicity unchanged). Some of the residual is therefore slow
  post-refresh relaxation rather than discrete spikes; the LR-dependent
  component survives both settings.
