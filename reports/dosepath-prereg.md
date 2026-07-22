# Program #15 — dose-path geometry: pre-registration (Phase A, measurement)

2026-07-22. Committed before any run. Dev-phase; adjudication of any
follow-on intervention is HUMAN. Motivated by program #13's P5 violation
(same-seed LR-ladder endpoints follow a graded path, rel-dist 0.53–0.72,
vs the unrelated-destinations geometry 0.95–0.99 of spectral
interventions) and by the user's direction to exploit geometric
structure rather than the LR axis per se.

## Runs

B = 1000, rungs {0.24, 0.37, 0.48, 0.64} × seeds 1476–1485 (fresh
block), endpoint hooks on: 40 runs ≈ 6 GPU-min. Checkpoints scratch-only
(deleted after measurement; durable record is the derived JSON).

## Pre-registered measurements

- **M1 (mode connectivity):** per seed, tta accuracy at 5 evenly spaced
  points on the linear path W(α) between the peak (0.24) and shoulder
  (0.48) endpoints (BatchNorm stats recomputed on the train slab before
  eval at each α — the standard LMC repair). Scalar: barrier = min
  path accuracy − min(endpoint accuracies). Reading bars: barrier
  ≥ −0.5pp = "one basin" (connected); ≤ −5pp = barriered.
- **M2 (straightness):** per seed, cos between (W_0.37 − W_0.24) and
  (W_0.48 − W_0.24), and between (W_0.48 − W_0.24) and
  (W_0.64 − W_0.24), on concatenated filter+head weights. Bars: mean
  cos ≥ 0.7 = "chord-like path"; ≤ 0.3 = curved/independent.
- **M3 (sharp-or-sloppy):** per seed, Rayleigh quotient of the
  normalized dose direction (W_0.48 − W_0.24) in the shoulder
  endpoint's Hessian (fp32 functional pattern, fixed 2000-example
  batch) divided by that endpoint's λ1 (20-iteration power iteration).
  Bar: ratio ≤ 0.1 = the dose displacement lives in sloppy directions
  (repair-plausible); ≥ 0.5 = sharp-aligned.
- **M4 (layer profile):** per-matrix share of ‖dose vector‖² —
  descriptive.

Interpretation gate (pre-committed): a follow-on repair-operator
experiment (subtracting/constraining the dose component) is proposed to
the human ONLY if M1 reads connected AND M3 reads sloppy; otherwise the
program reports geometry and stops. Cross-seed weight-space comparisons
are NOT made (permutation-symmetry confound; disclosed).

Novelty sweep (launched in parallel, dated 2026-07-22): linear mode
connectivity across hyperparameters (Frankle et al. LMC lineage is
across seeds/data order; the LR axis is the question), task-arithmetic /
model-editing for hyperparameter-induced damage, and any published
"subtract the large-LR damage direction" precedent. The sweep's verdict
gates the *claims*, not these measurements.
