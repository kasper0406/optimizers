# Program #11 results — the frontier invariant is not in the instrumented span

2026-07-22. Pre-registration `reports/invariant-search-prereg.md` (commit
0ecd919, before any fit). Offline; zero new runs. Script
`scripts/analyze_invariant_search.py`; per-cell features cached in
`reports/invariant-search-features.json` (61 cells, seed-averaged).

## Headline

**No single feature and no log-linear pair passes the pre-registered
tracking signature on held-out validation cells** (0 / 8 singles, 0 / 40
evaluable pairs; permutation null: 0.00 expected passes, q95 = 0 — the
criterion is stringent, so the zero is informative). The frontier-
invariant null, previously established for 4 single quantities on the
coarse grid, now extends to the pair span of 8 features including the
never-before-tested spike-rate, under held-out validation.

## The structural finding behind the zero

The failure mode is not near-misses — it is that **within the frontier
region, every instrumented dynamical quantity is nearly flat along the
lr axis** (within-batch max/min across the dense rungs: occupancy
1.11–1.19, sigma 1.02–1.36, grad-norm 1.03–1.28, spectral smoothness
1.19–1.80, HVP q90 1.30–1.67, spike-rate 1.32–1.59 — nothing reaches
even 2, against the pre-registered bar of 3) — while accuracy falls
~1pp+ across the same rungs. Two corollaries:

1. Apparent equalizations are illusions of flatness: occupancy's
   shoulder ratio of 1.22 (under the 1.5 bar!) coexists with a
   fixed-lr cross-batch ratio of ~1.28 — it is approximately a function
   of batch alone here, tracking nothing.
2. The trajectory's time-averaged statistics are *locally blind* to the
   thing that degrades accuracy across the shoulder. This matches
   program #10 (mid-run ±40% LR perturbations are absorbed at no cost)
   and the equivalent-destinations mechanism: the dynamics look the
   same on both sides of the shoulder; the damage lives somewhere none
   of these scalars integrates over.

## Reading

The frontier-setting quantity is not merely unidentified — it is not in
the span of per-step trajectory scalars of the kind this project
instruments (magnitudes, curvatures, smoothness, serial statistics,
spike rates, occupancies). Candidate objects that survive this
exclusion, for any successor program: end-state properties (what the
extra noise at high lr does to the *learned solution* rather than to
the trajectory — e.g., feature quality/memorization measures),
anneal-phase interactions (the deficit is cooldown-concentrated on the
LM side), and genuinely non-scalar objects (distributions over
directions, the loop transfer function of program #9's closing note).

## Disclosures

rho_mean was effectively excluded by the log-space positivity
requirement (it is negative); an |rho| variant was not pre-registered
and was not fit. hvp_q90 is absent at B = 8000 (HVP off, amendment A1),
so pairs involving it validate on 3 batches. The unrestricted 8-weight
exploratory fit was not pursued once all pairs failed criterion 2 — with
every feature flat in lr, no linear combination can satisfy it (a
capacity argument, not a fit result).
