# Program #11 — composite frontier-invariant search: pre-registration

2026-07-22. Offline measurement program (zero GPU): search the span of
already-measured trajectory quantities for a scalar conserved along the
Muon useful-LR frontier — the open question every prior program sharpened
(all four *single* instrumented quantities fail the pre-registered
tracking signature; `reports/stability-frontier.md`). Committed before
any fit is computed.

## Data

Existing instrumentation sidecars from programs #6/#6b (results/, seeds
1400–1414): coarse grid (B ∈ {500, 1000, 2000, 4000} × 8 rungs, n = 2) =
**fit set**; dense ladders (B ∈ {1000, 2000, 4000}, ×1.15 rungs, n = 5)
plus the step-matched B = 8000 arm = **validation set** (held out from
all fitting). Shoulder cells: fit-set shoulders as recorded in
`reports/stability-frontier.json`; validation shoulders recomputed from
the dense accuracy data by the program-#6b interpolated-crossing
convention, taking the nearest measured rung.

## Features (per run; seed-averaged per cell)

f1 occupancy (β = 0.9, mature n_since_reset ≥ 10, pooled ρ < −0.2
fraction — the prior convention); f2 lr·D_smooth_frobenius plateau
(last-50%, matrix-mean); f3 lr·D_smooth_spectral plateau; f4 hvp_q90
(q90 of lr·λ_HVP; absent at B = 8000, disclosed); f5 grad_fro_norm
(last-50% matrix-mean); f6 sigma level (per-direction window σ,
last-50% pooled); f7 spike_rate (|z| > 3 rate, MAD-z, 5-step
post-refresh burn-in — the program-#8 intermittency statistic, never
tested against the frontier); f8 rho_mean (pooled mature per-direction
ρ, β = 0.9). Learning rate and batch size themselves are **excluded**
as features: the invariant must be a measured dynamical quantity, not a
re-encoding of the grid coordinates.

## Candidate space and fit

All 28 ordered pairs (f_i, f_j), composite I = log f_i + a·log f_j with
the single exponent a fitted to minimize the max/min ratio of exp(I)
across the **fit-set shoulder cells**. Singles (a = 0) are the known-
failing baselines. An unrestricted 8-weight fit is computed but reported
as exploratory only (capacity ≈ number of shoulder cells; assumed
overfit).

## Pre-registered pass criterion (evaluated on validation cells only)

A pair PASSES iff, with the exponent frozen from the fit set:
1. shoulder ratio across validation batches (incl. B = 8000 where
   features permit) **≤ 1.5** (the original program-#6 threshold), and
2. within-B variation along the lr axis (max/min across that batch's
   dense rungs) **≥ 3.0** at every validation batch.

Multiplicity control: the same fit-and-validate pipeline is run on 1,000
permutation draws (shoulder rung replaced by a uniformly random rung per
batch, fit set and validation set alike); the expected number of
false-passing pairs under the null is reported next to the real count.
If no pair passes, that extends the frontier-invariant null from 4
single quantities to the ~30-dimensional pair space — itself a
publishable tightening.
