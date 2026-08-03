# Pre-registration — Program #21: Muon Central Flow, Stages 0–1

Registered 2026-07-24. Source: `reports/ideation-geometry-theory-2026-07-24.md`
§1 program #3 (human GO for the portfolio). This document is the Stage-0
machine-readable spec (committed before any lab-data contact by this program)
and the Stage-1 kill-switch criteria. Stage 2 (hash-committed numeric
predictions vs the held-out frontier tables) gets its own registration after
— and only if — Stage 1 passes.

## Stage 0 — definitions and data-hole handling (spec before contact)

- **Goal object:** a deterministic "central flow" for spectral-LMO training —
  a self-consistent time-averaged description of Muon dynamics
  M ← βM + ∇L + ε, W ← W − η·msign(M) in the oscillatory (EoS-like) regime,
  with the oscillation second moment Σ(t) determined self-consistently.
- **EoS-occupancy definition (for regime assignment on lab data, fixed now):**
  a matrix at step t is "oscillatory" iff the lag-1 autocorrelation of its
  tracked-direction gradient projections is < −0.2 (the lab's standard
  instrumented statistic); a run cell is EoS-occupied iff the occupancy
  fraction ≥ 0.5 at the mid-training instrumented window.
- **Regime-assignment rule for frontier cells:** each (lr, batch) cell of the
  two frontier tables is assigned {sub-EoS, EoS, over-shoulder} using only
  quantities already logged in the frontier runs (occupancy where present;
  the lr position relative to the measured accuracy/loss shoulder otherwise).
  Assignment happens before any flow prediction is computed, and is frozen.
- **Data holes, handling registered now:** (1) airbench B=8000 has no HVP
  spectra (OOM, amendment A1) — the flow's spectral inputs there come from
  moment-matched extrapolation of the B≤4000 spectra, flagged as such;
  (2) nanogpt has zero stored HVP spectra — if Stage 2 needs them, new
  Lanczos probes run on the stored step-963 prefix checkpoints (dev seeds
  1511–1513 only), pre-registered as measurement-only.
- **Held-out targets (never fitted):** the entire airbench frontier table
  (programs #6/#6b: lr* per batch, exponent 0.35 [0.30, 0.42]) and the
  nanogpt frontier table (program #7: batch-invariant across 8×). Flow inputs
  may use only: synthetic calibration, and spectra/statistics from the three
  designated dev checkpoints per harness. Any Stage-2 prediction file is
  hash-committed before comparison ("unblinding") against the tables.

## Stage 1 — kill-switch (synthetic validation before any lab data)

- **Test family (registered):** matrix quadratics L(W) = ½ tr((W−W*)ᵀ A (W−W*) B)
  with SPD A, B of controlled spectra (curvature eigenvalues λ_i(A)·μ_j(B)),
  plus i.i.d. gradient noise of scale σ; Muon dynamics with β = 0.95, exact
  msign (SVD) and the record's 5-step Newton-Schulz both simulated (the flow
  must match the NS variant; exact msign is diagnostic).
- **Grid (registered):** (η, σ) ∈ 3×3 with η spanning sub-EoS to η·λ_max > 20,
  σ from 0 to noise-dominated; matrix size 64×48; A, B spectra log-uniform
  with condition numbers (10, 100); 5 simulation replicates per cell.
- **KILL-SWITCH:** the derived central flow must reproduce the *time-averaged*
  simulated trajectory (EMA half-life 20 steps, comparison over the first
  2000 steps) within **10% relative trajectory error** (time-averaged
  Frobenius error / trajectory norm) on **≥ 8 of 9 grid cells**, including
  every η·λ > 20 cell, for both msign variants. **Failure after two
  derivation iterations stops the program before any lab-data contact**
  (fallback: the benched NRSE program per the ideation portfolio).
- **Reference bar:** the naive flow dW/dt = −η·msign(∇L(W)) (no oscillation
  correction) is run on the same grid first; the derived flow must beat it on
  every cell where the naive flow exceeds 10% error — otherwise the "central"
  machinery adds nothing and the program stops.
- Deliverable either way: `reports/central-flow-stage1.md` with the grid
  table, plus the simulator (`src/theory/muon_flow_sim.py`, unit-tested).

Costs: Stage 1 is CPU/GPU-light (minutes of simulation); the cost is analyst
time on the derivation (budgeted 1–2 weeks; kill-switch caps it).

## Amendments

Both made 2026-07-24, **before any candidate flow was derived or scored**
(no Stage-1 result existed at amendment time), on findings from an internal
adversarial review of the harness. Disclosed here rather than silently
applied.

**A1 — kill-switch metric was defective (matched filtering added).** As
originally registered, the 10% criterion compared the EMA-averaged simulated
trajectory (half-life 20) against an *unfiltered* flow trajectory. The EMA
has a ~29-step time constant that is never de-lagged, so lag alone produces
~0.21 error: a **perfect** flow would have failed the kill-switch on the
harness's own sub-EoS consistency case. `trajectory_error` now applies the
same causal EMA to the flow before comparison (`match_filter_half_life`),
which takes that case to 0.0017. The 10% bar and all other Stage-1 terms are
unchanged. Rationale for amending rather than accepting: the criterion as
written could only produce false negatives, and a false kill would have
retired the program on an artifact.

**A2 — the "first Stage-1 measurement" was vacuous and has been replaced.**
The originally-committed `stability_and_floor_vs_curvature` varied an overall
scale multiplier on the curvature operator A and reported a loss-floor ratio
"constant to three decimals across 1000x curvature" (commit `890650e`). That
constancy is an **identity, not a measurement**: msign(s·M) = msign(M), so an
overall scale leaves the entire iterate sequence invariant and the ratio is
s-invariant by construction; the sweep measured the scale-invariance of the
polar factor. It is retracted. The corrected sweep varies the axes the
dynamics are actually sensitive to — curvature **conditioning** and **eta** —
and yields a non-trivial target: the floor ratio
L_floor / (0.5*lam_max*(eta*sqrt(r))^2) is **invariant in eta across 100x**
(0.419/0.421/0.423) but varies **7.4x with conditioning** (0.42 at cond
(3,3); 0.093 at (10,100); 0.057 at (100,1000)). A candidate flow must
reproduce both the eta-invariance and the conditioning dependence; this
replaces the retracted claim as the registered first measurement.

## Relation to standing results

The flow, if it survives, must be consistent with (and would explain):
Muon stability at HVP η·λ ≈ 65 with no cliff (graveyard #3); oscillation
amplitude saturating at O(η) independent of curvature (the structural claim
motivating the program — testable in Stage 1 simulation *before* any
derivation, and registered as the first Stage-1 measurement); the failure of
all instrumented trajectory scalars as frontier invariants (null #5 — the
flow-level invariant, if any, must be a different object).
