# Gate 1 Decision Record

Date: 2026-07-19 · Decided by: agent under explicit full delegation (Kasper
Nielsen, in-session, 2026-07-19: "Please just drive by yourself - no human
confirmations needed"), with mandatory adversarial review per the same
session's instruction. Evidence at commits `a8e40dc` (measurement) and
`ec6cf7b` (disambiguation).

## Decision

**PROCEED to Phase 2 with OSCILLATION-FOCUSED scope** (plan §1.4 middle
branch / §2.4 contingency): the primary pre-registered Phase-2 claim attaches
to the oscillation channel only. The noise channel and full three-channel
routing run as explicitly **exploratory arms** with no pre-registered claim.
"Stop" was rejected: the negative-ρ population is real, large, phase-
structured, and survived its artifact test at trivial forward cost (~$5).

## Evidence basis (what actually carries weight)

1. **Excess over null, not the formal criteria.** The adversarial review
   demonstrated (via white-noise streams through the identical pipeline) that
   both pre-registered criteria as literally worded are satisfied under the
   null: BIC prefers k≥2 on white noise at these sample sizes, and the null's
   ρ<−0.2 fraction is 0.22–0.29. The gate therefore rests on the
   **non-pre-registered excess**: observed ρ<−0.2 fractions 0.60–0.89 in
   phases 1–3 (vs 0.22–0.29 null), consistent with an AR(1) population at
   ρ ≈ −0.3…−0.5, phase-structured (falls to 0.37–0.46 in the anneal tail),
   and unchanged under with-replacement sampling (max |diff| 0.05, most ≤0.03).
   This is recorded as a deviation-in-spirit from the pre-registration: the
   registered statistics were too weak, and the decision uses stronger,
   post-hoc-but-adversarially-audited statistics. Both are reported.
2. **No detectable signal population.** At mature classification
   (n_since_reset ≥ 50): frac(|t| ≥ 4) = 0.0000; the instrument's structural
   t-ceiling (|t| ≤ SNR·√ess) cannot reach τ_sig = 4 at observed SNR (q90 ≈
   0.26) and these β. The 68/27/5 occupancy figure is an artifact of the
   start-in-signal confidence window; the truly classified mature population
   is ~85% noise / ~10–16% oscillating / ~0.3% signal. **Signal/noise
   separation was not observed** → noise-channel routing has no empirical
   basis and maximal misclassification-asymmetry exposure (plan risk 4).
3. **ηλ calibration is a negative result.** Implied ηλ (amplitude ratio)
   saturates at ~2–4 on oscillating-labeled directions — statistically
   indistinguishable from the pure-noise null of the ratio statistic (~2.05)
   — while HVP-measured lr·λ spans 0–65 (Pearson −0.06/−0.14). Partial
   excuse: training is stable at lr·λ ≫ 2, so GD eigendynamics (the basis of
   the amplitude model) demonstrably do not govern Muon's per-direction
   stability; lr·λ_HVP is not the operative multiplier. But the estimator
   also fails against its own noise null, so **g_osc as v0-coded is a ≈0.53
   near-constant attenuator, not adaptive damping**, and its decay-escape
   gate fired on 0/12,851 snapshots (dead code as configured).
4. **Timescale instability:** the oscillating sets at β=0.9 vs 0.99 overlap
   at Jaccard 0.36 — "oscillating" is a (direction, window) property.
5. **Anomaly, open:** bulk probes out-oscillate top directions in phase 1
   (0.80–0.83 vs 0.39–0.56), inverting the sharp-direction EoS prediction —
   consistent with a broadband (e.g. momentum-overshoot) mechanism.

## Binding amendments (from the adversarial review, accepted in full)

A1. Phase-2 primary claim = oscillation channel only; noise/full-routing arms
    exploratory. Phase-2 success criteria to be drafted accordingly before
    the eval-seed comparison runs.
A2. Add a **constant-g_osc arm** (g_osc ∈ {0.25, 0.5, 0.75} fixed) so
    "adaptive vs constant attenuation" is adjudicated, since v0's adaptive
    values match the noise null. Report the decay-gate dead-code fact.
A3. This record documents the null-satisfaction of the literal pre-registered
    criteria (see 1 above) — required for pre-registration honesty and the
    eventual paper.
A4. **Mechanism probes** (~$0.30, before/alongside WP2.2): instrumented
    momentum=0 run and LR-ladder (×0.5, ×0.25) — does the negative-ρ
    population track momentum/LR as overshoot predicts? Bulk-vs-top anomaly
    reported either way.
A5. WP2.2 comparison runs must log **treated-fraction and per-channel gain
    distributions** (n_min=50 leaves the first 50 of 200 steps untreated and
    ~67% of directions reset per refresh at align_min=0.9 — a null result
    without these logs would be uninterpretable). β sensitivity: osc-arm dev
    sweep includes β=0.9 alongside default 0.99.
A6. `scripts/analyze_phase1.py` distinct-directions key omits the matrix
    (640 = 20 runs × 32 dirs collapsed across 6 matrices) — fix, cosmetic.

## What this changes downstream

- WP2.2 execution order: mechanism probes + stage-A tuning first; primary
  eval tables built around muon vs routed(osc-only) vs constant-g_osc vs
  baselines; full-routed and noise-only demoted to exploratory appendix arms.
- The paper's Phase-1 section gains two negative results (curvature-for-free
  fails under Muon; literal pre-registered criteria null-satisfiable) and one
  robust positive (large artifact-tested negative-ρ excess with phase
  structure).
