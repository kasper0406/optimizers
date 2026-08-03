# Program #9 — FIRMuon (process-matched momentum filter)

Phase A (offline design + kill-tests) report and Phase B pre-registration.
2026-07-22. Dev-phase document; nothing here is an evaluation-gate claim.
Code `src/optim/firmuon.py`; kill-test scripts in the session scratchpad
(`wiener_killtest.py`), to be promoted into `scripts/` with the results
report.

## 0. Decision record (why this direction)

Prompted by the user's request for radical approaches with significant-gain
potential. Four candidates were novelty-swept in parallel on 2026-07-22:

1. **Process-matched momentum filtering — OPEN and pursued.** No published
   work measures gradient autocovariance online and synthesizes momentum
   filter weights from it (beyond selecting β inside the EMA family), in
   any optimizer. Nearest neighbors to differentiate: Greedy Alignment
   (2512.06370; the filter framing, selects β only), SGDF/DMR
   (2311.02818/2603.06120; "Wiener" gain from lag-0 variances under a
   white-noise assumption — owns the keywords, misses the temporal
   structure), Kühn & Rosenow (2306.05300; the anti-correlation phenomenon,
   analysis only), Muon-IGT (2605.05577; same benefit claim via transport —
   must be a baseline eventually), AdEMAMix/ADANA (richer fixed kernels).
   Time-sensitive: two active groups are one step away.
2. **Cross-layer subspace whitening — open but empirically hollow here**:
   our offline kill-test found cross-matrix correlations of tracked-direction
   projections at the sampling noise floor (|corr| 0.052 vs floor 0.053;
   top joint mode ~4% of trace). Parked; LM substrate (residual stream)
   left as the only plausible revival.
3. **Optimizer-state-informed teleportation — open but poor headroom**:
   4 years / ~6 groups, no practical-scale end-task gain, and a proven
   no-benefit theorem in the strongly convex regime. Parked.
4. **Schedule solver from measured constants — partially published**
   (Multi-Power Law fits+optimizes schedules and slightly beats tuned WSD);
   margins near tuned optima are small. Parked.

## 1. Offline kill-tests (existing sidecars; zero GPU)

Measured per-direction ACF (post-refresh burn-in 5, MAD-window pipeline,
25,920 directions): ≈ (1, −0.29, ~0, …) top, (1, −0.14, ~0, …) bulk — a
clean negative-lag-1 process.

Variance of the lag-constrained linear estimate at matched mean lag
(medians over 3,456 direction cells):

- optimal filter vs truncated plain EMA: **1.39–1.84×** lower variance;
- optimal filter vs the record's Nesterov-EMA kernel: **2.3–4.7×**;
- **decomposition: the kernel family is the entire effect.**
  nesterov→white-optimal: 2.27–3.43×; white→ρ-matched: **1.01–1.06×**.
  The online measurement is nearly cosmetic on this substrate; what the
  EMA family is missing is the *shape* (spread, lag-constrained,
  min-variance) rather than knowledge of the anti-correlation.

Design consequence, fixed before any training run: the **primary treatment
is the fixed white-noise-optimal kernel** (`force_rho: 0`), and the
measured-ρ variant runs as a secondary arm expected to be
indistinguishable from it. This inverts the original "Wiener" framing and
the pre-registration says so now, not after the results.

Honest mechanism caveats: (i) variance-at-matched-mean-lag is a
stationary-mean proxy; Nesterov's current-gradient spike implements
extrapolation whose curvature benefit the proxy cannot see — the training
experiment, not the proxy, is the arbiter. (ii) The record's β was tuned,
but only within the EMA family; the family itself was never varied on this
recipe. (iii) A null is informative: it would say momentum-estimator
variance is not a binding constraint on this substrate.

## 2. Phase B pre-registration (dev seeds 1442–1451, n = 10 paired)

Recipe: airbench_smoke record config; endpoint tta_val_acc. Arms:

| arm | sweep |
|---|---|
| stock Muon | momentum β ∈ {0.45, 0.6, 0.75} (family-best control, equal tuning effort) |
| FIRMuon white (`force_rho: 0`) | τ ∈ {0.8, 1.5, 2.5} |
| FIRMuon measured-ρ | τ = 1.5 (decomposition check only) |

each × lr ∈ {0.24 (1×), 0.48 (2×)}. 140 runs, local, retry drivers.

Predictions (committed before any run):

- **P1 (primary):** best-vs-best at 1×: FIR-white ≥ stock. Given the
  2.5–3.4× variance mechanism, we predict a positive Δ but pre-commit no
  minimum; the decision-relevant outcomes are (a) Δ > +0.10pp (mechanism
  bites — proceed toward eval/n=100 and LM), (b) |Δ| ≤ 0.10pp (momentum
  noise not binding at 1× — check P2 before closing), (c) Δ < −0.10pp
  (the extrapolation caveat dominates — negative result, report).
- **P2:** at 2× LR, FIR-white's degradation is smaller than stock's
  (variance headroom should matter more when hot). Directional.
- **P3 (decomposition):** FIR-measured ≈ FIR-white within ±0.10pp at
  matched τ (the offline 1–6% should be invisible at this n).
- **P4 (telemetry):** measured ρ̂₁ per matrix lands in [−0.45, −0.15]
  mid-training (consistency with the offline ACF), and the τ-best for FIR
  differs from the lag implied by stock's β-best (the families should not
  collapse onto each other).
- **P5 (failure signature):** if FIR loses specifically early (per-epoch
  val curves) while matching late, the warm-up/kernel-switch transient is
  implicated, not the kernel — report, don't re-tune within Phase B.

Any re-tuning after seeing Phase-B accuracy is Phase B′ on fresh seeds,
labeled. External baselines (Muon-IGT, AdEMAMix-style kernels) are
required before any comparative publication claim; not part of Phase B.
