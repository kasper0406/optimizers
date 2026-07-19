# Gate 2 Decision Record — PROVISIONAL (secondary criterion pending)

Date: 2026-07-19 · Decided by: agent under explicit full delegation, amended
per mandatory adversarial review (verdict: AMEND; all seven amendments
accepted). Evidence: `reports/wp22-comparison-table.md` @ `b323ed6`.

## Primary criterion (plan §2.3): **FAIL — final**

- Registered: time/steps-to-94% improvement over stock Muon ≥ 2σ at fair
  tuning, and not matched by ablations. **Metric substitution disclosed:**
  the harness runs a fixed 200-step budget and non-TTA val never crosses 94%,
  so the evaluated surrogate is accuracy-at-fixed-budget (tta_val_acc);
  on the time axis routed is strictly worse (+10.0% wall at equal accuracy),
  so the substitution is conservative in routed's favor.
- Result (n=100 eval seeds, seed-paired): routed−muon = +0.011pp, paired
  t=0.69, p=0.49; 95% CI [−0.020, +0.042]pp — **effects above 0.042pp are
  excluded at 97.5% confidence vs the 0.272pp (2σ) bar**. 0/45 within-tier
  pairwise contrasts reach nominal p<0.05 (min p=0.109, below the expected
  family-max under the global null). Fair-tuned contrast is sign-negative
  (−0.020pp). Ablations: 4a (retuned WD) and 4c/4d (rho-ignored exploratory,
  random-gating placebo) ran and are all within the tier; 4b (LR bump +
  grad clip) never ran (missing harness hook, deviation 6) — moot under the
  null but recorded. Pipeline sensitivity is proven by the baseline arms
  (dynmuon −0.717pp, paired t=−35).
- The registered "≥ the DynMuon gap on the same harness" clause is
  **degenerate**: WP0.3 was never executed (no reference number), and the
  measured on-harness DynMuon gap is negative (−0.72pp), making the clause
  trivially satisfied. The FAIL rests solely on the 2σ clause. WP0.3
  non-execution is recorded as a deviation.

## Secondary criterion (stability margin ≥ 1.3): **NOT YET EVALUATED — completion runs pre-registered here**

The draft claim "no divergence regime exists" was an over-claim from 3 LR
points ≤ 2× at dev n=10. The adversarial review surfaced an omitted signal:
at 2× record LR, active-routed retains **+0.144pp paired over stock (t=3.52,
p=0.006, dev n=10)** — the §2.2.5 prediction, nominally significant at the
only point tested deep in degradation. Pre-registered completion (BEFORE
looking at any new data; ~$1.50):

1. Extended LR ladder, dev n=10 per point, both optimizers: multipliers
   {3, 4, 6}× record (lr 0.72, 0.96, 1.44), extending until stock breaks.
2. Eval-n=100 confirmation at 2× (lr 0.48), muon vs routed(active),
   seed-paired.
3. **Operationalization (pre-declared):** "stable at LR m" ⇔ zero divergence
   (all runs produce finite results) AND mean tta_val_acc ≥ 93.0% (one full
   point below tier). Secondary SUCCESS ⇔ max-stable-LR(routed)/
   max-stable-LR(muon) ≥ 1.3, OR the 2×-LR eval confirmation shows routed
   ahead by ≥ 2σ of the paired difference. Secondary success reopens Phase-3
   gating per the plan's "primary or secondary" wording; failure closes
   Gate 2 as FAIL overall.

## Consequences (contingent on secondary outcome)

- Primary-track consequences (final): no Phase-3 spend justified by the
  primary; measurement-paper deliverable proceeds and **must include** the
  paired/equivalence statistics and placebo-controlled null (the strongest
  part of the record), plus scoping: the null covers airbench-8-epoch record
  config, 200-step horizon, and the tracked-subspace intervention class.
- Baselines claim rescoped: "at documented light tuning (LR-only 5-seed dev
  probes, grids published) on the muon-co-adapted record config, none of
  DynMuon/AdaMuon/NorMuon matched stock Muon (gaps 0.29–0.82pp)" — a
  tuning-effort-qualified observation, never an optimizer ranking; says
  nothing about home-scale/home-metric claims of those methods.
- Temporal-trust-ratio pivot (litreview A+B): **conditionally approved** —
  protocol must be committed to `criteria/` before any spend, containing the
  falsifiable LR-recovery/setpoint predictions, the mechanism rationale
  (per-direction actuator inert at fixed 4× attenuation, global-LR actuator
  potent, occupancy LR-monotone), and litreview B's mandatory baselines
  (OrScale, NAMO/LANTON-style noise scaling, GALA, Prodigy, hand-tuned
  schedule). Airbench outcomes framed as robustness/parameter-freeness.
- Housekeeping: aggregations over dev stress/tuneA routed grids must filter
  by git SHA (pre-fix routing-inactive duplicates retained by design).

## Secondary criterion — RESOLVED (2026-07-20): **FAIL. Gate 2 = FAIL, FINAL.**

Completion runs executed exactly as pre-registered above (260 runs; one
GPU-availability interruption mid-batch, remedied by targeted rerun of the
212 missing (variant, seed) pairs; zero duplicates; results in `results/`).

1. Extended ladder {3,4,6}x, dev n=10/point: zero divergence for either
   optimizer at any LR (graceful degradation persists to 6x — itself a
   finding for the stability-law writeup), but both cross the pre-declared
   93.0% floor between 2x and 3x (muon 92.68% / routed 92.55% at 3x).
   Max-stable-LR = 0.48 (2x) for BOTH. Ratio = 1.0 < 1.3.
2. 2x-LR eval confirmation, n=100 seed-paired: routed - muon = -0.024pp
   (sd 0.176pp, t = -1.34). The dev-n=10 +0.144pp signal did not replicate;
   sign inverted. Recorded as a multiple-comparisons false positive caught
   by pre-registered confirmation.

Consequences now final: Phase 3 not reopened; the plan's Phase-2 track ends
in a fully-adjudicated FAIL (primary: paired equivalence, effect <= 0.042pp
at 97.5% confidence; secondary: no stability margin, ratio 1.0). Deliverables
proceed per the amended consequences: measurement paper (incl. the graceful-
degradation-to-6x observation) and the pre-registered temporal-trust-ratio
protocol before any new spend.
