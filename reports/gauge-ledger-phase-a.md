# Program #20 — Gauge Ledger, Phase A results

2026-07-24. Pre-registration: `reports/gauge-ledger-prereg.md` (committed
`14de93a`, before any decomposition). Registered quantities:
`reports/gauge-ledger-phase-a.json` via `scripts/analyze_gauge_ledger.py`.
Data: stored Wave-1 fp32 artifacts (seeds 1511–1513) + 3 gauge-probed replay
tails (~1.7 GPU-h) + 7 forward-pass evaluations.

## Verdict: registered criteria evaluate to FAIL — gauge-artifact route closes

*(Process note, added 2026-07-24 after internal review: the criteria below
were fixed numerically in the prereg and the script implements them
faithfully, so the FAIL is mechanically determined rather than judged — but
the gate call itself is the human's, per CLAUDE.md ground rule 1, and no gate
record was written for program #20 the way `reports/wave1-phase-b-gate.md`
was for #18. The consequences in §"Consequences" are therefore proposals
pending that record, not decisions already taken.)*

**Registered deviation (disclosed late, 2026-07-24):** the prereg §2 roster
says "144 blocks of shape (128, 768)" (12 layers × 6 heads × 2). Block 7 has
no attention (`src/nanogpt/model.py:242`), so the actual roster is **132
blocks over 11 layers**. `roster_blocks` enumerates the true set and every
number in this report is over 132; only the prereg's arithmetic was wrong.
No criterion depends on the count.

| criterion | registered bar | measured | verdict |
|---|---|---|---|
| (a) radial dominance of D on roster | ≥ 0.5, all seeds | **0.311 / 0.310 / 0.311** | FAIL |
| (b) tangential de-anti-alignment | \|cos\| < 0.2 on ≥2/3 seeds; CI excl < −0.4 | **cos(v_tan, D_tan) = −0.588 / −0.586 / −0.587**, CI ≈ [−0.59, −0.58] | FAIL |
| (c) perpendicular-update growth law | ≥80% blocks perp ≤ 0.2 & 5% RMS | **0/132 and 0/132 blocks** | FAIL |
| (d) function-nullness of roster scale | \|Δval\| < 0.0025 at ±10% | **max Δ = 1.0e-5** | PASS |

**H_gauge is refuted.** The cos(v, D) ≈ −0.6…−0.7 anti-alignment between the
anneal displacement and the constant-LR tail drift is **genuine functional
opposition, not a weight-norm (gauge) artifact**:

1. Projecting out the radial direction barely moves the cosine
   (−0.64 full → −0.59 tangential) on the exactly scale-invariant Q/K
   head-blocks.
2. The non-invariant control group (V/c_proj/MLP) shows the *same* radial
   fraction (0.35) and the same tangential anti-alignment (−0.60) — the
   effect has no invariance structure at all, which H_gauge required.
3. Criterion (d) confirms the invariance itself is real (±10% roster rescale
   moves full-val by ~1e-5, and transporting the constant-LR Polyak endpoint
   to the annealed per-block norms changes nothing: 3.31021 vs 3.31022). The
   norms genuinely don't matter — and the anneal still doesn't use that
   freedom: its motion is functional, and it opposes the drift.

## Secondary findings (descriptive, registered as exploratory)

- **The hidden LR schedule exists but is weak.** In the constant-LR tail,
  per-matrix weight norms grow ×1.42 (range 1.37–1.44 across all 46 Muon
  matrices, remarkably uniform) and the effective angular rate
  η_eff = η‖ΔW‖/‖W‖ decays only to **0.713×** its tail-start value — versus
  the explicit anneal's ~18× decay (WSD replay: η_eff ratio 0.054).
  *(Window disclosed after internal review: ratios are mean(last 20 tail
  steps)/mean(first 20). An earlier version quoted 0.715 from an unstated
  window; all secondary numbers here are now emitted by
  `scripts/analyze_gauge_ledger.py` into the results JSON rather than
  computed ad hoc.)* The
  WD=0 gauge dynamics supply a mild implicit anneal, far too small to explain
  Wave-1's constant-LR/anneal equivalence-under-averaging.
- **Muon updates are strongly non-perpendicular to the weights** even on
  exactly scale-invariant blocks: median 2|⟨W,V⟩|/‖V‖² = **1.02**
  (constant-LR, per roster block; block range [0.785, 1.356]) and **0.57**
  (WSD), vs ≤ 0.2 for the perpendicular model. *(Correction 2026-07-24: an
  earlier version quoted 1.30 / 0.77 here — those are the medians over the
  46 whole Muon matrices, a population that is mostly NOT scale-invariant,
  so they were attached to the wrong unit. Both sets are ≫ 0.2 and criterion
  (c) fails 132/132 blocks either way.)* Note the registered ratio uses V
  (the Newton-Schulz output) rather than ΔW = η_eff·V; that makes the
  perpendicularity bar ~1/η more lenient, so the FAIL is robust. The
  gradient of a scale-invariant block is exactly ⟂ W, but momentum plus
  Newton-Schulz on the *merged* QKV matrix produces block-level updates with
  large radial components. The textbook "norm growth = Ση²‖ΔW‖²"
  scale-invariance story quantitatively fails for Muon (0/132 blocks) — a
  caution for any analysis importing SGD/BN-era gauge results into
  spectral-LMO training.

## Consequences

- The Wave-1 "anneal as retraction" interpretation survives its best
  deflationary explanation and is strengthened: the anneal actively undoes
  function-relevant drift accumulated at constant LR.
- Per the ideation-report ledger discipline: **Explicit-Gauge Muon stays
  benched permanently** (its premise needed a strong hidden schedule);
  Gauge Ledger **Phase B does not launch** (gated on PASS).
- The frontier-theory programs (BBP noise-side vs Central Flow
  curvature-side) inherit a useful constraint: whatever sets the anneal's
  behavior is not gauge kinematics.
- Program #20 closed at Phase A: ~1.7 GPU-h + one day, three informative
  sub-results (route closed with mechanism, hidden-schedule magnitude
  measured, perpendicularity law refuted for Muon).
