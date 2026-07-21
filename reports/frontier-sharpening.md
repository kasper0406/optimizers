# Program #6b — Frontier sharpening: descriptive report

Pre-registration: `reports/frontier-sharpening-preregistration.md`
(committed ba5bad3 before any run). Tables:
`reports/frontier-sharpening-tables.md`; data
`reports/frontier-sharpening.json`; figures
`reports/figures/sharpening_*.png`. Produced by
`scripts/analyze_frontier_dense.py`. Descriptive only.

## Run ledger

121/121 runs (105 dense-ladder + 16 step-matched), local 2× RTX 5090,
seeds 1410–1414, `git_dirty: false` at ba5bad3 throughout. Two transient
failures at (B=1000, lr 0.32, seeds 1412/1414) succeeded unchanged on
immediate rerun (their tracebacks were lost to launcher log truncation;
both reruns are the only copies in `results/`). `cost_usd` null pending the
owned-hardware convention.

## Part 1 — α with the quantization removed

Interpolated floor crossings (pre-registered estimator, n=5 per cell, all
curves monotone within their bands, every crossing interior):

| B | lr*_cross |
|---|---|
| 1000 | 0.475 |
| 2000 | 0.612 |
| 4000 | 0.772 |

**α = 0.350, seed-bootstrap CI95 [0.297, 0.425]** (2000/2000 draws valid).

Against the pre-registered regions:

- **Noise-governed (α ≥ 0.25): the entire CI lies inside it.** The
  deterministic region (|α| < 0.1) is decisively excluded — the frontier is
  batch-coupled.
- **The √B point prediction (0.5) is also excluded by the CI.** Program
  #6's coarse α = 0.50 was rung-quantization luck, exactly the possibility
  #6b was pre-registered to adjudicate. The measured scaling is
  **lr\* ∝ B^0.35**, distinctly sub-√B. (Proximity to B^1/3 is noted as an
  observation, not a claim; no 1/3-exponent hypothesis was pre-registered.)

Caveat: floor(B) inherits the ref(B) estimate (the anchor-rung mean; e.g.
the B=2000 anchor came in at 93.89 here vs 94.14 on program #6's n=2 —
within seed noise, but it shifts all of that B's crossings together). The
bootstrap propagates per-cell seed noise including the anchor's; a
systematic ref bias would move the intercept far more than the slope.

## Part 2 — step-matched B=8000: the fallback was undertraining

At epochs 32 (~200 steps, matching B=2000's step count; 4× the sample
budget, so comparable only to its own ladder):

| lr | 0.12 | 0.24 | 0.36 | 0.48 | 0.60 | 0.72 | 0.96 | 1.44 |
|---|---|---|---|---|---|---|---|---|
| acc | 90.64 | 92.15 | 93.28 | 93.69 | **93.75** | 93.69 | 93.32 | 92.31 |

- Accuracy fully recovers (peak 93.75 vs 72.7 undertrained).
- **Peak-referenced shoulder = 0.96 ≥ 0.72**: the pre-registered read is
  that undertraining explained the program-#6 fallback — the rightward
  shift *continues* through B=8000 (0.72 at B=4000 → 0.96 here), it does
  not saturate.
- The interpolated lr\*_cross is undefined by its own rule: the lowest rung
  (0.5×) is 3.1pp below the floor. That is the "low lr is the losing side"
  signature again, now surviving step-matching — at B=8000 even with 200
  steps, halving the record lr costs 3pp while tripling it costs 0.4pp.
  The asymmetry of the curve around its peak keeps sharpening with batch.

## Combined picture after #6 + #6b

Muon's useful-lr frontier on airbench94 is batch-coupled with exponent
**α ≈ 0.35 (CI [0.30, 0.42])** at fixed sample budget — solidly in the
noise-governed regime, decisively not batch-independent, and measurably
shallower than the √B reference; the shift continues through B=8000 once
undertraining is removed; no instrumented curvature or occupancy scalar is
conserved along the frontier (program #6 P2); and there is no divergence
cliff anywhere in the measured region. Together with the standing η·λ ≈ 65
result, this is a self-consistent, pre-registered characterization of the
momentum+minibatch stability regime the ICML'26 line leaves open — with
the frontier-setting invariant still unidentified as the sharpest open
question.
