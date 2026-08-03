# Program #12 Phase A results — the momentum-referenced selection signal barely exists

2026-07-22. Pre-registration `reports/data-selection-prereg.md` (commit
dc0985e, before any instrumented run). 8 probe runs (lr ∈ {1×, 3×} ×
seeds 1462–1465) + smoke; probe instrument non-perturbing (accuracy and
wall time unchanged).

## Measurements vs the pre-registered viability gates

- **M2 (persistence): FAIL.** ICC of per-example momentum alignment
  (between-example variance share) = **0.07–0.08**: ~92% of the signal
  is transient step-to-step fluctuation, not a stable property of the
  example. The prereg required persistence for any Phase B; it is absent.
- **M3 (loss-redundancy): adverse.** corr(alignment, example loss) =
  −0.38…−0.39 — a large fraction of the little structure that exists is
  loss-selection in disguise (the strip-mined baseline).
- **M4 (spectral concentration): inverted.** Per-example gradients put
  **1.6–1.7%** of their energy in the momentum's top-8 singular
  subspace, ~8× *less* than the random-subspace baseline — the
  "momentum-spectrum-referenced" variant has nothing to grab.
- **M1/M5:** the population is compact (sd ≈ 0.03, no exploitable heavy
  tail; mean slightly negative ≈ −0.02, a post-update artifact — the
  optimizer just stepped along M, deflating current-batch alignments)
  and essentially identical at 1× and 3× LR.

## Why — and why this predicts the literature's shape

The scale of the signal is set by arithmetic: momentum is an average
over roughly N ≈ batch × ESS ≈ 2000 × 4 ≈ 8000 example-gradients, so a
single example's cosine against it is O(1/√N) ≈ 0.01 before structure —
matching the measured sd. Per-example-vs-aggregate alignment is
SNR-doomed at practical batch sizes, which retroactively explains the
field's actual trajectory: the methods that work at scale (OPUS, LESS,
GREATS) all compare *aggregates to aggregates* (per-example gradients
against validation/proxy-pool gradients), and the one
momentum-as-reference precedent (GMC) lives at toy scale where N is
small. The unpublished cell we identified is unpublished because it is
empty of signal, not because nobody looked — a finding worth one
paragraph in the paper's data-measure discussion, not a program.

## Ledger

Program #12: 9 runs, ~10 GPU-minutes, killed at Phase A by its own
gates. The probe instrument (`recipe.example_probe`) is retained —
per-example gradient telemetry may serve future end-state-property
measurements (the program-#11 successor direction). Dev seeds 1462–1465
consumed.
