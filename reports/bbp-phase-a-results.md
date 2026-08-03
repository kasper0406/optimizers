# Program #22 — BBP Alignment Frontier, Phase A-empirical results

2026-07-24. Pre-registration: `reports/bbp-prereg.md` (commit `303b70a`,
before any probe), with amendments **A1** (held-out arm) and **A2** (repaired
criterion), both registered before any curve was read. Registered quantities:
`reports/bbp-phase-a.json` via `scripts/analyze_bbp.py`. 8 frozen-checkpoint
probes, 512 streamed record-chunks each, no training. Gate call is human.

## Registered outcome

**(V) vacuity guard: PASS.** All 46 Muon matrices on every probe show
â-dynamic-range ≥ 0.2 across the grid (the curves span roughly 0.07 → 0.72).
The measurement is informative rather than flat.

**(S′) saturation at the training batch: FAIL** (held-out arm, mid
checkpoint, both seeds).

| arm | seed | â(128) | â(256) | ratio (bar ≥ 0.90) |
|---|---|---|---|---|
| **held-out (shard 8)** | 1511 | 0.580 | 0.702 | **0.829** |
| **held-out (shard 8)** | 1512 | 0.643 | 0.762 | **0.844** |
| seen (shard 3) | 1511 | 0.713 | 0.818 | 0.871 |
| seen (shard 3) | 1512 | — | — | 0.902 |

Median held-out ratio **0.836**; per-matrix range 0.770–0.890 — the FAIL is
uniform across matrices, not driven by outliers.

**Interpretation, as registered:** under the BBP/noise-side account,
nanogpt's measured batch-invariant useful-LR frontier requires the
momentum-corrected alignment to be saturated at the batch the optimizer
effectively sees. It is not — the alignment curve is **still climbing** into
the training point. Per prereg §3, this **kills the noise-side account of
the nanogpt frontier**, which was the pre-registered informative-negative
branch. The curvature-side competitor (program #21, Muon Central Flow) is
not adjudicated by this result; it is now the surviving candidate of the two.

Calibration for reading 0.836: pure sampling noise with no saturation
(a ∝ √b) predicts a doubling ratio of 1/√2 = **0.707**; full saturation
predicts **1.0**. The measurement sits between — partial saturation, closer
to the noise-limited end.

## Descriptive findings (no criteria attached)

1. **Alignment at the record batch is low: â ≈ 0.21–0.25 (held-out).** The
   orthogonalized update the optimizer actually applies at the record's
   393,216-token batch is only ~20–25% aligned with its own large-batch
   limit. A representative per-matrix curve (b in chunks):

   | b | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
   |---|---|---|---|---|---|---|---|---|---|
   | â | 0.070 | 0.094 | 0.127 | 0.175 | 0.242 | 0.331 | 0.450 | 0.585 | 0.718 |

   Near-perfect √b scaling at small b (each doubling ×1.36 vs √2 = 1.41),
   bending only in the top octave.
2. **Alignment decreases over training** (seen arm, shard 3): early
   â(record) 0.290/0.277 → mid 0.283/0.327 → **annealed 0.176**. The
   annealed model's gradients are the noisiest relative to their own
   large-batch limit.
3. **Amendment A1's registered prediction was confirmed.** I predicted before
   unblinding that memorized data would look *more* saturated, biasing the
   as-registered protocol toward PASS. Measured: seen arm 0.886 vs held-out
   0.836, **+0.050** in exactly the predicted direction. The original shard-3
   protocol would have moved the result 5 points toward the bar (though not
   over it — the seen arm fails too, 0.886 < 0.90).

## Process record

Two defects in this program's own design were caught before unblinding and
are disclosed in the prereg rather than silently fixed:

- **A1**: the probe stream (shard 3) had been *trained on* at the mid and
  annealed checkpoints but not at the early one, so "matched data" did not
  mean matched seen/unseen status. Held-out arm added on shard 8; criterion
  moved to it.
- **A2**: criterion (S) as originally registered evaluated the
  momentum-corrected points b_eff = 312 and 2496, both **off** the b ≤ 256
  grid. Both clamped to the same value, making the ratio identically 1.0 —
  an unconditional PASS that could never have produced the FAIL branch. Found
  by writing synthetic-curve unit tests for the analysis script. Repaired to
  on-grid flatness (S′); the analysis now refuses off-grid criterion reads.

Cost: ~2.5 GPU-h (2 truncated hot runs + 8 probes). Suite: 750 tests green.

## What this stage does not claim

No RMT comparison, no zero-parameter free-probability curve, no airbench
exponent prediction, no lr\*(B) claim. Those are Phase A-theory and require
their own pre-registration. This stage measured a(B) and adjudicated (S′)
only.

## Proposed next steps (human gate)

1. **Record the FAIL and close the noise-side branch**, or commission the
   Phase A-theory RMT computation anyway to test whether the *measured*
   curve matches the zero-parameter prediction (the curve is now in hand and
   the theory is cheap to evaluate against it — arguably the highest-value
   use of this data even with (S′) failed).
2. **Program #21 (Central Flow) becomes the sole surviving frontier
   theory.** Its Stage-1 derivation now has two targets: the η-invariant /
   conditioning-dependent loss-floor ratio, and — new from this program — an
   alignment curve that is still √b-limited at the training batch.
3. The alignment-decreases-over-training observation (finding 2) is
   unexplained and cheap to extend to more checkpoints if it is wanted for
   the measurement paper.
