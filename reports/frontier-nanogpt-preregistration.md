# Program #7 — Frontier transfer test on nanogpt: pre-registration

Status: **pre-registered before any run** (committed prior to first launch;
runs must carry a SHA at or after this commit). Human directive 2026-07-21:
proceed with the highest-value method-agenda item — this is the transfer of
the airbench frontier law (programs #6/#6b) to the validated local nanogpt
testbed (`reports/nanogpt-local-baseline.md`). Dev-seed measurement
program; descriptive; no gate.

## Question

Airbench measured the Muon useful-lr frontier as batch-coupled with
**α = 0.350, CI95 [0.297, 0.425]** (lr\* ∝ B^α at fixed sample budget).
Does the law — and the exponent — transfer to a transformer LM on FineWeb
under the record recipe? Transfer at a similar α makes it a two-workload
empirical scaling rule; failure to transfer bounds the law's domain. Both
outcomes are contributions.

## Design

Batch axis (`chunks_per_step`, new port knob committed with this file):
49,152-token record chunks per optimizer step ∈ **{2, 4, 8, 16}** →
tokens/step ∈ {98,304 / 196,608 / 393,216 (record) / 786,432} — an 8×
range like airbench's 500→4000.

**Fixed token budget** 7,040 chunks = **346,030,080 tokens** (~half the
record run; the budget, not the record length, is the controlled quantity):

| chunks | tokens/step | num_iterations | val_loss_every | momentum_warmup |
|---|---|---|---|---|
| 2 | 98,304 | 3520 | 440 | 603 |
| 4 | 196,608 | 1760 | 220 | 302 |
| 8 | 393,216 | 880 | 110 | 151 |
| 16 | 786,432 | 440 | 55 | 75 |

Schedule shape is step-fraction-normalized by the recipe (cooldown_frac
0.45, min_lr_frac 0.05 unchanged); momentum warmup is the record's 300/1750
fraction of the run, scaled and rounded. Muon lr ladder (√2-spaced, 6
rungs): **0.05 × {0.5, 0.707, 1, 1.414, 2, 2.828}**. The Adam side
(embeddings/scalars/head) stays stock — the same "spectral-lr axis only"
isolation as programs #6/#6b. Harness-standard local deviations
(fp32_embed_grad_accum, head_chunk_rows 8192) on all runs; endpoint = final
val loss at the fixed budget (the validated uncensored endpoint).

n = 2 fresh dev seeds **1720, 1721** per cell → 48 runs, ~15 h on the two
5090s under `scripts/babysit_nanogpt.sh` supervision.

## Pre-registered definitions (loss convention — lower is better)

- valley(B) = min over rungs of mean final val loss; ref rung = argmin.
- floor(B) = valley(B) + **0.010** loss (≈ 4–8× the per-cell seed sd at
  these step counts; the airbench convention was likewise ~5–7× seed sd).
- **lr\*_cross(B)**: scanning rungs upward in lr from the valley rung, at
  the FIRST adjacent pair with mean_i ≤ floor(B) < mean_{i+1}, interpolate
  linearly in (log lr, loss). Undefined if no upward crossing in the band.
- **shoulder(B)** (secondary): largest rung ≥ valley rung with mean ≤
  floor(B), scanning up to the first crossing (the #6b Part-2 convention).
- **α** = OLS slope of log lr\*_cross vs log tokens-per-step over the four
  batch sizes. Seed bootstrap (resample the n=2 per-cell values, 2000
  draws, rng seed 0) reported; with n=2 it is indicative, and the rung
  spacing (√2) bounds quantization of the secondary shoulder at ±½ rung.
- Low-lr side: mean loss of the 0.5× rung minus valley, per B — the
  "low lr becomes the losing side at large B" signature from #6/#6b,
  predicted to grow with B.

## Pre-registered predictions

- **P1d (primary):** noise-governed α ≥ 0.25; deterministic |α| < 0.1;
  between = ambiguous. **Transfer point prediction: α ≈ 0.35** (inside the
  airbench CI [0.30, 0.42]). The report states which region α lands in and
  whether the airbench CI covers it.
- **P2d:** the valley (best-lr) rung shifts right with B — monotone
  non-decreasing across the four batch sizes.
- **P3d:** no divergence cliff: no cell's mean final loss exceeds 4.0
  (far-degraded but not collapsed); graceful degradation on the 2.83×
  rung at every B.

## Analysis

`scripts/analyze_frontier_nanogpt.py`, deterministic; tables/plot; report
`reports/frontier-nanogpt.md` (descriptive). Caveats pre-declared: n=2 per
cell (rung-level reads, not fine effects); the record recipe was tuned at
chunks=8, so absolute cross-B levels are not comparable (per-B valley
reference handles this, as on airbench); half-budget runs shift all
valleys vs the record's full budget — the within-program comparison is
what is controlled.
