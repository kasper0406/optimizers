# Program #7 — nanogpt frontier transfer test: descriptive report

Pre-registration: `reports/frontier-nanogpt-preregistration.md` (committed
ca9eb93 before any run). Tables/data/figure:
`reports/frontier-nanogpt-tables.md`, `.json`,
`reports/figures/frontier_nanogpt_transfer.png`, produced by
`scripts/analyze_frontier_nanogpt.py`. Descriptive; the pre-registered
region calls are stated as facts about where the estimates landed.

## Headline: the airbench frontier law does NOT transfer

**α = −0.29, seed-bootstrap CI95 [−0.35, +0.30]** (lr\*_cross vs
tokens/step over the 8× batch range 98,304 → 786,432):

| tokens/step | valley (lr) | lr\*_cross | low-lr penalty |
|---|---|---|---|
| 98,304 (c2) | 3.4304 @ 0.050 | 0.128 | 0.021 |
| 196,608 (c4) | 3.4131 @ 0.035 | 0.069 | 0.0007 |
| 393,216 (c8 = record) | 3.4404 @ 0.035 | 0.073 | 0.0012 |
| 786,432 (c16) | 3.5103 @ 0.035 | 0.065 | 0.0024 |

Against the pre-registered regions:

- **The noise-governed region (α ≥ 0.25) is not supported** — the point
  estimate is negative and the CI's upper edge is 0.299.
- **The airbench transfer point (0.35) is excluded by the CI.** The
  airbench exponent does not carry to this workload.
- **The batch-independent region (|α| < 0.1) is not excluded** — with the
  c2 arm's noise the CI is wide — and it is the visually obvious reading
  of the c4–c16 rows: valley pinned at the 0.71× rung and crossings within
  ±7% of each other across a 4× batch range. The c2 arm alone elevates the
  crossing (its curve is much flatter near its valley and its seed noise is
  ~10× the other arms'; see caveats).
- **P2d refuted**: the valley does not shift right with batch (0.050 at c2,
  then 0.035 flat — if anything it moves left then pins).
- **P3d holds**: no divergence cliff anywhere (worst mean loss 3.574 at
  c16 × 2.83×); degradation is graceful at every batch size.

## What this means

The two pre-registered frontier measurements now bracket the law's domain:

- **airbench94 (CNN, 500–4,000 images/batch, fixed epochs): batch-coupled,
  lr\* ∝ B^0.35 [0.30, 0.42]** — solidly noise-governed (#6b).
- **modded-nanogpt record recipe (LM, 98K–786K tokens/step, fixed token
  budget): batch-invariant within measurement resolution** — the useful
  Muon-lr band does not move across an 8× token-batch range.

One coherent post-hoc reading (interpretation, not pre-registered): the LM
grid sits entirely at token batches orders of magnitude above airbench's
sample batches — plausibly past the critical-batch-size crossover where
gradient noise stops setting the frontier — while airbench's range
straddles it. Under that reading both results are consistent with a single
noise-governed mechanism whose batch-coupling saturates; testing it would
require pushing the LM to much *smaller* token batches (the c2 arm's rising
noise and its 30× low-lr penalty already point in that direction). That is
a follow-up design, not a claim.

Practical corollary for this testbed: **the record's Muon lr needs no
rescaling when the token batch is changed by grad-accumulation count in
the 98K–786K range** — lr ≈ 0.7–1× record stays within 0.010 of each
arm's optimum. (At fixed half budget the optimum sits at ~0.7× the record
lr at every batch, consistent with the record lr being tuned for the full
1750-step budget.)

## Caveats (pre-declared + observed)

- n=2 seeds/cell. The c2 arm's per-cell seed spread reaches 0.036 —
  ~15–20× the c8-arm spread — so its crossing (the α fit's leftmost point)
  is the least reliable; 305/2000 bootstrap draws produced an undefined
  crossing, nearly all on c2 resamples. The α CI honestly reflects this.
- Half token budget (346M): all valleys sit left of the record lr, as
  expected for a shorter run; within-program comparisons are controlled,
  absolute levels are not comparable to full-budget runs.
- The three checkpoint-collision results and one duplicated cell run are
  tombstoned/deduped (`results/INVALID_RUNS.json`; analyzer keeps the
  latest per cell). 48/48 valid cells entered this analysis.

## Run ledger

48 cells + 2 catch-up + 1 discarded duplicate, seeds 1720–1721, SHAs
ca9eb93/42e3b8e/f8fed15 (pre-registration → checkpoint-collision fix →
done-check fix; the incident trail is in the git log and
`results/INVALID_RUNS.json`), `git_dirty: false` throughout, babysitter
zero GPU-failure retries. `cost_usd` null (owned hardware).
