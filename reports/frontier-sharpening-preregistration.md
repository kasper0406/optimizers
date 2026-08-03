# Program #6b — Frontier sharpening: pre-registration

Status: **pre-registered before any run** (committed prior to first launch;
runs must carry a SHA at or after this commit). Follows program #6
(`reports/stability-frontier.md`), whose two pre-declared limits this pass
attacks: rung quantization on α, and the undertraining confound on the
B=8000 row. Human go-ahead: 2026-07-21 ("Please go ahead and run it").
Dev-seed measurement program; descriptive output only.

## Part 1 — dense ladder (α sharpening)

**Question:** the same P1 as program #6 — the scaling exponent α in
lr* ∝ B^α — measured with a continuous estimator on a dense ladder.

**Grid** (`configs/dev/frontier_dense_b{1000,2000,4000}.yaml`), n=5 fresh
dev seeds **1410–1414** per cell (disjoint from 1400–1402 and all earlier
blocks); instrumentation identical to program #6 (HVP + smoothness on;
these batch sizes fit in 32 GB):

- B=1000: lr ∈ {0.24 (peak anchor), 0.32, 0.37, 0.42, 0.48, 0.55, 0.64}
- B=2000: lr ∈ {0.24 (peak anchor), 0.42, 0.48, 0.55, 0.64, 0.73, 0.84}
- B=4000: lr ∈ {0.36 (peak anchor), 0.55, 0.64, 0.73, 0.84, 0.97, 1.11}

Dense rungs are ×1.15-spaced and bracket the program-#6 crossing region for
that B with at least one rung of margin on each side; the anchor rung is
that B's observed accuracy peak from #6 and serves only to define the
reference level. 105 runs.

**Pre-registered definitions (Part 1):**

- ref(B) = max over that B's rung list of mean TTA accuracy.
- floor(B) = ref(B) − 1.0pp (program-#6 convention).
- **lr\*_cross(B)** (primary, replaces the rung-quantized shoulder): scan
  rungs in ascending lr; at the FIRST adjacent pair (lr_i, lr_{i+1}) with
  mean acc_i ≥ floor(B) > mean acc_{i+1}, interpolate linearly in
  (log lr, acc):
  log lr\* = log lr_i + (acc_i − floor)/(acc_i − acc_{i+1}) · (log lr_{i+1} − log lr_i).
  If the lowest rung is already below floor, or no crossing occurs within
  the band, lr\*_cross(B) is undefined and reported as such (the bands were
  sized to make this unlikely).
- **α** = OLS slope of log lr\*_cross vs log B over the three batch sizes.
  Uncertainty: seed bootstrap (resample the 5 per-cell accuracies with
  replacement, 2000 draws, rng seed 0, recompute crossings and α), 95% CI.

**Pre-registered prediction P1b:** noise-governed α ≥ 0.25 (point 0.5);
deterministic |α| < 0.1; between = ambiguous. Program #6 landed at 0.50 on
the coarse estimator; this pass either confirms it with a CI that excludes
the deterministic region or exposes it as quantization luck.

## Part 2 — step-matched B=8000 arm (undertraining deconfound)

**Question:** program #6's B=8000 row broke the rightward trend
(peak-referenced shoulder 0.48 vs 0.72 at B=4000) while being deeply
undertrained (50 steps, peak 72.7%). Does matching the *step count* instead
of the sample budget restore the trend?

**Grid** (`configs/dev/frontier_b8000_stepmatched.yaml`): B=8000,
`epochs: 32` (≈ 200 steps, matching B=2000's step count; 4× the sample
budget — this arm is therefore NOT sample-budget-comparable to program #6
and is analyzed only against its own ladder), the same 8-rung ladder as
program #6 (0.12…1.44), n=2 seeds (1410, 1411), `hvp: false` (32 GB OOM,
amendment A1 precedent), smoothness on. 16 runs.

**Pre-registered definitions (Part 2):** peak-referenced shoulder exactly as
the program-#6 post-hoc readout (largest rung within 1.0pp of the ladder
maximum, scanning down from the peak to the first crossing), now declared
in advance. Also reported: interpolated lr\*_cross on the same ladder using
the Part-1 estimator.

**Pre-registered prediction P2b:** if undertraining explains the fallback,
the step-matched B=8000 peak-referenced shoulder lands at ≥ 0.72 (the
B=4000 value), continuing the trend; if it stays ≤ 0.48 despite 4× the
steps, the rightward shift genuinely saturates by B=8000 at this workload.

## Analysis & caveats

`scripts/analyze_frontier_dense.py`, deterministic, from results JSONs;
descriptive report `reports/frontier-sharpening.md`. No gate. Caveats
pre-declared: mean-acc crossings with n=5 have ~0.07pp SE per cell near the
floor (seed sd ~0.15pp), so the bootstrap CI on α is expected ~±0.1–0.15;
the anchor rung means ref(B) is estimated from one cell — if a dense rung
exceeds the anchor, ref uses it automatically (max over the list). The
step-matched arm changes two things at once relative to #6's B=8000 row
(steps AND sample budget); it deconfounds undertraining only, by design.
