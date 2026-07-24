# Pre-registration — Program #22: BBP Alignment Frontier, Phase A-empirical

Registered 2026-07-24, before any gradient-probe run. Source:
`reports/ideation-geometry-theory-2026-07-24.md` §1 program #2 (human GO for
the portfolio, "Please proceed" after #20 closed). This document registers the
**empirical stage only**: measuring the msign alignment curve a(B) at frozen
checkpoints. The zero-free-parameter RMT prediction (free-probability curve
from measured spectra + noise covariance) is Phase A-theory, registered
separately before its computation touches these curves' held-out use.

## 1. Objects

At a frozen parameter point W (no optimizer steps), per Muon matrix m:
G_B = mean gradient over B disjoint 49,152-token record chunks (the record's
own BOS-aligned chunking); msign(·) = the record's 5-step Newton-Schulz
orthogonalization in bf16 (what Muon actually computes, not exact msign).

**Estimator (split-half doubling tree):** stream M = 512 chunk gradients;
merge pairwise (binary-counter streaming). At each merge of two independent
half-sums A, B each aggregating b chunks, record
c_m(b) = ⟨msign(A_m), msign(B_m)⟩_F / (‖msign(A_m)‖_F · ‖msign(B_m)‖_F).
The alignment estimate is â_m(b) = sqrt(max(mean_merges c_m(b), 0)) — the
standard split-half debiasing under independent fluctuation halves.
b grid: 1, 2, 4, …, 256 chunks (49K – 12.6M tokens; the record batch is 8
chunks, so the grid spans 1/8× to 32× the training batch, ≥ 16× as the
ideation spec requires). Estimates per level: 512/2b merges; propagate
split-half scatter as the error band.

**Momentum correction:** e_m(b) = â_m(b_eff) with b_eff = b·(1+β)/(1−β),
β = 0.95 (EMA variance-reduction factor), read off the measured curve by
log-b interpolation; b_eff = 39b. Registered before measurement.

## 2. Runs

Checkpoints (3 per seed, seeds **1511 and 1512**):
- **early**: step 321 (new truncated hot runs, max_steps 321, checkpoint kept);
- **mid**: step 963 (the stored Wave-1 prefix checkpoints);
- **annealed**: step 1750 (the stored arm-A final-weight artifacts).

Data stream: train shards, fixed registered offset (file index 3, position 0),
identical for every checkpoint/seed — gradient statistics at matched data.
Attention window set to the checkpoint's record-equivalent step. 512 chunks
per probe; 6 probes total; ~1 GPU-h expected all-in.

## 3. Registered quantities and criteria

Primary per probe: the per-matrix curves â_m(b) with bands; the aggregate
median curve â(b); the **saturation statistic** s = â(8)/â(64) aggregated as
the median over matrices (record batch vs 8× record batch).

- **(V) Vacuity guard:** the measurement is informative iff ≥ 50% of Muon
  matrices show dynamic range max_b â_m − min_b â_m ≥ 0.2 across the grid on
  every probe. If violated at the low end, extend the grid downward (b < 1
  chunk is impossible → declare the low-b regime unmeasurable at this token
  granularity and report as such).
- **(S) Saturation consistency (the falsifiable claim of this stage):** the
  lab measured Muon's useful-LR band batch-INVARIANT on nanogpt across 8×.
  Under the BBP account this requires the momentum-corrected alignment to be
  saturated at the training batch: **PASS iff median_m e_m(8)/e_m(64) ≥ 0.9
  at the mid checkpoint** (the training-relevant one) on both seeds.
  **FAIL** (e(b) still rising steeply at the record batch while the measured
  frontier is invariant) kills the noise-side account of the nanogpt
  frontier — registered as an informative negative for program #22, exactly
  the "non-saturated a(B) with measured invariance" branch of the ideation
  spec.
- Descriptive (no criteria): early-vs-mid-vs-annealed curve evolution;
  per-matrix-class (attn vs MLP) breakdown; raw c distributions.

## 3b. AMENDMENT A1 — held-out-data arm (2026-07-24, before unblinding)

Registered **before any probe result was opened** (5 of 6 as-registered
probes had completed; none had been read). Raised by an internal adversarial
review of the harness.

**Defect.** §2 fixes the probe stream at train shard index 3, position 0, on
the stated intent "identical for every checkpoint/seed — gradient statistics
at matched data". The *data* is matched, but the **seen/unseen status is
not**. Reading the loader state out of the stored checkpoints:

- mid (step 963, where the primary criterion (S) is registered):
  `{file_index: 3, pos: 92,635,277}` — the probe's 512 chunks (25.2M tokens
  from pos 0) were consumed around training steps 762–826, i.e. **already
  trained on**;
- early (step 321): `{file_index: 1, pos: 30.8M}` — shard 3 **unseen**;
- annealed (step 1750): shard 3 fully consumed.

So criterion (S) would be evaluated on memorized data, and the descriptive
early→mid→annealed evolution confounds training progress with memorization.
Gradient noise structure on trained-on data is not the noise structure the
optimizer actually faced, which is exactly what a(B) is meant to measure.

**Amendment.** Add a **held-out arm** at the mid checkpoint, both seeds,
identical in every other respect but reading from **train shard index 8**
(training consumes ~688M tokens ≈ shards 0–6, so shard 8 is unseen at every
checkpoint). Criterion (S) is evaluated on the **held-out arm**; the
as-registered shard-3 arm is retained and reported alongside as the
seen-data comparison. The (S) threshold, the estimator, the b-grid, the
momentum correction and the vacuity guard (V) are all unchanged.

Registered prediction, made before unblinding either arm: if memorization
matters, the seen arm should show *higher* alignment at small b (smaller,
more consistent gradients) and therefore look **more** saturated than the
held-out arm — i.e. the as-registered protocol would have been biased
*toward* PASS on (S). If the two arms agree, the confound is empirically
immaterial and both are reportable.

## 3c. AMENDMENT A2 — criterion (S) was vacuous as registered (2026-07-24, before unblinding)

Registered **before any probe curve was read** (found by writing synthetic
unit tests for the analysis script, per an internal review finding that
Wave-1-era analysis scripts shipped untested; the test asserting "a still-
rising curve must FAIL" failed, which is how the defect surfaced).

**Defect.** §1 fixes the b-grid at 1…256 chunks (512 streamed chunks ⇒ top
merge level b = 256) and §3 evaluates (S) as
`median_m e_m(8)/e_m(64)` with `e_m(b) = â_m(39b)`. But 39·8 = **312** and
39·64 = **2496**, both **above the measured grid**. Interpolation clamps
both to â(256), so the ratio is **identically 1.0 for any input** —
criterion (S) as written passes unconditionally and measures nothing. It
could never have produced the registered FAIL branch.

**Why not just extend the grid.** Reaching b = 2496 needs ≈ 5000 streamed
chunks per probe (≈ 245M tokens, ~10× the current cost per probe, ×8 probes)
— out of proportion to a Phase-A stage.

**Repaired criterion (S′).** The momentum-corrected training point
b_eff(8) = 312 sits just above the grid top (256), i.e. within 1.22× — so
alignment *at* the training point is measurable to good approximation, while
the "8× beyond training batch" comparison is **not measurable at this probe
cost** and is withdrawn. (S′) instead tests whether the curve has flattened
approaching the training point:

> **(S′) PASS iff median over matrices of â(128)/â(256) ≥ 0.9**, on the
> held-out arm (A1), at the mid checkpoint, on both seeds.

Interpretation is unchanged in kind: under the BBP account, nanogpt's
measured batch-invariant LR frontier requires alignment to be saturated at
the batch the optimizer effectively sees. FAIL (curve still rising steeply
into the training point) kills the noise-side account of that frontier, as
originally registered.

**Guard.** `scripts/analyze_bbp.py` now refuses to evaluate any criterion at
an off-grid b and marks clamped readings, so this class of error cannot recur
silently. The raw â(8)/â(64) ratio and the full curves are reported
descriptively alongside.

## 4. What this stage does NOT claim

No RMT comparison, no airbench exponent prediction, no lr*(b) claim — those
require the Phase A-theory computation and airbench probes, each with its own
pre-registration. This stage only measures a(B) and adjudicates (S).

Cost: ~1.5 GPU-h (incl. two 321-step truncated runs) + half an analyst-day.
