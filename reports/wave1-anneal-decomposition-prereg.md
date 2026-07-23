# Wave 1 pre-registration — anneal decomposition (programs #17, #18, #19)

Registered 2026-07-23, before any governed run. Source: the vetted ideation
report (`reports/ideation-post-muon-2026-07-23.md`, top-5 items #3, #1, #4)
following the user's go-ahead on the two-wave portfolio plan. This document
fixes designs, arms, seeds, endpoints, and pass/fail criteria for the three
Wave-1 programs. Analysis scripts must read only quantities named here for the
gated conclusions; anything else is exploratory and will be labeled so.

**The shared question.** The nanogpt record's LR anneal (cooldown_frac 0.45,
steps 963–1750) is where the reproduction deficit concentrates and where
end-state objects acquire their LR dependence. Three mechanistically
independent accounts of what the anneal does are tested:

- **#17 (drift completion):** the anneal completes a *bias* — remaining drift
  down the valley — that a linear extrapolation of the constant-LR tail mean
  can synthesize for free.
- **#18 (schedule-free tail graft):** the anneal's benefit is descent *at the
  readout point*; closed-loop anchored averaging (gradients evaluated at the
  interpolate toward the Polyak average) replaces it at constant LR.
- **#19 (batch-annealed Muon, Stage 1):** the anneal's benefit is *noise
  quenching*; growing the token batch at constant LR replaces it.

Program #16 refuted the pure variance-reduction account (open-loop tail
readout averaging) on airbench. Wave 1 puts the whole decomposition on the
nanogpt harness. Three nulls upgrade "the anneal is irreducible" to a
nanogpt-scale finding with a mechanism decomposition; any positive is a
tokens-to-loss win at zero or negative marginal compute.

## 0. Harness, seeds, endpoints

- Harness: local 2× RTX 5090 port (`reports/nanogpt-local-baseline.md`),
  standard local config = `configs/dev/nanogpt_local_baseline.yaml`
  (device_count 1, fp32_embed_grad_accum, head_chunk_rows 8192). Eval-seed
  baseline (n=10, seeds 1710–1719): final val **3.28888 ± 0.00125**.
- Dev seeds for all Wave-1 development: **1511–1519** (verified fresh:
  max consumed seed field in `results/` is 1510). Eval seeds only in a
  promoted Phase-B, only via `--seed`, per CLAUDE.md ground rule 2.
- Primary endpoint everywhere: **final val loss at step 1750** (matched
  tokens: 1750 × 393,216). For #19's ramp arm, whose step count differs,
  the endpoint is final val at the **matched token budget** (see §3).
- Decay onset: with cooldown_frac 0.45, LR factor w = 1 through training
  step 962 and decays from step 963. **T_c = 963**: the fork state is the
  iterate after training steps 0–962, identical (bit-for-bit modulo
  nondeterministic kernels) across all arms sharing (seed, hot config).

### Shared infrastructure (implemented once, used by all three)

1. **Constant-LR runs**: `min_lr_frac: 1.0` makes `get_lr` ≡ 1.0 at every
   step; through step 962 this is the baseline trajectory exactly.
2. **Prefix forking**: a `nanogpt.fork_from: <checkpoint>` config key resumes
   a run from another config's checkpoint iff (a) the checkpoint's
   **hot fingerprint** (config fingerprint excluding cooldown_frac,
   min_lr_frac, max_steps, checkpoint block, and the tail block) matches, and
   (b) the fork step is ≤ the stable-phase end of both configs. Guards the
   program-#7 collision class while allowing deliberate cross-config forks.
3. **Tail accumulators** (`tail.accumulate: true`): streaming fp32 means of
   the post-update iterate — W1 over steps 1450–1599, W2 over 1600–1749, and
   a Polyak (equal-weight) mean from step 963 — plus the raw final iterate,
   saved as an fp32 artifact next to the results JSON. **Spike gate** on all
   accumulators: a step is excluded when its train loss z-score against a
   running EMA (β=0.9, first 20 tail steps unconditionally included) exceeds
   4; gate decisions logged. The update path is never touched.
4. **Schedule-free tail** (`tail.mode: schedule_free`, from `tail.start_step`
   = 963): LR factor pinned to `tail.kappa`; per step the iterate z is
   stashed, gradients are evaluated at y = (1−ρ)·z + ρ·x̄ (ρ = `tail.rho`),
   z is restored, the stock Muon/DistAdam step updates z, and x̄ ←
   x̄ + (z − x̄)/t (t = tail step index). Validation in the tail is computed
   at **both** x̄ and z; the run's primary endpoint is val(x̄).
5. **Batch-ramp tail** (`tail.mode: batch_ramp`, #19): constant LR factor
   `tail.kappa`; token batch grows with record-equivalent progress
   u = (tail tokens consumed)/(total tail token budget):
   B(u) = B₀ / max(w(u), 1/8) with w(u) = 1 − 0.95·u, rounded to whole
   49,152-token chunks (8→64 chunks/step, i.e. capped at 8×B₀); runs until
   the tail token budget 787 × 393,216 is consumed. No LR compensation.

Verification before any governed run: unit tests (CPU tiny-model) proving
(a) `min_lr_frac: 1.0` + fork reproduces an unforked constant-LR run;
(b) `schedule_free` with ρ=0 equals the constant-LR run exactly;
(c) accumulator means equal brute-force means on a synthetic run and the
spike gate excludes a planted spike step; (d) hot-fingerprint forking refuses
a muon_lr-variant checkpoint; (e) batch_ramp consumes exactly the registered
token budget and its B(u) matches the formula. Plus one GPU smoke
(max_steps ≈ 30, dev seed 1999) checking peak memory < 31 GiB with
accumulators and with the SF tail.

## 1. Program #17 — drift-completion readout

**Hypothesis.** The anneal's endpoint gain over a constant-LR run is largely
a completable linear drift: W(α) = W2 + α·(W2 − W1) recovers ≥ 40% of the
constant-LR-to-annealed gap for some α > 0.5, α selected once on a selection
seed/shard and frozen.

**Runs (dev):**
- Constant-LR arm C: full 1750-step runs, `min_lr_frac: 1.0`,
  `tail.accumulate: true`, seeds **1511, 1512, 1513**.
- Annealed control A (stock WSD baseline + `tail.accumulate: true` for its
  final iterate): same seeds **1511, 1512, 1513**.

**Readout evaluation** (forward passes only, deterministic script):
val split into shard-1 (val chunks 0–19) and shard-2 (chunks 20–39).
α grid: {0, 0.5, 1, 2, 4, 8, 16, 32}. α* = argmin val loss of W(α) on
**seed 1511, shard 1**, frozen. Held-out evaluation: W(α*) on seeds 1512,
1513 × shards 1, 2.

**Pre-registered quantities:** per seed, recovery
r = (L(W2) − L(W(α*))) / (L(W2) − L_WSD,same-seed), computed per shard with
L_WSD from the same-seed arm-A final iterate evaluated on that shard.
Mechanism readouts, reported regardless of outcome: cos(v, D) and ‖D‖/‖v‖
with v = W2 − W1, D = W_WSD_final − W2, per seed (same-seed pairs share the
trajectory to step 962, so D is the anneal displacement). Also reported:
the full α curve on the selection cell, and a WSM-style convex-merge readout
(w·W2 + (1−w)·Polyak, w ∈ {0.25, 0.5, 0.75}) as a labeled exploratory arm.

**Criteria.** **PASS** iff r ≥ 0.40 on all four held-out cells (seeds 1512,
1513 × both shards). **FAIL** iff any held-out r < 0.40 **or** α* ≤ 0.5.
A PASS promotes to an eval-seed confirmation design (human gate). The null
kills *linear* drift completion only; that scope limit is registered now.

## 2. Program #18 — schedule-free tail graft

**Hypothesis.** Grafting closed-loop anchored averaging onto the record at
T_c (constant LR κ·1.0, gradient evaluation at y, Polyak readout) matches or
beats the tuned anneal, and beats the open-loop control (same constant-LR
tail, passive Polyak readout — arm C's Polyak accumulator), isolating
*feedback* as the active ingredient.

**Runs (dev, Phase A):** all forked from the seed-1511 constant-LR run's
step-963 checkpoint (identical hot phase for every arm):
- SF sweep: (κ, ρ) ∈ {0.5, 1.0} × {0.7, 0.9}, 4 tail runs, seed 1511.
- Confirm: best (κ*, ρ*) by val(x̄) at 1750, one tail run on seed 1512's
  prefix.

**Dev decision rule** (descriptive, gates only whether to propose Phase B):
promising iff val(x̄, κ*, ρ*) at 1750 (i) beats the same-seed constant-LR
raw-iterate and Polyak readouts by > 0.005 and (ii) is within +0.005 of the
same-seed WSD control, on both dev seeds. Phase B (eval seeds 1710–1713,
n=4 paired arms B and C, regenerated prefixes) uses the ideation report's
registered criteria verbatim: **WIN** if paired mean(B−A) ≤ −0.0025 with
95% CI excluding 0; **ANNEAL-REPLACED** if within +0.0025 of A AND
B < C_polyak − 0.005 AND val(x̄) at steps 1375 and 1500 within 0.01 of its
endpoint; **KILL** if paired delta ≥ +0.0025. Phase B launches only after a
human look at the dev table.

## 3. Program #19 — batch-annealed Muon, Stage 1

**Hypothesis.** The no-decay deficit is noise-limited: a constant-LR batch
ramp (§0.5) recovers most of the anneal's endpoint gain.

**Runs (dev):** arm B = batch_ramp tails forked from the step-963 checkpoints
of seeds **1511, 1512** (constant-LR prefixes, shared with #17/#18); arms A
(WSD) and C (constant LR, fixed batch) are #17's runs, shared.

**Pre-registered endpoint:** recovered fraction f = (C − B)/(C − A), paired
per seed, using final val at matched token budget (B's endpoint is its last
validation, run at exactly 1750 × 393,216 cumulative tokens; A and C at step
1750). Trajectory guard: within the tail, at matched cumulative tokens, arm
B's val curve must track arm A's within paired seed noise (|Δ| ≤ 0.004 at
every shared val point mapped by tokens) for the noise-quenching account to
survive, regardless of f — the 8× cap is not an escape hatch.

**Criteria.** **DEAD** iff mean f ≤ 0.2 (registered expectation: this is the
likely outcome; it transfers #16's anneal-irreducibility to nanogpt and is
reportable). **PARTIAL** iff 0.2 < f < 0.9 → no more BAM seeds; a hybrid
ramp+short-decay Phase B may be *proposed* (human gate). **PROMOTE** iff
f ≥ 0.9 → eval-seed n=4 design proposed with WIN bar mean(B) ≤ 3.28638.

## 4. Compute budget and lane plan (dev phase)

| Runs | Count × time | GPU-h |
|---|---|---|
| Constant-LR full (C), seeds 1511–1513 | 3 × 75 min | 3.75 |
| WSD control (A), seeds 1511–1513 | 3 × 75 min | 3.75 |
| SF tails (κ,ρ sweep + confirm) | 5 × ~35 min | ~2.9 |
| Batch-ramp tails, seeds 1511–1512 | 2 × ~40 min | ~1.3 |
| GPU smoke | 2 × ~6 min | 0.2 |

Total ≈ **12 GPU-h** ≈ 6 wall-hours on the two lanes. Artifacts: ~2.6 GB
fp32 tail artifact per accumulating run (~16 GB total, disk has 1.5 TB free).

## 5. What would count as a Wave-1 conclusion

- Any single PASS/WIN/PROMOTE → that program's Phase-B proposal goes to the
  human gate with the others' outcomes as context.
- Triple null at the registered bars → "the record's anneal is irreducible on
  this harness against linear drift completion, anchored-averaging feedback,
  and batch-growth noise quenching at matched tokens" — a reportable
  mechanism decomposition; Wave 2 (anneal-handoff #2, Spectral-SAM A1 #5)
  then proceeds per the portfolio plan, with #2's prior updated downward.
- Registered scope limits: dev-seed-only conclusions are tuning evidence, not
  effect claims; all effect claims require the Phase-B eval-seed designs.
