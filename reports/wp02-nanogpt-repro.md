# WP0.2 — nanogpt port vs pinned record (descriptive)

Record: **2025-07-12_BosAlign**, n=20 same-script validation runs in `vendor/modded-nanogpt/records/track_1_short/2025-07-12_BosAlign` (script md5 `5ffba04f`).
Record final val loss: mean 3.2791, std 0.0013, min 3.2770, max 3.2819, n=20.

All comparisons below are against the **ensemble** of those 20 logs, not against any single one. A single record log carries the record's own between-run seed noise, which is 5x larger at step 125 than at step 1750; measuring against it attributes that noise to our port.

## Runs

| seed | GPU | git | device_count x accum | tokens/step | precision | final val | steps→3.28 | train s | $ |
|---|---|---|---|---|---|---|---|---|---|
| 1701 | NVIDIA H100 PCIe | 90e414f0 | 1 x 8 | 393,216 | fp8 | 3.2903 | — | 1964 | 1.6 |

**Correction to the stored `record_faithful` flag.** seed 1701: the results JSON records `record_faithful: true`, but the run uses gradient accumulation (`device_count` != the record's world size) and its own `deviations` dict lists `grad_accumulation`. That flag was computed by a predicate that did not inspect `device_count` — a code defect, since accumulation changes the gradient *reduction order* and at D<8 the bf16 embedding grads make that a genuine precision difference (docs/nanogpt-port.md §2). The predicate is fixed in `src/nanogpt/config.py` (D != 8 is never record-faithful) and pinned by a regression test. `results/` is append-only, so the JSON keeps the pre-fix value; **the correct reading is that these runs are NOT record-faithful**, and this report is where that correction lives.

**Deviation flags active** (these runs are NOT record-faithful):

- seed 1701 — `grad_accumulation`: 1 device(s) x 8 micro-batches = 8 record chunks per optimizer step; tokens/step 393216 (record 393216)

## Overlay vs the record ensemble (headline)

Deviation = our val loss − the mean of the n=20 record logs at the same step (token checkpoints of 393,216 tokens/step). `sigma` divides that deviation by the record's **between-run sd at that step**.

Read `sigma` with care: it is *the record's* sigma, not ours. We have n=1 and therefore **no estimate of our own harness's seed variance**, so these are not z-scores for our run. The record's sd also shrinks by ~5x over training, so a constant absolute deviation grows in sigma by yardstick shrinkage alone.

### seed 1701 — deviation trajectory

| step | ours | record mean | record sd | dev (loss) | dev (sigma) |
|---|---|---|---|---|---|
| 0 | 10.8258 | 10.8258 | 0.00000 | +0.0000 | — |
| 125 | 4.6350 | 4.6386 | 0.00751 | -0.0036 | -0.5 |
| 250 | 4.1038 | 4.0972 | 0.00602 | +0.0066 | +1.1 |
| 375 | 3.9021 | 3.8953 | 0.00671 | +0.0068 | +1.0 |
| 500 | 3.7501 | 3.7473 | 0.00442 | +0.0028 | +0.6 |
| 625 | 3.6674 | 3.6600 | 0.00384 | +0.0074 | +1.9 |
| 750 | 3.6010 | 3.5965 | 0.00270 | +0.0045 | +1.7 |
| 875 | 3.5559 | 3.5470 | 0.00256 | +0.0090 | +3.5 |
| 1000 | 3.5143 | 3.5058 | 0.00232 | +0.0085 | +3.7 |
| 1125 | 3.4630 | 3.4534 | 0.00208 | +0.0096 | +4.6 |
| 1250 | 3.4169 | 3.4071 | 0.00153 | +0.0098 | +6.4 |
| 1375 | 3.3765 | 3.3661 | 0.00165 | +0.0104 | +6.3 |
| 1500 | 3.3406 | 3.3300 | 0.00137 | +0.0105 | +7.7 |
| 1625 | 3.3109 | 3.3000 | 0.00145 | +0.0109 | +7.5 |
| 1750 | 3.2903 | 3.2791 | 0.00134 | +0.0112 | +8.4 |

At the final step our loss is **+0.0112** from the record ensemble mean and **+0.0084** from the record's observed MAXIMUM (3.2819) — **outside the observed support of the record's n=20 distribution**.

*(Single-log statistic, non-headline: against `0c5449cc` alone the max |dev| is 0.0179 at step 125.
  At that step the record's between-run sd is 0.0075 and our deviation from the ensemble mean is only -0.0036 (-0.5 sigma) — i.e. the single-log number is mostly the record run's own seed noise, not our port's. That is why it was dropped as the headline.)*

## Where the deviation accumulates (phase decomposition)

The record's LR cooldown begins at `num_iterations * (1 - cooldown_frac)` (RECORD:670-684). Splitting training there separates a deviation that accrues during the stable-LR phase from one that accrues while the LR anneals. The deficit is stated in absolute loss AND as a fraction of the loss that phase actually removes — the second is the meaningful one, since the two phases remove very different amounts.

### seed 1701

Cooldown onset: step 962.5 exactly (= 1750 x (1 − 0.45)), snapped to the nearest validation step **1000**.

| phase | steps | our drop | record drop | deficit | deficit % of phase drop |
|---|---|---|---|---|---|
| stable | 0→1000 | 7.3116 | 7.3200 | +0.0085 | 0.12% |
| cooldown | 1000→1750 | 0.2239 | 0.2266 | +0.0027 | 1.20% |

Per unit of loss removed, the deficit is **10.4x denser in the cooldown phase** than in the stable phase.

**Per-eval-segment drops within cooldown** (ratio < 1 = we remove less loss than the record over that interval):

| segment | our drop | record drop | ratio |
|---|---|---|---|
| 1000→1125 | 0.0512 | 0.0524 | 0.9781 |
| 1125→1250 | 0.0461 | 0.0463 | 0.9974 |
| 1250→1375 | 0.0404 | 0.0410 | 0.9855 |
| 1375→1500 | 0.0359 | 0.0361 | 0.9947 |
| 1500→1625 | 0.0297 | 0.0300 | 0.9886 |
| 1625→1750 | 0.0205 | 0.0209 | 0.9846 |

Sign test: we remove less loss than the record in **6 of 6** cooldown segments, two-sided p = 0.031. Consecutive segments share one curve, so treat this as a consistency measure, not an inferential test.

## Distributions

Ours: final val loss mean 3.2903 (n=1, no std), n=1.
Record: mean 3.2791, std 0.0013, n=20.

### Steps-to-3.28

Record steps-to-3.28 (interpolated from its own 125-step trace): mean 1740.5, std 4.5 of 1750 total — **over n=14, not n=20**.

> **CENSORING DISCLOSURE — steps-to-3.28 is not a sound primary endpoint here.**
> **6 of the record's own 20 runs never reach 3.28** (their finals: 3.2801, 3.2803, 3.2804, 3.2805, 3.2817, 3.2819). The record's "steps-to-3.28 mean 1740.5, std 4.5" is therefore computed over the **n=14 survivors only** — the runs that happened to clear the bar — and is a **censored** statistic, not the ensemble's steps-to-target.
> It is also **biased between arms**: the target sits inside the record's own final-loss distribution, so an arm straddling 3.28 drops its slow runs from the average and is flattered relative to an arm that clears it outright. The direction of that bias depends on where each arm's distribution sits, so it does not cancel. **Unsuitable as a primary endpoint** at this target; final val loss at fixed steps is the uncensored alternative.

**None of our runs reached 3.28** — our steps-to-target is undefined and is NOT extrapolated. Note that our run is censored by the same mechanism as the 6 record runs above, which is precisely why the metric cannot carry this comparison.

## Reading of these numbers (descriptive)

1. **Our one run sits +0.0112 above the record's n=20 mean (3.2791), and +0.0084 above the record's observed maximum (3.2819)** — that is, outside the observed support of the record's distribution, not merely in its upper tail.
2. **The "8.4 sigma" is 8.4x the RECORD's sigma, and is partly a yardstick artefact.** We have n=1 and therefore no estimate of our own harness's seed variance; the record's sd shrinks from 0.0075 (step 125) to 0.00134 (step 1750) while our absolute deviation is roughly flat after step 875, so most of the growth in the sigma column is the denominator shrinking, not our run drifting.
3. **The deviation is cooldown-concentrated** — see the phase table: per unit of loss removed, the deficit is an order of magnitude denser during the LR cooldown, and every cooldown segment underperforms.
4. **Leading suspect: bf16 embedding-gradient accumulation at D<8.** At `device_count: 1` the port sums 8 chunk gradients sequentially into bf16 `p.grad` (embeddings are bf16, RECORD:628-630) where the record does an 8-way `ReduceOp.AVG` across ranks. docs/nanogpt-port.md §2 already names this "the least-controlled numeric deviation in the port". It is not the only candidate — torch-version/kernel drift vs the record's 2025 nightly is unmeasured — but it is the one we can test with a single one-variable run.

No pass/fail is drawn from any of this.

## PRE-REGISTERED next diagnostic (written BEFORE the run)

**This section was written before the probe run was launched and must not be revised after seeing its result.**

Probe: `configs/wp02_nanogpt_fp32embed.yaml` — the port doc's §6.1 diagnostic. fp32 master-buffer accumulation of embedding gradients across the 8 micro-batches (cast back to bf16 once per step) at `device_count: 1`, **seed 1701, everything else identical** to `configs/wp02_nanogpt_repro.yaml`. One variable changes.

Read, fixed in advance:

| final deficit vs record mean | conclusion |
|---|---|
| **<= +0.006** | bf16-accumulation suspect **confirmed** |
| **unchanged at ~+0.011** | suspect **excluded**; residual is torch-version / kernel / hardware |

An intermediate outcome (between +0.006 and +0.011) is partial attribution and is to be reported as partial, not rounded to either verdict. The probe run is NOT record-faithful (two deviation flags: `grad_accumulation` and `fp32_embed_grad_accum`) and never enters a reproduction table.

## Cost reconciliation

Summed from `cost_usd` across `results/`: **$12.00 project total**, of which **$3.20 is WP0.2 nanogpt** (2603 costed run(s); 6 run(s) carry no `cost_usd` and are excluded).

---
Descriptive only; no pass/fail. Reproduction quality is judged by the human against `criteria/nanogpt_tolerance.yaml`.
