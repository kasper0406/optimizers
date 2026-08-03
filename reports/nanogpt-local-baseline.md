# nanogpt local testbed validation — baseline seed set (next-step #4)

2026-07-21, local 2× RTX 5090, harness-standard config
`configs/dev/nanogpt_local_baseline.yaml` (record hyperparameters +
`fp32_embed_grad_accum` + `head_chunk_rows: 8192`; PORT CHANGE P5). n = 10
dev seeds (1710–1719), all completed on the first attempt under
`scripts/babysit_nanogpt.sh` supervision (zero retries — no GPU instability
during this set). Tables/data: `reports/nanogpt-local-baseline-tables.md`,
`.json`, produced by `scripts/analyze_local_baseline.py`. Descriptive.

## Headline

**final val loss = 3.28888 ± 0.00125 (sd), SE(mean) 0.0004,
sd CI95 [0.00086, 0.00228].**

The testbed is self-consistent and as quiet as the record itself: the
record's own n=20 native-hardware sd is 0.0013, ours is 0.00125 — the port
plus new hardware adds **no seed-noise inflation**. The across-seed sd
shrinks 5.6× from step 125 (0.0070) to step 1750 (0.0012), matching the
record's ~5× shrinkage profile, so mid-training comparisons face the same
yardstick behavior as at the end.

## Endpoint

Steps-to-target(3.28) is **censored 10/10** on this harness — no run
reaches the record's target (min final 3.2876). This settles the endpoint
question WP0.2 raised: on this testbed the primary endpoint for any A/B
comparison is **final val loss at fixed 1750 steps** (uncensored, sd
0.00125); steps-to-target is unusable.

## Offset vs the record (context, not a goal)

Our mean sits +0.0098 above the record's n=20 ensemble mean (3.2791), vs
+0.0062 for the single H100-PCIe fp32-embed run of WP0.2. The extra
+0.0036 is the aggregate of: consumer-Blackwell vs H100 numerics, torch
2.13.0+cu130 vs the record's nightly, and the chunked-head fp32 summation
order — inseparable without ablations we have no reason to run:
record-faithfulness off native hardware is a documented non-goal
(project-state §3.6). What the offset does NOT do is inflate variance —
which is the property A/B comparisons need.

## Power (what this testbed can resolve)

End-of-run slope 1.64e-4 loss/step converts loss effects to
steps-equivalents. 80% power at α = .05 (two-sided, unpaired):

| loss effect | ≈ steps | ≈ % of run | seeds/arm |
|---|---|---|---|
| 0.0010 | 6 | 0.35% | 25 |
| 0.0025 | 15 | 0.87% | 4 |
| 0.0050 | 31 | 1.7% | 2 |
| 0.0086 | 52 | 3.0% (plan's smallest effect of interest) | 2 |

The plan's 3–5% effect band costs 2 seeds/arm; even sub-1% effects are
affordable (4/arm ≈ 4.5 GPU-hours). Seed-pairing (same seeds both arms,
paired test) would tighten these further; these baselines are the control
arm for exactly that design.

## Provenance notes

- All 10 runs at SHA ecd48f1 (includes the run.py seed-injection fix this
  set surfaced — without it, sweep-launched nanogpt runs all trained as
  seed 1000; caught before any result was written).
- `git_dirty: true` on all 10: the dirt is the then-untracked operational
  directories `checkpoints/` and `logs/` (both since gitignored), not code.
- `cost_usd` null (owned hardware; convention pending). Wall ≈ 65–68
  min/run; GPU0 consistently ~3.5% slower than GPU1 (4,050 s vs 3,915 s
  train time) — worth remembering when reading paired wall-clock numbers.

## Testbed status

Validated. A/B method work on this box can proceed against these controls:
same config, same seeds (1710–1719) for the treatment arm, final val at
1750 steps as endpoint, babysitter supervision for card flakiness.
