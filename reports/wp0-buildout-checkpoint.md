# WP0 Buildout — Checkpoint Report

Date: 2026-07-18 · Git: `a9c1975` (buildout) · Author: agent (workflow `routed-muon-wp0-buildout`, 8 agents, all green)

## Status summary

| Work package | Status | Evidence |
|---|---|---|
| WP0.0 skeleton & environment | **Done** | 287-test suite green; smoke run writes valid results JSON (`max_param_delta == 0.0`); submodules pinned (airbench `4c1b6d1e`, modded-nanogpt `edf47a05`, DynMuon `89baa666`) |
| WP0.5 stats validation suite | **Done — unblocks instrumented runs** | 127 tests: ρ within ±0.05 (all 40 AR(1) cells), analytic t-stat crossings at SNR {0.1, 1, 10}, ηλ within ±0.1 for r {0.8, 1.0, 1.1}, reclassification within N_min, both β, small-t bias correction. Report: `reports/wp05-stats-validation.md` |
| WP0.4 baseline zoo | **Code done; GPU smoke pending** | Muon/AdamW/DynMuon/AdaMuon/NorMuon verified vs vendor code and arXiv:2507.11005 / 2510.05491; 77 tests incl. hand-computed ≥2-step trajectories. Airbench smoke configs in `configs/dev/` await a CUDA box |
| WP1.1 instrumentation | **Code done; overhead DoD pending** | Tracked pairs + bulk probes + projections through `src/stats` (verified: no reimplementation, no HVP reachable from update paths); Phase-1 plot scripts run from results JSONs. ⚠ see "Overhead risk" below |
| WP0.1 airbench baseline | **Done** | n=100 eval seeds (0–99) on one RTX A6000 spot VM: **tta_val_acc mean 0.94003, std 0.00141, 95% CI ±0.00028** (published A100 reference ≈ 0.9401). Timed train mean 13.8 s/run; sweep wall 36.6 min; sweep cost $0.25. Aggregates: `reports/baseline_airbench_aggregate.{md,csv}`; per-seed JSONs in `results/`. Awaiting human-authored `criteria/airbench_tolerance.yaml` for the formal pass/fail |
| WP0.2 record candidates | **Report ready — human pick needed** | `reports/wp02-record-candidates.md` |
| WP0.3 DynMuon repro plan | **Plan ready — human checkpoint after run** | `reports/wp03-dynmuon-repro-plan.md` |

## Standing verification (independent agent, full detail in `reports/wp0-verify-notes.md`)

All six checks pass: results append-only · no eval-seed literals in configs/tests · criteria/ = blank-threshold templates only · submodules at pinned SHAs, clean · no stats reimplementation in instrumentation · no HVP reachable from any optimizer update path. Docker build validated to the dependency layer locally (timed out on Mac bandwidth, no defect); full build happens on the cloud VM.

## Decisions that are yours (agent will not proceed on these)

1. **Author `criteria/airbench_tolerance.yaml`** from the four blank templates once the WP0.1 baseline distribution lands — the agent reports mean/std/CI only.
2. **Pick the pinned nanogpt record** from the three candidates in `reports/wp02-record-candidates.md`. Note: `2025-07-12_BosAlign` ships an in-repo n=20 seed distribution (mean 3.2791, std 0.0013) — the strongest basis for authoring `criteria/nanogpt_tolerance.yaml`.
3. **Gate WP0.3**: the repro plan targets the 127M/10B-token DynMuon config (their 21.9% cell, est. ~6.4 h/run on H200-class); approve before any cloud spend on it.
4. **Overhead-risk decision (Phase 1, not urgent):** the instrumentation agent measured ~76 µs per scalar classifier update; at airbench shapes (6 matrices × 32 directions × 2 β) that is ~20–30 ms/step of serial CPU work vs ~2–17 ms GPU steps — the <10% overhead DoD will very likely fail as-is. The fix is array-mode classification in `src/stats` (~100× reduction, already supported by `DirectionStats`); it touches WP0.5-validated code, so it needs your sign-off and a re-run of the WP0.5 suite after vectorization.

## Cloud campaign log (2026-07-18/19, VM `rm-wp01-airbench`, RTX A6000 spot, $0.4067/h)

Total VM cost **$1.24** (98 billed minutes; rate from the billing API, not estimated). Includes driver upgrade (image shipped R535, too old for torch cu12.8 — upgraded in place to 580-server), one stalled CIFAR-10 download from the origin server (now cached under `data/` and mounted into all runs), image rebuilds for two fixes (missing gcc for triton; `bench_overhead` vendor API bug). Per-run costs in results JSONs amortize the sweep window evenly (methodology: window × billing rate / n).

## WP0.4 zoo smoke table (dev seeds, A6000, untuned single runs)

| optimizer | seed | val_acc | tta_val_acc | timed s |
|---|---|---|---|---|
| muon | 1500 | 0.9321 | 0.9410 | 8.1 |
| adamw | 1501 | 0.8938 | 0.9022 | 7.9 |
| dynmuon | 1502 | 0.5662 | **0.5684 ⚠** | 8.3 |
| adamuon | 1503 | 0.9108 | 0.9233 | 8.2 |
| normuon | 1504 | 0.8797 | 0.8908 | 8.6 |

⚠ **Open item:** DynMuon's smoke lands far outside a plausible band. The update rule passes hand-computed unit tests against their reference code, so the suspect is the untuned smoke config (lr=0.05 with `adjust_lr=spectral_norm`, logistic p-anneal to p_min=−0.25 compressed into 200 steps — a schedule shaped for LLM-length runs). Needs a dev-seed tuning pass before the WP2.2 comparison; not silently tuned per ground rule 6.

## WP1.1 overhead measurement (GPU, `results/bench_overhead_airbench.json`)

Instrumented vs stock airbench step on A6000: median **37.3% overhead** (stock 28.95 ms → instrumented 39.74 ms; measurement only — pass/fail vs the <10% DoD is a human judgment, but it confirms the predicted scalar-stats bottleneck and motivates decision 4 below).

## Discrepancy worth knowing (from the quote-extraction pass)

DynMuon's paper claims a cosine LR decay (App. D p.18) but their released code implements a linear trapezoid schedule (`train_gpt.py:795-803`). Recorded verbatim in `reports/dynmuon-quotes.md`; interpretation is yours.
