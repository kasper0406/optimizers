# WP0 Buildout — Checkpoint Report

Date: 2026-07-18 · Git: `a9c1975` (buildout) · Author: agent (workflow `routed-muon-wp0-buildout`, 8 agents, all green)

## Status summary

| Work package | Status | Evidence |
|---|---|---|
| WP0.0 skeleton & environment | **Done** | 287-test suite green; smoke run writes valid results JSON (`max_param_delta == 0.0`); submodules pinned (airbench `4c1b6d1e`, modded-nanogpt `edf47a05`, DynMuon `89baa666`) |
| WP0.5 stats validation suite | **Done — unblocks instrumented runs** | 127 tests: ρ within ±0.05 (all 40 AR(1) cells), analytic t-stat crossings at SNR {0.1, 1, 10}, ηλ within ±0.1 for r {0.8, 1.0, 1.1}, reclassification within N_min, both β, small-t bias correction. Report: `reports/wp05-stats-validation.md` |
| WP0.4 baseline zoo | **Code done; GPU smoke pending** | Muon/AdamW/DynMuon/AdaMuon/NorMuon verified vs vendor code and arXiv:2507.11005 / 2510.05491; 77 tests incl. hand-computed ≥2-step trajectories. Airbench smoke configs in `configs/dev/` await a CUDA box |
| WP1.1 instrumentation | **Code done; overhead DoD pending** | Tracked pairs + bulk probes + projections through `src/stats` (verified: no reimplementation, no HVP reachable from update paths); Phase-1 plot scripts run from results JSONs. ⚠ see "Overhead risk" below |
| WP0.1 airbench baseline | **In progress** | `airbench` experiment = stock vendored recipe (vendor Muon lr=0.24/momentum=0.6/nesterov, compile on) registered; 100-eval-seed sweep launching on Hyperstack A6000-spot (user-authorized provisioning) |
| WP0.2 record candidates | **Report ready — human pick needed** | `reports/wp02-record-candidates.md` |
| WP0.3 DynMuon repro plan | **Plan ready — human checkpoint after run** | `reports/wp03-dynmuon-repro-plan.md` |

## Standing verification (independent agent, full detail in `reports/wp0-verify-notes.md`)

All six checks pass: results append-only · no eval-seed literals in configs/tests · criteria/ = blank-threshold templates only · submodules at pinned SHAs, clean · no stats reimplementation in instrumentation · no HVP reachable from any optimizer update path. Docker build validated to the dependency layer locally (timed out on Mac bandwidth, no defect); full build happens on the cloud VM.

## Decisions that are yours (agent will not proceed on these)

1. **Author `criteria/airbench_tolerance.yaml`** from the four blank templates once the WP0.1 baseline distribution lands — the agent reports mean/std/CI only.
2. **Pick the pinned nanogpt record** from the three candidates in `reports/wp02-record-candidates.md`. Note: `2025-07-12_BosAlign` ships an in-repo n=20 seed distribution (mean 3.2791, std 0.0013) — the strongest basis for authoring `criteria/nanogpt_tolerance.yaml`.
3. **Gate WP0.3**: the repro plan targets the 127M/10B-token DynMuon config (their 21.9% cell, est. ~6.4 h/run on H200-class); approve before any cloud spend on it.
4. **Overhead-risk decision (Phase 1, not urgent):** the instrumentation agent measured ~76 µs per scalar classifier update; at airbench shapes (6 matrices × 32 directions × 2 β) that is ~20–30 ms/step of serial CPU work vs ~2–17 ms GPU steps — the <10% overhead DoD will very likely fail as-is. The fix is array-mode classification in `src/stats` (~100× reduction, already supported by `DirectionStats`); it touches WP0.5-validated code, so it needs your sign-off and a re-run of the WP0.5 suite after vectorization.

## Discrepancy worth knowing (from the quote-extraction pass)

DynMuon's paper claims a cosine LR decay (App. D p.18) but their released code implements a linear trapezoid schedule (`train_gpt.py:795-803`). Recorded verbatim in `reports/dynmuon-quotes.md`; interpretation is yours.
