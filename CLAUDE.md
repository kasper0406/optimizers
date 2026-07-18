# Claude Code Handoff — Routed Muon

Companion to `routed-muon-research-plan.md` (the strategy document; read it first, it is authoritative on scientific intent). This file defines the executable workflow: work packages, verification, human checkpoints, and guardrails.

## Ground rules for the agent

1. **Human checkpoints are hard stops.** When a work package ends at a checkpoint, produce the checkpoint report and stop. Never evaluate a scientific gate yourself, never adjust pre-registered criteria, thresholds, or binning to change a gate outcome, never proceed past a gate on your own judgment.
2. **Seed discipline.** Seeds 0–99 are the evaluation set for any comparison table. Development, debugging, and hyperparameter exploration use seeds ≥ 1000 only. Never tune on evaluation seeds.
3. **No silent metric changes.** Metric definitions, tolerances, and success criteria live in `criteria/` as versioned files. Changing them requires an explicit human instruction referencing the file.
4. **Compute boundary.** You write code, launch and verify anything that runs on the local machine in ≤ ~15 min, and prepare entrypoint scripts for everything larger. Cloud (Hyperstack) runs are launched by the human using your scripts; you consume their synced results from `results/`. Never assume a cloud run happened — check for its results file.
5. **Reproducibility.** Every run gets: pinned config file, git SHA, seed, GPU type string, wall time, and cost field (human-filled for cloud) written into its results JSON. A results file without these is invalid.
6. **When blocked, report — don't improvise.** Missing baseline numbers, failed reproductions, and ambiguous results are findings to report, never gaps to fill with estimates.

## Repo skeleton (WP0.0)

```
routed-muon/
  pyproject.toml            # pinned deps
  Dockerfile                # the Hyperstack image; entrypoint runs any config
  configs/                  # one YAML per experiment, no CLI-arg science
  criteria/                 # pre-registered gates & tolerances (human-authored)
  src/
    optim/                  # optimizer interface + implementations
      interface.py          # per-matrix hooks: pre_step(G), shape_spectrum(O, state), post_step
      muon.py  dynmuon.py  adamuon.py  normuon.py  routed.py
    stats/                  # tracked pairs, EMAs, classifier  ← unit-test target
    instrument/             # logging hooks, scatter/occupancy plotting
  tests/                    # synthetic-signal suite (WP0.5)
  vendor/                   # airbench, modded-nanogpt, DynMuon as submodules @ pinned SHAs
  scripts/                  # launch_local.sh, launch_cloud.sh, sweep.py, aggregate.py
  results/                  # synced JSONs (append-only; never edit)
  reports/                  # checkpoint reports (agent-written)
```

## Work packages

### WP0.0 — Skeleton & environment
Build the structure above; Dockerfile builds; `pytest` runs (empty suite passes); submodules pinned.
**DoD:** `docker build` succeeds locally; `scripts/launch_local.sh configs/smoke.yaml` runs a 10-step training no-op and writes a valid results JSON.

### WP0.5 — Statistics validation suite (BLOCKS all instrumented runs)
Implement `src/stats/` and a synthetic test suite. Generators (no GPU, no training):
- AR(1) processes with ρ ∈ {−0.8, −0.4, 0, 0.4, 0.8}, various noise scales → classifier must recover ρ within ±0.05 after burn-in and assign the correct regime label.
- Drifting mean + noise at SNR ∈ {0.1, 1, 10} → t-statistic crosses signal threshold iff SNR and window predict it (analytic expectation in test).
- Pure oscillation s(t) = A·(−r)^t for r ∈ {0.8, 1.0, 1.1} → implied ηλ from amplitude ratio within ±0.1 of ground truth; classified oscillating; amplitude-decay flag correct.
- Regime *switches* mid-stream → confidence reset logic re-classifies within N_min steps.
- Both β values (0.9, 0.99); bias-corrected estimates tested at small t.
**DoD:** all tests pass; a short `reports/wp05-stats-validation.md` with recovered-vs-true plots. **No Phase-1 run may launch before this passes.**

### WP0.1 — Airbench baseline
Reproduce airbench 94% tier via vendored submodule; run seed sweep (eval seeds 0–99) locally or emit cloud script per compute boundary.
**DoD:** `results/baseline_airbench.json` with per-seed accuracy/time; aggregate report with mean/std/CI; check against `criteria/airbench_tolerance.yaml` (human authors the tolerance; agent reports pass/fail, does not set it).

### WP0.2 — modded-nanogpt baseline
Pin a historical record (human picks which; propose 3 candidates with rationale in a report first). Port config to target GPU count via grad accumulation; verify loss-vs-tokens overlays the record log.
**DoD:** overlay plot + max deviation at fixed token checkpoints vs `criteria/nanogpt_tolerance.yaml`; 3-seed variance table. **→ HUMAN CHECKPOINT** (reproduction quality judgment).

### WP0.3 — DynMuon reproduction
Smallest config showing their Muon-vs-DynMuon gap; equal tuning effort documented.
**DoD:** report stating observed gap ± CI vs claimed 10.6–26.5%, with configs. **→ HUMAN CHECKPOINT.**

### WP0.4 — Baseline zoo
AdamW, DynMuon, AdaMuon, NorMuon behind `interface.py`. Each verified by: (a) unit test of the update rule on a fixed tiny matrix against a hand-computed step; (b) airbench smoke run (dev seeds) landing in a plausible accuracy band.
**DoD:** tests pass; smoke table in report.

### WP1.1 — Instrumentation
Tracked-pair machinery per plan §1.1 (top-k₁ subspace iteration + k₂ bulk probes, projections, EMAs, HVP-per-refresh). Overhead measured: instrumented vs stock step time.
**DoD:** overhead < 10% on airbench; instrumented run produces the per-direction log schema; stats module is the WP0.5-tested code (no reimplementation).

### WP1.2 — Phase-1 measurement runs
**Precondition:** human has committed `criteria/phase1_preregistration.md` (cluster criteria per plan §1.3). Agent verifies file exists and predates first run's timestamp; refuses to launch otherwise.
Runs: airbench ≥20 instrumented seeds (dev seeds acceptable — measurement, not comparison); nanogpt 2–3 seeds via cloud scripts.
**DoD:** the three plots of plan §1.2, generated by deterministic scripts from results JSONs; a *descriptive* report (what the scatters show, GMM/BIC outputs) that explicitly does **not** conclude pass/fail. **→ HUMAN CHECKPOINT: Gate 1.** Human decides: full routing / oscillation-only / stop.

### WP2.x — Routed optimizer & airbench comparison (only after Gate 1, scoped by its outcome)
- WP2.1: `routed.py` per plan §2.1; unit tests: synthetic momentum matrices with planted signal/noise/oscillating directions → correct per-direction gains applied. Distributed invariants enforced structurally (stats owned per-matrix; no full-gradient gathers; no HVP in the update path — CI greps for it).
- WP2.2: experiment matrix of plan §2.2 as configs; local smoke each config on dev seeds; cloud sweep scripts for eval seeds; aggregate into the comparison table with CIs.
- WP2.3: ablation table (§2.2.4) incl. random-gating placebo; LR stress test (§2.2.5); λ-tracking plot (§2.2.6).
**DoD:** tables/plots + descriptive report vs `criteria/phase2_success.yaml`. **→ HUMAN CHECKPOINT: Gate 2.**

### WP3.x — nanogpt scale (after Gate 2; human-launched cloud runs throughout)
Configs, launch scripts, power analysis from WP0.2 seed variance (agent computes required n for 3–5% effect at 80% power *before* proposing the run set), aggregation, draft results section. Feasibility deliverable: the distributed cost-accounting table (analysis only, per plan).
**→ HUMAN CHECKPOINT: paper go/no-go.**

## Tasks that are human-only (agent must not attempt)
- Authoring anything in `criteria/` (agent may draft *templates* with blank thresholds).
- Reading DynMuon's theory section / writing the differentiation memo (agent may extract quotes and structure; interpretation is human).
- Gate decisions; pinned-record selection; cloud provisioning; cost entries.

## Standing verification the agent runs every session
- `pytest` green before and after changes.
- No file in `results/` modified (append-only check via git status).
- No occurrence of eval seeds (0–99) in any config under active development (`grep` check).
- `criteria/` untouched by agent commits (git blame check on changed lines).
