# Routed Muon

Research repo for the Routed Muon project: per-direction regime classification
(signal / noise / edge-of-stability oscillation) and routed spectral shaping for
Muon-family optimizers.

- Strategy: `routed-muon-research-plan.md` (authoritative on scientific intent)
- Workflow, work packages, guardrails: `CLAUDE.md`

## Quick start

```bash
uv sync
uv run pytest
bash scripts/launch_local.sh configs/smoke.yaml
```

## Layout

```
configs/      one YAML per experiment, no CLI-arg science
criteria/     pre-registered gates & tolerances (HUMAN-authored)
src/optim/    optimizer interface + implementations
src/stats/    tracked pairs, EMAs, classifier (unit-test target)
src/instrument/  logging hooks, scatter/occupancy plotting
tests/        synthetic-signal suite and sanity tests
vendor/       airbench, modded-nanogpt, DynMuon submodules @ pinned SHAs
scripts/      launch_local.sh, launch_cloud.sh, run.py, sweep, aggregate
results/      synced JSONs (append-only; never edit)
reports/      checkpoint reports
```

## Seed discipline

Seeds 0–99 are the evaluation set for comparison tables and are resolved by
launch tooling only. All development, debugging, and hyperparameter exploration
uses seeds >= 1000.
