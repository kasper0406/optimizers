# Routed Muon

Research repo for the Routed Muon project: per-direction regime classification
(signal / noise / edge-of-stability oscillation) and routed spectral shaping
for Muon-family optimizers — executed end to end 2026-07-18/20, both scientific
gates closed. Full narrative: `reports/paper-draft.md`.

- Strategy: `routed-muon-research-plan.md` (authoritative on scientific intent)
- Workflow, work packages, guardrails: `CLAUDE.md`
- Gate records (adversarially audited): `reports/gate1-decision.md`,
  `reports/gate2-decision.md`

## Key findings

**Affirmative measurements** (novel; nothing equivalent published as of the
2026-07-19 literature sweep, `docs/litreview/`):

1. **A structured negative-autocorrelation population in per-direction
   gradient projections under Muon.** 60–89% of direction-snapshots have
   lag-1 ρ < −0.2 against a ~25% white-noise floor (airbench94, 20 seeds);
   the population is *phase-structured* (peaks mid-training, collapses in the
   LR anneal), *LR-driven* (monotone in LR across configs and within runs),
   *momentum-independent* (momentum=0 leaves it intact), and *robust to
   batch-sampling artifacts* (with-replacement ablation). Bulk directions
   participate as much as top directions. `reports/wp12-phase1-measurement.md`,
   `reports/wp22-mechanism-probes.md`.
2. **Curvature does not govern Muon's stability.** Training is stable at
   HVP-measured η·λ ≈ 65 along tracked directions (GD's bound: 2), and no
   divergence occurs even at 6× the record LR — accuracy degrades gracefully
   (94.0 → 90.5%), with the useful-LR shoulder between 2× and 3×. This is
   empirical data on the momentum+minibatch regime that the non-Euclidean
   edge-of-stability line (ICML'26 oral, arXiv:2603.05002) names as open.
3. **Amplitude ratios do not measure curvature under Muon.** Implied η·λ from
   oscillation amplitude saturates at the noise floor (~2–4) while HVP-measured
   η·λ spans 0–65 (Pearson ≈ 0) — "curvature-for-free from oscillation
   amplitude" is dead for normalized optimizers. `reports/wp12-disambiguation.md`.
4. **Per-direction persistent signal is structurally unmeasurable at normal
   batch SNR** (t-statistic ceiling: |t| ≤ SNR·√ess; observed SNR q90 ≈ 0.26)
   — per-direction "signal routing" has nothing to route on at batch 2000;
   coarser granularity or much larger batches are required.

**The intervention null** (well-powered, placebo-controlled, routing verified
active at 13.6% treated fraction):

5. **Per-direction oscillation damping does nothing on airbench.** Across
   adaptive gains, constant gains {0.25, 0.5, 0.75}, full three-channel
   routing, and a random-gating placebo (n=100 eval seeds each, seed-paired):
   any effect above **0.042pp is excluded at 97.5% confidence** (pre-registered
   bar: 0.27pp). Stability margin ratio: exactly 1.0. Gate 2: FAIL, final.
   Scope: 200-step airbench record config, tracked-subspace interventions.
   `reports/wp22-comparison-table.md`.
6. **A pre-registration case study:** a promising dev-seed signal at 2× LR
   (+0.144pp, p=0.006) was killed by its own pre-registered n=100
   confirmation (−0.024pp, sign flipped) — and the original pre-registered
   GMM/BIC cluster criteria turned out to be satisfiable by white noise
   (documented in `reports/gate1-decision.md`).
7. **Qualified baselines observation:** at documented light tuning (LR-only,
   5-seed dev probes) on the Muon-co-adapted record config, none of DynMuon /
   AdaMuon / NorMuon matched stock Muon (gaps 0.29–0.82pp). Not an optimizer
   ranking; says nothing about their home-scale claims.

**Follow-up programs (2026-07-20, `reports/brainstorm-programs.md`):** four
pre-registered predictions derived from the findings above were all refuted —
occupancy is schedule-position-dependent rather than an LR state function; the
routing null is *not* a compounding failure (interventions drive training to
essentially unrelated weights of equal quality — stock-vs-stock control diverges
exactly 0.000, so this is real); spectral directional smoothness does not
equilibrate at an lr-invariant constant (ratio 2.57 across the ladder); and
frozen full-run probes find no hidden persistent per-direction signal
(median |t| 0.61, slope ~0.05 vs √t's 0.5).

**Where this points next** (`docs/litreview/README.md`): the temporal trust
ratio — per-matrix/global LR control from negative-ρ occupancy, the one
combination the literature sweep found unclaimed — and the Muon stability-law
measurement (finding 2's open problem). Total project cloud spend: $9.72.

## Quick start

```bash
uv sync
uv run pytest
bash scripts/launch_local.sh configs/smoke.yaml
```

## Layout

```
configs/      one YAML per experiment, no CLI-arg science
criteria/     pre-registered gates & tolerances (human-authored + audited)
docs/         run plans, runbook, literature knowledge base (litreview/)
src/optim/    optimizer interface + zoo (muon, routed, dynmuon, adamuon, normuon)
src/stats/    EMAs, autocorrelation, regime classifier (WP0.5-validated)
src/instrument/  tracked-pair instrumentation, HVP probe, Phase-1 plots
tests/        505 tests: synthetic-signal suite, contracts, invariants
vendor/       airbench, modded-nanogpt, DynMuon submodules @ pinned SHAs
scripts/      run.py, sweep.py, aggregate.py, launch_*, analysis scripts
results/      2,600+ result JSONs (append-only; full provenance: git SHA,
              seed, gpu_type, wall time, cost)
reports/      gate records, measurement reports, paper draft, project-state.md, figures
```

## Seed discipline

Seeds 0–99 are the evaluation set for comparison tables and are resolved by
launch tooling only. All development, debugging, and hyperparameter exploration
uses seeds >= 1000.
