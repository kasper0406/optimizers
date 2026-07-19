# Phase 1 Pre-registration — Cluster-existence criteria

Authored: Kasper Nielsen, 2026-07-19 (content selected by explicit instruction:
research plan §1.3 verbatim, chosen via in-session decision; typeset by agent,
committed by the human). Per CLAUDE.md WP1.2 this file must exist and predate
the timestamp of the first instrumented Phase-1 run.

## Pre-registered criteria (routed-muon-research-plan.md §1.3, verbatim)

> Pre-register (in the repo README, dated) the cluster-existence criterion
> before looking: e.g., GMM/BIC prefers ≥2 components in (SNR, ρ) space in
> ≥70% of (matrix, phase) cells, and the ρ<−0.2 population is non-empty
> during the high-LR phase.

Adopted as binding:

1. **Cluster existence:** GMM/BIC prefers ≥ 2 components in (SNR, ρ) space in
   ≥ 70% of (matrix, phase) cells.
2. **Oscillation population:** the ρ < −0.2 population is non-empty during
   the high-LR phase.

## Scope notes

- Statistics are computed by the WP0.5-validated `src/stats` module from
  instrumented-but-stock Muon runs (zero behavior change), airbench ≥ 20
  seeds; nanogpt seeds follow after the BosAlign record port (WP0.2).
- The agent's WP1.2 report is descriptive only (scatters, GMM/BIC outputs,
  occupancy, ηλ agreement) and makes no pass/fail claim. Evaluation of these
  criteria and the Gate-1 decision (full routing / oscillation-only / stop)
  are human-only.
- Anything this file does not specify (e.g., the exact phase binning) is
  reported descriptively across reasonable choices, never selected post hoc
  to change an outcome (CLAUDE.md ground rule 1).
