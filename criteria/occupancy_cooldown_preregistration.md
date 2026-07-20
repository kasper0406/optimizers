# Pre-registration — Occupancy-Triggered Cooldown (post-Gate-2 direction)

Authored: agent under explicit full delegation (Kasper Nielsen, 2026-07-19:
"Please just drive by yourself — no human confirmations needed"; direction
approved 2026-07-20). Required by `reports/gate2-decision.md` before any spend
on a new direction. Written BEFORE any run of this protocol.

## Motivation (what the completed work established)

1. **Update-direction shaping is inert on our workload; update *scale* is not.**
   Nine per-direction interventions landed within 0.03pp (Gate 2), while the
   same harness moves 0.5–3.5pp across the LR ladder. Twin probes show the
   interventions do move training — to unrelated-but-equivalent weights
   (`reports/brainstorm-programs.md` #1).
2. **Oscillation occupancy is a schedule-position indicator, not an LR state
   function** (`reports/occupancy-lr-law.md`): it rises early, peaks
   mid-training, collapses in the anneal, and reproduces that shape relative to
   each run's own peak LR. It is therefore a candidate *progress* signal, not a
   control setpoint — the constant-occupancy hypothesis is already refuted.
3. Literature (`docs/litreview/f-schedules-as-control.md`): nobody derives
   schedule *timing* from a measured temporal-statistic; the WSD cooldown-start
   fraction is a hand-set hyperparameter that must be chosen before the budget
   is known.

## Hypothesis

The occupancy trajectory carries information about when a run has exhausted the
stable phase, sufficient to trigger the WSD cooldown automatically — matching a
tuned fixed-fraction cooldown without knowing the token budget in advance.

## Pre-registered predictions (falsifiable, in order of strength)

- **P1 (primary, matching):** at a fixed token budget, occupancy-triggered
  cooldown reaches the target val loss in a number of steps within 1% of the
  best fixed-fraction cooldown (itself tuned over cooldown fractions on dev
  seeds). Equivalence, not superiority, is the claim.
- **P2 (transfer, the actual value):** a single occupancy trigger rule, with
  parameters fixed at one token budget, matches the *separately tuned* fixed
  fraction at 0.5× and 2× that budget — i.e. it removes the hyperparameter.
  Failure of P2 with P1 holding means the trigger is a re-parameterization of
  the fraction, not an improvement.
- **P3 (null to exclude):** the trigger must beat a "trigger at a random step
  drawn from the same distribution as the triggered steps" placebo — otherwise
  any apparent match is timing luck.

## What counts as refutation

P1 failing by >1% at the tuning budget, or P2 failing at either transfer point
while a fixed fraction succeeds, refutes the direction and it is reported as a
negative result. No re-parameterization of the trigger after seeing transfer
results (the trigger rule is frozen after the tuning budget).

## Protocol

- **Testbed:** modded-nanogpt 124M (pinned record 2025-07-12_BosAlign port,
  WP0.2), metric = steps to val loss 3.28, per `docs/litreview/i-benchmark-headroom.md`.
  Airbench is disqualified for method claims by our own equivalence result.
- **Instrumentation:** existing tracked-pair machinery at its validated
  settings; occupancy = fraction of tracked directions with lag-1 ρ < −0.2 at
  β=0.9 (the cleaner probe per `reports/occupancy-lr-law.md`), computed over
  the same tier configuration as WP1.2.
- **Trigger rule (frozen before runs):** cooldown starts at the first step
  where the smoothed occupancy has fallen a fraction δ below its running
  maximum, with δ and the smoothing window fixed at the tuning budget only.
- **Seeds:** dev seeds (≥1000) for all tuning and trigger fixing; eval seeds
  0–99 only for a final comparison table, if the direction survives P1–P3.
- **Baselines (mandatory, from `docs/litreview/f-schedules-as-control.md` and
  `b-layer-temporal-trust-ratio.md`):** tuned fixed-fraction WSD cooldown; the
  record's own schedule; and the P3 random-timing placebo. GALA/Prodigy-style
  adaptive-LR baselines are out of scope for a *timing* claim and are named as
  such rather than run.
- **Power:** the record's in-repo n=20 seed distribution (mean 3.2791, std
  0.0013) is the variance basis; the required n for a 1% steps effect is
  computed from it BEFORE the comparison runs and recorded here as an
  amendment.

## Budget

Hard cap for this direction: $15 of remaining Hyperstack credit, including the
WP0.2 port validation runs. Exceeding it stops the direction pending review.

## Standing constraints

Seed discipline, append-only results, provenance fields, adversarial review of
any gate-equivalent decision, and honest reporting of refutation all carry over
from the main arc unchanged.
