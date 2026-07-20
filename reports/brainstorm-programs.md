# Brainstorm Programs #1–#5 — Results

Date: 2026-07-20 · Agent under full delegation · All five directions were
literature-reviewed first (`docs/litreview/`, round 2) and each experimental
program carried a pre-registered prediction written into its configs before
any run. **All four testable predictions were refuted.** That is the result.

## #2 — Is oscillation occupancy a state function of learning rate?

**Prediction:** occupancy(t) collapses onto one curve of instantaneous lr, so
schedules could be derived from a constant-occupancy setpoint.
**Refuted.** Zero new compute (32 existing sidecars, `scripts/analyze_occupancy_lr.py`).
The lr-ladder configs break the within-run lr↔time confound and show occupancy
tracks *schedule position* (lr/lr₀) as much as instantaneous lr — lrquarter sits
+0.29 above the baseline curve at matched lr. The record schedule does **not**
hold occupancy constant (0.75 → 0.41 across phases). momentum=0 rides a parallel
offset; with-replacement and HVP configs land exactly on-curve (a measurement-
stability check that passed). Detail: `reports/occupancy-lr-law.md`.
**Consequence:** a setpoint controller needs schedule-relative calibration; the
naive "good schedules hold occupancy constant" story is dead.

## #1 — Does the null come from output-side interventions not compounding?

**Prediction (pre-registered in configs):** state-mode divergence ≥ 10× output-
mode at matched gain; if both are small or both large, the compounding
explanation is refuted.
**Refuted — "both large".** Twin-trajectory probes (`scripts/probe_divergence.py`,
identical init and batch sequence, one process):

| arm | rel_dist @50 | @100 | final | twin accs |
|---|---|---|---|---|
| control: stock vs stock | **0.000** | **0.000** | **0.000** | 94.02 / 94.02 |
| output-side, adaptive | 0.390 | 0.892 | 0.954 | 93.98 / 94.12 |
| state-side, adaptive | 0.391 | 0.896 | 0.952 | 93.96 / 93.92 |
| output-side, g=0.5 | 0.508 | 0.956 | 0.993 | 94.19 / 94.06 |
| state-side, g=0.5 | 0.370 | 0.885 | 0.947 | 93.72 / 93.65 |

Matched-gain ratio state/output = **0.95×** (predicted ≥10×). The control is
exactly zero at every step, so the divergence is entirely attributable to the
intervention and the metric has full discriminating power.
**Reframes the Gate-2 null:** per-direction damping does not fail by being
inert — it drives training to *essentially unrelated weights* (‖ΔW‖/‖W‖ ≈ 1)
that are equally good. The landscape supplies an abundance of equivalent
destinations; steering among them costs 10% wall time and buys nothing.

## #3 — Does spectral directional smoothness equilibrate at c/lr?

**Prediction:** the dimensionless product lr·D_smooth_spectral plateaus at an
lr-invariant constant (the Muon analogue of GD's 2 / Adam's ~38) while the
Euclidean quantity does not.
**Refuted.** Measured along the actual trajectory (9 runs, lr ladder ×0.5/1/2):

| quantity | lr 0.12 | lr 0.24 | lr 0.48 | max/min |
|---|---|---|---|---|
| lr·D_smooth **spectral** | 123.6 | 222.6 | 317.6 | **2.57** |
| lr·D_smooth Euclidean | 1.68 | 2.81 | 3.79 | 2.26 |

Neither norm yields an lr-invariant constant; the spectral quantity is, if
anything, *less* invariant than the Euclidean one. Whatever sets Muon's maximum
stable lr, it is not a plateau of directional smoothness in either norm on this
workload. Caveats (in `reports/smoothness-plateau.md`): minibatch estimate of a
full-batch quantity, sum-reduced loss, per-matrix perturbation, 2 seeds/rung.

## #4 — Is there persistent per-direction signal at long integration?

**Prediction:** along frozen, never-refreshed probes, |t| grows ~√t if
persistent signal exists (drift ~T vs noise ~√T); some probes cross |t| ≥ 4.
**Refuted.** 864 frozen probes, full-run unbounded-window accumulation:
median final |t| = 0.61 (naive) / 0.87 (Newey–West); only 0.1% / 0.9% cross
|t| ≥ 4; growth slope ≈ 0.05 against the 0.5 expected under √t. Removing the
integration-window ceiling did **not** reveal hidden signal — the earlier
"structurally unmeasurable" finding was not merely an instrument artifact.

## #5 — Benchmark headroom

Literature-only (`docs/litreview/i-benchmark-headroom.md`). Our powered
equivalence on a record config is itself novel evidence of saturation. Ranked
testbeds for any future method claim: modded-nanogpt 124M steps-to-target
(1–2% deltas demonstrably resolvable) > EPFL 124M harness (pre-tuned baselines)
> CIFAR *time-to-94%* (sub-$10, but reviewers weight LM evidence higher).
No runs; this is a protocol decision, not an experiment.

## What the four refutations jointly say

The per-direction oscillation program is closed on this workload, and closed
for a specific, now-measured reason: the signal is real but (a) not a state
function of lr, (b) not attached to any invariant stability constant, (c) not
hiding recoverable per-direction signal, and (d) actionable only in the weak
sense that acting on it moves you a long way to somewhere equally good.
Cost of the four programs: ~$2.2 (one A6000-spot session, VM deleted).
