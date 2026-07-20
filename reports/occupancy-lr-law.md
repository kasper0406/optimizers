# Occupancy-LR law (descriptive; no pass/fail)

Is the negative-autocorrelation occupancy — the fraction of tracked
direction-snapshots with lag-1 rho < -0.2 — a *state function* of the
instantaneous learning rate? Computed from the existing airbench
instrumented sidecars only (zero new compute) by
`scripts/analyze_occupancy_lr.py`; numbers below are generated from
the same computation that writes `occupancy-lr-law.json`.

## What was computed

- Per run, per beta, per snapshot step t (t in {1, 5, ..., 200}):
  occupancy(t) = fraction of direction-snapshots with rho < -0.2,
  pooled over all 6 tracked matrices x 32 directions, with the same
  snapshot filters as `scripts/analyze_phase1.py` (var > 0, rho not
  None), in two variants: raw and burn-in (n_since_reset >= 10).
  Steps with < 20 surviving
  direction-snapshots are dropped.
- Instantaneous LR: the airbench filter groups anneal linearly
  (`src/optim/airbench_zoo.py`: `lr = initial_lr * (1 - step/200)`,
  set with the 0-based loop step before `optimizer.step()`; the hub's
  sidecar step counter is 1-based), so a snapshot at sidecar step s
  was produced by an update at lr(s) = lr0 * (1 - (s-1)/200).
  The whiten-bias group of optimizer1 has its own shorter schedule;
  all tracked matrices are filter matrices, so only the filter LR is
  used. lr0 is read per run from metrics.optimizer_lr.
- Config identity is derived per run from the sidecar's sibling
  results JSON (probe_overrides / optimizer_lr / sampling / recipe),
  never from a seed listing.

## Data

- **baseline (lr0 0.24)**: 20 run(s), seeds [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019]
- **lr0 0.12 (x0.5)**: 2 run(s), seeds [1302, 1303]
- **lr0 0.06 (x0.25)**: 2 run(s), seeds [1304, 1305]
- **momentum 0 (lr0 0.24)**: 2 run(s), seeds [1300, 1301]
- **with-replacement (lr0 0.24)**: 3 run(s), seeds [1100, 1101, 1102]
- **HVP on / compile off (lr0 0.24)**: 3 run(s), seeds [1200, 1201, 1202]

## Collapse statistics

R^2 of a single monotone (isotonic, non-decreasing in lr) curve
fitted to the pooled (lr, occupancy) points of ALL runs and configs;
decile decomposition: between-bin share = variance explained by the
lr decile alone; the within-bin remainder splits into a between-config
part (configs differing at matched lr) and a residual. The last two
columns repeat the decomposition binned on training step t instead
of lr — the path-dependence counterpart (within a run t and lr are
the same variable; across the lr0 ladder they differ).

| variant | beta | n points | R^2 isotonic (pooled) | R^2 isotonic (baseline only) | R^2 between lr-deciles | config share of within-lr-bin var | R^2 between t-deciles | config share of within-t-bin var |
|---|---|---|---|---|---|---|---|---|
| raw | 0.9 | 1309 | 0.748 | 0.770 | 0.762 | 0.251 | 0.774 | 0.244 |
| raw | 0.99 | 1307 | 0.608 | 0.596 | 0.727 | 0.254 | 0.729 | 0.333 |
| burn_in_10 | 0.9 | 1248 | 0.765 | 0.796 | 0.804 | 0.322 | 0.843 | 0.367 |
| burn_in_10 | 0.99 | 1248 | 0.618 | 0.601 | 0.837 | 0.469 | 0.809 | 0.496 |

Note the isotonic R^2 is depressed relative to the (unconstrained)
decile R^2 wherever the pooled relationship is non-monotone: the
highest lr values only ever occur in the first ~30 steps of a run,
where occupancy is still climbing out of the start-of-training
transient, so the top of the lr range carries a dip that a monotone
fit cannot represent (see the decile table below and the collapse
figure).

### Occupancy by lr decile and config (burn_in_10)

Per-decile mean occupancy per config — the collapse test in table
form: a state function of lr means matching columns within a row.

**beta 0.9** (columns: mean occupancy; n in JSON):

| lr bin | baseline (lr0 0.24) | lr0 0.12 (x0.5) | lr0 0.06 (x0.25) | momentum 0 (lr0 0.24) | with-replacement (lr0 0.24) | HVP on / compile off (lr0 0.24) |
|---|---|---|---|---|---|---|
| 0.0003–0.0192 | 0.280 | 0.234 | 0.280 | 0.306 | 0.256 | 0.261 |
| 0.0192–0.0372 | 0.372 | 0.370 | 0.531 | 0.433 | 0.356 | 0.369 |
| 0.0372–0.0552 | 0.442 | 0.509 | 0.746 | 0.504 | 0.443 | 0.455 |
| 0.0552–0.0792 | 0.557 | 0.678 | 0.475 | 0.629 | 0.546 | 0.536 |
| 0.0792–0.1029 | 0.685 | 0.801 | — | 0.749 | 0.671 | 0.679 |
| 0.1029–0.1272 | 0.685 | 0.589 | — | 0.712 | 0.662 | 0.680 |
| 0.1272–0.1512 | 0.790 | — | — | 0.809 | 0.789 | 0.794 |
| 0.1512–0.1812 | 0.812 | — | — | 0.855 | 0.799 | 0.790 |
| 0.1812–0.2052 | 0.817 | — | — | 0.855 | 0.830 | 0.817 |
| 0.2052–0.2292 | 0.609 | — | — | 0.640 | 0.643 | 0.631 |

**beta 0.99** (columns: mean occupancy; n in JSON):

| lr bin | baseline (lr0 0.24) | lr0 0.12 (x0.5) | lr0 0.06 (x0.25) | momentum 0 (lr0 0.24) | with-replacement (lr0 0.24) | HVP on / compile off (lr0 0.24) |
|---|---|---|---|---|---|---|
| 0.0003–0.0192 | 0.382 | 0.237 | 0.273 | 0.459 | 0.368 | 0.381 |
| 0.0192–0.0372 | 0.436 | 0.456 | 0.626 | 0.540 | 0.442 | 0.457 |
| 0.0372–0.0552 | 0.499 | 0.606 | 0.678 | 0.612 | 0.529 | 0.514 |
| 0.0552–0.0792 | 0.688 | 0.830 | 0.414 | 0.804 | 0.663 | 0.697 |
| 0.0792–0.1029 | 0.770 | 0.707 | — | 0.848 | 0.730 | 0.762 |
| 0.1029–0.1272 | 0.816 | 0.537 | — | 0.865 | 0.800 | 0.802 |
| 0.1272–0.1512 | 0.889 | — | — | 0.909 | 0.873 | 0.886 |
| 0.1512–0.1812 | 0.886 | — | — | 0.926 | 0.876 | 0.880 |
| 0.1812–0.2052 | 0.654 | — | — | 0.675 | 0.671 | 0.656 |
| 0.2052–0.2292 | 0.555 | — | — | 0.585 | 0.576 | 0.566 |


## Matched-lr cross-config comparison (the confound-breaking statistic)

Within a run, lr(t) and training progress t are perfectly confounded;
only the cross-config comparison at matched instantaneous lr breaks
this. For every non-baseline point, Delta = occupancy minus the mean
of the 20 baseline trajectories linearly interpolated at the SAME lr.

| variant | beta | config | n points | mean Delta | mean abs Delta | sd over run means |
|---|---|---|---|---|---|---|
| raw | 0.9 | lr0 0.12 (x0.5) | 80 | 0.020 | 0.097 | 0.023 |
| raw | 0.9 | lr0 0.06 (x0.25) | 80 | 0.138 | 0.179 | 0.000 |
| raw | 0.9 | momentum 0 (lr0 0.24) | 80 | 0.047 | 0.048 | 0.006 |
| raw | 0.9 | with-replacement (lr0 0.24) | 120 | -0.001 | 0.028 | 0.005 |
| raw | 0.9 | HVP on / compile off (lr0 0.24) | 120 | -0.001 | 0.026 | 0.005 |
| raw | 0.99 | lr0 0.12 (x0.5) | 80 | -0.036 | 0.135 | 0.028 |
| raw | 0.99 | lr0 0.06 (x0.25) | 79 | 0.059 | 0.172 | 0.007 |
| raw | 0.99 | momentum 0 (lr0 0.24) | 80 | 0.061 | 0.061 | 0.004 |
| raw | 0.99 | with-replacement (lr0 0.24) | 120 | -0.002 | 0.025 | 0.007 |
| raw | 0.99 | HVP on / compile off (lr0 0.24) | 120 | 0.002 | 0.019 | 0.009 |
| burn_in_10 | 0.9 | lr0 0.12 (x0.5) | 76 | 0.030 | 0.080 | 0.022 |
| burn_in_10 | 0.9 | lr0 0.06 (x0.25) | 76 | 0.145 | 0.174 | 0.002 |
| burn_in_10 | 0.9 | momentum 0 (lr0 0.24) | 78 | 0.043 | 0.046 | 0.007 |
| burn_in_10 | 0.9 | with-replacement (lr0 0.24) | 117 | -0.004 | 0.028 | 0.005 |
| burn_in_10 | 0.9 | HVP on / compile off (lr0 0.24) | 117 | -0.004 | 0.028 | 0.005 |
| burn_in_10 | 0.99 | lr0 0.12 (x0.5) | 76 | -0.027 | 0.127 | 0.027 |
| burn_in_10 | 0.99 | lr0 0.06 (x0.25) | 76 | 0.066 | 0.162 | 0.008 |
| burn_in_10 | 0.99 | momentum 0 (lr0 0.24) | 78 | 0.061 | 0.062 | 0.006 |
| burn_in_10 | 0.99 | with-replacement (lr0 0.24) | 117 | -0.005 | 0.024 | 0.006 |
| burn_in_10 | 0.99 | HVP on / compile off (lr0 0.24) | 117 | 0.002 | 0.020 | 0.008 |

## Early-vs-late at the same lr (path-dependence check)

Points with lr in [0.75 lr0_probe, lr0_probe] occur EARLY in the
probe runs (first ~50 steps) but LATE in the baseline runs (which
only anneal down to that lr after 100+/150+ steps). If occupancy were
a pure state function of lr, the two groups would match.

| variant | beta | probe | lr window | probe occ (mean t) | baseline occ (mean t) | probe - baseline |
|---|---|---|---|---|---|---|
| raw | 0.9 | lr0 0.12 (x0.5) | [0.090, 0.120] | 0.710 (t≈25) | 0.711 (t≈115) | -0.002 |
| raw | 0.9 | lr0 0.06 (x0.25) | [0.045, 0.060] | 0.713 (t≈25) | 0.502 (t≈158) | 0.211 |
| raw | 0.99 | lr0 0.12 (x0.5) | [0.090, 0.120] | 0.607 (t≈25) | 0.794 (t≈115) | -0.187 |
| raw | 0.99 | lr0 0.06 (x0.25) | [0.045, 0.060] | 0.576 (t≈26) | 0.577 (t≈158) | -0.001 |
| burn_in_10 | 0.9 | lr0 0.12 (x0.5) | [0.086, 0.115] | 0.718 (t≈32) | 0.705 (t≈118) | 0.012 |
| burn_in_10 | 0.9 | lr0 0.06 (x0.25) | [0.043, 0.057] | 0.713 (t≈32) | 0.427 (t≈160) | 0.286 |
| burn_in_10 | 0.99 | lr0 0.12 (x0.5) | [0.086, 0.115] | 0.611 (t≈32) | 0.788 (t≈118) | -0.177 |
| burn_in_10 | 0.99 | lr0 0.06 (x0.25) | [0.043, 0.057] | 0.591 (t≈32) | 0.529 (t≈160) | 0.062 |

## Config offsets vs the baseline curve

Mean residual of each config's points against the baseline-only
isotonic curve f(lr) (positive = above the baseline curve).

| variant | beta | config | n points | mean residual | sd over run means |
|---|---|---|---|---|---|
| raw | 0.9 | baseline (lr0 0.24) | 817 | -0.000 | 0.013 |
| raw | 0.9 | lr0 0.12 (x0.5) | 82 | 0.019 | 0.022 |
| raw | 0.9 | lr0 0.06 (x0.25) | 82 | 0.134 | 0.001 |
| raw | 0.9 | momentum 0 (lr0 0.24) | 82 | 0.046 | 0.006 |
| raw | 0.9 | with-replacement (lr0 0.24) | 123 | -0.001 | 0.005 |
| raw | 0.9 | HVP on / compile off (lr0 0.24) | 123 | -0.001 | 0.005 |
| raw | 0.99 | baseline (lr0 0.24) | 816 | 0.000 | 0.016 |
| raw | 0.99 | lr0 0.12 (x0.5) | 82 | -0.024 | 0.028 |
| raw | 0.99 | lr0 0.06 (x0.25) | 81 | 0.052 | 0.007 |
| raw | 0.99 | momentum 0 (lr0 0.24) | 82 | 0.060 | 0.004 |
| raw | 0.99 | with-replacement (lr0 0.24) | 123 | -0.002 | 0.006 |
| raw | 0.99 | HVP on / compile off (lr0 0.24) | 123 | 0.002 | 0.009 |
| burn_in_10 | 0.9 | baseline (lr0 0.24) | 780 | 0.000 | 0.014 |
| burn_in_10 | 0.9 | lr0 0.12 (x0.5) | 78 | 0.030 | 0.022 |
| burn_in_10 | 0.9 | lr0 0.06 (x0.25) | 78 | 0.141 | 0.003 |
| burn_in_10 | 0.9 | momentum 0 (lr0 0.24) | 78 | 0.043 | 0.007 |
| burn_in_10 | 0.9 | with-replacement (lr0 0.24) | 117 | -0.004 | 0.005 |
| burn_in_10 | 0.9 | HVP on / compile off (lr0 0.24) | 117 | -0.004 | 0.005 |
| burn_in_10 | 0.99 | baseline (lr0 0.24) | 780 | -0.000 | 0.017 |
| burn_in_10 | 0.99 | lr0 0.12 (x0.5) | 78 | -0.023 | 0.028 |
| burn_in_10 | 0.99 | lr0 0.06 (x0.25) | 78 | 0.059 | 0.009 |
| burn_in_10 | 0.99 | momentum 0 (lr0 0.24) | 78 | 0.061 | 0.006 |
| burn_in_10 | 0.99 | with-replacement (lr0 0.24) | 117 | -0.005 | 0.006 |
| burn_in_10 | 0.99 | HVP on / compile off (lr0 0.24) | 117 | 0.002 | 0.008 |

## Baseline occupancy-vs-lr curve (would-be controller calibration)

The baseline isotonic curve sampled on a fixed lr grid, and at the
record schedule's phase midpoints (t = 25/75/125/175). If occupancy
is treated as a controlled variable, this curve is the calibration
between an occupancy setpoint and the LR that produces it — subject
to every caveat below. (In the raw variant the value at lr = 0.24
reflects only the degenerate t = 1 snapshot — lr0 occurs exactly
once per run, at the first step, where the EMA statistics are
essentially unformed; use the burn_in_10 rows.)

**raw, beta 0.9** — f(lr) on the grid:

| lr | 0.240 | 0.210 | 0.180 | 0.150 | 0.120 | 0.090 | 0.060 | 0.030 | 0.012 | 0.001 |
|---|---|---|---|---|---|---|---|---|---|---|
| occupancy | 1.000 | 0.741 | 0.741 | 0.741 | 0.740 | 0.685 | 0.580 | 0.402 | 0.307 | 0.250 |

Phase midpoints: t=25 (lr 0.211): 0.741, t=75 (lr 0.151): 0.741, t=125 (lr 0.091): 0.685, t=175 (lr 0.031): 0.408

**raw, beta 0.99** — f(lr) on the grid:

| lr | 0.240 | 0.210 | 0.180 | 0.150 | 0.120 | 0.090 | 0.060 | 0.030 | 0.012 | 0.001 |
|---|---|---|---|---|---|---|---|---|---|---|
| occupancy | 1.000 | 0.751 | 0.751 | 0.751 | 0.751 | 0.751 | 0.697 | 0.451 | 0.398 | 0.362 |

Phase midpoints: t=25 (lr 0.211): 0.751, t=75 (lr 0.151): 0.751, t=125 (lr 0.091): 0.751, t=175 (lr 0.031): 0.455

**burn_in_10, beta 0.9** — f(lr) on the grid:

| lr | 0.240 | 0.210 | 0.180 | 0.150 | 0.120 | 0.090 | 0.060 | 0.030 | 0.012 | 0.001 |
|---|---|---|---|---|---|---|---|---|---|---|
| occupancy | 0.752 | 0.752 | 0.752 | 0.752 | 0.737 | 0.677 | 0.558 | 0.402 | 0.307 | 0.250 |

Phase midpoints: t=25 (lr 0.211): 0.752, t=75 (lr 0.151): 0.752, t=125 (lr 0.091): 0.677, t=175 (lr 0.031): 0.408

**burn_in_10, beta 0.99** — f(lr) on the grid:

| lr | 0.240 | 0.210 | 0.180 | 0.150 | 0.120 | 0.090 | 0.060 | 0.030 | 0.012 | 0.001 |
|---|---|---|---|---|---|---|---|---|---|---|
| occupancy | 0.759 | 0.759 | 0.759 | 0.759 | 0.759 | 0.759 | 0.685 | 0.451 | 0.398 | 0.362 |

Phase midpoints: t=25 (lr 0.211): 0.759, t=75 (lr 0.151): 0.759, t=125 (lr 0.091): 0.759, t=175 (lr 0.031): 0.455

## Figures

`reports/figures/occupancy_lr/`: `collapse_<variant>.png` (occupancy
vs lr, all runs, colored by config, baseline isotonic fit dashed) and
`residuals_vs_t_<variant>.png` (residuals against the baseline curve
vs training step, i.e. the path-dependence view).

## Reading (descriptive; refers to the 32-run 2026-07-19 sidecar set)

- Configs sharing the record LR trajectory (with-replacement,
  HVP/compile-off) sit on the baseline curve in every decile — the
  collapse is exact when the lr *path* is identical, so the
  measurement itself is stable.
- momentum=0 runs a small, roughly constant amount above the curve
  at all lr (the level shift already reported in
  `wp22-mechanism-probes.md`), i.e. approximately a parallel curve,
  not a reshaped one.
- The lr0-ladder runs are NOT on the baseline curve at matched lr:
  each ladder run reproduces the baseline's *shape in scheduled
  time* (early dip near its own lr0, mid-run hump, anneal-tail
  collapse) shifted to its own lr scale. They sit above the
  baseline at lrs just below their lr0 (their own hump vs the
  baseline's anneal) and below it near their lr0 (their own early
  transient vs the baseline's hump). At the very bottom of the
  anneal (lr < ~0.02) the configs reconverge at beta 0.9, while at
  beta 0.99 the ladder runs remain ~0.10-0.15 below the baseline
  (the slow EMA carries their lower recent history into the tail).
- Binning on training step t explains the pooled cloud about as
  well as binning on lr, and neither dominates (collapse table).
  This is expected: 26 of the 32 runs share lr0 = 0.24, so t and
  lr are collinear in most of the pooled data and pooled R^2
  cannot separate them. The separation lives entirely in the
  ladder runs' matched-lr deviations (above) together with the
  matched-phase lr0 dependence already shown in
  `wp22-mechanism-probes.md`: each is nonzero, so the data are
  consistent with occupancy depending on BOTH the instantaneous lr
  level and the position in the schedule (equivalently lr/lr0),
  not on instantaneous lr alone. With n=2 ladder runs per rung
  this is a description, not an estimate of the two contributions.
- For a setpoint controller this means the baseline curve below is
  a calibration of occupancy against the *record schedule*, valid
  along that trajectory; it is not a schedule-free lr-occupancy
  law, and transferring it to a different schedule shape is not
  supported by this data.

## Caveats (read before using any number above)

- **Within-run confound.** Inside any single run, lr(t) and training
  progress t are perfectly confounded; the within-run collapse can
  never distinguish 'occupancy tracks lr' from 'occupancy tracks
  time'. Only the lr0-ladder configs (0.12, 0.06) break this, and
  they have n = 2 runs each — the matched-lr and early-vs-late
  tables are the load-bearing statistics, at probe-side n of 2.
- **What CAN be concluded** is limited to: whether the 2+2 ladder
  runs' (lr, occupancy) points lie on / off the 20-run baseline
  curve at overlapping lr, plus the analogous check for momentum=0,
  with-replacement and HVP/compile-off at lr0 = 0.24. What CANNOT:
  any claim about schedules not in the data (e.g. LR increases,
  constant-LR long runs), any per-direction claim (occupancy is a
  population fraction), or causality of lr vs co-annealed dynamics.
- **Estimator memory.** rho is an EMA-lag-1 statistic; at beta 0.99
  the effective window (~100 steps) spans a large stretch of the
  anneal, so 'instantaneous' lr attribution is smeared — beta 0.9
  (~10-step window) is the cleaner state-function probe; treat the
  beta 0.99 columns as slow-averaged.
- **Snapshot cadence and resets.** Occupancy is sampled every 5
  steps; subspace refreshes at t = 50/100/150 reset the EMAs and
  transiently depress negative-rho fractions (the burn_in_10 variant
  drops n_since_reset < 10 snapshots and is the cleaner one; raw is
  reported for sensitivity). Early-t points also carry re-tracked
  directions whose identity changed at refresh.
- **Single task / arch / scale.** airbench94 (CIFAR-10, 200 steps,
  batch 2000) only; nothing here says nanogpt or any other scale
  behaves the same.
- **Occupancy definition.** frac(rho < -0.2) over the tracked
  subspace (top-16 + 16 bulk probes per matrix); the threshold is
  the pre-registered -0.2, not tuned here; top and bulk directions
  are pooled (they differ in level, see wp22-mechanism-probes.md).
- **n = 2-3 per probe config**; run-to-run sd columns are computed
  over 2-3 run means and are indicative only. No inferential claims
  are made anywhere in this file.

