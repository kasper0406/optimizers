# Central-flow Muon v0 — does the explicit term stand in for the high-LR oscillation? (`airbench_centralflow`)

Descriptive output of `scripts/analyze_centralflow.py` (docs/litreview/j-theory-theorem-sweep.md §6, program item 2). No pass/fail judgment is made here.

Seed-paired 2x2: n=10 dev seeds (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009), GPU NVIDIA L40. Every seed is present in all four arms.

| arm | LR scale | CF term | final val | final TTA | |
| --- | --- | --- | --- | --- | --- |
| A | 1.00 | off | 0.9313 | 0.9403 | stock LR, term off (reference) |
| B | 0.25 | off | 0.9118 | 0.9208 | cold LR (0.25x), term off |
| C | 0.25 | on | 0.9108 | 0.9199 | cold LR (0.25x) + CF term (mechanism arm) |
| D | 1.00 | on | 0.1000 | 0.1000 | stock LR + CF term |

## Paired differences (per-seed, mean ± 95% CI)

| difference | val | TTA | reading |
| --- | --- | --- | --- |
| C − B | -0.0010 ± 0.0017 | -0.0010 ± 0.0021 | mechanism effect (term at cold LR) |
| C − A | -0.0205 ± 0.0014 | -0.0204 ± 0.0014 | recovery gap vs stock |
| B − A | -0.0195 ± 0.0011 | -0.0195 ± 0.0020 | cost of the cold LR |
| D − A | -0.8313 ± 0.0006 | -0.8403 ± 0.0009 | term on top of stock LR |

recovery_fraction = mean(C − B) / mean(A − B) = -0.051 (-5.1% of the accuracy the cold LR gave up is recovered by the explicit term).

## Central-flow term telemetry (term-on arms)

- Arm C (cold LR (0.25x) + CF term (mechanism arm)): penalty grad norm 10.410259 (first logged step) → 0.014714 (last); mid-run mean curvature 23.373314; 10/10 seeds logged a timeseries.
- Arm D (stock LR + CF term): penalty grad norm 56.394515 (first logged step) → 13793.425137 (last); mid-run mean curvature -0.092644; 10/10 seeds logged a timeseries.
