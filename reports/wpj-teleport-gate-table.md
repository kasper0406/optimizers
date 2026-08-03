# Teleportation gate — gradient-size variation along the loss-invariant orbit (`airbench_teleport_gate`)

Descriptive output of `scripts/analyze_teleport_gate.py` (docs/litreview/j-theory-theorem-sweep.md §6, program item 4 / T6). No pass/fail judgment is made here.

Runs: n=6 dev seeds (1000, 1001, 1002, 1003, 1004, 3001), GPU NVIDIA L40; level-set tolerance |rel_dloss| < 0.001.

Ratios are relative to the un-teleported base point at the same training state (1.0000 = no change). `random p50/p90` are the per-seed median / 90th percentile of the drawn nuclear-sum ratios, averaged over seeds; `best random` and `best refined` are mean ± 95% CI over seeds.

| snapshot step | n_feasible (mean) | worst \|rel_dloss\| | random p50 nuc | random p90 nuc | best random nuc | best refined nuc | best refined grad |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 64.0 | 0.000011 | 1.1077 | 1.1246 | 1.1402 ± 0.0063 | 1.2292 ± 0.0105 | 1.3295 ± 0.0179 |
| 100 | 64.0 | 0.000012 | 1.1043 | 1.1215 | 1.1364 ± 0.0056 | 1.2228 ± 0.0137 | 1.3230 ± 0.0139 |
| 150 | 64.0 | 0.000010 | 1.0980 | 1.1146 | 1.1277 ± 0.0036 | 1.2140 ± 0.0140 | 1.3126 ± 0.0207 |

Smallest best-refined nuclear-sum ratio over snapshot steps (the most conservative training state): 1.2140 (+21.40% vs the base point).
