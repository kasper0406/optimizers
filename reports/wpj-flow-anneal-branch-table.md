# Anneal dissection — accuracy vs anneal length (`airbench_anneal_branch`)

Descriptive output of `scripts/analyze_anneal_branch.py` (docs/litreview/j-theory-theorem-sweep.md §6, program item 1). No pass/fail judgment is made here.

Seed-paired runs: n=21 dev seeds, GPU NVIDIA L40; stock budget 200 steps.

Baseline = each seed's OWN stock linear-schedule final (`airbench_ema`, lr_schedule=linear): val 0.9317 (TTA 0.9399).

Constant-arm final at the full budget (no anneal at all): val 0.6733 (TTA 0.6937).

k* rule: smallest anneal length k whose paired mean val delta is >= -0.0020.

## Branch step 100 (pre-anneal base val 0.6543)

| k | val delta vs stock final | TTA delta vs stock final |
| --- | --- | --- |
| 0 | -0.2775 ± 0.0290 | -0.2672 ± 0.0319 |
| 5 | -0.0657 ± 0.0021 | -0.0635 ± 0.0018 |
| 10 | -0.0440 ± 0.0014 | -0.0425 ± 0.0012 |
| 25 | -0.0257 ± 0.0013 | -0.0248 ± 0.0010 |
| 50 | -0.0141 ± 0.0011 | -0.0135 ± 0.0009 |

k* = none — no tested anneal length reaches within 0.0020 of the stock final at branch step 100.

## Branch step 150 (pre-anneal base val 0.6544)

| k | val delta vs stock final | TTA delta vs stock final |
| --- | --- | --- |
| 0 | -0.2773 ± 0.0388 | -0.2663 ± 0.0408 |
| 5 | -0.0591 ± 0.0021 | -0.0565 ± 0.0022 |
| 10 | -0.0411 ± 0.0020 | -0.0390 ± 0.0014 |
| 25 | -0.0240 ± 0.0012 | -0.0224 ± 0.0008 |
| 50 | -0.0113 ± 0.0010 | -0.0120 ± 0.0008 |

k* = none — no tested anneal length reaches within 0.0020 of the stock final at branch step 150.

## Branch step 200 (pre-anneal base val 0.6733)

| k | val delta vs stock final | TTA delta vs stock final |
| --- | --- | --- |
| 0 | -0.2584 ± 0.0336 | -0.2462 ± 0.0353 |
| 5 | -0.0557 ± 0.0023 | -0.0528 ± 0.0017 |
| 10 | -0.0386 ± 0.0015 | -0.0368 ± 0.0012 |
| 25 | -0.0220 ± 0.0011 | -0.0211 ± 0.0007 |
| 50 | -0.0108 ± 0.0012 | -0.0104 ± 0.0009 |

k* = none — no tested anneal length reaches within 0.0020 of the stock final at branch step 200.

