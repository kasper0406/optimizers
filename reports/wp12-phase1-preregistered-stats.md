# Phase-1 pre-registered quantities (descriptive; no pass/fail)

Runs: 20 · phases: [(0, 50), (50, 100), (100, 150), (150, 200)] · GMM k∈{1,2,3}, random_state=0, fit on (log10 SNR, ρ) pooled across runs

## Variant: raw

- β=0.9: BIC prefers ≥2 components in **100%** of 24 (matrix, phase) cells
- β=0.99: BIC prefers ≥2 components in **100%** of 24 (matrix, phase) cells

| β | phase | snapshots | frac ρ<−0.2 | distinct dirs ρ<−0.2 |
|---|---|---|---|---|
| 0.9 | 1 | 38891 | 0.682 | 640 |
| 0.9 | 2 | 38400 | 0.804 | 640 |
| 0.9 | 3 | 38400 | 0.676 | 640 |
| 0.9 | 4 | 38400 | 0.382 | 640 |
| 0.99 | 1 | 38848 | 0.598 | 640 |
| 0.99 | 2 | 38400 | 0.886 | 640 |
| 0.99 | 3 | 38400 | 0.768 | 640 |
| 0.99 | 4 | 38400 | 0.458 | 640 |

## Variant: burn_in_10

- β=0.9: BIC prefers ≥2 components in **100%** of 24 (matrix, phase) cells
- β=0.99: BIC prefers ≥2 components in **100%** of 24 (matrix, phase) cells

| β | phase | snapshots | frac ρ<−0.2 | distinct dirs ρ<−0.2 |
|---|---|---|---|---|
| 0.9 | 1 | 34555 | 0.701 | 640 |
| 0.9 | 2 | 35793 | 0.799 | 640 |
| 0.9 | 3 | 35838 | 0.665 | 640 |
| 0.9 | 4 | 35841 | 0.365 | 640 |
| 0.99 | 1 | 34555 | 0.599 | 639 |
| 0.99 | 2 | 35793 | 0.886 | 640 |
| 0.99 | 3 | 35838 | 0.764 | 640 |
| 0.99 | 4 | 35841 | 0.444 | 639 |

