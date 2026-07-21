# Stability frontier (lr × batch) — tables (descriptive)

Runs: 80 · pre-registration: `reports/stability-frontier-preregistration.md` · no verdict here

## Accuracy vs lr rung per batch size

| B | 0.12 | 0.24 | 0.36 | 0.48 | 0.6 | 0.72 | 0.96 | 1.44 | ref | floor | shoulder | recoveries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 94.06 | 93.36 | 92.6 | 92.06 | 91.45 | 90.82 | 89.65 | 88.74 | 94.06 | 93.06 | 0.24 | — |
| 1000 | 93.9 | 93.94 | 93.47 | 92.82 | 92.39 | 91.85 | 91.14 | 89.69 | 93.94 | 92.94 | 0.36 | — |
| 2000 | 93.56 | 94.14 | 93.72 | 93.47 | 92.98 | 92.68 | 91.94 | 90.16 | 94.14 | 93.14 | 0.48 | — |
| 4000 | 91.46 | 92.74 | 92.81 | 92.7 | 92.53 | 92.11 | 91.06 | 88.98 | 92.74 | 91.74 | — | 0.24,0.36,0.48,0.6,0.72 |
| 8000 | 63.38 | 70.58 | 71.56 | 72.69 | 69.95 | 68.27 | 67.25 | 57.94 | 70.58 | 69.58 | — | 0.24,0.36,0.48,0.6 |

## P1 — shoulder scaling

alpha = 0.5 over 3 batch sizes (quantization half-width ±0.5; seed bootstrap median 0.5, CI95 [0.5, 1])

Pre-registered regions: noise-governed alpha ≥ 0.25 · deterministic |alpha| < 0.1 · between = ambiguous.

## P2 — invariant candidates (at-shoulder vs at-1× control)

### At the per-B shoulder rung

| B | occupancy | spectral | euclidean | hvp_q90 |
|---|---|---|---|---|
| 500 | 0.4155 | 127.1 | 1.37 | 0.6159 |
| 1000 | 0.5745 | 195.5 | 2.369 | 1.571 |
| 2000 | 0.6937 | 298.4 | 3.665 | 2.978 |

### At the fixed 1× rung (control)

| B | occupancy | spectral | euclidean | hvp_q90 |
|---|---|---|---|---|
| 500 | 0.4155 | 127.1 | 1.37 | 0.6159 |
| 1000 | 0.514 | 166.6 | 1.961 | 1.294 |
| 2000 | 0.6464 | 207.8 | 2.787 | 2.199 |
| 4000 | 0.7368 | 203.8 | 2.606 | 3.849 |

### Across-B max/min ratios

| candidate | at shoulder | at 1× | tracking signature (<1.5 and ≥1.5) |
|---|---|---|---|
| occupancy | 1.67 | 1.773 | False |
| spectral | 2.348 | 1.635 | False |
| euclidean | 2.675 | 2.034 | False |
| hvp_q90 | 4.836 | 6.249 | False |

## P3 — graceful degradation

| B | min mean acc across rungs |
|---|---|
| 500 | 88.74 |
| 1000 | 89.69 |
| 2000 | 90.16 |
| 4000 | 88.98 |
| 8000 | 57.94 |

Reads are descriptive; region calls and any interpretation live in
`reports/stability-frontier.md`.

