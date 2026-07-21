# Frontier sharpening (#6b) — tables (descriptive)

Pre-registration: `reports/frontier-sharpening-preregistration.md` · no verdict here

## Part 1 — dense ladders

### B = 1000  (ref 94, floor 93, lr*_cross 0.4754)

| lr | 0.24 | 0.32 | 0.37 | 0.42 | 0.48 | 0.55 | 0.64 |
|---|---|---|---|---|---|---|---|
| mean acc | 94 | 93.74 | 93.43 | 93.23 | 92.98 | 92.56 | 92.33 |
| n | 5 | 5 | 5 | 5 | 5 | 5 | 5 |

### B = 2000  (ref 93.89, floor 92.89, lr*_cross 0.6116)

| lr | 0.24 | 0.42 | 0.48 | 0.55 | 0.64 | 0.73 | 0.84 |
|---|---|---|---|---|---|---|---|
| mean acc | 93.89 | 93.62 | 93.37 | 93.17 | 92.77 | 92.49 | 92.16 |
| n | 5 | 5 | 5 | 5 | 5 | 5 | 5 |

### B = 4000  (ref 92.88, floor 91.88, lr*_cross 0.7725)

| lr | 0.36 | 0.55 | 0.64 | 0.73 | 0.84 | 0.97 | 1.11 |
|---|---|---|---|---|---|---|---|
| mean acc | 92.88 | 92.59 | 92.38 | 92.05 | 91.63 | 91.04 | 90.69 |
| n | 5 | 5 | 5 | 5 | 5 | 5 | 5 |

## alpha

alpha = 0.3502 over 3 batch sizes · seed bootstrap median 0.3516, CI95 [0.2967, 0.4247] (2000/2000 draws valid)

Pre-registered regions: noise-governed alpha ≥ 0.25 · deterministic |alpha| < 0.1.

## Part 2 — step-matched B=8000 (epochs 32, ~200 steps)

| lr | 0.12 | 0.24 | 0.36 | 0.48 | 0.6 | 0.72 | 0.96 | 1.44 |
|---|---|---|---|---|---|---|---|---|
| mean acc | 90.64 | 92.15 | 93.28 | 93.69 | 93.75 | 93.69 | 93.32 | 92.31 |

peak-referenced shoulder: 0.96 · interpolated lr*_cross: — · ref 93.75

Pre-registered read: shoulder ≥ 0.72 → undertraining explained the
program-#6 fallback; ≤ 0.48 → the rightward shift saturates by B=8000.

Reads are descriptive; interpretation lives in `reports/frontier-sharpening.md`.

