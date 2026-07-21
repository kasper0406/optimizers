# nanogpt local baseline — own-harness sigma (descriptive)

Config tag: `nanogpt_local_baseline` · n = 10 seeds 1710–1719 · SHA(s): ecd48f1d586a7f37fc074ed08d4fb0fe84eda135

**final val loss = 3.28888 ± 0.001249 (sd)**, SE(mean) 0.000395, sd CI95 [0.000859, 0.00228]

steps-to-target(3.28) censored: 10/10 — final-val-at-1750-steps is the uncensored endpoint on this harness.

## Per-seed finals

| seed | final val |
|---|---|
| 1710 | 3.28762 |
| 1711 | 3.28789 |
| 1712 | 3.28949 |
| 1713 | 3.28815 |
| 1714 | 3.29138 |
| 1715 | 3.28835 |
| 1716 | 3.28768 |
| 1717 | 3.28858 |
| 1718 | 3.29041 |
| 1719 | 3.28929 |

## Across-seed sd by checkpoint

| step | mean | sd |
|---|---|---|
| 0 | 10.82584 | 0.0 |
| 125 | 4.63733 | 0.007035 |
| 250 | 4.09952 | 0.00778 |
| 375 | 3.89997 | 0.006014 |
| 500 | 3.75326 | 0.004529 |
| 625 | 3.66496 | 0.002468 |
| 750 | 3.60197 | 0.003298 |
| 875 | 3.55481 | 0.002228 |
| 1000 | 3.51314 | 0.001817 |
| 1125 | 3.46094 | 0.001584 |
| 1250 | 3.41489 | 0.001277 |
| 1375 | 3.37519 | 0.001394 |
| 1500 | 3.33871 | 0.001269 |
| 1625 | 3.30939 | 0.001221 |
| 1750 | 3.28888 | 0.001249 |

End-of-run slope: 0.00016405 loss/step (mean of last-two-checkpoint differences).

## Power (80% power, alpha .05 two-sided, unpaired)

| loss effect | ~steps equiv | seeds/arm |
|---|---|---|
| 0.001 | 6.1 | 25 |
| 0.00125 | 7.6 | 16 |
| 0.0025 | 15.2 | 4 |
| 0.005 | 30.5 | 2 |
| 0.01 | 61.0 | 2 |

Descriptive only; testbed interpretation lives in `reports/nanogpt-local-baseline.md`.

