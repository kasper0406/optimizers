# Channel audit, Phase B - frozen tier (descriptive)

Output of `scripts/analyze_channel_audit_frozen.py`, the Phase B
producer registered in `reports/channel-audit-preregistration.md` 6b.

**Descriptive only.** Every registered quantity is printed next to the
threshold the pre-registration proposes for it, and the row of each
registered outcome map that the numbers mechanically fall in is a
LOOKUP, not a verdict: adjudication is HUMAN (CLAUDE.md ground rule 1).

Threshold status: **FROZEN 2026-08-31 (human, as proposed)**.

| input | value |
| --- | --- |
| runs | 15 |
| series (frozen probes + tracked segments) | 17280 |
| core pool (B = 2000, seeds (1300, 1301, 1302, 1310, 1311)) | 3456 frozen probes |
| rider pool (seeds (1320, 1321)) | 2304 frozen probes |
| burn-in (primary / sweep) | 5 / [5, 15, 25] |
| ladder max lag / NW cap | 64 / 8 (bandwidth L in [3, 4], identical at both caps: True) |
| null reps / seed | 200000 / 4242 (209 draws) |
| tau reference seed | 4243 |
| bootstrap block / reps | 64 (tracked 4) / 2000 |
| segment-start parity (prereg 1) | odd, 17280 odd of 17280 |

## P1 - band artifact (frozen tier, 9-run B = 2000 core)

| quantity | value | CI95 | proposed threshold |
| --- | --- | --- | --- |
| ratio_alt | 1.116 | [1.084, 1.161] | - |
| ratio_dc | 1.422 | [1.345, 1.525] | - |
| **band_contrast** | 0.784 | [0.726, 0.841] | >= 1.3 (middle band from 1.15) |
| theta (dc units) | 6.272 | - | |t| >= 4.0 |
| theta, alt raw threshold | 4.696 | - | - |
| frac_alt | 0.00000 | - | >= 0.01 |
| frac_dc | 0.00781 | - | - |
| tail_contrast | 0.000 | - | >= 3.0 |
| DC exceedance events | 27 | - | guard at 10 |

Registered outcome-map row: **6** (FAIL); P1 read: True.

Raw (uncalibrated) companions, printed so P1's deliberate FAIL-ward
suppression stays visible: median |t_alt| 0.837, median |t_dc| 0.911, frac(|t_alt| >= 4) 0.00087, frac(|t_dc| >= 4) 0.00781.

## P2 - frame gain (tracked `top` / frozen, same runs)

| quantity | value | CI95 | proposed threshold |
| --- | --- | --- | --- |
| frame_gain | 1.106 | [1.024, 1.188] | >= 3.0 |
| bulk_gain | 0.701 | [0.653, 0.749] | <= 1.3 or >= 2.0 |

Registered outcome-map row: **D** (no gain).

## P3 - integrated autocorrelation time (re-read + the K > 4 extension)

| K | tau_hat | tau_white | tau_ar1 | tau_cal | CI95 | branch |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.4139 | 0.9981 | 0.5432 | 0.4147 | [0.3922, 0.4375] | DECISIVE |
| 16 | 0.4667 | 0.9945 | 0.5882 | 0.4693 | [0.4485, 0.4909] | DECISIVE |
| 32 | 0.5918 | 1.0059 | 0.6684 | 0.5883 | [0.5664, 0.6121] | DECISIVE |
| 64 | 0.8859 | 1.0000 | 0.8299 | 0.8859 | [0.8572, 0.9162] | DECISIVE |

K-stable: True; branch at K = 32: **DECISIVE**. Consistency clause tau_hat(K)/tau_ar1(K) = 0.885 against the proposed band [0.75, 1.3]: True.

## Riders (secondary; batch axis, seeds 1320/1321)

| B | excess_dc | median T_dc | ESS/n | phi_hat | probes | Nyquist = epoch harmonic |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 0.106 | 1.106 | 1.474 | -0.157 | 768 | True |
| 2000 | 0.338 | 1.338 | 1.909 | -0.315 | 768 | False |
| 8000 | 0.387 | 1.387 | 2.737 | -0.488 | 768 | False |

Rider-1 ratio 3.662 against the proposed pass band [2.8, 5.6] / flat bar 1.5 -> **PASS**.
Rider-2 ESS/n max/min 1.857 against 1.3 -> **MIXED**.

## Kill-clause diagnostics (reported whether or not they fire)

| clause | quantity | value | proposed bar | fires |
| --- | --- | --- | --- | --- |
| K2 (alt) | NW-floored fraction | 0.00000 | 0.05 | False |
| K2 (dc) | NW-floored fraction | 0.00000 | 0.05 | False |
| K3 | per-matrix phi_hat spread | 0.1707 | 0.35 (window [-0.6, -0.15]) | False |
| K4 | frozen median T_dc | 1.422 | 2.0 | False |
| K6 | max channel-shape divergence | 0.0474 | 0.15 | False |

## K1 - the estimator's own controls, run through this pipeline

| control | band_contrast | tail_contrast (null q75) | tau_cal by K | within 1.00 +/- 0.10 |
| --- | --- | --- | --- | --- |
| ar1_phi_m0.34 | 1.024 | 1.014 | 8:0.544, 16:0.576, 32:0.657, 64:0.824 | contrasts True |
| white | 1.020 | 0.994 | 8:1.012, 16:0.996, 32:0.994, 64:0.989 | contrasts True, tau True |

## Sensitivities (they can qualify a number, never create one)

| burn-in | band_contrast | tail_contrast | frame_gain | tau_cal | phi_hat |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.784 | 0.000 | 1.106 | 0.588 | -0.3318 |
| 15 | 0.850 | 0.050 | 1.141 | 0.591 | -0.3130 |
| 25 | 0.871 | 0.000 | 1.019 | 0.594 | -0.2937 |

| null phi | ratio_alt | ratio_dc | band_contrast | frac_alt | frac_dc |
| --- | --- | --- | --- | --- | --- |
| phi_fixed_-0.34 | 1.120 | 1.428 | 0.785 | 0.00000 | 0.00781 |
| phi_hat+0.05 | 1.137 | 1.405 | 0.809 | 0.00000 | 0.00781 |
| phi_hat-0.05 | 1.097 | 1.441 | 0.761 | 0.00000 | 0.00781 |

## Descriptive (no criteria, no thresholds)

ESS/n on the core pool: median 1.953 (published, PEEKED: 1.9495); per-matrix phi_hat -0.419, -0.404, -0.390, -0.296, -0.253, -0.248.

Mirror check (logged accumulator statistic vs `spectral.channel_t` on the same series): 64 probes, max |dev| t_nw 0.0.
