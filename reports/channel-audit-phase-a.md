# Per-direction channel audit (tracked tier) — EXPLORATORY, PHASE A

**EXPLORATORY. These numbers are a prior, not evidence.** This report
is offline analysis of sidecars that have ALREADY BEEN PEEKED: the
lag-ladder and channel readings recorded as A1–A5 in
`reports/channel-audit-preregistration.md` §2 were produced by ad-hoc
in-session analysis over exactly this data, before any criterion
existed. Nothing here is unblinded, controlled for multiplicity, or
separable from the tracked tier's period-50 refresh cadence — whose
25th harmonic lands exactly on the 0.5 cyc/step frequency the
alternating channel reads. The confirmatory surface is the
frozen-probe tier re-run on GPU (prereg §3), which does not exist on
disk.

**Phase A output, at the Phase A path.** Prereg §6a registers this
file as `reports/channel-audit-phase-a.{md,json}` and §6b reserves
`reports/channel-audit.{md,json}` for the CONFIRMATORY Phase B
producer, which is a different script. Writing to the reserved names
is refused by the argument parser.

Descriptive output of `scripts/analyze_channel_audit.py`: quantities
only. No gate is evaluated and no pass/fail is stated; gate decisions
are human-only (CLAUDE.md ground rule 1).

## 0. Inputs and estimator

- sidecars discovered: 218; selected: 218; runs used: 218; skipped: 0
- selection filters: run prefix `airbench_instrumented_`, min seed 1000, `results/INVALID_RUNS.json` tombstones — excluded 0 (invalid_runs: 0, min_seed: 0, run_prefix: 0)
- observations read: 12621312; direction slots: 41856; matrices: 6
- lr grid: [0.12, 0.24, 0.32, 0.36, 0.37, 0.42, 0.48, 0.55, 0.6, 0.64, 0.72, 0.73, 0.84, 0.96, 0.97, 1.11, 1.44]
- batch grid: [500, 1000, 2000, 4000, 8000]; seeds: [1400, 1401, 1410, 1411, 1412, 1413, 1414]
- **the grid is not a full factorial**: cells populated 112 of 170 (kind × lr × batch); the empty combinations are absent from §4, not zero
- unit of analysis: **one refresh segment** (`--segment-at refresh`), burn-in [0, 5, 15, 25] (registered sweep [5, 15, 25]; 0 added for the A2 transient anchor), primary 5; lag ladder to 8; segments with fewer than 10 post-burn-in observations dropped
- segment statistics used: 1015296 (dropped short: 0), summed over the burn-in sweep; slot statistics used: 167424
- AR(1) surrogate null: 2000 reps per (phi_hat, **each distinct raw segment length in the cell**, burn-in), mixed at the cell's own length frequencies, seeded from those parameters and base seed 4242
- block bootstrap: block 16 (= one run × matrix × segment cluster of direction slots), 2000 reps at every pool size, seed 4242; the registered direction-block (4 segments of one direction, prereg §5.8) is reported beside it in §1

### What `ratio` does not mean

`ratio` = observed median |t_nw| / null median |t_nw| is a **scale**
correction, not an identification. Its numerator is the
zero-frequency content of a 45-observation window; its denominator
integrates lags 1..L with Bartlett weights and L = 3 at n = 45. Any
zero-mean component slower than ~3 steps is therefore absent from
the denominator and fully present in the numerator, and an AR(1)
null fitted to ρ₁ has a correlation time under one step by
construction. A **zero-mean** stream (fast AR(1) φ = −0.40 plus slow
AR(1) φ = 0.97 at variance 0.15) run through this exact pipeline
returns ρ₁ = −0.33, dc median |t| = 1.78, dc ratio = 2.71, alt ratio
= 1.04 — numerically the pooled `top` cell below, with no mean
anywhere in it. It is control **K3** in the estimator-control table
of §8, and §7 is the discriminator on the real data.

### Deviations from the DRAFT pre-registration, stated up front

1. **Unit.** Prereg §5.1 registers the SLOT-level estimator (a
   slot's post-burn-in segments concatenated, N ≈ 720 at burn-in 5).
   Sections 1–5 use the single segment (N ≈ 45). |t| scales like
   √N, so those magnitudes are not comparable rung-for-rung with a
   slot-level reading; the null calibration is matched to the same
   short window, so the *ratios* are. §7 restores the registered
   slot-level unit as a diagnostic.
2. **`--null-reps` = 2000, not the registered 200000**
   (§5.6). The registration states why: exceedance rates of order
   1e-4 cannot be resolved by 2000 draws. `null frac>=4` is printed
   as `<floor` when it sits at or below its own 1/reps resolution
   floor, and the null median's Monte-Carlo error is carried into
   every `ratio CI95`.
3. **`--bootstrap-block` = 16 is not the
   registered block** (§5.8 registers "the 4 segments of one
   direction" for tracked per-segment statistics). The two are
   orthogonal clusterings of the same table; both are computed and
   §1 prints them side by side.
4. **Nulls are mixed over each cell's actual raw segment lengths**,
   not drawn once at its median length (§5.6 asks for every distinct
   length in the design). This is a repair: the median put every
   b04000 cell's null at raw length 48, which does not occur in that
   cell at all.
5. **Burn-in 0 is carried** alongside the registered
   [5, 15, 25] (§5.3), because it is the only value
   that can exhibit the A2 transient. Every criterion is still read
   at 5.
6. **Batched kernel.** The per-segment and per-slot statistics are
   computed by batched NumPy kernels and checked against
   `src.stats.spectral` on a seeded uniform sample whose coverage
   (slot positions, raw segment lengths) is printed in §6.

## 0b. Phase A reproduction obligation — **OPEN**

Prereg §2 registers, verbatim: *"Before any Phase B run,
`scripts/analyze_channel_audit.py` … must reproduce A1–A5
deterministically … Any disagreement between the numbers quoted
above and the reproduced ones is reported as an amendment to this
file."* This section is that reproduction. It states quantities
and an obligation status; it evaluates no gate.

| anchor | registered claim | reproduced | status |
| --- | --- | --- | --- |
| A1 | clean AR(1), phi ~ -0.34 on the tracked lag ladder, LR-INVARIANT across the ladder | phi_hat_pooled_top = -0.343; phi_hat_top_max = -0.186; phi_hat_top_min = -0.439; phi_hat_top_span = 0.252 | **DISAGREES** |
| A2 | burn-in is load-bearing: at burn-in 0 rho_2 reads ~ -0.01 against the AR(1) prediction +0.116 at phi = -0.34 | ar1_prediction_at_burn_in_5 = 0.118; rho_2_burn_in_0 = 0.009; rho_2_burn_in_5 = 0.071 | **AGREES** |
| A3 | alternating channel at the null: median |t_alt,nw| ~ 0.75-0.85 across every lr, burn-in and kind | alt_median_abs_t_max = 0.992; alt_median_abs_t_min = 0.762; n_cells = 102 | **DISAGREES** |
| A4 | DC excess monotone in lr, peaking at 3.77 at lr = 0.96, 2.83 at burn-in 25; confined to top | dc_median_abs_t_at_lr_0.96 = 3.698; dc_median_abs_t_at_lr_0.96_burn_in_25 = 2.452; dc_median_abs_t_peak = 3.698; dc_median_abs_t_peak_lr = 0.960 | **AGREES** |
| A5 | the published beta = 0.9 vs 0.99 tier contrast (0.596 vs 0.400) is ~20% reproduced by a zero-mean AR(1) surrogate | — | **NOT_REPRODUCIBLE** |

- **A1**: phi_hat is NOT lr-invariant: over the lr ladder it runs -0.439 to -0.186, a span of 0.252, against A1's single '-0.34, LR-invariant'. The pooled value may still land near -0.34; the invariance clause is what disagrees, and it needs an amendment. Separately, this column is an ESTIMATOR value: the +1/n ladder correction leaves a measured +0.014 residual at phi = -0.34, n = 45, so a reproduced -0.343 corresponds to a true phi of about -0.358 (src/stats/spectral.py module docstring has the measured map).
- **A2**: burn-in 0 is carried in the sweep for exactly this anchor; the registered sweep {5, 15, 25} cannot exhibit a transient confined to the head of a segment.
- **A3**: range over every kind x lr cell at the REGISTERED burn-ins [5, 15, 25]; burn-in 0 is excluded because A3 is stated on burn-in-cleaned segments.
- **A4**: A4's 3.77 is a median |t_dc|, not a ratio; compare it against the median |t| column, not the null-calibrated one. The lr of the peak is a separate claim from its height and both are listed.
- **A5**: the beta = 0.9 vs 0.99 tier contrast is a frozen-probe-tier quantity; this script reads tracked-direction sidecars only and has no producer for it. Prereg 6b assigns it to scripts/analyze_channel_audit_frozen.py, which does not exist.

**The §2 obligation is OPEN** (unreproduced: A1, A3, A5). This script's unit of analysis is the refresh segment (N ~ 45); prereg 5.1 registers the slot (N ~ 720). |t| scales like sqrt(N), so A3/A4 magnitudes are not rung-for-rung comparable with the registered unit even when they numerically agree, and A5 has no producer. The obligation is therefore NOT discharged by running this script, and the disagreements listed above are amendments owed to reports/channel-audit-preregistration.md 2.

## 1. Lag ladder, kind × lr (burn-in 5)

Bias-corrected `rho_k = c_k/c_0 + 1/n` per segment, median over
segments; `rho_1 raw` is the same median without the +1/n correction
(computed from the raw ladder, so variance-floored rows read 0 and
not −1/n).

`L` is the Newey-West bandwidth actually in force at this window
(`min(max_lag, floor(4(n/100)^(2/9)), n−2)`), and **`tau_nw(L)` is
the integrated autocorrelation time that bandwidth implies**:
`1 + 2*sum_{j<=L} (1 − j/(L+1)) rho_j`, Bartlett-weighted. Its
reciprocal is the inflation factor that bandwidth applies (1.889 on
pooled `top`, against the 1.854 the null's own ESS/n column
measures); the two are not algebraically equal, because ESS/n is a
median of per-segment ratios and this is a ratio of medians, so read
the agreement as a sanity check and not as an identity. `tau(4)` is
the flat, unweighted
lag-4 sum, kept for continuity with the pre-registration's τ(K)
family — it is **not** what the estimator uses, and at the tracked
window the bandwidth is L = 3, not 4 (L = 4 is the frozen tier's, at
n ∈ {187, 195}). On pooled `top` the two read 0.529 and 0.405, 30%
apart.

`rho_1` is an estimator value, not φ: the +1/n correction leaves a
measured +0.014 residual at φ = −0.34, n = 45 (2× the printed CI
half-width), so −0.343 corresponds to a true φ of about −0.358. See
the `src/stats/spectral.py` module docstring for the measured map.

| group | n_runs | n_seg | seg len | n_kept | L | rho_1 | rho_2 | rho_3 | rho_4 | rho_1 raw | rho_1 CI95 | rho_1 CI95 (dir block) | tau_nw(L) | tau(4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulk/lr00.12/bALL | 15 | 11328 | 50 | 45.0 | 3 | -0.075 | 0.003 | -0.028 | 0.004 | -0.098 | [-0.082, -0.067] | [-0.080, -0.070] | 0.876 | 0.806 |
| bulk/lr00.24/bALL | 25 | 17088 | 50 | 45.0 | 3 | -0.127 | 0.016 | -0.030 | 0.002 | -0.150 | [-0.135, -0.120] | [-0.131, -0.123] | 0.810 | 0.721 |
| bulk/lr00.32/bALL | 5 | 3840 | 50 | 45.0 | 3 | -0.141 | 0.019 | -0.025 | 0.002 | -0.163 | [-0.155, -0.127] | [-0.148, -0.133] | 0.795 | 0.710 |
| bulk/lr00.36/bALL | 20 | 12288 | 50 | 45.0 | 3 | -0.157 | 0.025 | -0.028 | 0.004 | -0.179 | [-0.166, -0.147] | [-0.162, -0.151] | 0.776 | 0.689 |
| bulk/lr00.37/bALL | 5 | 3840 | 50 | 45.0 | 3 | -0.156 | 0.011 | -0.019 | -0.003 | -0.178 | [-0.170, -0.141] | [-0.163, -0.149] | 0.767 | 0.664 |
| bulk/lr00.42/bALL | 10 | 5760 | 50 | 45.0 | 3 | -0.183 | 0.026 | -0.030 | 0.002 | -0.205 | [-0.195, -0.171] | [-0.188, -0.177] | 0.737 | 0.631 |
| bulk/lr00.48/bALL | 24 | 15552 | 50 | 45.0 | 3 | -0.177 | 0.028 | -0.028 | 0.006 | -0.199 | [-0.185, -0.170] | [-0.181, -0.173] | 0.749 | 0.658 |
| bulk/lr00.55/bALL | 15 | 6720 | 50 | 45.0 | 3 | -0.220 | 0.043 | -0.037 | 0.015 | -0.242 | [-0.232, -0.208] | [-0.226, -0.213] | 0.694 | 0.601 |
| bulk/lr00.60/bALL | 14 | 9792 | 50 | 45.0 | 3 | -0.177 | 0.024 | -0.030 | 0.008 | -0.199 | [-0.185, -0.168] | [-0.182, -0.172] | 0.744 | 0.650 |
| bulk/lr00.64/bALL | 15 | 6720 | 50 | 45.0 | 3 | -0.218 | 0.040 | -0.032 | 0.008 | -0.240 | [-0.228, -0.209] | [-0.223, -0.214] | 0.697 | 0.598 |
| bulk/lr00.72/bALL | 14 | 9792 | 50 | 45.0 | 3 | -0.186 | 0.026 | -0.027 | 0.009 | -0.209 | [-0.194, -0.178] | [-0.191, -0.181] | 0.733 | 0.644 |
| bulk/lr00.73/bALL | 10 | 2880 | 50 | 45.0 | 3 | -0.277 | 0.068 | -0.041 | 0.025 | -0.300 | [-0.292, -0.264] | [-0.287, -0.271] | 0.632 | 0.550 |
| bulk/lr00.84/bALL | 10 | 2880 | 50 | 45.0 | 3 | -0.274 | 0.063 | -0.034 | 0.020 | -0.296 | [-0.287, -0.260] | [-0.281, -0.265] | 0.635 | 0.550 |
| bulk/lr00.96/bALL | 14 | 9792 | 50 | 45.0 | 3 | -0.186 | 0.024 | -0.028 | 0.010 | -0.209 | [-0.193, -0.179] | [-0.191, -0.182] | 0.730 | 0.639 |
| bulk/lr00.97/bALL | 5 | 960 | 48 | 43.0 | 3 | -0.336 | 0.108 | -0.057 | 0.030 | -0.359 | [-0.348, -0.318] | [-0.345, -0.320] | 0.576 | 0.490 |
| bulk/lr01.11/bALL | 5 | 960 | 48 | 43.0 | 3 | -0.318 | 0.101 | -0.045 | 0.023 | -0.342 | [-0.330, -0.305] | [-0.330, -0.306] | 0.601 | 0.520 |
| bulk/lr01.44/bALL | 12 | 6720 | 50 | 45.0 | 3 | -0.190 | 0.037 | -0.025 | 0.010 | -0.212 | [-0.198, -0.180] | [-0.196, -0.183] | 0.740 | 0.665 |
| top/lr00.12/bALL | 15 | 11328 | 50 | 45.0 | 3 | -0.186 | -0.012 | -0.049 | 0.001 | -0.209 | [-0.200, -0.174] | [-0.193, -0.180] | 0.684 | 0.508 |
| top/lr00.24/bALL | 25 | 17088 | 50 | 45.0 | 3 | -0.299 | 0.033 | -0.052 | 0.016 | -0.322 | [-0.313, -0.285] | [-0.306, -0.292] | 0.558 | 0.397 |
| top/lr00.32/bALL | 5 | 3840 | 50 | 45.0 | 3 | -0.352 | 0.052 | -0.050 | 0.015 | -0.375 | [-0.374, -0.331] | [-0.363, -0.343] | 0.498 | 0.329 |
| top/lr00.36/bALL | 20 | 12288 | 50 | 45.0 | 3 | -0.338 | 0.056 | -0.057 | 0.019 | -0.361 | [-0.351, -0.324] | [-0.346, -0.331] | 0.520 | 0.360 |
| top/lr00.37/bALL | 5 | 3840 | 50 | 45.0 | 3 | -0.362 | 0.063 | -0.054 | 0.028 | -0.384 | [-0.383, -0.339] | [-0.373, -0.349] | 0.493 | 0.350 |
| top/lr00.42/bALL | 10 | 5760 | 50 | 45.0 | 3 | -0.397 | 0.089 | -0.058 | 0.032 | -0.420 | [-0.412, -0.382] | [-0.405, -0.389] | 0.464 | 0.331 |
| top/lr00.48/bALL | 24 | 15552 | 50 | 45.0 | 3 | -0.369 | 0.078 | -0.059 | 0.029 | -0.392 | [-0.379, -0.359] | [-0.375, -0.364] | 0.495 | 0.358 |
| top/lr00.55/bALL | 15 | 6720 | 50 | 45.0 | 3 | -0.418 | 0.116 | -0.069 | 0.036 | -0.440 | [-0.431, -0.406] | [-0.426, -0.410] | 0.454 | 0.329 |
| top/lr00.60/bALL | 14 | 9792 | 50 | 45.0 | 3 | -0.356 | 0.073 | -0.056 | 0.034 | -0.378 | [-0.367, -0.345] | [-0.364, -0.349] | 0.511 | 0.391 |
| top/lr00.64/bALL | 15 | 6720 | 50 | 45.0 | 3 | -0.403 | 0.107 | -0.065 | 0.042 | -0.425 | [-0.415, -0.391] | [-0.411, -0.396] | 0.470 | 0.363 |
| top/lr00.72/bALL | 14 | 9792 | 50 | 45.0 | 3 | -0.351 | 0.081 | -0.055 | 0.043 | -0.374 | [-0.361, -0.340] | [-0.357, -0.344] | 0.527 | 0.438 |
| top/lr00.73/bALL | 10 | 2880 | 50 | 45.0 | 3 | -0.439 | 0.159 | -0.076 | 0.044 | -0.461 | [-0.450, -0.428] | [-0.446, -0.431] | 0.463 | 0.376 |
| top/lr00.84/bALL | 10 | 2880 | 50 | 45.0 | 3 | -0.424 | 0.157 | -0.065 | 0.050 | -0.447 | [-0.438, -0.407] | [-0.434, -0.411] | 0.489 | 0.437 |
| top/lr00.96/bALL | 14 | 9792 | 50 | 45.0 | 3 | -0.321 | 0.085 | -0.046 | 0.045 | -0.343 | [-0.331, -0.312] | [-0.328, -0.314] | 0.581 | 0.526 |
| top/lr00.97/bALL | 5 | 960 | 48 | 43.0 | 3 | -0.438 | 0.196 | -0.057 | 0.056 | -0.460 | [-0.459, -0.422] | [-0.453, -0.424] | 0.511 | 0.516 |
| top/lr01.11/bALL | 5 | 960 | 48 | 43.0 | 3 | -0.417 | 0.196 | -0.063 | 0.058 | -0.441 | [-0.436, -0.394] | [-0.430, -0.399] | 0.540 | 0.548 |
| top/lr01.44/bALL | 12 | 6720 | 50 | 45.0 | 3 | -0.288 | 0.097 | -0.027 | 0.050 | -0.311 | [-0.301, -0.276] | [-0.296, -0.280] | 0.652 | 0.666 |

## 2. Channels, kind × lr (burn-in 5)

`ratio` is the null-calibrated statistic: observed median |t_nw|
divided by the median |t_nw| of an AR(1) surrogate null matched to
that cell's own `phi_hat` and mixed over its own raw segment
lengths, pushed through the identical estimator. Under a correct
null `ratio ≈ 1` — but read `ratio CI95`, not `ratio`: the
denominator is a Monte-Carlo median over `--null-reps` draws and
carries ~3% error of its own, which on the pooled cells exceeds the
numerator's CI half-width. A ratio whose CI95 contains 1.0 is not a
measurement of anything.

Two further limits of the denominator, neither of them in that CI:

- **The null is matched on ρ₁ only.** Fitting a Yule-Walker AR(8) to
  a pooled ladder and re-drawing (20k zero-mean streams through the
  identical estimator) raises the null median |t| by 0.4–4.4%
  depending on cell and channel, so these ratios are *overstated* by
  up to ~4.5%, by a channel-dependent amount (measured at this
  report's own pooled ladders: top dc 4.4%, top alt 4.1%, bulk dc
  0.4%, bulk alt 3.9%). No contrast changes sign, but `ratio ≈ 1` is
  not attainable to better than a few percent.
- **`phi_hat` is plugged in, not inverted.** The null is drawn with
  the cell's measured `rho_1` as its TRUE φ, so running this ladder
  estimator on the null's own streams returns about `rho_1 + 0.014`
  rather than `rho_1` — the surrogate does not reproduce the
  statistic it is matched on. Redrawing at the bias-inverted φ moves
  the null median |t| by −0.4% (dc) and +0.8% (alt) at the `top`
  ladder, i.e. inside the Monte-Carlo floor above.

| group | n_runs | n_seg | ch | median \|t\| | CI95 | null median \|t\| | null CI95 | ratio | ratio CI95 | frac \|t\|>=2 | null frac>=2 | frac \|t\|>=4 | null frac>=4 | ESS/n | null ESS/n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulk/lr00.12/bALL | 15 | 11328 | alt | 0.766 | [0.750, 0.781] | 0.747 | [0.709, 0.797] | 1.024 | [0.963, 1.086] | 0.094 | 0.074 | 0.003 | 0.00350 | 0.99 | 0.98 |
| bulk/lr00.12/bALL | 15 | 11328 | dc | 0.645 | [0.630, 0.663] | 0.719 | [0.688, 0.760] | 0.897 | [0.846, 0.948] | 0.054 | 0.073 | 0.001 | <0.00050 | 1.29 | 1.25 |
| bulk/lr00.24/bALL | 25 | 17088 | alt | 0.766 | [0.754, 0.779] | 0.776 | [0.731, 0.810] | 0.987 | [0.932, 1.043] | 0.094 | 0.088 | 0.002 | 0.00100 | 0.91 | 0.90 |
| bulk/lr00.24/bALL | 25 | 17088 | dc | 0.661 | [0.647, 0.671] | 0.662 | [0.630, 0.702] | 0.999 | [0.944, 1.054] | 0.055 | 0.066 | 0.001 | 0.00200 | 1.42 | 1.32 |
| bulk/lr00.32/bALL | 5 | 3840 | alt | 0.771 | [0.737, 0.796] | 0.765 | [0.723, 0.806] | 1.008 | [0.940, 1.076] | 0.095 | 0.092 | 0.003 | 0.00600 | 0.89 | 0.91 |
| bulk/lr00.32/bALL | 5 | 3840 | dc | 0.707 | [0.682, 0.729] | 0.682 | [0.648, 0.710] | 1.037 | [0.975, 1.098] | 0.068 | 0.060 | 0.002 | 0.00100 | 1.45 | 1.36 |
| bulk/lr00.36/bALL | 20 | 12288 | alt | 0.762 | [0.747, 0.781] | 0.755 | [0.714, 0.806] | 1.009 | [0.937, 1.081] | 0.091 | 0.095 | 0.003 | 0.00250 | 0.87 | 0.89 |
| bulk/lr00.36/bALL | 20 | 12288 | dc | 0.706 | [0.688, 0.721] | 0.685 | [0.651, 0.730] | 1.030 | [0.971, 1.090] | 0.076 | 0.061 | 0.002 | 0.00100 | 1.48 | 1.38 |
| bulk/lr00.37/bALL | 5 | 3840 | alt | 0.767 | [0.743, 0.797] | 0.751 | [0.720, 0.798] | 1.021 | [0.959, 1.083] | 0.095 | 0.093 | 0.002 | 0.00150 | 0.88 | 0.89 |
| bulk/lr00.37/bALL | 5 | 3840 | dc | 0.729 | [0.703, 0.765] | 0.695 | [0.665, 0.739] | 1.049 | [0.978, 1.119] | 0.084 | 0.068 | 0.002 | <0.00050 | 1.49 | 1.38 |
| bulk/lr00.42/bALL | 10 | 5760 | alt | 0.775 | [0.749, 0.801] | 0.740 | [0.699, 0.781] | 1.047 | [0.978, 1.116] | 0.095 | 0.088 | 0.003 | 0.00200 | 0.84 | 0.84 |
| bulk/lr00.42/bALL | 10 | 5760 | dc | 0.737 | [0.710, 0.766] | 0.711 | [0.668, 0.740] | 1.037 | [0.972, 1.103] | 0.094 | 0.064 | 0.005 | 0.00100 | 1.57 | 1.46 |
| bulk/lr00.48/bALL | 24 | 15552 | alt | 0.785 | [0.770, 0.797] | 0.769 | [0.728, 0.816] | 1.020 | [0.958, 1.081] | 0.098 | 0.091 | 0.003 | 0.00100 | 0.85 | 0.85 |
| bulk/lr00.48/bALL | 24 | 15552 | dc | 0.762 | [0.744, 0.777] | 0.698 | [0.658, 0.734] | 1.091 | [1.029, 1.153] | 0.097 | 0.068 | 0.003 | 0.00100 | 1.54 | 1.43 |
| bulk/lr00.55/bALL | 15 | 6720 | alt | 0.787 | [0.766, 0.806] | 0.802 | [0.767, 0.843] | 0.981 | [0.930, 1.032] | 0.099 | 0.083 | 0.003 | 0.00250 | 0.80 | 0.80 |
| bulk/lr00.55/bALL | 15 | 6720 | dc | 0.794 | [0.769, 0.816] | 0.706 | [0.674, 0.739] | 1.124 | [1.064, 1.183] | 0.108 | 0.067 | 0.006 | 0.00100 | 1.68 | 1.52 |
| bulk/lr00.60/bALL | 14 | 9792 | alt | 0.780 | [0.764, 0.798] | 0.769 | [0.729, 0.816] | 1.014 | [0.952, 1.076] | 0.097 | 0.091 | 0.002 | 0.00100 | 0.85 | 0.85 |
| bulk/lr00.60/bALL | 14 | 9792 | dc | 0.783 | [0.765, 0.800] | 0.701 | [0.665, 0.735] | 1.117 | [1.055, 1.179] | 0.106 | 0.068 | 0.005 | 0.00100 | 1.54 | 1.43 |
| bulk/lr00.64/bALL | 15 | 6720 | alt | 0.803 | [0.780, 0.824] | 0.738 | [0.694, 0.782] | 1.089 | [1.016, 1.163] | 0.101 | 0.102 | 0.003 | 0.00250 | 0.81 | 0.79 |
| bulk/lr00.64/bALL | 15 | 6720 | dc | 0.821 | [0.790, 0.848] | 0.693 | [0.660, 0.737] | 1.184 | [1.101, 1.267] | 0.121 | 0.059 | 0.007 | 0.00100 | 1.66 | 1.55 |
| bulk/lr00.72/bALL | 14 | 9792 | alt | 0.796 | [0.776, 0.814] | 0.780 | [0.745, 0.820] | 1.021 | [0.967, 1.075] | 0.099 | 0.082 | 0.002 | 0.00400 | 0.85 | 0.84 |
| bulk/lr00.72/bALL | 14 | 9792 | dc | 0.841 | [0.817, 0.866] | 0.682 | [0.649, 0.716] | 1.233 | [1.165, 1.301] | 0.120 | 0.057 | 0.007 | <0.00050 | 1.56 | 1.43 |
| bulk/lr00.73/bALL | 10 | 2880 | alt | 0.783 | [0.757, 0.817] | 0.786 | [0.754, 0.824] | 0.996 | [0.937, 1.055] | 0.119 | 0.110 | 0.006 | 0.00350 | 0.74 | 0.74 |
| bulk/lr00.73/bALL | 10 | 2880 | dc | 0.838 | [0.791, 0.877] | 0.687 | [0.655, 0.723] | 1.220 | [1.130, 1.310] | 0.135 | 0.066 | 0.011 | <0.00050 | 1.86 | 1.69 |
| bulk/lr00.84/bALL | 10 | 2880 | alt | 0.798 | [0.768, 0.831] | 0.766 | [0.719, 0.805] | 1.043 | [0.971, 1.115] | 0.109 | 0.096 | 0.003 | 0.00400 | 0.75 | 0.74 |
| bulk/lr00.84/bALL | 10 | 2880 | dc | 0.834 | [0.796, 0.864] | 0.685 | [0.640, 0.708] | 1.217 | [1.139, 1.295] | 0.134 | 0.056 | 0.012 | 0.00100 | 1.83 | 1.67 |
| bulk/lr00.96/bALL | 14 | 9792 | alt | 0.788 | [0.768, 0.808] | 0.780 | [0.745, 0.820] | 1.010 | [0.956, 1.064] | 0.094 | 0.082 | 0.001 | 0.00400 | 0.85 | 0.84 |
| bulk/lr00.96/bALL | 14 | 9792 | dc | 0.850 | [0.828, 0.876] | 0.682 | [0.649, 0.716] | 1.246 | [1.177, 1.315] | 0.136 | 0.057 | 0.009 | <0.00050 | 1.56 | 1.43 |
| bulk/lr00.97/bALL | 5 | 960 | alt | 0.901 | [0.825, 0.980] | 0.830 | [0.790, 0.878] | 1.085 | [0.971, 1.199] | 0.128 | 0.117 | 0.002 | 0.00400 | 0.69 | 0.68 |
| bulk/lr00.97/bALL | 5 | 960 | dc | 0.871 | [0.807, 0.974] | 0.647 | [0.622, 0.685] | 1.346 | [1.203, 1.489] | 0.147 | 0.054 | 0.007 | <0.00050 | 2.00 | 1.84 |
| bulk/lr01.11/bALL | 5 | 960 | alt | 0.836 | [0.779, 0.896] | 0.799 | [0.757, 0.837] | 1.046 | [0.951, 1.140] | 0.121 | 0.111 | 0.003 | 0.00500 | 0.69 | 0.70 |
| bulk/lr01.11/bALL | 5 | 960 | dc | 0.942 | [0.890, 1.021] | 0.671 | [0.640, 0.705] | 1.405 | [1.296, 1.513] | 0.188 | 0.051 | 0.019 | <0.00050 | 1.95 | 1.83 |
| bulk/lr01.44/bALL | 12 | 6720 | alt | 0.788 | [0.766, 0.813] | 0.795 | [0.756, 0.836] | 0.991 | [0.936, 1.046] | 0.094 | 0.091 | 0.003 | 0.00300 | 0.85 | 0.84 |
| bulk/lr01.44/bALL | 12 | 6720 | dc | 0.892 | [0.863, 0.912] | 0.705 | [0.669, 0.740] | 1.265 | [1.192, 1.338] | 0.140 | 0.066 | 0.009 | 0.00150 | 1.54 | 1.46 |
| top/lr00.12/bALL | 15 | 11328 | alt | 0.774 | [0.755, 0.796] | 0.781 | [0.744, 0.818] | 0.991 | [0.938, 1.043] | 0.098 | 0.082 | 0.005 | 0.00400 | 0.86 | 0.84 |
| top/lr00.12/bALL | 15 | 11328 | dc | 0.571 | [0.554, 0.591] | 0.683 | [0.649, 0.716] | 0.835 | [0.787, 0.883] | 0.058 | 0.056 | 0.005 | <0.00050 | 1.77 | 1.43 |
| top/lr00.24/bALL | 25 | 17088 | alt | 0.797 | [0.782, 0.812] | 0.838 | [0.802, 0.883] | 0.951 | [0.901, 1.001] | 0.107 | 0.105 | 0.005 | 0.00400 | 0.76 | 0.72 |
| top/lr00.24/bALL | 25 | 17088 | dc | 0.951 | [0.910, 0.990] | 0.665 | [0.639, 0.705] | 1.430 | [1.333, 1.528] | 0.237 | 0.049 | 0.045 | <0.00050 | 2.19 | 1.75 |
| top/lr00.32/bALL | 5 | 3840 | alt | 0.795 | [0.769, 0.821] | 0.865 | [0.827, 0.905] | 0.919 | [0.868, 0.971] | 0.109 | 0.108 | 0.007 | 0.00450 | 0.74 | 0.66 |
| top/lr00.32/bALL | 5 | 3840 | dc | 1.581 | [1.331, 1.801] | 0.648 | [0.619, 0.686] | 2.440 | [2.052, 2.828] | 0.422 | 0.047 | 0.161 | <0.00050 | 2.40 | 1.89 |
| top/lr00.36/bALL | 20 | 12288 | alt | 0.812 | [0.793, 0.829] | 0.824 | [0.784, 0.855] | 0.986 | [0.936, 1.037] | 0.111 | 0.102 | 0.004 | 0.00300 | 0.75 | 0.67 |
| top/lr00.36/bALL | 20 | 12288 | dc | 1.437 | [1.314, 1.548] | 0.676 | [0.640, 0.706] | 2.125 | [1.924, 2.327] | 0.397 | 0.058 | 0.174 | <0.00050 | 2.32 | 1.86 |
| top/lr00.37/bALL | 5 | 3840 | alt | 0.821 | [0.787, 0.858] | 0.788 | [0.742, 0.841] | 1.042 | [0.963, 1.122] | 0.121 | 0.121 | 0.006 | 0.00600 | 0.74 | 0.64 |
| top/lr00.37/bALL | 5 | 3840 | dc | 1.988 | [1.699, 2.235] | 0.653 | [0.621, 0.682] | 3.047 | [2.613, 3.480] | 0.499 | 0.044 | 0.240 | <0.00050 | 2.47 | 1.92 |
| top/lr00.42/bALL | 10 | 5760 | alt | 0.826 | [0.795, 0.857] | 0.811 | [0.774, 0.861] | 1.019 | [0.949, 1.089] | 0.122 | 0.129 | 0.006 | 0.00500 | 0.71 | 0.62 |
| top/lr00.42/bALL | 10 | 5760 | dc | 2.380 | [2.129, 2.592] | 0.629 | [0.597, 0.661] | 3.783 | [3.365, 4.201] | 0.552 | 0.040 | 0.290 | <0.00050 | 2.60 | 2.06 |
| top/lr00.48/bALL | 24 | 15552 | alt | 0.833 | [0.815, 0.847] | 0.822 | [0.782, 0.869] | 1.014 | [0.958, 1.070] | 0.115 | 0.120 | 0.005 | 0.00400 | 0.75 | 0.65 |
| top/lr00.48/bALL | 24 | 15552 | dc | 2.376 | [2.205, 2.522] | 0.659 | [0.631, 0.698] | 3.607 | [3.304, 3.911] | 0.550 | 0.051 | 0.315 | 0.00100 | 2.45 | 1.97 |
| top/lr00.55/bALL | 15 | 6720 | alt | 0.856 | [0.824, 0.888] | 0.794 | [0.757, 0.836] | 1.078 | [1.012, 1.144] | 0.131 | 0.120 | 0.009 | 0.00500 | 0.70 | 0.59 |
| top/lr00.55/bALL | 15 | 6720 | dc | 2.630 | [2.405, 2.871] | 0.637 | [0.607, 0.680] | 4.126 | [3.699, 4.552] | 0.594 | 0.041 | 0.341 | <0.00050 | 2.67 | 2.12 |
| top/lr00.60/bALL | 14 | 9792 | alt | 0.823 | [0.795, 0.843] | 0.824 | [0.782, 0.882] | 1.000 | [0.926, 1.074] | 0.120 | 0.110 | 0.007 | 0.00400 | 0.78 | 0.65 |
| top/lr00.60/bALL | 14 | 9792 | dc | 2.852 | [2.634, 3.100] | 0.662 | [0.630, 0.702] | 4.307 | [3.872, 4.742] | 0.600 | 0.044 | 0.378 | 0.00100 | 2.34 | 1.93 |
| top/lr00.64/bALL | 15 | 6720 | alt | 0.828 | [0.797, 0.861] | 0.804 | [0.754, 0.841] | 1.030 | [0.961, 1.099] | 0.126 | 0.123 | 0.006 | 0.00500 | 0.73 | 0.62 |
| top/lr00.64/bALL | 15 | 6720 | dc | 3.010 | [2.761, 3.309] | 0.612 | [0.582, 0.647] | 4.918 | [4.403, 5.434] | 0.630 | 0.042 | 0.392 | <0.00050 | 2.56 | 2.04 |
| top/lr00.72/bALL | 14 | 9792 | alt | 0.817 | [0.794, 0.841] | 0.828 | [0.782, 0.872] | 0.986 | [0.930, 1.042] | 0.124 | 0.111 | 0.007 | 0.00350 | 0.81 | 0.66 |
| top/lr00.72/bALL | 14 | 9792 | dc | 3.232 | [3.030, 3.488] | 0.681 | [0.655, 0.720] | 4.748 | [4.336, 5.159] | 0.645 | 0.050 | 0.423 | <0.00050 | 2.27 | 1.90 |
| top/lr00.73/bALL | 10 | 2880 | alt | 0.860 | [0.824, 0.908] | 0.828 | [0.786, 0.883] | 1.039 | [0.962, 1.116] | 0.145 | 0.121 | 0.008 | 0.01000 | 0.67 | 0.59 |
| top/lr00.73/bALL | 10 | 2880 | dc | 2.712 | [2.441, 3.031] | 0.651 | [0.623, 0.692] | 4.168 | [3.646, 4.691] | 0.627 | 0.036 | 0.342 | <0.00050 | 2.62 | 2.17 |
| top/lr00.84/bALL | 10 | 2880 | alt | 0.890 | [0.843, 0.927] | 0.847 | [0.801, 0.887] | 1.050 | [0.977, 1.124] | 0.136 | 0.114 | 0.005 | 0.00350 | 0.68 | 0.60 |
| top/lr00.84/bALL | 10 | 2880 | dc | 2.848 | [2.620, 3.139] | 0.647 | [0.612, 0.686] | 4.398 | [3.918, 4.878] | 0.656 | 0.048 | 0.350 | <0.00050 | 2.48 | 2.14 |
| top/lr00.96/bALL | 14 | 9792 | alt | 0.810 | [0.788, 0.831] | 0.811 | [0.764, 0.848] | 0.998 | [0.941, 1.055] | 0.116 | 0.101 | 0.007 | 0.00300 | 0.85 | 0.69 |
| top/lr00.96/bALL | 14 | 9792 | dc | 3.698 | [3.456, 3.945] | 0.663 | [0.625, 0.703] | 5.580 | [5.092, 6.067] | 0.695 | 0.057 | 0.467 | <0.00050 | 2.08 | 1.83 |
| top/lr00.97/bALL | 5 | 960 | alt | 0.896 | [0.825, 0.968] | 0.846 | [0.805, 0.888] | 1.060 | [0.965, 1.154] | 0.145 | 0.127 | 0.009 | 0.00750 | 0.62 | 0.58 |
| top/lr00.97/bALL | 5 | 960 | dc | 2.261 | [2.140, 2.431] | 0.651 | [0.622, 0.688] | 3.474 | [3.179, 3.770] | 0.590 | 0.042 | 0.120 | <0.00050 | 2.34 | 2.18 |
| top/lr01.11/bALL | 5 | 960 | alt | 0.905 | [0.834, 0.983] | 0.823 | [0.789, 0.861] | 1.099 | [0.993, 1.204] | 0.134 | 0.118 | 0.006 | 0.00700 | 0.65 | 0.61 |
| top/lr01.11/bALL | 5 | 960 | dc | 2.573 | [2.384, 2.738] | 0.638 | [0.601, 0.668] | 4.032 | [3.693, 4.370] | 0.637 | 0.044 | 0.151 | 0.00100 | 2.20 | 2.10 |
| top/lr01.44/bALL | 12 | 6720 | alt | 0.821 | [0.794, 0.845] | 0.774 | [0.729, 0.827] | 1.061 | [0.986, 1.135] | 0.117 | 0.098 | 0.006 | 0.00350 | 0.88 | 0.72 |
| top/lr01.44/bALL | 12 | 6720 | dc | 3.337 | [3.091, 3.554] | 0.664 | [0.639, 0.698] | 5.028 | [4.604, 5.452] | 0.684 | 0.058 | 0.412 | <0.00050 | 1.84 | 1.73 |

## 3. Channels, kind × batch (burn-in 5)

| group | n_runs | n_seg | ch | median \|t\| | CI95 | null median \|t\| | null CI95 | ratio | ratio CI95 | frac \|t\|>=2 | null frac>=2 | frac \|t\|>=4 | null frac>=4 | ESS/n | null ESS/n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulk/lrALL/b00500 | 33 | 50688 | alt | 0.755 | [0.747, 0.762] | 0.774 | [0.734, 0.807] | 0.976 | [0.927, 1.024] | 0.087 | 0.080 | 0.002 | 0.00300 | 0.94 | 0.92 |
| bulk/lrALL/b00500 | 33 | 50688 | dc | 0.736 | [0.727, 0.745] | 0.702 | [0.666, 0.732] | 1.048 | [0.997, 1.098] | 0.090 | 0.067 | 0.003 | <0.00050 | 1.38 | 1.31 |
| bulk/lrALL/b01000 | 51 | 39168 | alt | 0.768 | [0.760, 0.778] | 0.768 | [0.726, 0.812] | 1.000 | [0.942, 1.059] | 0.094 | 0.097 | 0.003 | 0.00250 | 0.87 | 0.86 |
| bulk/lrALL/b01000 | 51 | 39168 | dc | 0.745 | [0.735, 0.755] | 0.737 | [0.707, 0.778] | 1.012 | [0.960, 1.063] | 0.091 | 0.070 | 0.004 | <0.00050 | 1.50 | 1.41 |
| bulk/lrALL/b02000 | 51 | 19584 | alt | 0.795 | [0.780, 0.807] | 0.785 | [0.738, 0.827] | 1.012 | [0.954, 1.070] | 0.102 | 0.094 | 0.003 | 0.00250 | 0.79 | 0.79 |
| bulk/lrALL/b02000 | 51 | 19584 | dc | 0.770 | [0.755, 0.785] | 0.693 | [0.653, 0.726] | 1.111 | [1.049, 1.172] | 0.109 | 0.058 | 0.008 | <0.00050 | 1.70 | 1.58 |
| bulk/lrALL/b04000 | 51 | 9792 | alt | 0.855 | [0.833, 0.879] | 0.799 | [0.756, 0.842] | 1.070 | [1.005, 1.135] | 0.126 | 0.111 | 0.003 | 0.00250 | 0.67 | 0.69 |
| bulk/lrALL/b04000 | 51 | 9792 | dc | 0.814 | [0.795, 0.835] | 0.659 | [0.620, 0.699] | 1.236 | [1.153, 1.318] | 0.122 | 0.052 | 0.007 | <0.00050 | 2.03 | 1.85 |
| bulk/lrALL/b08000 | 32 | 7680 | alt | 0.896 | [0.871, 0.926] | 0.845 | [0.796, 0.884] | 1.061 | [0.997, 1.125] | 0.126 | 0.116 | 0.004 | 0.00400 | 0.64 | 0.66 |
| bulk/lrALL/b08000 | 32 | 7680 | dc | 0.798 | [0.775, 0.819] | 0.604 | [0.570, 0.634] | 1.322 | [1.242, 1.402] | 0.102 | 0.046 | 0.004 | 0.00100 | 2.07 | 1.96 |
| top/lrALL/b00500 | 33 | 50688 | alt | 0.784 | [0.775, 0.793] | 0.791 | [0.752, 0.829] | 0.990 | [0.939, 1.041] | 0.104 | 0.096 | 0.005 | 0.00200 | 0.88 | 0.75 |
| top/lrALL/b00500 | 33 | 50688 | dc | 1.744 | [1.647, 1.844] | 0.644 | [0.610, 0.674] | 2.710 | [2.504, 2.917] | 0.467 | 0.052 | 0.276 | <0.00050 | 1.96 | 1.60 |
| top/lrALL/b01000 | 51 | 39168 | alt | 0.805 | [0.794, 0.816] | 0.836 | [0.797, 0.870] | 0.963 | [0.917, 1.009] | 0.113 | 0.122 | 0.006 | 0.00600 | 0.77 | 0.66 |
| top/lrALL/b01000 | 51 | 39168 | dc | 2.174 | [2.058, 2.284] | 0.644 | [0.608, 0.676] | 3.378 | [3.139, 3.618] | 0.523 | 0.050 | 0.300 | 0.00100 | 2.38 | 1.91 |
| top/lrALL/b02000 | 51 | 19584 | alt | 0.844 | [0.829, 0.860] | 0.827 | [0.795, 0.879] | 1.021 | [0.964, 1.077] | 0.126 | 0.133 | 0.005 | 0.00500 | 0.68 | 0.59 |
| top/lrALL/b02000 | 51 | 19584 | dc | 2.424 | [2.296, 2.545] | 0.653 | [0.626, 0.686] | 3.710 | [3.455, 3.964] | 0.561 | 0.043 | 0.314 | <0.00050 | 2.66 | 2.16 |
| top/lrALL/b04000 | 51 | 9792 | alt | 0.919 | [0.895, 0.942] | 0.885 | [0.843, 0.934] | 1.038 | [0.981, 1.095] | 0.147 | 0.131 | 0.005 | 0.00750 | 0.57 | 0.56 |
| top/lrALL/b04000 | 51 | 9792 | dc | 1.732 | [1.666, 1.803] | 0.642 | [0.623, 0.672] | 2.698 | [2.542, 2.854] | 0.429 | 0.041 | 0.077 | <0.00050 | 2.64 | 2.32 |
| top/lrALL/b08000 | 32 | 7680 | alt | 0.934 | [0.907, 0.965] | 0.916 | [0.867, 0.969] | 1.020 | [0.953, 1.086] | 0.153 | 0.152 | 0.009 | 0.01650 | 0.54 | 0.54 |
| top/lrALL/b08000 | 32 | 7680 | dc | 1.139 | [1.079, 1.210] | 0.616 | [0.587, 0.653] | 1.849 | [1.699, 1.999] | 0.290 | 0.035 | 0.099 | <0.00050 | 2.95 | 2.47 |

## 4. Full cells, kind × lr × batch (burn-in 5)

112 of 170 kind × lr × batch cells exist on disk; the rest are absent, not zero.

| group | n_runs | n_seg | ch | median \|t\| | CI95 | null median \|t\| | null CI95 | ratio | ratio CI95 | frac \|t\|>=2 | null frac>=2 | frac \|t\|>=4 | null frac>=4 | ESS/n | null ESS/n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulk/lr00.12/b00500 | 5 | 7680 | alt | 0.736 | [0.715, 0.755] | 0.759 | [0.715, 0.794] | 0.970 | [0.910, 1.029] | 0.081 | 0.086 | 0.002 | <0.00050 | 1.04 | 1.05 |
| bulk/lr00.12/b00500 | 5 | 7680 | dc | 0.628 | [0.613, 0.646] | 0.740 | [0.704, 0.776] | 0.848 | [0.801, 0.896] | 0.047 | 0.083 | 0.000 | <0.00050 | 1.22 | 1.17 |
| bulk/lr00.12/b01000 | 2 | 1536 | alt | 0.683 | [0.641, 0.730] | 0.736 | [0.705, 0.771] | 0.928 | [0.851, 1.005] | 0.072 | 0.085 | 0.001 | 0.00300 | 0.99 | 1.00 |
| bulk/lr00.12/b01000 | 2 | 1536 | dc | 0.656 | [0.617, 0.692] | 0.743 | [0.709, 0.779] | 0.883 | [0.821, 0.945] | 0.048 | 0.079 | 0.000 | 0.00100 | 1.31 | 1.24 |
| bulk/lr00.12/b02000 | 2 | 768 | alt | 0.866 | [0.808, 0.922] | 0.740 | [0.685, 0.774] | 1.170 | [1.065, 1.276] | 0.113 | 0.084 | 0.007 | 0.00350 | 0.88 | 0.87 |
| bulk/lr00.12/b02000 | 2 | 768 | dc | 0.582 | [0.535, 0.644] | 0.703 | [0.663, 0.736] | 0.827 | [0.732, 0.922] | 0.040 | 0.064 | 0.000 | 0.00100 | 1.45 | 1.39 |
| bulk/lr00.12/b04000 | 2 | 384 | alt | 0.955 | [0.853, 1.082] | 0.838 | [0.799, 0.877] | 1.139 | [0.989, 1.290] | 0.151 | 0.108 | 0.013 | 0.00450 | 0.69 | 0.70 |
| bulk/lr00.12/b04000 | 2 | 384 | dc | 0.714 | [0.614, 0.788] | 0.692 | [0.652, 0.734] | 1.032 | [0.877, 1.186] | 0.091 | 0.054 | 0.003 | <0.00050 | 1.90 | 1.80 |
| bulk/lr00.12/b08000 | 4 | 960 | alt | 1.057 | [0.990, 1.121] | 0.796 | [0.759, 0.836] | 1.329 | [1.226, 1.432] | 0.192 | 0.113 | 0.007 | 0.00650 | 0.61 | 0.63 |
| bulk/lr00.12/b08000 | 4 | 960 | dc | 0.799 | [0.749, 0.874] | 0.658 | [0.625, 0.701] | 1.215 | [1.100, 1.329] | 0.117 | 0.054 | 0.003 | <0.00050 | 2.11 | 2.01 |
| bulk/lr00.24/b00500 | 5 | 7680 | alt | 0.757 | [0.737, 0.775] | 0.746 | [0.704, 0.771] | 1.015 | [0.961, 1.069] | 0.087 | 0.071 | 0.001 | 0.00100 | 0.98 | 0.98 |
| bulk/lr00.24/b00500 | 5 | 7680 | dc | 0.658 | [0.637, 0.672] | 0.713 | [0.680, 0.753] | 0.922 | [0.868, 0.976] | 0.055 | 0.073 | 0.001 | 0.00150 | 1.32 | 1.25 |
| bulk/lr00.24/b01000 | 7 | 5376 | alt | 0.753 | [0.736, 0.775] | 0.775 | [0.726, 0.803] | 0.972 | [0.913, 1.030] | 0.089 | 0.086 | 0.003 | 0.00100 | 0.91 | 0.90 |
| bulk/lr00.24/b01000 | 7 | 5376 | dc | 0.651 | [0.630, 0.672] | 0.662 | [0.630, 0.708] | 0.983 | [0.919, 1.048] | 0.050 | 0.065 | 0.000 | 0.00150 | 1.41 | 1.32 |
| bulk/lr00.24/b02000 | 7 | 2688 | alt | 0.760 | [0.727, 0.798] | 0.734 | [0.695, 0.772] | 1.035 | [0.963, 1.107] | 0.098 | 0.085 | 0.003 | 0.00250 | 0.81 | 0.82 |
| bulk/lr00.24/b02000 | 7 | 2688 | dc | 0.669 | [0.643, 0.696] | 0.682 | [0.647, 0.713] | 0.981 | [0.914, 1.049] | 0.054 | 0.058 | 0.000 | 0.00100 | 1.56 | 1.48 |
| bulk/lr00.24/b04000 | 2 | 384 | alt | 0.962 | [0.844, 1.051] | 0.851 | [0.813, 0.887] | 1.131 | [1.002, 1.259] | 0.128 | 0.127 | 0.000 | 0.00650 | 0.65 | 0.66 |
| bulk/lr00.24/b04000 | 2 | 384 | dc | 0.798 | [0.697, 0.893] | 0.680 | [0.649, 0.716] | 1.173 | [0.996, 1.349] | 0.068 | 0.049 | 0.000 | <0.00050 | 2.05 | 1.90 |
| bulk/lr00.24/b08000 | 4 | 960 | alt | 0.889 | [0.825, 0.977] | 0.836 | [0.782, 0.895] | 1.064 | [0.943, 1.184] | 0.149 | 0.128 | 0.002 | 0.00700 | 0.58 | 0.62 |
| bulk/lr00.24/b08000 | 4 | 960 | dc | 0.693 | [0.654, 0.752] | 0.675 | [0.639, 0.714] | 1.026 | [0.938, 1.115] | 0.080 | 0.049 | 0.003 | 0.00150 | 2.15 | 2.07 |
| bulk/lr00.32/b01000 | 5 | 3840 | alt | 0.771 | [0.737, 0.796] | 0.765 | [0.723, 0.806] | 1.008 | [0.940, 1.076] | 0.095 | 0.092 | 0.003 | 0.00600 | 0.89 | 0.91 |
| bulk/lr00.32/b01000 | 5 | 3840 | dc | 0.707 | [0.682, 0.729] | 0.682 | [0.648, 0.710] | 1.037 | [0.975, 1.098] | 0.068 | 0.060 | 0.002 | 0.00100 | 1.45 | 1.36 |
| bulk/lr00.36/b00500 | 5 | 7680 | alt | 0.734 | [0.715, 0.754] | 0.749 | [0.718, 0.782] | 0.979 | [0.929, 1.030] | 0.083 | 0.077 | 0.003 | <0.00050 | 0.94 | 0.94 |
| bulk/lr00.36/b00500 | 5 | 7680 | dc | 0.694 | [0.671, 0.717] | 0.692 | [0.659, 0.727] | 1.003 | [0.944, 1.062] | 0.075 | 0.069 | 0.001 | 0.00100 | 1.37 | 1.30 |
| bulk/lr00.36/b01000 | 2 | 1536 | alt | 0.755 | [0.705, 0.796] | 0.776 | [0.751, 0.817] | 0.973 | [0.902, 1.044] | 0.084 | 0.089 | 0.003 | 0.00150 | 0.87 | 0.88 |
| bulk/lr00.36/b01000 | 2 | 1536 | dc | 0.696 | [0.664, 0.745] | 0.683 | [0.653, 0.714] | 1.019 | [0.950, 1.088] | 0.076 | 0.068 | 0.001 | 0.00100 | 1.51 | 1.42 |
| bulk/lr00.36/b02000 | 2 | 768 | alt | 0.824 | [0.757, 0.891] | 0.803 | [0.752, 0.848] | 1.026 | [0.919, 1.134] | 0.115 | 0.100 | 0.005 | 0.00150 | 0.77 | 0.78 |
| bulk/lr00.36/b02000 | 2 | 768 | dc | 0.658 | [0.621, 0.737] | 0.671 | [0.636, 0.704] | 0.981 | [0.871, 1.090] | 0.073 | 0.062 | 0.004 | 0.00150 | 1.70 | 1.56 |
| bulk/lr00.36/b04000 | 7 | 1344 | alt | 0.839 | [0.796, 0.903] | 0.816 | [0.779, 0.860] | 1.029 | [0.939, 1.119] | 0.121 | 0.120 | 0.004 | 0.00450 | 0.64 | 0.65 |
| bulk/lr00.36/b04000 | 7 | 1344 | dc | 0.719 | [0.667, 0.766] | 0.684 | [0.656, 0.717] | 1.051 | [0.955, 1.147] | 0.073 | 0.042 | 0.002 | 0.00100 | 2.07 | 1.94 |
| bulk/lr00.36/b08000 | 4 | 960 | alt | 0.858 | [0.801, 0.911] | 0.851 | [0.806, 0.889] | 1.008 | [0.928, 1.087] | 0.107 | 0.117 | 0.006 | 0.00800 | 0.61 | 0.63 |
| bulk/lr00.36/b08000 | 4 | 960 | dc | 0.812 | [0.770, 0.853] | 0.643 | [0.610, 0.673] | 1.264 | [1.175, 1.352] | 0.096 | 0.056 | 0.003 | <0.00050 | 2.11 | 2.03 |
| bulk/lr00.37/b01000 | 5 | 3840 | alt | 0.767 | [0.743, 0.797] | 0.751 | [0.720, 0.798] | 1.021 | [0.959, 1.083] | 0.095 | 0.093 | 0.002 | 0.00150 | 0.88 | 0.89 |
| bulk/lr00.37/b01000 | 5 | 3840 | dc | 0.729 | [0.703, 0.765] | 0.695 | [0.665, 0.739] | 1.049 | [0.978, 1.119] | 0.084 | 0.068 | 0.002 | <0.00050 | 1.49 | 1.38 |
| bulk/lr00.42/b01000 | 5 | 3840 | alt | 0.763 | [0.732, 0.792] | 0.776 | [0.751, 0.817] | 0.983 | [0.922, 1.044] | 0.093 | 0.089 | 0.004 | 0.00150 | 0.87 | 0.88 |
| bulk/lr00.42/b01000 | 5 | 3840 | dc | 0.737 | [0.706, 0.778] | 0.683 | [0.653, 0.714] | 1.079 | [1.009, 1.150] | 0.091 | 0.068 | 0.004 | 0.00100 | 1.51 | 1.42 |
| bulk/lr00.42/b02000 | 5 | 1920 | alt | 0.798 | [0.751, 0.850] | 0.785 | [0.738, 0.827] | 1.016 | [0.933, 1.100] | 0.099 | 0.094 | 0.002 | 0.00250 | 0.78 | 0.79 |
| bulk/lr00.42/b02000 | 5 | 1920 | dc | 0.738 | [0.678, 0.781] | 0.693 | [0.653, 0.726] | 1.064 | [0.971, 1.157] | 0.099 | 0.058 | 0.005 | <0.00050 | 1.71 | 1.58 |
| bulk/lr00.48/b00500 | 4 | 6144 | alt | 0.764 | [0.741, 0.788] | 0.771 | [0.734, 0.805] | 0.990 | [0.931, 1.049] | 0.088 | 0.073 | 0.003 | 0.00250 | 0.92 | 0.90 |
| bulk/lr00.48/b00500 | 4 | 6144 | dc | 0.767 | [0.738, 0.790] | 0.700 | [0.671, 0.745] | 1.095 | [1.029, 1.162] | 0.097 | 0.072 | 0.002 | 0.00100 | 1.42 | 1.34 |
| bulk/lr00.48/b01000 | 7 | 5376 | alt | 0.772 | [0.745, 0.796] | 0.781 | [0.734, 0.813] | 0.988 | [0.929, 1.047] | 0.093 | 0.097 | 0.002 | 0.00200 | 0.86 | 0.85 |
| bulk/lr00.48/b01000 | 7 | 5376 | dc | 0.756 | [0.732, 0.779] | 0.693 | [0.660, 0.727] | 1.090 | [1.029, 1.152] | 0.097 | 0.065 | 0.002 | <0.00050 | 1.53 | 1.42 |
| bulk/lr00.48/b02000 | 7 | 2688 | alt | 0.808 | [0.775, 0.856] | 0.771 | [0.729, 0.807] | 1.049 | [0.970, 1.127] | 0.115 | 0.100 | 0.002 | 0.00400 | 0.77 | 0.77 |
| bulk/lr00.48/b02000 | 7 | 2688 | dc | 0.769 | [0.727, 0.806] | 0.677 | [0.634, 0.715] | 1.137 | [1.045, 1.229] | 0.106 | 0.054 | 0.007 | <0.00050 | 1.73 | 1.61 |
| bulk/lr00.48/b04000 | 2 | 384 | alt | 0.791 | [0.709, 0.913] | 0.813 | [0.786, 0.861] | 0.972 | [0.845, 1.100] | 0.120 | 0.113 | 0.003 | 0.00600 | 0.63 | 0.65 |
| bulk/lr00.48/b04000 | 2 | 384 | dc | 0.740 | [0.640, 0.818] | 0.656 | [0.617, 0.690] | 1.128 | [0.981, 1.276] | 0.078 | 0.045 | 0.003 | <0.00050 | 2.09 | 1.92 |
| bulk/lr00.48/b08000 | 4 | 960 | alt | 0.900 | [0.832, 0.975] | 0.840 | [0.796, 0.879] | 1.071 | [0.965, 1.176] | 0.141 | 0.117 | 0.006 | 0.00650 | 0.62 | 0.63 |
| bulk/lr00.48/b08000 | 4 | 960 | dc | 0.766 | [0.697, 0.817] | 0.629 | [0.602, 0.654] | 1.216 | [1.103, 1.330] | 0.085 | 0.049 | 0.003 | <0.00050 | 2.21 | 2.00 |
| bulk/lr00.55/b01000 | 5 | 3840 | alt | 0.783 | [0.754, 0.807] | 0.743 | [0.700, 0.779] | 1.054 | [0.990, 1.119] | 0.101 | 0.089 | 0.003 | 0.00150 | 0.86 | 0.84 |
| bulk/lr00.55/b01000 | 5 | 3840 | dc | 0.785 | [0.748, 0.814] | 0.698 | [0.664, 0.730] | 1.123 | [1.053, 1.194] | 0.103 | 0.057 | 0.006 | <0.00050 | 1.54 | 1.47 |
| bulk/lr00.55/b02000 | 5 | 1920 | alt | 0.778 | [0.744, 0.817] | 0.798 | [0.754, 0.842] | 0.975 | [0.902, 1.049] | 0.098 | 0.102 | 0.002 | 0.00300 | 0.76 | 0.78 |
| bulk/lr00.55/b02000 | 5 | 1920 | dc | 0.812 | [0.757, 0.860] | 0.693 | [0.657, 0.730] | 1.171 | [1.077, 1.265] | 0.115 | 0.060 | 0.007 | <0.00050 | 1.76 | 1.60 |
| bulk/lr00.55/b04000 | 5 | 960 | alt | 0.805 | [0.765, 0.867] | 0.813 | [0.786, 0.861] | 0.989 | [0.911, 1.068] | 0.095 | 0.113 | 0.004 | 0.00600 | 0.64 | 0.65 |
| bulk/lr00.55/b04000 | 5 | 960 | dc | 0.804 | [0.746, 0.860] | 0.656 | [0.617, 0.690] | 1.226 | [1.111, 1.341] | 0.117 | 0.045 | 0.004 | <0.00050 | 2.07 | 1.92 |
| bulk/lr00.60/b00500 | 4 | 6144 | alt | 0.761 | [0.737, 0.781] | 0.740 | [0.696, 0.775] | 1.028 | [0.967, 1.090] | 0.090 | 0.083 | 0.002 | 0.00150 | 0.91 | 0.90 |
| bulk/lr00.60/b00500 | 4 | 6144 | dc | 0.775 | [0.750, 0.797] | 0.689 | [0.654, 0.721] | 1.125 | [1.064, 1.186] | 0.106 | 0.060 | 0.004 | 0.00100 | 1.44 | 1.36 |
| bulk/lr00.60/b01000 | 2 | 1536 | alt | 0.796 | [0.735, 0.835] | 0.766 | [0.723, 0.820] | 1.039 | [0.948, 1.131] | 0.104 | 0.086 | 0.003 | 0.00300 | 0.84 | 0.85 |
| bulk/lr00.60/b01000 | 2 | 1536 | dc | 0.796 | [0.751, 0.839] | 0.720 | [0.682, 0.756] | 1.105 | [1.023, 1.187] | 0.109 | 0.064 | 0.007 | 0.00100 | 1.57 | 1.44 |
| bulk/lr00.60/b02000 | 2 | 768 | alt | 0.776 | [0.703, 0.839] | 0.786 | [0.744, 0.845] | 0.986 | [0.873, 1.100] | 0.109 | 0.108 | 0.003 | 0.00150 | 0.78 | 0.78 |
| bulk/lr00.60/b02000 | 2 | 768 | dc | 0.748 | [0.672, 0.812] | 0.684 | [0.653, 0.716] | 1.093 | [0.965, 1.222] | 0.121 | 0.046 | 0.012 | <0.00050 | 1.72 | 1.57 |
| bulk/lr00.60/b04000 | 2 | 384 | alt | 0.879 | [0.791, 0.952] | 0.871 | [0.802, 0.914] | 1.009 | [0.887, 1.131] | 0.122 | 0.133 | 0.003 | 0.00600 | 0.61 | 0.63 |
| bulk/lr00.60/b04000 | 2 | 384 | dc | 0.861 | [0.800, 0.943] | 0.648 | [0.615, 0.683] | 1.328 | [1.187, 1.470] | 0.104 | 0.052 | 0.005 | <0.00050 | 2.17 | 2.02 |
| bulk/lr00.60/b08000 | 4 | 960 | alt | 0.889 | [0.815, 0.972] | 0.800 | [0.762, 0.834] | 1.111 | [1.002, 1.220] | 0.106 | 0.111 | 0.004 | 0.00450 | 0.61 | 0.64 |
| bulk/lr00.60/b08000 | 4 | 960 | dc | 0.800 | [0.738, 0.881] | 0.662 | [0.627, 0.701] | 1.209 | [1.074, 1.344] | 0.089 | 0.045 | 0.002 | <0.00050 | 2.16 | 2.00 |
| bulk/lr00.64/b01000 | 5 | 3840 | alt | 0.776 | [0.747, 0.802] | 0.785 | [0.747, 0.821] | 0.989 | [0.932, 1.045] | 0.093 | 0.085 | 0.003 | 0.00300 | 0.85 | 0.85 |
| bulk/lr00.64/b01000 | 5 | 3840 | dc | 0.819 | [0.780, 0.858] | 0.686 | [0.655, 0.720] | 1.195 | [1.117, 1.273] | 0.115 | 0.053 | 0.005 | <0.00050 | 1.56 | 1.43 |
| bulk/lr00.64/b02000 | 5 | 1920 | alt | 0.821 | [0.782, 0.853] | 0.818 | [0.773, 0.853] | 1.003 | [0.939, 1.067] | 0.097 | 0.103 | 0.003 | 0.00300 | 0.78 | 0.78 |
| bulk/lr00.64/b02000 | 5 | 1920 | dc | 0.846 | [0.784, 0.901] | 0.725 | [0.691, 0.758] | 1.166 | [1.072, 1.260] | 0.133 | 0.068 | 0.009 | <0.00050 | 1.72 | 1.62 |
| bulk/lr00.64/b04000 | 5 | 960 | alt | 0.873 | [0.823, 0.933] | 0.848 | [0.805, 0.892] | 1.030 | [0.947, 1.113] | 0.136 | 0.111 | 0.003 | 0.00700 | 0.66 | 0.67 |
| bulk/lr00.64/b04000 | 5 | 960 | dc | 0.786 | [0.739, 0.846] | 0.654 | [0.623, 0.691] | 1.202 | [1.091, 1.312] | 0.123 | 0.056 | 0.010 | <0.00050 | 2.05 | 1.89 |
| bulk/lr00.72/b00500 | 4 | 6144 | alt | 0.779 | [0.752, 0.799] | 0.727 | [0.690, 0.755] | 1.071 | [1.012, 1.130] | 0.095 | 0.084 | 0.002 | 0.00350 | 0.90 | 0.89 |
| bulk/lr00.72/b00500 | 4 | 6144 | dc | 0.834 | [0.804, 0.867] | 0.646 | [0.616, 0.675] | 1.291 | [1.211, 1.370] | 0.120 | 0.062 | 0.006 | <0.00050 | 1.46 | 1.37 |
| bulk/lr00.72/b01000 | 2 | 1536 | alt | 0.811 | [0.756, 0.844] | 0.776 | [0.739, 0.815] | 1.046 | [0.962, 1.130] | 0.102 | 0.088 | 0.001 | 0.00300 | 0.84 | 0.84 |
| bulk/lr00.72/b01000 | 2 | 1536 | dc | 0.867 | [0.816, 0.907] | 0.754 | [0.720, 0.781] | 1.150 | [1.072, 1.229] | 0.117 | 0.077 | 0.008 | 0.00400 | 1.61 | 1.50 |
| bulk/lr00.72/b02000 | 2 | 768 | alt | 0.807 | [0.750, 0.891] | 0.769 | [0.730, 0.814] | 1.049 | [0.940, 1.158] | 0.074 | 0.107 | 0.001 | 0.00150 | 0.76 | 0.76 |
| bulk/lr00.72/b02000 | 2 | 768 | dc | 0.860 | [0.792, 0.958] | 0.694 | [0.649, 0.735] | 1.239 | [1.098, 1.381] | 0.135 | 0.067 | 0.007 | 0.00100 | 1.76 | 1.60 |
| bulk/lr00.72/b04000 | 2 | 384 | alt | 0.945 | [0.829, 1.058] | 0.783 | [0.742, 0.821] | 1.207 | [1.053, 1.360] | 0.156 | 0.112 | 0.000 | 0.00650 | 0.64 | 0.66 |
| bulk/lr00.72/b04000 | 2 | 384 | dc | 0.805 | [0.682, 0.927] | 0.680 | [0.642, 0.716] | 1.183 | [0.977, 1.388] | 0.112 | 0.044 | 0.016 | <0.00050 | 2.10 | 1.93 |
| bulk/lr00.72/b08000 | 4 | 960 | alt | 0.837 | [0.766, 0.919] | 0.832 | [0.782, 0.876] | 1.005 | [0.905, 1.106] | 0.114 | 0.120 | 0.003 | 0.00300 | 0.65 | 0.67 |
| bulk/lr00.72/b08000 | 4 | 960 | dc | 0.832 | [0.749, 0.903] | 0.650 | [0.619, 0.695] | 1.281 | [1.144, 1.418] | 0.120 | 0.060 | 0.003 | <0.00050 | 2.08 | 1.88 |
| bulk/lr00.73/b02000 | 5 | 1920 | alt | 0.763 | [0.734, 0.799] | 0.769 | [0.730, 0.814] | 0.993 | [0.923, 1.062] | 0.110 | 0.107 | 0.007 | 0.00150 | 0.78 | 0.76 |
| bulk/lr00.73/b02000 | 5 | 1920 | dc | 0.863 | [0.812, 0.899] | 0.694 | [0.649, 0.735] | 1.243 | [1.142, 1.344] | 0.138 | 0.067 | 0.013 | 0.00100 | 1.76 | 1.60 |
| bulk/lr00.73/b04000 | 5 | 960 | alt | 0.838 | [0.766, 0.907] | 0.825 | [0.768, 0.862] | 1.016 | [0.914, 1.118] | 0.138 | 0.120 | 0.005 | 0.00550 | 0.67 | 0.68 |
| bulk/lr00.73/b04000 | 5 | 960 | dc | 0.791 | [0.732, 0.858] | 0.643 | [0.606, 0.682] | 1.230 | [1.116, 1.344] | 0.129 | 0.053 | 0.007 | <0.00050 | 2.07 | 1.87 |
| bulk/lr00.84/b02000 | 5 | 1920 | alt | 0.794 | [0.750, 0.828] | 0.779 | [0.737, 0.817] | 1.019 | [0.948, 1.089] | 0.095 | 0.092 | 0.004 | 0.00500 | 0.78 | 0.78 |
| bulk/lr00.84/b02000 | 5 | 1920 | dc | 0.837 | [0.788, 0.877] | 0.681 | [0.647, 0.717] | 1.231 | [1.138, 1.323] | 0.134 | 0.062 | 0.013 | 0.00100 | 1.75 | 1.59 |
| bulk/lr00.84/b04000 | 5 | 960 | alt | 0.817 | [0.760, 0.892] | 0.831 | [0.793, 0.872] | 0.984 | [0.890, 1.077] | 0.138 | 0.118 | 0.002 | 0.00450 | 0.68 | 0.69 |
| bulk/lr00.84/b04000 | 5 | 960 | dc | 0.816 | [0.761, 0.875] | 0.651 | [0.621, 0.684] | 1.253 | [1.148, 1.359] | 0.133 | 0.054 | 0.010 | <0.00050 | 2.03 | 1.81 |
| bulk/lr00.96/b00500 | 4 | 6144 | alt | 0.768 | [0.747, 0.792] | 0.705 | [0.666, 0.755] | 1.090 | [1.011, 1.170] | 0.089 | 0.086 | 0.001 | 0.00150 | 0.89 | 0.87 |
| bulk/lr00.96/b00500 | 4 | 6144 | dc | 0.844 | [0.819, 0.876] | 0.707 | [0.674, 0.740] | 1.194 | [1.125, 1.264] | 0.134 | 0.067 | 0.007 | <0.00050 | 1.47 | 1.39 |
| bulk/lr00.96/b01000 | 2 | 1536 | alt | 0.802 | [0.755, 0.860] | 0.768 | [0.731, 0.810] | 1.044 | [0.959, 1.130] | 0.098 | 0.085 | 0.002 | <0.00050 | 0.85 | 0.83 |
| bulk/lr00.96/b01000 | 2 | 1536 | dc | 0.870 | [0.794, 0.920] | 0.656 | [0.614, 0.698] | 1.325 | [1.192, 1.457] | 0.138 | 0.052 | 0.012 | <0.00050 | 1.59 | 1.46 |
| bulk/lr00.96/b02000 | 2 | 768 | alt | 0.782 | [0.709, 0.866] | 0.758 | [0.728, 0.803] | 1.031 | [0.915, 1.148] | 0.099 | 0.091 | 0.005 | 0.00300 | 0.79 | 0.77 |
| bulk/lr00.96/b02000 | 2 | 768 | dc | 0.863 | [0.802, 0.952] | 0.705 | [0.670, 0.749] | 1.224 | [1.097, 1.351] | 0.156 | 0.063 | 0.018 | <0.00050 | 1.71 | 1.58 |
| bulk/lr00.96/b04000 | 2 | 384 | alt | 0.918 | [0.772, 0.996] | 0.824 | [0.791, 0.861] | 1.115 | [0.967, 1.262] | 0.120 | 0.102 | 0.000 | 0.00300 | 0.67 | 0.69 |
| bulk/lr00.96/b04000 | 2 | 384 | dc | 0.877 | [0.743, 0.991] | 0.672 | [0.638, 0.709] | 1.304 | [1.097, 1.511] | 0.156 | 0.057 | 0.003 | <0.00050 | 1.97 | 1.83 |
| bulk/lr00.96/b08000 | 4 | 960 | alt | 0.867 | [0.789, 0.932] | 0.803 | [0.759, 0.857] | 1.080 | [0.963, 1.196] | 0.104 | 0.117 | 0.001 | 0.00400 | 0.69 | 0.70 |
| bulk/lr00.96/b08000 | 4 | 960 | dc | 0.846 | [0.765, 0.934] | 0.674 | [0.640, 0.707] | 1.254 | [1.104, 1.404] | 0.123 | 0.052 | 0.010 | <0.00050 | 1.93 | 1.79 |
| bulk/lr00.97/b04000 | 5 | 960 | alt | 0.901 | [0.825, 0.980] | 0.830 | [0.790, 0.878] | 1.085 | [0.971, 1.199] | 0.128 | 0.117 | 0.002 | 0.00400 | 0.69 | 0.68 |
| bulk/lr00.97/b04000 | 5 | 960 | dc | 0.871 | [0.807, 0.974] | 0.647 | [0.622, 0.685] | 1.346 | [1.203, 1.489] | 0.147 | 0.054 | 0.007 | <0.00050 | 2.00 | 1.84 |
| bulk/lr01.11/b04000 | 5 | 960 | alt | 0.836 | [0.779, 0.896] | 0.799 | [0.757, 0.837] | 1.046 | [0.951, 1.140] | 0.121 | 0.111 | 0.003 | 0.00500 | 0.69 | 0.70 |
| bulk/lr01.11/b04000 | 5 | 960 | dc | 0.942 | [0.890, 1.021] | 0.671 | [0.640, 0.705] | 1.405 | [1.296, 1.513] | 0.188 | 0.051 | 0.019 | <0.00050 | 1.95 | 1.83 |
| bulk/lr01.44/b00500 | 2 | 3072 | alt | 0.752 | [0.726, 0.783] | 0.737 | [0.693, 0.779] | 1.020 | [0.948, 1.092] | 0.086 | 0.091 | 0.003 | 0.00100 | 0.90 | 0.90 |
| bulk/lr01.44/b00500 | 2 | 3072 | dc | 0.889 | [0.836, 0.921] | 0.688 | [0.649, 0.722] | 1.292 | [1.201, 1.384] | 0.138 | 0.066 | 0.007 | <0.00050 | 1.42 | 1.35 |
| bulk/lr01.44/b01000 | 2 | 1536 | alt | 0.788 | [0.751, 0.854] | 0.785 | [0.747, 0.821] | 1.004 | [0.920, 1.088] | 0.102 | 0.085 | 0.004 | 0.00300 | 0.85 | 0.85 |
| bulk/lr01.44/b01000 | 2 | 1536 | dc | 0.906 | [0.859, 0.965] | 0.686 | [0.655, 0.720] | 1.322 | [1.224, 1.419] | 0.156 | 0.053 | 0.011 | <0.00050 | 1.53 | 1.43 |
| bulk/lr01.44/b02000 | 2 | 768 | alt | 0.821 | [0.757, 0.876] | 0.767 | [0.727, 0.809] | 1.070 | [0.964, 1.176] | 0.104 | 0.091 | 0.004 | 0.00400 | 0.82 | 0.80 |
| bulk/lr01.44/b02000 | 2 | 768 | dc | 0.878 | [0.780, 1.004] | 0.667 | [0.628, 0.694] | 1.315 | [1.144, 1.487] | 0.154 | 0.059 | 0.014 | <0.00050 | 1.65 | 1.53 |
| bulk/lr01.44/b04000 | 2 | 384 | alt | 0.769 | [0.676, 0.880] | 0.772 | [0.730, 0.809] | 0.996 | [0.858, 1.134] | 0.091 | 0.103 | 0.003 | 0.00350 | 0.74 | 0.74 |
| bulk/lr01.44/b04000 | 2 | 384 | dc | 0.962 | [0.899, 1.037] | 0.686 | [0.660, 0.727] | 1.402 | [1.273, 1.531] | 0.146 | 0.062 | 0.008 | <0.00050 | 1.77 | 1.68 |
| bulk/lr01.44/b08000 | 4 | 960 | alt | 0.897 | [0.844, 0.966] | 0.784 | [0.745, 0.834] | 1.143 | [1.042, 1.245] | 0.099 | 0.112 | 0.001 | 0.00400 | 0.72 | 0.71 |
| bulk/lr01.44/b08000 | 4 | 960 | dc | 0.835 | [0.769, 0.907] | 0.681 | [0.651, 0.715] | 1.226 | [1.102, 1.349] | 0.106 | 0.062 | 0.007 | <0.00050 | 1.86 | 1.78 |
| top/lr00.12/b00500 | 5 | 7680 | alt | 0.734 | [0.716, 0.753] | 0.726 | [0.688, 0.780] | 1.012 | [0.945, 1.078] | 0.076 | 0.078 | 0.002 | 0.00150 | 0.93 | 0.90 |
| top/lr00.12/b00500 | 5 | 7680 | dc | 0.529 | [0.510, 0.549] | 0.678 | [0.637, 0.717] | 0.780 | [0.725, 0.836] | 0.040 | 0.059 | 0.001 | <0.00050 | 1.62 | 1.35 |
| top/lr00.12/b01000 | 2 | 1536 | alt | 0.722 | [0.677, 0.783] | 0.761 | [0.722, 0.795] | 0.948 | [0.867, 1.029] | 0.089 | 0.087 | 0.003 | 0.00200 | 0.82 | 0.79 |
| top/lr00.12/b01000 | 2 | 1536 | dc | 0.588 | [0.541, 0.651] | 0.691 | [0.662, 0.724] | 0.852 | [0.763, 0.941] | 0.059 | 0.063 | 0.001 | <0.00050 | 1.92 | 1.54 |
| top/lr00.12/b02000 | 2 | 768 | alt | 0.924 | [0.814, 1.011] | 0.788 | [0.742, 0.841] | 1.173 | [1.019, 1.326] | 0.143 | 0.121 | 0.009 | 0.00600 | 0.67 | 0.64 |
| top/lr00.12/b02000 | 2 | 768 | dc | 0.621 | [0.548, 0.744] | 0.653 | [0.621, 0.682] | 0.952 | [0.800, 1.104] | 0.062 | 0.044 | 0.005 | <0.00050 | 2.36 | 1.92 |
| top/lr00.12/b04000 | 2 | 384 | alt | 0.936 | [0.873, 1.048] | 0.852 | [0.809, 0.900] | 1.099 | [0.984, 1.215] | 0.156 | 0.129 | 0.010 | 0.00750 | 0.59 | 0.57 |
| top/lr00.12/b04000 | 2 | 384 | dc | 0.804 | [0.703, 0.965] | 0.660 | [0.626, 0.685] | 1.220 | [1.032, 1.407] | 0.107 | 0.043 | 0.013 | <0.00050 | 2.63 | 2.29 |
| top/lr00.12/b08000 | 4 | 960 | alt | 1.121 | [1.035, 1.241] | 0.898 | [0.844, 0.950] | 1.249 | [1.108, 1.391] | 0.233 | 0.163 | 0.021 | 0.01150 | 0.52 | 0.52 |
| top/lr00.12/b08000 | 4 | 960 | dc | 0.916 | [0.808, 1.032] | 0.609 | [0.582, 0.642] | 1.504 | [1.304, 1.704] | 0.174 | 0.038 | 0.041 | <0.00050 | 3.03 | 2.56 |
| top/lr00.24/b00500 | 5 | 7680 | alt | 0.770 | [0.751, 0.792] | 0.824 | [0.785, 0.866] | 0.935 | [0.879, 0.991] | 0.098 | 0.099 | 0.003 | 0.00450 | 0.85 | 0.79 |
| top/lr00.24/b00500 | 5 | 7680 | dc | 0.941 | [0.880, 1.015] | 0.677 | [0.650, 0.713] | 1.391 | [1.274, 1.508] | 0.243 | 0.051 | 0.054 | <0.00050 | 1.90 | 1.52 |
| top/lr00.24/b01000 | 7 | 5376 | alt | 0.789 | [0.762, 0.814] | 0.802 | [0.772, 0.842] | 0.983 | [0.930, 1.037] | 0.096 | 0.110 | 0.004 | 0.00450 | 0.75 | 0.70 |
| top/lr00.24/b01000 | 7 | 5376 | dc | 0.997 | [0.913, 1.096] | 0.713 | [0.678, 0.748] | 1.399 | [1.252, 1.545] | 0.261 | 0.056 | 0.049 | <0.00050 | 2.27 | 1.76 |
| top/lr00.24/b02000 | 7 | 2688 | alt | 0.818 | [0.778, 0.860] | 0.874 | [0.840, 0.926] | 0.935 | [0.868, 1.003] | 0.119 | 0.134 | 0.003 | 0.00650 | 0.61 | 0.58 |
| top/lr00.24/b02000 | 7 | 2688 | dc | 1.035 | [0.950, 1.107] | 0.655 | [0.621, 0.697] | 1.579 | [1.425, 1.733] | 0.236 | 0.044 | 0.030 | <0.00050 | 2.74 | 2.20 |
| top/lr00.24/b04000 | 2 | 384 | alt | 1.060 | [0.927, 1.181] | 0.927 | [0.878, 0.977] | 1.143 | [1.003, 1.283] | 0.182 | 0.149 | 0.003 | 0.00750 | 0.52 | 0.52 |
| top/lr00.24/b04000 | 2 | 384 | dc | 0.706 | [0.611, 0.854] | 0.631 | [0.600, 0.667] | 1.118 | [0.920, 1.316] | 0.128 | 0.040 | 0.018 | <0.00050 | 3.10 | 2.58 |
| top/lr00.24/b08000 | 4 | 960 | alt | 0.977 | [0.898, 1.056] | 0.907 | [0.860, 0.968] | 1.077 | [0.966, 1.188] | 0.168 | 0.166 | 0.021 | 0.01150 | 0.48 | 0.49 |
| top/lr00.24/b08000 | 4 | 960 | dc | 0.740 | [0.675, 0.835] | 0.645 | [0.613, 0.678] | 1.147 | [1.002, 1.293] | 0.111 | 0.039 | 0.007 | <0.00050 | 3.34 | 2.84 |
| top/lr00.32/b01000 | 5 | 3840 | alt | 0.795 | [0.769, 0.821] | 0.865 | [0.827, 0.905] | 0.919 | [0.868, 0.971] | 0.109 | 0.108 | 0.007 | 0.00450 | 0.74 | 0.66 |
| top/lr00.32/b01000 | 5 | 3840 | dc | 1.581 | [1.331, 1.801] | 0.648 | [0.619, 0.686] | 2.440 | [2.052, 2.828] | 0.422 | 0.047 | 0.161 | <0.00050 | 2.40 | 1.89 |
| top/lr00.36/b00500 | 5 | 7680 | alt | 0.774 | [0.751, 0.799] | 0.831 | [0.792, 0.882] | 0.931 | [0.873, 0.990] | 0.095 | 0.115 | 0.005 | 0.00350 | 0.83 | 0.75 |
| top/lr00.36/b00500 | 5 | 7680 | dc | 1.601 | [1.450, 1.827] | 0.701 | [0.670, 0.736] | 2.285 | [1.993, 2.577] | 0.440 | 0.067 | 0.215 | <0.00050 | 2.08 | 1.68 |
| top/lr00.36/b01000 | 2 | 1536 | alt | 0.771 | [0.715, 0.834] | 0.820 | [0.770, 0.858] | 0.940 | [0.851, 1.029] | 0.115 | 0.121 | 0.003 | 0.00550 | 0.75 | 0.66 |
| top/lr00.36/b01000 | 2 | 1536 | dc | 1.872 | [1.426, 2.301] | 0.661 | [0.627, 0.690] | 2.833 | [2.178, 3.488] | 0.475 | 0.044 | 0.219 | <0.00050 | 2.45 | 1.93 |
| top/lr00.36/b02000 | 2 | 768 | alt | 0.818 | [0.768, 0.900] | 0.855 | [0.817, 0.890] | 0.957 | [0.871, 1.043] | 0.120 | 0.146 | 0.004 | 0.01050 | 0.63 | 0.57 |
| top/lr00.36/b02000 | 2 | 768 | dc | 1.794 | [1.413, 2.125] | 0.643 | [0.608, 0.681] | 2.787 | [2.207, 3.367] | 0.462 | 0.043 | 0.155 | <0.00050 | 2.89 | 2.28 |
| top/lr00.36/b04000 | 7 | 1344 | alt | 0.997 | [0.916, 1.056] | 0.869 | [0.816, 0.924] | 1.147 | [1.036, 1.258] | 0.157 | 0.147 | 0.002 | 0.01400 | 0.52 | 0.52 |
| top/lr00.36/b04000 | 7 | 1344 | dc | 0.972 | [0.855, 1.062] | 0.645 | [0.605, 0.670] | 1.507 | [1.333, 1.682] | 0.192 | 0.029 | 0.013 | <0.00050 | 3.07 | 2.56 |
| top/lr00.36/b08000 | 4 | 960 | alt | 1.011 | [0.922, 1.070] | 0.949 | [0.896, 0.998] | 1.065 | [0.968, 1.162] | 0.158 | 0.168 | 0.007 | 0.01100 | 0.50 | 0.49 |
| top/lr00.36/b08000 | 4 | 960 | dc | 0.855 | [0.770, 0.971] | 0.623 | [0.594, 0.655] | 1.372 | [1.209, 1.535] | 0.161 | 0.038 | 0.016 | <0.00050 | 3.33 | 2.77 |
| top/lr00.37/b01000 | 5 | 3840 | alt | 0.821 | [0.787, 0.858] | 0.788 | [0.742, 0.841] | 1.042 | [0.963, 1.122] | 0.121 | 0.121 | 0.006 | 0.00600 | 0.74 | 0.64 |
| top/lr00.37/b01000 | 5 | 3840 | dc | 1.988 | [1.699, 2.235] | 0.653 | [0.621, 0.682] | 3.047 | [2.613, 3.480] | 0.499 | 0.044 | 0.240 | <0.00050 | 2.47 | 1.92 |
| top/lr00.42/b01000 | 5 | 3840 | alt | 0.819 | [0.780, 0.857] | 0.844 | [0.803, 0.885] | 0.970 | [0.902, 1.038] | 0.118 | 0.114 | 0.007 | 0.00550 | 0.76 | 0.64 |
| top/lr00.42/b01000 | 5 | 3840 | dc | 2.378 | [2.033, 2.711] | 0.650 | [0.619, 0.682] | 3.656 | [3.105, 4.207] | 0.546 | 0.049 | 0.311 | <0.00050 | 2.49 | 1.96 |
| top/lr00.42/b02000 | 5 | 1920 | alt | 0.842 | [0.793, 0.886] | 0.867 | [0.828, 0.903] | 0.971 | [0.900, 1.043] | 0.130 | 0.128 | 0.004 | 0.00700 | 0.64 | 0.58 |
| top/lr00.42/b02000 | 5 | 1920 | dc | 2.385 | [2.093, 2.658] | 0.637 | [0.603, 0.670] | 3.747 | [3.263, 4.232] | 0.563 | 0.036 | 0.247 | <0.00050 | 2.85 | 2.24 |
| top/lr00.48/b00500 | 4 | 6144 | alt | 0.826 | [0.807, 0.855] | 0.813 | [0.772, 0.851] | 1.016 | [0.961, 1.071] | 0.112 | 0.102 | 0.004 | 0.00200 | 0.84 | 0.70 |
| top/lr00.48/b00500 | 4 | 6144 | dc | 2.472 | [2.177, 2.773] | 0.637 | [0.603, 0.683] | 3.882 | [3.362, 4.402] | 0.558 | 0.046 | 0.334 | <0.00050 | 2.16 | 1.76 |
| top/lr00.48/b01000 | 7 | 5376 | alt | 0.808 | [0.780, 0.837] | 0.846 | [0.803, 0.890] | 0.955 | [0.898, 1.012] | 0.110 | 0.109 | 0.006 | 0.00550 | 0.76 | 0.64 |
| top/lr00.48/b01000 | 7 | 5376 | dc | 2.790 | [2.471, 3.109] | 0.669 | [0.634, 0.702] | 4.169 | [3.655, 4.682] | 0.595 | 0.051 | 0.364 | 0.00100 | 2.50 | 1.94 |
| top/lr00.48/b02000 | 7 | 2688 | alt | 0.853 | [0.807, 0.901] | 0.856 | [0.822, 0.906] | 0.997 | [0.925, 1.069] | 0.116 | 0.124 | 0.006 | 0.00400 | 0.65 | 0.57 |
| top/lr00.48/b02000 | 7 | 2688 | dc | 2.610 | [2.367, 2.892] | 0.643 | [0.608, 0.674] | 4.062 | [3.615, 4.509] | 0.594 | 0.043 | 0.309 | <0.00050 | 2.82 | 2.27 |
| top/lr00.48/b04000 | 2 | 384 | alt | 0.818 | [0.744, 0.914] | 0.926 | [0.864, 0.979] | 0.883 | [0.776, 0.991] | 0.107 | 0.149 | 0.003 | 0.01550 | 0.52 | 0.52 |
| top/lr00.48/b04000 | 2 | 384 | dc | 1.200 | [1.069, 1.425] | 0.644 | [0.615, 0.678] | 1.864 | [1.564, 2.165] | 0.286 | 0.034 | 0.026 | <0.00050 | 3.00 | 2.53 |
| top/lr00.48/b08000 | 4 | 960 | alt | 0.968 | [0.895, 1.055] | 0.906 | [0.848, 0.963] | 1.069 | [0.958, 1.180] | 0.168 | 0.170 | 0.007 | 0.01200 | 0.50 | 0.49 |
| top/lr00.48/b08000 | 4 | 960 | dc | 0.996 | [0.899, 1.124] | 0.582 | [0.545, 0.609] | 1.712 | [1.496, 1.929] | 0.232 | 0.041 | 0.047 | <0.00050 | 3.26 | 2.77 |
| top/lr00.55/b01000 | 5 | 3840 | alt | 0.851 | [0.809, 0.895] | 0.871 | [0.814, 0.911] | 0.977 | [0.904, 1.050] | 0.135 | 0.131 | 0.012 | 0.00500 | 0.76 | 0.63 |
| top/lr00.55/b01000 | 5 | 3840 | dc | 3.075 | [2.718, 3.471] | 0.642 | [0.614, 0.679] | 4.789 | [4.163, 5.414] | 0.630 | 0.049 | 0.401 | <0.00050 | 2.53 | 2.01 |
| top/lr00.55/b02000 | 5 | 1920 | alt | 0.825 | [0.772, 0.871] | 0.839 | [0.802, 0.871] | 0.982 | [0.912, 1.052] | 0.115 | 0.149 | 0.005 | 0.00750 | 0.67 | 0.58 |
| top/lr00.55/b02000 | 5 | 1920 | dc | 2.956 | [2.609, 3.361] | 0.690 | [0.666, 0.720] | 4.281 | [3.716, 4.846] | 0.635 | 0.048 | 0.367 | <0.00050 | 2.79 | 2.20 |
| top/lr00.55/b04000 | 5 | 960 | alt | 0.950 | [0.890, 1.014] | 0.893 | [0.855, 0.943] | 1.063 | [0.971, 1.156] | 0.144 | 0.150 | 0.005 | 0.00850 | 0.55 | 0.54 |
| top/lr00.55/b04000 | 5 | 960 | dc | 1.546 | [1.430, 1.673] | 0.619 | [0.591, 0.662] | 2.497 | [2.262, 2.732] | 0.364 | 0.043 | 0.049 | <0.00050 | 2.98 | 2.46 |
| top/lr00.60/b00500 | 4 | 6144 | alt | 0.800 | [0.773, 0.830] | 0.802 | [0.772, 0.842] | 0.997 | [0.939, 1.055] | 0.117 | 0.110 | 0.008 | 0.00450 | 0.86 | 0.70 |
| top/lr00.60/b00500 | 4 | 6144 | dc | 3.209 | [2.895, 3.526] | 0.713 | [0.678, 0.748] | 4.499 | [4.001, 4.998] | 0.632 | 0.056 | 0.419 | <0.00050 | 2.17 | 1.76 |
| top/lr00.60/b01000 | 2 | 1536 | alt | 0.796 | [0.745, 0.850] | 0.851 | [0.793, 0.886] | 0.935 | [0.848, 1.021] | 0.113 | 0.116 | 0.003 | 0.00700 | 0.79 | 0.63 |
| top/lr00.60/b01000 | 2 | 1536 | dc | 3.441 | [2.848, 4.072] | 0.665 | [0.631, 0.703] | 5.177 | [4.232, 6.122] | 0.668 | 0.058 | 0.441 | <0.00050 | 2.44 | 1.99 |
| top/lr00.60/b02000 | 2 | 768 | alt | 0.855 | [0.774, 0.962] | 0.828 | [0.785, 0.865] | 1.033 | [0.901, 1.165] | 0.129 | 0.135 | 0.008 | 0.00650 | 0.69 | 0.58 |
| top/lr00.60/b02000 | 2 | 768 | dc | 3.310 | [2.717, 3.937] | 0.647 | [0.615, 0.679] | 5.116 | [4.093, 6.139] | 0.655 | 0.042 | 0.418 | <0.00050 | 2.77 | 2.21 |
| top/lr00.60/b04000 | 2 | 384 | alt | 1.045 | [0.926, 1.125] | 0.865 | [0.825, 0.917] | 1.209 | [1.076, 1.342] | 0.169 | 0.124 | 0.003 | 0.00750 | 0.55 | 0.54 |
| top/lr00.60/b04000 | 2 | 384 | dc | 1.620 | [1.493, 1.766] | 0.622 | [0.582, 0.653] | 2.607 | [2.328, 2.886] | 0.365 | 0.043 | 0.073 | <0.00050 | 2.97 | 2.44 |
| top/lr00.60/b08000 | 4 | 960 | alt | 0.860 | [0.779, 0.960] | 0.952 | [0.899, 1.008] | 0.904 | [0.797, 1.011] | 0.117 | 0.154 | 0.003 | 0.00850 | 0.51 | 0.52 |
| top/lr00.60/b08000 | 4 | 960 | dc | 1.267 | [1.095, 1.500] | 0.637 | [0.601, 0.665] | 1.990 | [1.637, 2.343] | 0.333 | 0.041 | 0.103 | <0.00050 | 3.10 | 2.56 |
| top/lr00.64/b01000 | 5 | 3840 | alt | 0.793 | [0.747, 0.840] | 0.844 | [0.806, 0.884] | 0.940 | [0.868, 1.011] | 0.115 | 0.113 | 0.008 | 0.00400 | 0.80 | 0.64 |
| top/lr00.64/b01000 | 5 | 3840 | dc | 3.583 | [3.238, 4.037] | 0.661 | [0.629, 0.687] | 5.420 | [4.755, 6.086] | 0.671 | 0.052 | 0.462 | <0.00050 | 2.42 | 1.97 |
| top/lr00.64/b02000 | 5 | 1920 | alt | 0.835 | [0.789, 0.896] | 0.827 | [0.795, 0.879] | 1.009 | [0.928, 1.090] | 0.128 | 0.133 | 0.004 | 0.00500 | 0.69 | 0.59 |
| top/lr00.64/b02000 | 5 | 1920 | dc | 3.283 | [2.859, 3.748] | 0.653 | [0.626, 0.686] | 5.025 | [4.266, 5.784] | 0.646 | 0.043 | 0.417 | <0.00050 | 2.70 | 2.16 |
| top/lr00.64/b04000 | 5 | 960 | alt | 0.930 | [0.848, 1.006] | 0.857 | [0.809, 0.910] | 1.085 | [0.977, 1.193] | 0.166 | 0.132 | 0.003 | 0.00600 | 0.56 | 0.55 |
| top/lr00.64/b04000 | 5 | 960 | dc | 1.759 | [1.641, 1.903] | 0.612 | [0.591, 0.654] | 2.877 | [2.591, 3.163] | 0.432 | 0.036 | 0.060 | <0.00050 | 2.80 | 2.41 |
| top/lr00.72/b00500 | 4 | 6144 | alt | 0.807 | [0.777, 0.836] | 0.758 | [0.722, 0.798] | 1.065 | [0.995, 1.134] | 0.123 | 0.095 | 0.007 | 0.00250 | 0.88 | 0.70 |
| top/lr00.72/b00500 | 4 | 6144 | dc | 3.630 | [3.307, 3.984] | 0.640 | [0.608, 0.678] | 5.675 | [5.053, 6.298] | 0.676 | 0.045 | 0.464 | <0.00050 | 2.10 | 1.75 |
| top/lr00.72/b01000 | 2 | 1536 | alt | 0.825 | [0.776, 0.886] | 0.788 | [0.761, 0.823] | 1.047 | [0.964, 1.129] | 0.126 | 0.107 | 0.005 | 0.00500 | 0.82 | 0.64 |
| top/lr00.72/b01000 | 2 | 1536 | dc | 4.219 | [3.440, 5.022] | 0.647 | [0.615, 0.680] | 6.524 | [5.244, 7.805] | 0.715 | 0.051 | 0.517 | <0.00050 | 2.39 | 1.97 |
| top/lr00.72/b02000 | 2 | 768 | alt | 0.886 | [0.802, 0.970] | 0.843 | [0.794, 0.888] | 1.050 | [0.938, 1.162] | 0.138 | 0.128 | 0.009 | 0.00800 | 0.71 | 0.60 |
| top/lr00.72/b02000 | 2 | 768 | dc | 3.390 | [2.705, 4.007] | 0.619 | [0.592, 0.657] | 5.476 | [4.337, 6.615] | 0.664 | 0.041 | 0.431 | <0.00050 | 2.64 | 2.10 |
| top/lr00.72/b04000 | 2 | 384 | alt | 0.846 | [0.744, 0.997] | 0.912 | [0.868, 0.962] | 0.927 | [0.792, 1.061] | 0.130 | 0.143 | 0.003 | 0.00900 | 0.56 | 0.55 |
| top/lr00.72/b04000 | 2 | 384 | dc | 1.944 | [1.828, 2.130] | 0.638 | [0.611, 0.662] | 3.049 | [2.772, 3.325] | 0.479 | 0.040 | 0.073 | <0.00050 | 2.72 | 2.40 |
| top/lr00.72/b08000 | 4 | 960 | alt | 0.815 | [0.752, 0.906] | 0.916 | [0.867, 0.969] | 0.889 | [0.797, 0.981] | 0.119 | 0.152 | 0.006 | 0.01650 | 0.54 | 0.54 |
| top/lr00.72/b08000 | 4 | 960 | dc | 1.408 | [1.194, 1.779] | 0.616 | [0.587, 0.653] | 2.286 | [1.784, 2.787] | 0.382 | 0.035 | 0.139 | <0.00050 | 2.91 | 2.47 |
| top/lr00.73/b02000 | 5 | 1920 | alt | 0.839 | [0.801, 0.899] | 0.837 | [0.790, 0.880] | 1.002 | [0.924, 1.080] | 0.140 | 0.136 | 0.007 | 0.00300 | 0.71 | 0.59 |
| top/lr00.73/b02000 | 5 | 1920 | dc | 3.671 | [3.207, 4.156] | 0.649 | [0.615, 0.684] | 5.659 | [4.885, 6.434] | 0.690 | 0.048 | 0.465 | <0.00050 | 2.61 | 2.16 |
| top/lr00.73/b04000 | 5 | 960 | alt | 0.889 | [0.833, 0.968] | 0.901 | [0.855, 0.945] | 0.987 | [0.896, 1.078] | 0.157 | 0.157 | 0.009 | 0.00600 | 0.58 | 0.56 |
| top/lr00.73/b04000 | 5 | 960 | dc | 2.006 | [1.875, 2.140] | 0.632 | [0.606, 0.671] | 3.175 | [2.905, 3.444] | 0.502 | 0.038 | 0.095 | <0.00050 | 2.67 | 2.39 |
| top/lr00.84/b02000 | 5 | 1920 | alt | 0.897 | [0.841, 0.947] | 0.868 | [0.825, 0.909] | 1.033 | [0.954, 1.111] | 0.140 | 0.127 | 0.006 | 0.00450 | 0.73 | 0.61 |
| top/lr00.84/b02000 | 5 | 1920 | dc | 3.756 | [3.215, 4.310] | 0.682 | [0.640, 0.718] | 5.509 | [4.657, 6.361] | 0.702 | 0.052 | 0.474 | 0.00100 | 2.52 | 2.09 |
| top/lr00.84/b04000 | 5 | 960 | alt | 0.875 | [0.812, 0.944] | 0.865 | [0.820, 0.917] | 1.011 | [0.914, 1.107] | 0.129 | 0.120 | 0.004 | 0.00750 | 0.60 | 0.58 |
| top/lr00.84/b04000 | 5 | 960 | dc | 2.168 | [2.051, 2.335] | 0.650 | [0.606, 0.680] | 3.335 | [3.040, 3.631] | 0.562 | 0.043 | 0.101 | <0.00050 | 2.44 | 2.24 |
| top/lr00.96/b00500 | 4 | 6144 | alt | 0.792 | [0.766, 0.822] | 0.820 | [0.786, 0.851] | 0.965 | [0.915, 1.015] | 0.115 | 0.105 | 0.009 | 0.00600 | 0.91 | 0.72 |
| top/lr00.96/b00500 | 4 | 6144 | dc | 4.006 | [3.734, 4.264] | 0.657 | [0.622, 0.687] | 6.101 | [5.580, 6.622] | 0.722 | 0.054 | 0.502 | <0.00050 | 1.93 | 1.70 |
| top/lr00.96/b01000 | 2 | 1536 | alt | 0.819 | [0.779, 0.873] | 0.808 | [0.781, 0.846] | 1.013 | [0.942, 1.084] | 0.104 | 0.107 | 0.004 | 0.00200 | 0.88 | 0.68 |
| top/lr00.96/b01000 | 2 | 1536 | dc | 4.636 | [4.041, 5.335] | 0.636 | [0.605, 0.678] | 7.289 | [6.150, 8.429] | 0.760 | 0.048 | 0.572 | 0.00100 | 2.18 | 1.81 |
| top/lr00.96/b02000 | 2 | 768 | alt | 0.766 | [0.696, 0.850] | 0.820 | [0.779, 0.858] | 0.933 | [0.833, 1.034] | 0.099 | 0.116 | 0.004 | 0.00350 | 0.75 | 0.62 |
| top/lr00.96/b02000 | 2 | 768 | dc | 3.648 | [3.067, 4.706] | 0.675 | [0.639, 0.720] | 5.406 | [4.124, 6.688] | 0.728 | 0.046 | 0.469 | <0.00050 | 2.38 | 2.04 |
| top/lr00.96/b04000 | 2 | 384 | alt | 0.944 | [0.857, 1.038] | 0.858 | [0.811, 0.900] | 1.101 | [0.978, 1.224] | 0.161 | 0.150 | 0.008 | 0.00900 | 0.61 | 0.60 |
| top/lr00.96/b04000 | 2 | 384 | dc | 2.238 | [1.951, 2.538] | 0.667 | [0.639, 0.698] | 3.355 | [2.878, 3.832] | 0.560 | 0.042 | 0.102 | <0.00050 | 2.29 | 2.15 |
| top/lr00.96/b08000 | 4 | 960 | alt | 0.869 | [0.807, 0.954] | 0.837 | [0.795, 0.885] | 1.039 | [0.936, 1.142] | 0.134 | 0.134 | 0.003 | 0.00650 | 0.61 | 0.57 |
| top/lr00.96/b08000 | 4 | 960 | dc | 1.735 | [1.475, 2.144] | 0.637 | [0.616, 0.671] | 2.725 | [2.134, 3.316] | 0.446 | 0.041 | 0.223 | 0.00150 | 2.68 | 2.29 |
| top/lr00.97/b04000 | 5 | 960 | alt | 0.896 | [0.825, 0.968] | 0.846 | [0.805, 0.888] | 1.060 | [0.965, 1.154] | 0.145 | 0.127 | 0.009 | 0.00750 | 0.62 | 0.58 |
| top/lr00.97/b04000 | 5 | 960 | dc | 2.261 | [2.140, 2.431] | 0.651 | [0.622, 0.688] | 3.474 | [3.179, 3.770] | 0.590 | 0.042 | 0.120 | <0.00050 | 2.34 | 2.18 |
| top/lr01.11/b04000 | 5 | 960 | alt | 0.905 | [0.834, 0.983] | 0.823 | [0.789, 0.861] | 1.099 | [0.993, 1.204] | 0.134 | 0.118 | 0.006 | 0.00700 | 0.65 | 0.61 |
| top/lr01.11/b04000 | 5 | 960 | dc | 2.573 | [2.384, 2.738] | 0.638 | [0.601, 0.668] | 4.032 | [3.693, 4.370] | 0.637 | 0.044 | 0.151 | 0.00100 | 2.20 | 2.10 |
| top/lr01.44/b00500 | 2 | 3072 | alt | 0.791 | [0.759, 0.827] | 0.786 | [0.744, 0.845] | 1.005 | [0.923, 1.087] | 0.104 | 0.108 | 0.004 | 0.00150 | 0.94 | 0.78 |
| top/lr01.44/b00500 | 2 | 3072 | dc | 3.572 | [3.231, 3.902] | 0.684 | [0.653, 0.716] | 5.222 | [4.693, 5.751] | 0.714 | 0.046 | 0.440 | <0.00050 | 1.69 | 1.57 |
| top/lr01.44/b01000 | 2 | 1536 | alt | 0.829 | [0.782, 0.873] | 0.800 | [0.762, 0.847] | 1.035 | [0.953, 1.118] | 0.131 | 0.102 | 0.009 | 0.00500 | 0.91 | 0.71 |
| top/lr01.44/b01000 | 2 | 1536 | dc | 4.234 | [3.637, 4.817] | 0.634 | [0.605, 0.669] | 6.680 | [5.698, 7.661] | 0.734 | 0.056 | 0.529 | <0.00050 | 1.91 | 1.73 |
| top/lr01.44/b02000 | 2 | 768 | alt | 0.856 | [0.792, 0.946] | 0.832 | [0.772, 0.874] | 1.029 | [0.902, 1.155] | 0.125 | 0.114 | 0.008 | 0.00400 | 0.85 | 0.69 |
| top/lr01.44/b02000 | 2 | 768 | dc | 3.570 | [3.109, 4.052] | 0.672 | [0.633, 0.710] | 5.311 | [4.538, 6.085] | 0.746 | 0.052 | 0.424 | <0.00050 | 1.94 | 1.82 |
| top/lr01.44/b04000 | 2 | 384 | alt | 0.809 | [0.732, 0.885] | 0.845 | [0.796, 0.884] | 0.958 | [0.844, 1.071] | 0.112 | 0.124 | 0.003 | 0.00650 | 0.73 | 0.69 |
| top/lr01.44/b04000 | 2 | 384 | dc | 2.561 | [2.212, 2.882] | 0.692 | [0.661, 0.722] | 3.699 | [3.200, 4.197] | 0.630 | 0.051 | 0.182 | <0.00050 | 1.79 | 1.82 |
| top/lr01.44/b08000 | 4 | 960 | alt | 0.861 | [0.802, 0.950] | 0.838 | [0.792, 0.881] | 1.028 | [0.925, 1.130] | 0.129 | 0.119 | 0.005 | 0.00400 | 0.67 | 0.62 |
| top/lr01.44/b08000 | 4 | 960 | dc | 1.892 | [1.646, 2.274] | 0.684 | [0.653, 0.718] | 2.768 | [2.269, 3.268] | 0.478 | 0.049 | 0.216 | <0.00050 | 2.20 | 2.07 |

## 5. Burn-in sensitivity (pooled over lr and batch)

Every quantity above is read at burn-in 5; this table is the same
pooled read at [0, 5, 15, 25]. Burn-in 0 is the A2 row: the
re-anchoring transient at the head of a segment contaminates the
ladder and does not average out over segments, and 0 is the only
value that can show it — the {5, 15, 25} rows are nested windows of
the SAME segments (n_kept 45/35/25), so their |t| shrinks mechanically
like sqrt(n_kept) whether or not a transient exists.

| kind | burn-in | n_seg | n_kept | rho_1 | rho_2 | rho_3 | rho_4 | tau_nw(L) | dc median \|t\| | dc ratio | alt median \|t\| | alt ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulk/lrALL/bALL | 0 | 126912 | 50.0 | -0.168 | 0.025 | -0.029 | 0.006 | 0.759 | 0.760 | 1.091 | 0.771 | 1.012 |
| bulk/lrALL/bALL | 5 | 126912 | 45.0 | -0.172 | 0.027 | -0.029 | 0.007 | 0.755 | 0.753 | 1.081 | 0.780 | 1.020 |
| bulk/lrALL/bALL | 15 | 126912 | 35.0 | -0.168 | 0.029 | -0.028 | 0.009 | 0.763 | 0.746 | 1.047 | 0.794 | 1.041 |
| bulk/lrALL/bALL | 25 | 126912 | 25.0 | -0.161 | 0.031 | -0.023 | 0.011 | 0.806 | 0.702 | 0.981 | 0.817 | 1.013 |
| top/lrALL/bALL | 0 | 126912 | 50.0 | -0.265 | 0.009 | -0.043 | 0.001 | 0.589 | 1.626 | 2.432 | 0.883 | 1.100 |
| top/lrALL/bALL | 5 | 126912 | 45.0 | -0.343 | 0.071 | -0.054 | 0.029 | 0.529 | 1.897 | 2.917 | 0.818 | 1.022 |
| top/lrALL/bALL | 15 | 126912 | 35.0 | -0.341 | 0.071 | -0.053 | 0.029 | 0.532 | 1.750 | 2.666 | 0.834 | 1.028 |
| top/lrALL/bALL | 25 | 126912 | 25.0 | -0.325 | 0.071 | -0.045 | 0.032 | 0.614 | 1.371 | 1.988 | 0.860 | 1.016 |

## 6. Diagnostics

- mirror check against `src.stats.spectral` on 64 seeded-uniform segments and 8 slots — coverage: slot positions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 23, 25, 26, 27, 28, 29, 30, 31], raw segment lengths [42, 46, 48, 50]: abs_t_nw_alt_parity 1.776e-15, ess_alt 2.842e-14, ess_dc 7.390e-13, rho 3.886e-16, rho_raw 3.886e-16, slot_acf1_alt 1.110e-16, slot_acf1_dc 1.110e-16, slot_coherence_alt 0.000e+00, slot_coherence_dc 0.000e+00, slot_ess_alt 5.684e-14, slot_ess_dc 7.105e-14, slot_t_nw_alt 8.882e-16, slot_t_nw_dc 8.882e-16, t_nw_alt 1.776e-15, t_nw_dc 2.132e-14 (batched kernel vs canonical scalar functions)
- segment-start parity: 7932 of 7932 starts are odd — uniform: True. Mixed parity would flip the alternating channel's sign between segments of one slot and cancel a real signal in §7; the pooling refuses to run on a mixed-parity set.
- slot shapes (raw segment lengths per slot, ×slots): {'48': 12288, '50x46': 39168, '50x50x50x42': 12288, '50x50x50x50': 39168, '50x50x50x50x50x50x50x50': 39168, '50x50x50x50x50x50x50x50x50x50x50x50x50x50x50x50': 25344}
- refreshes that did not reset the direction (alignment >= align_min): 69605 of 211968 (0.328); with `--segment-at refresh` those are still cut, with `--segment-at reset` they are not
- Newey-West flooring, per channel (pooled, burn-in 5):
  - bulk/lrALL/bALL: dc 0.0000, alt 0.0000
  - top/lrALL/bALL: dc 0.0000, alt 0.0000
- the two channels are bootstrapped with the same seed and therefore the same resample index sets (common random numbers): deliberate, because the dc-vs-alt contrast is the point, but it means the two intervals are not independent of each other.

## 7. Slot-level read (prereg §5.1 unit) — the discriminator

Sections 1–6 cannot distinguish a persistent per-direction mean
from zero-mean low-frequency power inside a segment; §0's K3
control shows a zero-mean stream reproducing the whole `top` dc
cell. This section pools each slot's segments into the
pre-registered slot-level statistic (lag products never crossing
a segment boundary) and reports the **calibrated growth factor**
= slot ratio / segment ratio. The two hypotheses predict
different values:

- a mean that survives a subspace refresh → growth ≈ √k;
- zero-mean low-frequency power inside segments → growth ≈ 1,
  because the Newey-West bandwidth grows with N (L = 3 at n = 45,
  L = 4 at N = 180, L = 5 at N ≈ 360) and begins absorbing that
  power.

`seg-mean coherence` is the fraction of a slot's segment means
sharing the modal sign (null reference in the next column);
`seg-mean acf1` is their lag-1 autocorrelation about zero, whose
ceiling is (k−1)/k, not 1. Nulls are drawn at the exact slot
shapes present, never a modal or median shape.

### 7a. Pooled, per burn-in

burn-in 0:

| group | n_slots | k (median) | N | L | ch | slot median \|t\| | slot null \|t\| | slot ratio | growth (cal.) | sqrt(k) | seg-mean coherence | null coherence | seg-mean acf1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulk/lrALL/bALL | 20928 | 4.0 | 200 | 4 | alt | 0.760 | 0.732 | 1.039 | 1.026 | 2.000 | 0.625 | 0.625 | 0.006 |
| bulk/lrALL/bALL | 20928 | 4.0 | 200 | 4 | dc | 0.929 | 0.674 | 1.380 | 1.264 | 2.000 | 0.750 | 0.625 | 0.073 |
| top/lrALL/bALL | 20928 | 4.0 | 200 | 4 | alt | 1.660 | 0.725 | 2.290 | 2.082 | 2.000 | 0.938 | 0.625 | 0.380 |
| top/lrALL/bALL | 20928 | 4.0 | 200 | 4 | dc | 3.778 | 0.648 | 5.834 | 2.399 | 2.000 | 1.000 | 0.625 | 0.669 |

burn-in 5:

| group | n_slots | k (median) | N | L | ch | slot median \|t\| | slot null \|t\| | slot ratio | growth (cal.) | sqrt(k) | seg-mean coherence | null coherence | seg-mean acf1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulk/lrALL/bALL | 20928 | 4.0 | 180 | 4 | alt | 0.773 | 0.718 | 1.077 | 1.056 | 2.000 | 0.625 | 0.688 | 0.006 |
| bulk/lrALL/bALL | 20928 | 4.0 | 180 | 4 | dc | 0.909 | 0.665 | 1.367 | 1.265 | 2.000 | 0.750 | 0.688 | 0.072 |
| top/lrALL/bALL | 20928 | 4.0 | 180 | 4 | alt | 0.808 | 0.747 | 1.082 | 1.058 | 2.000 | 0.625 | 0.625 | 0.002 |
| top/lrALL/bALL | 20928 | 4.0 | 180 | 4 | dc | 3.596 | 0.607 | 5.926 | 2.031 | 2.000 | 0.938 | 0.625 | 0.540 |

burn-in 15:

| group | n_slots | k (median) | N | L | ch | slot median \|t\| | slot null \|t\| | slot ratio | growth (cal.) | sqrt(k) | seg-mean coherence | null coherence | seg-mean acf1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulk/lrALL/bALL | 20928 | 4.0 | 140 | 4 | alt | 0.773 | 0.732 | 1.056 | 1.014 | 2.000 | 0.625 | 0.625 | 0.001 |
| bulk/lrALL/bALL | 20928 | 4.0 | 140 | 4 | dc | 0.865 | 0.689 | 1.256 | 1.200 | 2.000 | 0.750 | 0.625 | 0.064 |
| top/lrALL/bALL | 20928 | 4.0 | 140 | 4 | alt | 0.818 | 0.795 | 1.029 | 1.000 | 2.000 | 0.625 | 0.688 | 0.005 |
| top/lrALL/bALL | 20928 | 4.0 | 140 | 4 | dc | 3.293 | 0.643 | 5.124 | 1.922 | 2.000 | 1.000 | 0.625 | 0.539 |

burn-in 25:

| group | n_slots | k (median) | N | L | ch | slot median \|t\| | slot null \|t\| | slot ratio | growth (cal.) | sqrt(k) | seg-mean coherence | null coherence | seg-mean acf1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulk/lrALL/bALL | 20928 | 4.0 | 100 | 4 | alt | 0.784 | 0.749 | 1.046 | 1.033 | 2.000 | 0.625 | 0.625 | -0.003 |
| bulk/lrALL/bALL | 20928 | 4.0 | 100 | 4 | dc | 0.797 | 0.678 | 1.175 | 1.199 | 2.000 | 0.688 | 0.625 | 0.049 |
| top/lrALL/bALL | 20928 | 4.0 | 100 | 4 | alt | 0.839 | 0.742 | 1.130 | 1.113 | 2.000 | 0.625 | 0.625 | 0.005 |
| top/lrALL/bALL | 20928 | 4.0 | 100 | 4 | dc | 2.712 | 0.663 | 4.088 | 2.056 | 2.000 | 0.875 | 0.625 | 0.508 |

### 7b. kind × lr (burn-in 5)

| group | n_slots | k (median) | N | L | ch | slot median \|t\| | slot null \|t\| | slot ratio | growth (cal.) | sqrt(k) | seg-mean coherence | null coherence | seg-mean acf1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulk/lr00.12/bALL | 1440 | 4.0 | 180 | 4 | alt | 0.789 | 0.712 | 1.108 | 1.082 | 2.000 | 0.625 | 0.625 | -0.016 |
| bulk/lr00.12/bALL | 1440 | 4.0 | 180 | 4 | dc | 0.747 | 0.684 | 1.092 | 1.218 | 2.000 | 0.625 | 0.625 | 0.005 |
| bulk/lr00.24/bALL | 2400 | 4.0 | 180 | 4 | alt | 0.755 | 0.688 | 1.097 | 1.111 | 2.000 | 0.625 | 0.625 | 0.005 |
| bulk/lr00.24/bALL | 2400 | 4.0 | 180 | 4 | dc | 0.806 | 0.663 | 1.216 | 1.217 | 2.000 | 0.750 | 0.625 | 0.054 |
| bulk/lr00.32/bALL | 480 | 8.0 | 360 | 5 | alt | 0.707 | 0.675 | 1.047 | 1.039 | 2.828 | 0.625 | 0.625 | 0.029 |
| bulk/lr00.32/bALL | 480 | 8.0 | 360 | 5 | dc | 0.873 | 0.687 | 1.270 | 1.225 | 2.828 | 0.625 | 0.625 | 0.080 |
| bulk/lr00.36/bALL | 1920 | 4.0 | 172 | 4 | alt | 0.758 | 0.731 | 1.037 | 1.028 | 2.000 | 0.625 | 0.625 | 0.007 |
| bulk/lr00.36/bALL | 1920 | 4.0 | 172 | 4 | dc | 0.844 | 0.729 | 1.158 | 1.124 | 2.000 | 0.688 | 0.625 | 0.050 |
| bulk/lr00.37/bALL | 480 | 8.0 | 360 | 5 | alt | 0.744 | 0.696 | 1.069 | 1.047 | 2.828 | 0.625 | 0.625 | 0.010 |
| bulk/lr00.37/bALL | 480 | 8.0 | 360 | 5 | dc | 0.951 | 0.666 | 1.429 | 1.363 | 2.828 | 0.625 | 0.625 | 0.129 |
| bulk/lr00.42/bALL | 960 | 6.0 | 270 | 4 | alt | 0.714 | 0.744 | 0.959 | 0.916 | 2.414 | 0.625 | 0.625 | 0.023 |
| bulk/lr00.42/bALL | 960 | 6.0 | 270 | 4 | dc | 0.917 | 0.696 | 1.317 | 1.270 | 2.414 | 0.750 | 0.625 | 0.127 |
| bulk/lr00.48/bALL | 2304 | 4.0 | 180 | 4 | alt | 0.743 | 0.752 | 0.989 | 0.970 | 2.000 | 0.625 | 0.625 | 0.001 |
| bulk/lr00.48/bALL | 2304 | 4.0 | 180 | 4 | dc | 0.939 | 0.671 | 1.399 | 1.283 | 2.000 | 0.750 | 0.625 | 0.092 |
| bulk/lr00.55/bALL | 1440 | 4.0 | 180 | 4 | alt | 0.772 | 0.738 | 1.047 | 1.067 | 2.000 | 0.625 | 0.625 | -0.011 |
| bulk/lr00.55/bALL | 1440 | 4.0 | 180 | 4 | dc | 0.949 | 0.659 | 1.440 | 1.282 | 2.000 | 0.750 | 0.625 | 0.089 |
| bulk/lr00.60/bALL | 1344 | 4.0 | 180 | 4 | alt | 0.774 | 0.743 | 1.042 | 1.027 | 2.000 | 0.625 | 0.625 | -0.012 |
| bulk/lr00.60/bALL | 1344 | 4.0 | 180 | 4 | dc | 0.959 | 0.675 | 1.420 | 1.271 | 2.000 | 0.750 | 0.688 | 0.090 |
| bulk/lr00.64/bALL | 1440 | 4.0 | 180 | 4 | alt | 0.821 | 0.723 | 1.136 | 1.043 | 2.000 | 0.625 | 0.625 | 0.012 |
| bulk/lr00.64/bALL | 1440 | 4.0 | 180 | 4 | dc | 0.995 | 0.693 | 1.437 | 1.213 | 2.000 | 0.750 | 0.625 | 0.103 |
| bulk/lr00.72/bALL | 1344 | 4.0 | 180 | 4 | alt | 0.776 | 0.733 | 1.058 | 1.036 | 2.000 | 0.688 | 0.688 | 0.006 |
| bulk/lr00.72/bALL | 1344 | 4.0 | 180 | 4 | dc | 0.969 | 0.657 | 1.474 | 1.196 | 2.000 | 0.750 | 0.625 | 0.082 |
| bulk/lr00.73/bALL | 960 | 3.0 | 133 | 4 | alt | 0.832 | 0.730 | 1.140 | 1.145 | 1.707 | 0.750 | 0.750 | 0.040 |
| bulk/lr00.73/bALL | 960 | 3.0 | 133 | 4 | dc | 0.996 | 0.656 | 1.519 | 1.246 | 1.707 | 0.750 | 0.750 | 0.087 |
| bulk/lr00.84/bALL | 960 | 3.0 | 133 | 4 | alt | 0.802 | 0.722 | 1.112 | 1.066 | 1.707 | 0.750 | 0.750 | -0.005 |
| bulk/lr00.84/bALL | 960 | 3.0 | 133 | 4 | dc | 0.970 | 0.654 | 1.482 | 1.218 | 1.707 | 0.750 | 0.750 | 0.069 |
| bulk/lr00.96/bALL | 1344 | 4.0 | 180 | 4 | alt | 0.794 | 0.733 | 1.083 | 1.072 | 2.000 | 0.625 | 0.688 | 0.024 |
| bulk/lr00.96/bALL | 1344 | 4.0 | 180 | 4 | dc | 0.911 | 0.657 | 1.385 | 1.112 | 2.000 | 0.688 | 0.625 | 0.076 |
| bulk/lr00.97/bALL | 480 | 2.0 | 86 | 3 | alt | 0.780 | 0.738 | 1.056 | 0.973 | 1.414 | 0.500 | 0.500 | -0.030 |
| bulk/lr00.97/bALL | 480 | 2.0 | 86 | 3 | dc | 1.086 | 0.627 | 1.731 | 1.286 | 1.414 | 1.000 | 0.500 | 0.046 |
| bulk/lr01.11/bALL | 480 | 2.0 | 86 | 3 | alt | 0.794 | 0.768 | 1.034 | 0.989 | 1.414 | 0.500 | 1.000 | -0.028 |
| bulk/lr01.11/bALL | 480 | 2.0 | 86 | 3 | dc | 1.098 | 0.655 | 1.676 | 1.193 | 1.414 | 1.000 | 1.000 | 0.039 |
| bulk/lr01.44/bALL | 1152 | 4.0 | 176 | 4 | alt | 0.800 | 0.723 | 1.106 | 1.116 | 2.000 | 0.750 | 0.750 | 0.021 |
| bulk/lr01.44/bALL | 1152 | 4.0 | 176 | 4 | dc | 0.929 | 0.662 | 1.402 | 1.108 | 2.000 | 0.750 | 0.750 | 0.050 |
| top/lr00.12/bALL | 1440 | 4.0 | 180 | 4 | alt | 0.844 | 0.723 | 1.168 | 1.179 | 2.000 | 0.625 | 0.625 | 0.004 |
| top/lr00.12/bALL | 1440 | 4.0 | 180 | 4 | dc | 0.835 | 0.658 | 1.269 | 1.519 | 2.000 | 0.688 | 0.625 | 0.247 |
| top/lr00.24/bALL | 2400 | 4.0 | 180 | 4 | alt | 0.798 | 0.717 | 1.112 | 1.170 | 2.000 | 0.625 | 0.625 | 0.003 |
| top/lr00.24/bALL | 2400 | 4.0 | 180 | 4 | dc | 1.787 | 0.656 | 2.726 | 1.906 | 2.000 | 0.750 | 0.625 | 0.461 |
| top/lr00.32/bALL | 480 | 8.0 | 360 | 5 | alt | 0.746 | 0.704 | 1.060 | 1.154 | 2.828 | 0.625 | 0.625 | -0.014 |
| top/lr00.32/bALL | 480 | 8.0 | 360 | 5 | dc | 3.892 | 0.625 | 6.224 | 2.551 | 2.828 | 0.750 | 0.625 | 0.687 |
| top/lr00.36/bALL | 1920 | 4.0 | 172 | 4 | alt | 0.793 | 0.716 | 1.107 | 1.122 | 2.000 | 0.625 | 0.625 | 0.009 |
| top/lr00.36/bALL | 1920 | 4.0 | 172 | 4 | dc | 2.303 | 0.664 | 3.470 | 1.633 | 2.000 | 0.875 | 0.625 | 0.481 |
| top/lr00.37/bALL | 480 | 8.0 | 360 | 5 | alt | 0.736 | 0.754 | 0.976 | 0.937 | 2.828 | 0.625 | 0.625 | 0.002 |
| top/lr00.37/bALL | 480 | 8.0 | 360 | 5 | dc | 4.673 | 0.672 | 6.959 | 2.284 | 2.828 | 0.875 | 0.625 | 0.721 |
| top/lr00.42/bALL | 960 | 6.0 | 270 | 4 | alt | 0.755 | 0.730 | 1.034 | 1.015 | 2.414 | 0.625 | 0.625 | 0.008 |
| top/lr00.42/bALL | 960 | 6.0 | 270 | 4 | dc | 4.588 | 0.661 | 6.936 | 1.834 | 2.414 | 0.875 | 0.625 | 0.638 |
| top/lr00.48/bALL | 2304 | 4.0 | 180 | 4 | alt | 0.805 | 0.764 | 1.054 | 1.040 | 2.000 | 0.625 | 0.625 | 0.004 |
| top/lr00.48/bALL | 2304 | 4.0 | 180 | 4 | dc | 4.759 | 0.626 | 7.602 | 2.107 | 2.000 | 0.875 | 0.625 | 0.643 |
| top/lr00.55/bALL | 1440 | 4.0 | 180 | 4 | alt | 0.779 | 0.770 | 1.012 | 0.939 | 2.000 | 0.625 | 0.625 | -0.020 |
| top/lr00.55/bALL | 1440 | 4.0 | 180 | 4 | dc | 4.618 | 0.625 | 7.392 | 1.792 | 2.000 | 1.000 | 0.625 | 0.552 |
| top/lr00.60/bALL | 1344 | 4.0 | 180 | 4 | alt | 0.761 | 0.731 | 1.040 | 1.040 | 2.000 | 0.625 | 0.625 | 0.015 |
| top/lr00.60/bALL | 1344 | 4.0 | 180 | 4 | dc | 4.967 | 0.632 | 7.864 | 1.826 | 2.000 | 1.000 | 0.688 | 0.648 |
| top/lr00.64/bALL | 1440 | 4.0 | 180 | 4 | alt | 0.886 | 0.760 | 1.166 | 1.132 | 2.000 | 0.625 | 0.625 | 0.037 |
| top/lr00.64/bALL | 1440 | 4.0 | 180 | 4 | dc | 5.030 | 0.615 | 8.172 | 1.662 | 2.000 | 1.000 | 0.625 | 0.560 |
| top/lr00.72/bALL | 1344 | 4.0 | 180 | 4 | alt | 0.797 | 0.761 | 1.047 | 1.061 | 2.000 | 0.625 | 0.625 | -0.006 |
| top/lr00.72/bALL | 1344 | 4.0 | 180 | 4 | dc | 5.275 | 0.681 | 7.743 | 1.631 | 2.000 | 1.000 | 0.625 | 0.658 |
| top/lr00.73/bALL | 960 | 3.0 | 133 | 4 | alt | 0.859 | 0.806 | 1.066 | 1.026 | 1.707 | 0.750 | 0.750 | -0.017 |
| top/lr00.73/bALL | 960 | 3.0 | 133 | 4 | dc | 4.149 | 0.633 | 6.549 | 1.571 | 1.707 | 1.000 | 0.750 | 0.479 |
| top/lr00.84/bALL | 960 | 3.0 | 133 | 4 | alt | 0.790 | 0.765 | 1.032 | 0.983 | 1.707 | 0.750 | 0.750 | -0.015 |
| top/lr00.84/bALL | 960 | 3.0 | 133 | 4 | dc | 4.313 | 0.638 | 6.759 | 1.537 | 1.707 | 1.000 | 0.750 | 0.475 |
| top/lr00.96/bALL | 1344 | 4.0 | 180 | 4 | alt | 0.837 | 0.749 | 1.118 | 1.120 | 2.000 | 0.688 | 0.688 | 0.002 |
| top/lr00.96/bALL | 1344 | 4.0 | 180 | 4 | dc | 5.852 | 0.655 | 8.934 | 1.601 | 2.000 | 1.000 | 0.688 | 0.672 |
| top/lr00.97/bALL | 480 | 2.0 | 86 | 3 | alt | 0.931 | 0.805 | 1.157 | 1.092 | 1.414 | 0.500 | 0.500 | -0.022 |
| top/lr00.97/bALL | 480 | 2.0 | 86 | 3 | dc | 2.734 | 0.612 | 4.470 | 1.286 | 1.414 | 1.000 | 0.500 | 0.243 |
| top/lr01.11/bALL | 480 | 2.0 | 86 | 3 | alt | 0.982 | 0.845 | 1.162 | 1.057 | 1.414 | 0.500 | 1.000 | -0.003 |
| top/lr01.11/bALL | 480 | 2.0 | 86 | 3 | dc | 2.892 | 0.621 | 4.657 | 1.155 | 1.414 | 1.000 | 1.000 | 0.197 |
| top/lr01.44/bALL | 1152 | 4.0 | 176 | 4 | alt | 0.775 | 0.759 | 1.021 | 0.962 | 2.000 | 0.750 | 0.750 | 0.015 |
| top/lr01.44/bALL | 1152 | 4.0 | 176 | 4 | dc | 4.922 | 0.661 | 7.447 | 1.481 | 2.000 | 1.000 | 0.750 | 0.635 |

### 7c. kind × batch (burn-in 5)

| group | n_slots | k (median) | N | L | ch | slot median \|t\| | slot null \|t\| | slot ratio | growth (cal.) | sqrt(k) | seg-mean coherence | null coherence | seg-mean acf1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulk/lrALL/b00500 | 3168 | 16.0 | 720 | 6 | alt | 0.667 | 0.673 | 0.991 | 1.016 | 4.000 | 0.562 | 0.562 | 0.001 |
| bulk/lrALL/b00500 | 3168 | 16.0 | 720 | 6 | dc | 0.925 | 0.679 | 1.362 | 1.300 | 4.000 | 0.625 | 0.562 | 0.061 |
| bulk/lrALL/b01000 | 4896 | 8.0 | 360 | 5 | alt | 0.695 | 0.693 | 1.003 | 1.002 | 2.828 | 0.625 | 0.625 | 0.006 |
| bulk/lrALL/b01000 | 4896 | 8.0 | 360 | 5 | dc | 0.940 | 0.648 | 1.451 | 1.434 | 2.828 | 0.625 | 0.625 | 0.097 |
| bulk/lrALL/b02000 | 4896 | 4.0 | 180 | 4 | alt | 0.781 | 0.738 | 1.058 | 1.045 | 2.000 | 0.750 | 0.750 | 0.015 |
| bulk/lrALL/b02000 | 4896 | 4.0 | 180 | 4 | dc | 0.921 | 0.656 | 1.405 | 1.265 | 2.000 | 0.750 | 0.750 | 0.089 |
| bulk/lrALL/b04000 | 4896 | 2.0 | 86 | 3 | alt | 0.827 | 0.764 | 1.082 | 1.011 | 1.414 | 0.500 | 0.500 | -0.006 |
| bulk/lrALL/b04000 | 4896 | 2.0 | 86 | 3 | dc | 0.902 | 0.630 | 1.431 | 1.158 | 1.414 | 1.000 | 0.500 | 0.036 |
| bulk/lrALL/b08000 | 3072 | 2.5 | 108 | 4 | alt | 0.923 | 0.787 | 1.172 | 1.105 | 1.500 | 1.000 | 1.000 | 0.020 |
| bulk/lrALL/b08000 | 3072 | 2.5 | 108 | 4 | dc | 0.859 | 0.660 | 1.301 | 0.984 | 1.500 | 1.000 | 1.000 | 0.069 |
| top/lrALL/b00500 | 3168 | 16.0 | 720 | 6 | alt | 0.700 | 0.705 | 0.992 | 1.002 | 4.000 | 0.562 | 0.562 | -0.008 |
| top/lrALL/b00500 | 3168 | 16.0 | 720 | 6 | dc | 6.813 | 0.635 | 10.722 | 3.956 | 4.000 | 0.812 | 0.562 | 0.755 |
| top/lrALL/b01000 | 4896 | 8.0 | 360 | 5 | alt | 0.747 | 0.753 | 0.993 | 1.031 | 2.828 | 0.625 | 0.625 | 0.001 |
| top/lrALL/b01000 | 4896 | 8.0 | 360 | 5 | dc | 5.439 | 0.656 | 8.286 | 2.453 | 2.828 | 0.875 | 0.625 | 0.735 |
| top/lrALL/b02000 | 4896 | 4.0 | 180 | 4 | alt | 0.796 | 0.753 | 1.056 | 1.035 | 2.000 | 0.750 | 0.750 | 0.022 |
| top/lrALL/b02000 | 4896 | 4.0 | 180 | 4 | dc | 4.594 | 0.638 | 7.198 | 1.940 | 2.000 | 1.000 | 0.750 | 0.559 |
| top/lrALL/b04000 | 4896 | 2.0 | 86 | 3 | alt | 0.916 | 0.821 | 1.115 | 1.074 | 1.414 | 0.500 | 1.000 | -0.007 |
| top/lrALL/b04000 | 4896 | 2.0 | 86 | 3 | dc | 2.198 | 0.629 | 3.494 | 1.295 | 1.414 | 1.000 | 1.000 | 0.260 |
| top/lrALL/b08000 | 3072 | 2.5 | 108 | 4 | alt | 0.928 | 0.848 | 1.094 | 1.073 | 1.500 | 1.000 | 1.000 | 0.024 |
| top/lrALL/b08000 | 3072 | 2.5 | 108 | 4 | dc | 1.477 | 0.613 | 2.411 | 1.304 | 1.500 | 1.000 | 1.000 | 0.447 |


## 8. Estimator control (synthetic, seeded)

4000 synthetic direction slots of 4 segments × 50 steps at burn-in 5, through the identical `segment_block` / `SlotAccumulator` / surrogate-null path. On a zero-mean stream there is nothing to detect, so both `ratio` columns read 1 whatever the autocorrelation (K1). K2a plants an alternating mean and must move `alt` only; K2b plants a per-slot DC mean and must move `dc` **and** drive the growth factor to √k = 2.000. **K3 has no mean at all** and still produces a large `dc ratio` — its growth factor is what separates it from K2b, and it is the reason §7 exists.

| control | rho_1 | tau(4) | dc \|t\| | dc ratio | dc growth | alt \|t\| | alt ratio | alt growth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K1a phi=0.00 | -0.003 | 0.991 | 0.731 | 0.999 | 0.979 | 0.723 | 0.988 | 0.987 |
| K1b phi=-0.34 | -0.325 | 0.577 | 0.659 | 0.973 | 1.024 | 0.811 | 1.032 | 0.993 |
| K1c phi=+0.50 | 0.474 | 2.526 | 0.866 | 0.993 | 1.015 | 0.622 | 0.956 | 1.048 |
| K2a phi=-0.34 +alt mean | -0.442 | 0.686 | 0.660 | 1.040 | 0.973 | 2.656 | 2.931 | 2.145 |
| K2b phi=-0.40 +slot dc mean | -0.384 | 0.528 | 1.925 | 2.875 | 1.976 | 0.840 | 1.006 | 1.026 |
| K3 zero-mean slow power | -0.334 | 0.803 | 1.790 | 2.639 | 0.881 | 0.838 | 1.001 | 1.010 |
