# Wave 1 dev results — anneal decomposition (programs #17/#18/#19)

2026-07-23. Pre-registration: `reports/wave1-anneal-decomposition-prereg.md`
(commit `8fed636`, before any run). Infrastructure: commit `62d0d8e`.
All runs on the single healthy RTX 5090 (bus-03; bus-01 wedged, "GPU requires
reset"), dev seeds 1511–1513, 16 science runs + 4 smokes, zero babysitter
retries. Readout evaluation: `reports/wave1-readout.json` via
`scripts/analyze_wave1_readout.py`.

**Dev-scope caveat (registered):** everything below is dev-seed evidence
(n=2–3, paired). Effect claims require the Phase-B eval-seed designs. Gate
calls are human.

## 0. The shared substrate

Per seed: hot prefix P to T_c=963 (checkpoint), then forked arms — A = stock
WSD anneal, C = constant LR (`min_lr_frac: 1.0`) with spike-gated tail
accumulators (W1 = mean 1450–1599, W2 = 1600–1749, Polyak from 963), plus the
program arms. Fork validity check passed at scale: arm A endpoints
3.28855 / 3.29059 / 3.28725 — all inside the n=10 baseline band
(3.28888 ± 0.00125).

**The anneal gap** (C_final − A_final, same seed): 0.0915 / 0.0909 / 0.0913.
Tight across seeds; this is the quantity the three programs try to recover.
Spike gate excluded 6–8 of 787 tail steps per run (~0.9%).

## 1. Program #17 — drift-completion readout: **registered FAIL branch**

α* = **0** on the selection cell (seed 1511, shard 1): every α > 0 hurts.

| α | 0 | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|
| L(W(α)) s1511/sh1 | 3.3232 | 3.3729 | 3.4749 | 3.9731 | 6.783 | 16.58 | 23.80 | 26.84 |

Registered criteria: FAIL iff α* ≤ 0.5 or any held-out r < 0.40 → **FAIL**
(held-out recovery ≡ 0 at α*=0). Kills *linear* drift completion (registered
scope limit: not the bias hypothesis in general).

**Mechanism readout (registered, the real yield):** cos(v, D) =
**−0.682 / −0.681 / −0.681** across seeds, ‖D‖/‖v‖ ≈ 1.45. The anneal
displacement is strongly ANTI-aligned with the constant-LR tail drift: the
anneal does not complete the tail's motion — it substantially retraces it.
The constant-LR iterate drifts somewhere the anneal walks back. This is a
settling/return picture, not a bias-completion picture, and it explains the
FAIL structurally rather than as a noise null.

## 2. Program #18 — schedule-free tail graft: **dev decision rule MET, propose Phase B**

Sweep (seed 1511, forked from one shared prefix; primary readout val(x̄) at
1750; same-seed references A = 3.28855, C_final = 3.38008, C_polyak = 3.31022):

| κ | ρ | val(x̄) | val(z) |
|---|---|--------|--------|
| **1.0** | **0.7** | **3.28802** | 4.010 |
| 1.0 | 0.9 | 3.31017 | 5.542 |
| 0.5 | 0.7 | 3.30194 | 3.592 |
| 0.5 | 0.9 | 3.31421 | 4.282 |

Confirm (winner κ=1, ρ=0.7 on seed 1512's prefix): **x̄ = 3.28983** vs
same-seed A = 3.29059.

Registered dev rule (§2): (i) beats same-seed constant-LR readouts by
> 0.005 — yes, by ~0.022 vs C_polyak on both seeds; (ii) within +0.005 of
same-seed WSD — yes, **beats it on both seeds** (−0.0005, −0.0008). The tuned
anneal is matched-or-beaten by a constant-full-LR tail whose gradients are
evaluated at the anchored interpolate, with the Polyak average as readout.

Honest caveats: (a) n=2 dev seeds, margins (~0.0006) are inside single-run
seed noise (0.00125) — only the paired Phase-B design can establish a WIN;
(b) x̄ is **not flat** at 1750 (still descending ~0.017/125 steps at the end),
so the anytime-checkpoint property is not yet demonstrated at this horizon —
and also suggests the margin may grow with tail length; (c) the raw iterate z
sits at ~4.0: all quality lives in the average, as the mechanism predicts.

**Phase-B proposal (per prereg §2, human gate):** arms B (SF κ=1, ρ=0.7) and
C (constant-LR + accumulators) forked from regenerated per-seed prefixes,
eval seeds 1710–1713, n=4 paired; A = existing n=10 baseline. Registered
criteria verbatim from the prereg (WIN ≤ −0.0025 paired, CI excl. 0;
ANNEAL-REPLACED within +0.0025 AND B < C_polyak − 0.005 AND flatness;
KILL ≥ +0.0025). Cost ≈ 4×(41+34+34+34) min ≈ 9.5 GPU-h.

## 3. Program #19 — batch-annealed Muon Stage 1: **registered PARTIAL branch**

f = (C−B)/(C−A), paired, matched token budget (688,128,000 exactly, 418 ramp
steps): **0.454 (seed 1511), 0.471 (seed 1512)** → mean 0.46, squarely in
the PARTIAL band (0.2, 0.9). The registered expectation was DEAD; batch
growth at constant LR in fact recovers ~46% of the anneal gap.

**Trajectory guard: FAILED** (registered consequence: the pure
noise-quenching account is killed regardless of f). At matched tokens the
ramp tracks A early (+0.003/+0.0045 at eq-step 1000) but diverges far beyond
the 0.004 band by the end (+0.050/+0.048); interpolated mid-tail ≈ +0.036.
Batch growth is not equivalent to LR decay along the trajectory — it just
ends somewhere ~half as good.

Per prereg: **no more BAM seeds**; a hybrid ramp + short-final-decay Phase B
may be proposed (human gate). Given #18's result, the hybrid is likely
dominated; recommend shelving unless the human wants the systems angle
(larger tail batches, 8× fewer NS/all-reduce per token).

## 4. Cross-program synthesis: what the anneal actually is (on this harness)

Same-seed ladder of anneal-gap recovery (fraction of C_final − A_final):

| readout | loss (s1511/s1512/s1513) | recovery |
|---|---|---|
| C_final (raw constant-LR iterate) | 3.3801 / 3.3815 / 3.3786 | 0% |
| ramp (constant LR, 8× batch growth) | 3.3385 / 3.3387 / — | ~46% |
| C_polyak ≈ W2 (passive tail average) | 3.3102 / 3.3120 / 3.3089 | ~76% |
| best convex merge (0.5·W2+0.5·Polyak, exploratory) | 3.2983 / 3.2995 / 3.2965 | ~89% |
| **A (tuned WSD anneal)** | 3.2886 / 3.2906 / 3.2873 | 100% |
| **SF tail x̄ (κ=1, ρ=0.7, closed loop)** | **3.2880 / 3.2898** / — | **~101%** |
| linear drift extrapolation (α>0) | worse than W2, monotone in α | <0% |

The decomposition: ~three quarters of the anneal is variance averaging
(recoverable open-loop — #16's airbench null does NOT transfer to nanogpt at
this readout), the last quarter needs descent at the readout point
(closed-loop feedback), none of it is linear drift completion (the anneal
*anti*-aligns with the tail drift, cos ≈ −0.68), and noise-quenching alone
gets stuck halfway off-trajectory.

## 5. Recommended next steps (human gates)

1. **#18 Phase B on eval seeds** (§2 above) — the registered path to the
   first method-level WIN/ANNEAL-REPLACED claim. Highest priority.
2. Longer-tail probe (dev): does x̄ keep descending past 1750 at constant LR?
   Bears on both the WIN margin and the anytime property. Cheap (extend one
   tail run).
3. #17 is closed (FAIL); fold cos(v,D) ≈ −0.68 into the measurement-paper
   narrative (the anneal as retraction, not completion).
4. #19 closed at Stage 1 (PARTIAL + guard fail); hybrid Phase B optional and
   probably dominated by #18.
5. Reset GPU bus-01 (`sudo nvidia-smi --gpu-reset -i 0`) before Phase B to
   restore two-lane throughput.
