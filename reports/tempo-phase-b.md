# Program #8 Phase B/B′ results — TempoMuon temporal trust ratio

2026-07-22. Dev-phase results (seeds 1420–1439; NOT an evaluation-gate
claim). Pre-registration: `reports/tempo-phase-a.md` §3 (commit 082a09d,
before any Phase-B run). All runs local (2×5090, timeout+retry drivers);
n=10 seed-paired per cell; endpoint `tta_val_acc`.

## 1. Phase B (seeds 1420–1429): stock vs controller arms

| lr | stock Muon | tempo-global | tempo-per-matrix | Δglobal (paired) | Δper-matrix (paired) |
|---|---|---|---|---|---|
| 0.24 (1×) | 0.9403±0.0013 | 0.9377 | 0.9345 | **−0.258pp ± 0.052** | −0.580pp ± 0.052 |
| 0.48 (2×) | 0.9342±0.0010 | 0.9384 | 0.9333 | +0.427pp ± 0.048 | −0.088pp ± 0.062 |
| 0.72 (3×) | 0.9256±0.0016 | 0.9367 | 0.9320 | **+1.110pp ± 0.048** | +0.641pp ± 0.063 |
| 0.96 (4×) | 0.9172±0.0020 | 0.9364 | 0.9321 | **+1.918pp ± 0.065** | +1.487pp ± 0.109 |

### Prediction scorecard (pre-registered P1–P5)

- **P1 (1× non-inferiority ≤0.10pp): FAILED** — global −0.26pp,
  per-matrix −0.58pp. Diagnosed from telemetry, not re-tuned post hoc:
  the healthy run's rho relaxes past the fixed −0.48 setpoint after
  ~step 60 (the Phase-A late-phase reversal), so the [25,100] window
  guaranteed late engagement at 1× (mean 1× gain fell to 0.36 by step
  100). Design flaw, mechanism understood.
- **P2 (rescue ≥ +0.5pp at 3–4×): CONFIRMED**, with margin — the global
  arm recovers ~84% of stock's 4× deficit.
- **P3 (frozen gain monotone in lr): FAILED** as stated (1× gain also
  fell — same root cause as P1). Within the early window the *rate* of
  gain descent was lr-monotone as designed.
- **P4 (granularity): global pooling beats per-matrix at every lr.**
  Cause: per-matrix baseline rho levels differ by more than the LR effect
  (−0.30 vs −0.43 at 1×), so a single setpoint mis-treats matrices;
  pooling matches the calibration. The pre-registered "per-matrix"
  novelty slot is thus NOT supported on this testbed — the temporal
  signal works, the granularity does not pay (continuing the repo's
  pattern: per-direction null → per-matrix worse → global works).
- **P5:** no 2× harm for the global arm (+0.43pp); the conflation
  signature appeared at 1× instead (see P1).

## 2. Pre-committed placebo (same seeds): open-loop replay of the global
arm's mean gain trajectory (kappa=0, `gain_schedule`)

| lr | Δreplay (paired) | Δglobal closed-loop |
|---|---|---|
| 0.24 | −0.185pp ± 0.035 | −0.258pp ± 0.052 |
| 0.48 | +0.457pp ± 0.041 | +0.427pp ± 0.048 |
| 0.72 | +1.131pp ± 0.075 | +1.110pp ± 0.048 |
| 0.96 | +1.807pp ± 0.079 | +1.918pp ± 0.065 |

**The placebo matches the controller within noise at every lr.** Reading:
within-run feedback beyond the mean trajectory contributes nothing here
(consistent with program #1's equivalent-destinations result). The
controller's value is *discovering* the lr-appropriate gain schedule
online from the temporal signal — the replay arm could only exist because
the closed loop found the trajectory first, and the trajectory differs
per lr in exactly the compensating direction. "Adaptive method as
schedule-discovery mechanism", stated plainly.

## 3. Phase B′ (labeled exploration, fresh seeds 1430–1439): one change,
active window 100 → 60 (ends before the healthy-regime setpoint crossing)

| lr | stock Muon | tempo-global (window 60) | Δ (paired) |
|---|---|---|---|
| 0.24 | 0.9406±0.0012 | 0.9399±0.0011 | **−0.064pp ± 0.044** |
| 0.48 | 0.9345±0.0009 | 0.9369±0.0008 | +0.240pp ± 0.044 |
| 0.72 | 0.9266±0.0019 | 0.9346±0.0021 | **+0.797pp ± 0.059** |
| 0.96 | 0.9171±0.0031 | 0.9332±0.0017 | **+1.602pp ± 0.102** |

The 1× cost collapses to statistical-zero while ~80–85% of the rescue
survives. This is the balanced config: `configs/dev/tempo_bprime_global.yaml`
(kappa −0.25, rho\* −0.48, window [25, 60], gain ∈ [0.2, 1], global pool).

## 4. What program #8 established (descriptive; gates are the human's)

1. **A ~free, matrix-level LR-excess dial exists early in Muon training:**
   EMA[cos(G_t,G_{t-1})], ~20σ LR separation at fixed step, with an
   *inverted* sign (healthy = deepest negative; hot = decorrelated) and a
   late-anneal reversal. New measurement; no per-direction machinery.
2. **A controller on that signal rescues mis-set LR** on airbench:
   +0.8pp at 3×, +1.6pp at 4×, −0.06pp at 1× (B′, dev seeds, n=10).
3. **Mechanism:** the whole effect is carried by the discovered gain
   *schedule* (open-loop replay reproduces it); feedback's role is
   finding it without knowing the lr is mis-set.
4. **Granularity result:** global pooling > per-matrix under a shared
   setpoint. The lit-review's empty cell ("per-matrix × temporal") is the
   part our own data argues against on this testbed; the defensible claim
   is closer to "occupancy-style temporal LR control for Muon, with
   placebo-grade mechanism decomposition" — differentiate vs GALA
   (global alignment → momentum), CLARA (global path length → LR), which
   the 2026-07-22 sweep confirmed are the nearest published neighbors.

## 5. Caveats / open

- Dev seeds; single task/recipe; single GPU type. Eval-seed confirmation
  would need human-authored criteria (`criteria/`) per ground rules.
- rho\* and the window are calibrated on this task's healthy 1× runs;
  transfer of the setpoint (or a self-calibrating variant: reference =
  own early-window rho of a short probe at a known-safe lr) is untested.
- nanogpt transfer untested (program #7 precedent warns airbench→nanogpt
  transfers can fail; the local harness is ready if pursued).
- 2× region is under-rescued (+0.24pp of a −0.61pp deficit at B′
  settings) — window-60 trades away part of the 2× correction.

Analysis: `scripts/analyze_tempo.py`; aggregates in scratchpad JSONs are
reproducible from `results/` (configs `tempo_b_*`, `tempo_placebo_*`,
`tempo_bprime_*`).
