# Program #8 — TempoMuon (per-matrix temporal trust ratio)

Phase A report + Phase B pre-registration. 2026-07-22. Dev-phase document:
all runs use dev seeds (≥1000); nothing here is an evaluation-gate claim.
Companion code: `src/optim/tempomuon.py`, analysis `scripts/analyze_tempo.py`.

## 0. Why this direction (decision record)

Three parallel literature sweeps (2026-07-22, this session) re-checked the
three candidate families for improving Muon:

1. **Newton-Schulz modification** — crowded and deflationary: Polar Express
   (ICLR'26 oral) owns optimal fixed coefficients, Muonᵖ (2606.13867) owns
   fractional powers, and the pair Kaon ("random spectra work just as well",
   2605.11181) + Huang ("how much orthogonalization does Muon need",
   2606.00371) shows training quality barely responds to polar accuracy in
   healthy regimes. Open gaps exist (online closed-loop NS adaptation, NS
   error feedback) but all carry that burden of proof. Not pursued.
2. **Meta-optimization / hypergradients on Muon** — genuinely unclaimed
   (arXiv "Muon"×"hypergradient" = zero hits) but fights the short-horizon
   bias literature, and global-LR hypergradients are likely unclaimed for
   lack of value (Muon's LR is robust). Not pursued now.
3. **Per-matrix temporal trust ratio** — the July-19 sweep's #1-ranked slot,
   re-verified still open as of 2026-07-22: no published per-matrix LR gain
   driven by serial/temporal statistics (lag-1 autocorrelation, sign-flip
   rate, oscillation occupancy) in any optimizer family. Nearest neighbors
   appeared within months (Greedy Alignment 2512.06370: global, momentum-
   only; MGUP-Muon 2606.17526: instantaneous per-parameter sign agreement;
   CLARA 2508.05408: global path-length LR) — the cell is being encircled.
   **Pursued.**

## 1. The signal

Per matrix, per step: `cos(G_t, G_{t-1})` on raw pre-momentum gradients
(fp32 dot; one extra buffer per matrix), accumulated into a bias-corrected
EMA (β=0.9), written `rho_hat`. This is the energy-weighted matrix-level
aggregate of the per-direction lag-1 autocorrelation population WP1.2
measured with the tracked-pair machinery. Cost: negligible (one dot + one
buffer per matrix); no HVP, no SVD, no per-direction tracking.

## 2. Phase A: passive measurement (kappa = 0, bit-identical to stock Muon)

12 runs: airbench_smoke (B=2000, 8 epochs, 200 steps, record recipe),
lr ∈ {0.24, 0.48, 0.72, 0.96} = {1×, 2×, 3×, 4×} record, seeds 1420–1422.
Configs `configs/dev/tempo_passive.yaml`; results in `results/` (post-fix
files, 2026-07-22T08:18Z onward).

### 2.1 Incidents (both fixed, committed)

- **fp16 overflow NaN collapse:** the airbench model trains in fp16;
  elementwise `G*prev` products overflow (max 65504) → inf cosine → NaN EMA
  → `exp(0·nan)` poisoned the gain *even in passive mode* at warmup expiry
  (~step 27), killing runs at 1× (where large early gradients persist
  longest) while 3–4× survived. Fix: fp32 cosine, non-finite guard,
  kappa=0 structurally never touches the gain. Regression test added.
- **GPU-0 hang:** one run hung 25 min at 0% util (the box's known
  flaky-card mode; nanogpt babysitter exists for this). Phase-A cells were
  driven to completion on GPU 1 with timeout+retry; 11/11 first-try clean.

### 2.2 Result: the window-averaged level is NOT a usable dial

Averaged over steps 50–150, agg rho_hat is non-monotone in lr
(−0.365, −0.401, −0.386, −0.358 at 1–4×) — a setpoint controller on the
window-averaged level is ill-posed (4× is indistinguishable from 1×).

### 2.3 Result: the fixed-step level IS a clean dial, with a sign inversion

Aggregate rho_hat at fixed steps (mean of 3 seeds, seed-std ≈ 0.01):

| step | 1× | 2× | 3× | 4× |
|---|---|---|---|---|
| 20  | −0.512 | −0.455 | −0.361 | −0.305 |
| 60  | −0.472 | −0.432 | −0.376 | −0.332 |
| 100 | −0.373 | −0.407 | −0.382 | −0.353 |
| 160 | −0.179 | −0.297 | −0.373 | −0.398 |
| 200 | −0.011 | −0.030 | −0.054 | −0.078 |

- **Early training (≤ step ~60): strictly monotone in lr, ~20σ separation
  — but inverted** vs the naive per-direction-occupancy intuition: the
  healthy 1× run is *deepest* negative; too-hot runs *decorrelate toward
  zero*. Reading: near-record LR produces coherent oscillation; excess LR
  destroys serial structure.
- **Late training: the ordering fully reverses** (healthy runs relax to 0
  with the anneal; hot runs stay deep) — the schedule-position behavior of
  program #2, reproduced at matrix level by a ~free statistic.
- Per-matrix spread (−0.29 vs −0.48 between matrices) exceeds the LR
  effect, so the controller must be *relative* (per-matrix EMA vs a common
  setpoint still works because all matrices order the same way vs lr; the
  global-pool arm tests whether granularity matters at all).

### 2.4 Derived controller (fixed before any Phase-B run)

`gain ← clip(gain · exp(kappa·(rho_hat − rho_star)), gain_min, 1)` with
**kappa = −0.25** (negative: shrink when rho_hat is *above* setpoint),
**rho_star = −0.48** (the healthy 1× early-window band), **gain_min = 0.2**
(4× × 0.2 < record lr, so full rescue is reachable), **warmup 25 steps**,
**active window ends at step 100** (before the late-phase sign reversal;
gain freezes at its reached value through the anneal). Self-stabilizing
argument: shrinking the gain deepens rho_hat (per the measured map),
closing the loop at rho_hat ≈ rho_star.

Honesty note: rho_star is calibrated on 1× dev runs of this task. The
claim under test is "given the healthy regime's signature, mis-set LR is
corrected online"; setpoint transfer across tasks is untested and out of
scope for Phase B.

## 3. Phase B pre-registration (dev-phase; descriptive, not a gate)

Arms (seed-paired, seeds 1420–1429, n=10, same recipe as Phase A):

| arm | config |
|---|---|
| stock Muon | `configs/dev/tempo_b_muon.yaml` |
| TempoMuon per-matrix | `configs/dev/tempo_b_permatrix.yaml` |
| TempoMuon global-pool | `configs/dev/tempo_b_global.yaml` |

each × lr ∈ {0.24, 0.48, 0.72, 0.96}. Endpoint: `tta_val_acc`.
120 runs, local (GPU 1, timeout+retry driver).

Predictions (written before any Phase-B run; seed-paired Δ = arm − stock):

- **P1 (non-inferiority at 1×):** |Δ| ≤ 0.10pp for both tempo arms at
  lr 0.24, and mean final gain at 1× ≥ 0.9 (controller idles).
- **P2 (rescue):** per-matrix Δ ≥ +0.50pp at 3× and 4× (stock deficits vs
  its own 1×: ~−1.4pp and ~−2.3pp on these dev seeds).
- **P3 (telemetry):** mean frozen gain monotone decreasing in lr;
  ballpark 1×≈1.0, 2×≈0.5–0.8, 3×≈0.25–0.45, 4×≈0.2–0.3.
- **P4 (granularity, exploratory):** no prediction — per-matrix vs global
  Δ difference is the reported quantity.
- **P5 (failure signature):** if the controller *hurts* at 2× (the only
  rung where stock is near-competitive and the controller half-engages),
  that is evidence the signal conflates progress with heat — report, do
  not re-tune within Phase B.

Analysis: `scripts/analyze_tempo.py compare` (+ gain summaries). Any
post-hoc re-tuning of (kappa, rho_star, window) after seeing Phase-B
accuracy invalidates these predictions and must be labeled Phase B′
exploration with fresh seeds.

## 4. Interpretation guardrails

Known confound for any positive result: a gain < 1 frozen after step 100
also *reduces total effective LR budget* — a rescue could be "just lower
LR". The per-matrix vs global comparison partially controls this; the
definitive control (a static gain schedule replayed without feedback,
random-gating-style) is pre-committed as the follow-up if P2 holds.
