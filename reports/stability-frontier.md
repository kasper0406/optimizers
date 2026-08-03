# Program #6 — Muon stability frontier in (lr × batch): descriptive report

Pre-registration: `reports/stability-frontier-preregistration.md` (committed
2e32a2d, before any run; amendment A1 recorded post-launch). Tables/figures:
`reports/stability-frontier-tables.md`, `reports/stability-frontier.json`,
`reports/figures/frontier_*.png`, all produced by
`scripts/analyze_frontier.py` from results JSONs. Descriptive only — where a
measurement lands relative to a pre-registered region is stated as fact;
nothing here is a gate verdict.

## Run ledger

80/80 grid cells complete on the local 2× RTX 5090 box (the machine's first
use; torch 2.13.0+cu130, eager). B ≤ 4000 at SHA 3395c99, B=8000 at f0a7138
with `hvp: false` (amendment A1: the fp32 create-graph HVP backward OOMs at
batch 8000 on 32 GB; probe is read-only w.r.t. training; no pre-registered
endpoint touches B=8000 sidecars). Two earlier aborted launches left 9 stray
result files with `git_dirty: true`; `results/` is append-only so they
remain, and the analyzer dedupes by (B, lr, seed) keeping the latest run.
`cost_usd` is null on all 97 new local files pending a human convention for
owned-hardware runs (prior local A6000 runs carry 0.02/run). Bridge cell
(B=2000, 1×): 94.14% mean TTA, consistent with the historic ~94.0 baseline
band — the new hardware/torch stack reproduces the old numbers at stock
settings (P2 quantities at that cell also land inside prior-program ranges:
occupancy 0.65 vs 0.68, spectral plateau 208 vs 208–228, Euclidean 2.79 vs
2.7–3.1).

## P1 — shoulder scaling

Mean TTA accuracy per rung (percent, n=2 seeds/cell):

| B | 0.5× | 1× | 1.5× | 2× | 2.5× | 3× | 4× | 6× | shoulder (pre-reg) |
|---|---|---|---|---|---|---|---|---|---|
| 500 | **94.06** | 93.36 | 92.60 | 92.06 | 91.45 | 90.82 | 89.65 | 88.74 | 0.24 (1×) |
| 1000 | 93.90 | **93.94** | 93.47 | 92.82 | 92.39 | 91.85 | 91.14 | 89.69 | 0.36 (1.5×) |
| 2000 | 93.56 | **94.14** | 93.72 | 93.47 | 92.98 | 92.68 | 91.94 | 90.16 | 0.48 (2×) |
| 4000 | 91.46 | 92.74 | **92.81** | 92.70 | 92.53 | 92.11 | 91.06 | 88.98 | undefined* |
| 8000 | 63.38 | 70.58 | 71.56 | **72.69** | 69.95 | 68.27 | 67.25 | 57.94 | undefined* |

*Where defined, the pre-registered shoulder moves right one rung per batch
doubling: 0.24 → 0.36 → 0.48 over B 500 → 2000. The OLS fit over those three
points gives **α = 0.50** — in the pre-registered noise-governed region
(α ≥ 0.25) and exactly the √B point prediction — but the pre-declared
rung-quantization envelope is ±0.5 at 3 points, so the deterministic region
(|α| < 0.1) is **not excluded by the pre-registered estimator alone**. Seed
bootstrap: median 0.50, CI95 [0.50, 1.0] (quantized — n=2 seeds barely move
cell means across rung boundaries).

*The pre-registered first-crossing definition **breaks at B ≥ 4000**, in an
informative direction: the 0.5× rung falls more than 1.0pp below the
reference (91.46 at B=4000; 63.38 at B=8000), i.e. the accuracy curve is no
longer monotone-decreasing in lr — **low lr is now the harmful direction**.
The convention scans for the first *downward* floor crossing and finds it
immediately, leaving the shoulder undefined (higher rungs are recorded as
flagged recoveries: up to 3× at B=4000, up to 2.5× at B=8000). The peak of
the accuracy curve itself moves right monotonically across all five batch
sizes: 0.5×, 1×, 1×, 1.5×, 2×.

**Post-hoc secondary readout** (labeled as such; not pre-registered): a
peak-referenced shoulder — largest rung within 1.0pp of the *ladder
maximum* — is defined at every B and gives 0.24 / 0.36 / 0.48 / 0.72 over
B 500→4000, a log-log slope of **0.53** over an 8× batch range. B=8000
(0.48) falls off that line; see the confound below.

**B=8000 confound:** at the fixed 8-epoch sample budget B=8000 is 50 steps
and deeply undertrained everywhere (peak 72.7%); its whole curve is
depressed and its rightward peak shift (2×) is measured on a training run
that never converges. The prereg fixed the sample budget deliberately, but
step-starvation this severe means the B=8000 row supports only the
qualitative reads (peak location; no cliff), not the scaling fit.

## P2 — invariant candidates

Across-B max/min ratios (pre-registered tracking signature: < 1.5 at the
per-B shoulder AND ≥ 1.5 at the fixed 1× control; shoulder column spans the
three B with a defined shoulder, control spans B ≤ 4000):

| candidate | at shoulder | at 1× | signature met |
|---|---|---|---|
| occupancy frac(ρ<−0.2) | 1.67 | 1.77 | no |
| lr·D_smooth_spectral | 2.35 | 1.64 | no |
| lr·D_smooth_euclid | 2.67 | 2.03 | no |
| HVP η·λ q90 | 4.84 | 6.25 | no |

**No candidate meets the pre-registered signature.** Occupancy comes
nearest the shoulder-flatness bar (1.67 vs 1.5) but is nearly as variable
at the control, so it fails the sensitivity half of the signature too. The
curvature quantities are the *least* equalized — HVP η·λ q90 spans ~5–6×
across batch at both rungs — consistent with the standing finding that
Euclidean curvature does not govern Muon's stability, and now also showing
it does not govern the batch-coupled frontier. Directionally, every
candidate *rises* with B at fixed lr (occupancy 0.42 → 0.74 from B=500 to
4000 at 1×), so whatever is equalized along the frontier, it is none of
these four as scalars.

## P3 — graceful degradation

No divergence cliff at any batch size: minimum mean accuracy across the
ladder is 88.7 / 89.7 / 90.2 / 89.0% for B ≤ 4000 and 57.9% at B=8000 —
degraded, never collapsed to random (10%). The cliff-free property
established at B=2000 persists across a 16× batch range.

## What this measured, in one paragraph

At fixed sample budget, Muon's useful-lr band shifts right with batch size
at close to the √B rate everywhere the pre-registered definition applies,
and the definition's own failure mode at large B (low lr becomes the losing
side) is the same rightward shift seen from the other flank. None of the
four instrumented candidates — oscillation occupancy, spectral or Euclidean
directional smoothness, HVP curvature — is constant along that frontier, so
the data supports batch-coupled (noise-side) frontier scaling while ruling
out all four measured quantities as the frontier-setting invariant in
scalar form. The quantization-dominated α uncertainty (±0.5 on 3 points)
is the main limit on the strength of the scaling claim.

## Limits and natural follow-ups (not launched)

- Rung quantization dominates α: a denser ladder (×1.15 spacing) around the
  per-B shoulders with n ≥ 5 seeds at B ∈ {1000, 2000, 4000} would shrink
  the envelope by ~4× at ~60 runs (~1 h on this box).
- A step-matched (rather than epoch-matched) large-B arm would deconfound
  the B=8000 row's undertraining from its frontier position.
- Per-direction SNR vs batch (open question #4, frozen-probe tier) shares
  the sweep geometry; the B-ladder sidecars produced here (minus B=8000
  HVP) are already suitable inputs for its pilot read.
