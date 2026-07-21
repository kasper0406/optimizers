# Program #6 — Muon stability frontier in (lr × batch): pre-registration

Status: **pre-registered before any run**. This file and its config
(`configs/dev/frontier_lrxbatch.yaml`) are committed before the first run
launches; every frontier results JSON must carry a git SHA at or after this
commit. Dev-seed measurement program (CLAUDE.md WP1.2 pattern; Gate-1 A4
probe-override sanction for the lr axis). Descriptive output only — no gate,
no pass/fail asserted by the agent.

## Question

Open question #1 of `reports/project-state.md`: **what sets Muon's maximum
useful learning rate?** Established so far (airbench94, B=2000):

- No divergence cliff to 6× the record lr; graceful degradation; useful-lr
  shoulder between 2× and 3× (Gate-2 stress, 93.0% floor = baseline − 1.0pp).
- Curvature does not bind: HVP-measured Euclidean η·λ reaches ≈ 65 while
  training is stable (GD's bound is 2).
- No smoothness plateau: lr·D_smooth is not lr-invariant in either the
  spectral or Euclidean norm (max/min 2.57 / 2.26 across a 4× lr ladder).

The un-probed axis is **batch size**. The two live accounts make opposite
predictions about it:

- **Noise-governed** (minibatch sampling noise sets the frontier): the
  shoulder lr rises with batch. The SDE/noise-scale account predicts
  lr* ∝ B^α with α ≈ 0.5 at fixed sample budget. This would also supply the
  mechanistic link to Muon's unexplained large-batch advantage (Essential
  AI observation; plan §"large-batch axis").
- **Deterministic/curvature-governed** (some landscape property sets it):
  the shoulder lr is batch-independent, α ≈ 0.

## Design

Grid, all cells n=2 dev seeds **1400, 1401** (fresh block, disjoint from all
previously used dev seeds):

- `train.batch_size` ∈ {500, 1000, 2000, 4000, 8000} (record = 2000).
- `probe_overrides.lr` ∈ 0.24 × {0.5, 1, 1.5, 2, 2.5, 3, 4, 6}
  = {0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.96, 1.44}. Muon parameter groups
  only; the recipe's SGD side (biases/whiten/head) stays stock, isolating the
  spectral-lr axis (same isolation as the Gate-2 stress test).
- Epochs fixed at 8 → **fixed sample budget** (8 × 50k images) at every B;
  steps scale ≈ 1/B (≈800 / 400 / 200 / 100 / 50). The vendored schedule is
  step-normalized and adapts; SGD weight decay is batch-scaled by the stock
  recipe itself (`wd = 2e-6 · B`).
- Instrumentation: the WP1.2 stack + HVP + spectral/Euclidean smoothness
  (eager, `compile: false`), identical settings to
  `configs/dev/instrumented_airbench_smoothness_lr_ladder.yaml`
  (k1=k2=16, t_refresh=50, snapshot_every=5, t_meas=5, betas {0.9, 0.99}).

80 runs total, local (2× RTX 5090). Results JSONs append-only to `results/`
with full provenance; per-direction sidecars stay local (git-excluded, as
per the existing sidecar policy).

## Pre-registered definitions

- **ref_acc(B)** = max over the two lowest rungs {0.5×, 1×} of mean TTA
  accuracy at that B. (Per-B reference because the stock-side recipe is
  tuned at B=2000; absolute cross-B accuracy is NOT the object of study —
  only the shape of the lr axis at each B.)
- **floor(B)** = ref_acc(B) − 1.0pp (the Gate-2 stress convention,
  94.0 → 93.0, applied per-B).
- **lr_shoulder(B)** = the largest rung whose mean acc ≥ floor(B) among the
  rungs below the first floor crossing (first-crossing convention). Rungs
  that recover above the floor beyond the first crossing are reported and
  flagged, not used for the shoulder.
- **α** = OLS slope of log lr_shoulder(B) vs log B over the 5 batch sizes.
  Dominant uncertainty is rung quantization (± half a rung ratio, ≈ ×1.15);
  report it alongside a seed bootstrap. Rung ratio ≈ 1.33 over a 16× B range
  gives α resolution ≈ ±0.1.

## Pre-registered predictions

- **P1 (primary — frontier scaling).** Noise-governed: α ≥ 0.25 (point
  prediction 0.5). Deterministic-governed: |α| < 0.1. Anything between is
  recorded as ambiguous. No prediction is "confirmed" by this report; the
  report states which region α lands in.
- **P2 (invariant candidates).** Restricted to B ∈ {500, 1000, 2000, 4000}
  (steps ≥ 100; at B=8000's 50 steps the EMAs and plateau windows are
  under-resolved — excluded in advance, not after looking). At each B's
  shoulder rung, pool per-run: (a) oscillation occupancy frac(ρ₁ < −0.2),
  β=0.9, burn-in 10; (b) plateau lr·D_smooth_spectral (last 50% of measured
  steps, the `analyze_smoothness.py` convention); (c) plateau
  lr·D_smooth_euclid; (d) HVP η·λ q90. A candidate is **frontier-tracking**
  if its across-B max/min ratio at the shoulder is < 1.5 **and** its ratio
  across B at the fixed 1× rung is ≥ 1.5 (i.e., it varies in general but is
  equalized where the frontier sits — a flat-everywhere quantity tracks
  nothing). Prediction under the noise account: the occupancy (a) is the
  best candidate to be frontier-tracking; the curvature quantities (b)–(d)
  are not. Under the deterministic account: (b) or (d) should track.
- **P3 (secondary — cliff persistence).** Graceful degradation (no collapse
  to ≈10% random accuracy at any rung ≤ 6×) persists at every B. An actual
  cliff appearing at some B is a headline finding either way.

## Analysis plan

`scripts/analyze_frontier.py` (deterministic, results JSONs + sidecars in,
tables/plots out): the acc(lr) curve per B with floor and shoulder marked;
the log-log shoulder-vs-B fit with α; the P2 two-table (at-shoulder vs
at-1×) candidate comparison; P3 min-accuracy table. Report:
`reports/stability-frontier.md`, descriptive.

## Amendment A1 (2026-07-20, post-launch, infrastructure)

All 16 B=8000 cells OOM'd on the 32 GB RTX 5090: the HVP probe's fp32
functional re-forward + create-graph backward does not fit at batch 8000.
B=8000 reruns use `configs/dev/frontier_lrxbatch_b8000.yaml` — identical
except `instrumentation.hvp: false` (smoothness stays on). The probe is
read-only w.r.t. training, and no pre-registered endpoint touches B=8000
sidecars (P2 excluded B=8000 in advance; P1/P3 are accuracy-only), so no
prediction, definition, or threshold changes. Recorded here for the run
ledger rather than silently.

## Caveats pre-declared

- n=2 per cell resolves a 1.0pp floor (seed sd ≈ 0.15pp at stock) but not
  fine effects; cells adjacent to a crossing may get a third seed (1402)
  before analysis if the crossing is ambiguous — allowed by this
  pre-registration, any other addition is not.
- B=8000 runs 50 steps: enters P1/P3 only.
- New hardware/software vs all prior sidecars (RTX 5090, torch 2.13+cu130 vs
  RTX A6000, prior pin): within-program comparisons only; no cross-program
  numeric comparison without a bridge run at stock settings (B=2000, 1×,
  seeds 1400–1401 — which is a grid cell, so the bridge is built in).
