# WP2.2 / WP2.3 — Airbench experiment matrix: run plan (Gate-1-amended)

Agent-authored plan for the plan-§2.2 experiment matrix (adapted for budget;
all deviations from the research plan are listed in §Deviations below and are
part of the Gate-2 record). Companion configs: `configs/wp22_*.yaml`,
`configs/wp23_lambda_tracking.yaml`. Config-consistency tests:
`tests/test_wp22_configs.py` (cross-checks the manifest in this file against
the actual configs).

**AMENDED per `reports/gate1-decision.md` (binding; oscillation-focused
scope, amendments A1/A2/A4/A5).** The pre-registered Phase-2 claim attaches
to the **oscillation channel only**: `wp22_headtohead_routed.yaml` is the
osc-only primary arm; three constant-attenuation arms
`wp22_goscconst_{025,050,075}.yaml` (A2) adjudicate adaptive vs constant
g_osc; full three-channel routing and noise-only are demoted to
**EXPLORATORY** appendix arms (`wp22_exploratory_fullrouted.yaml`,
`wp22_channel_noise_only.yaml`, plus companion null
`wp22_null_routed_rhoignored.yaml`) with no pre-registered claim; a β=0.9
osc-arm dev probe `wp22_beta09_oscarm.yaml` covers timescale sensitivity
(A5); all routed runs log treated-fraction / per-channel gain telemetry
(A5, `metrics["routing_stats"]` + `metrics["routing_timeseries"]`);
mechanism probes (A4, `configs/dev/instrumented_airbench_{mom0,lrhalf,
lrquarter}.yaml`) run before/alongside stage A. Config edits implementing
this scope were made by the agent under the same explicit full delegation
recorded in the Gate-1 decision record.

## Design constraints (all groups)

- **Harness:** `experiment: airbench_smoke` for every run; identical recipe
  everywhere (`normalize_filter_weights: true`, `compile: false`,
  `tta_level: 2`, 8 epochs, batch 2000). The only permitted recipe delta is
  `recipe.grad_clip` in the one ablation that is *about* clipping (G4b).
- **GPU type:** one A6000-class GPU for every run that enters any table;
  `scripts/aggregate.py` structurally refuses mixed-`gpu_type` tables.
- **Seeds:** comparison tables use the eval policy (seeds 0–99, resolved by
  `scripts/sweep.py` at launch time, never written in configs). Selection /
  stress stages use dev seeds (≥ 1000). No literal seed < 1000 appears in any
  config (enforced by sweep.py and by `tests/test_wp22_configs.py`).
- **Metric:** `accuracy` (aggregate.py default), mean ± std ± 95% CI per
  config; wall time reported alongside. Stress group additionally reports
  divergence rate (see G6 notes).

## Two-stage tuning (headline deviation)

The plan (§2.2) prescribes n = 100 seeds for *every* config, including the
3×3 LR×WD sweeps. A plan-faithful fair-tuning block alone would be
2 optimizers × 9 grid points × 100 seeds = 1800 runs ≈ 11 h — the entire
budget envelope. Instead:

- **Stage A (selection, dev seeds, n = 25):** the full 3×3 grid for muon and
  routed on dev seeds 1000–1024. Selection only; these numbers never enter a
  comparison table.
- **Stage B (evaluation, eval seeds, n = 100):** only the stage-A argmax
  point per optimizer is re-run on the full eval set. Placeholder configs
  `wp22_tuneB_{muon,routed}.yaml` carry `TBD-STAGE-A` markers and are filled
  by `scripts/plan_wp22.py fill-tuneB` from the stage-A aggregate CSV +
  sweep manifest.

This is a pre-registered deviation for the (delegated) gate record: the
fair-tuning comparison is nominally n = 100, but hyperparameter *selection*
used n = 25 dev seeds, so selection noise (~0.2σ/√25 per cell) is not
propagated into the stage-B CIs, and the argmax is mildly optimistic on the
selection set (selection bias affects both optimizers symmetrically since the
grids and seed sets are identical).

## Groups

| Group | Configs | Stage | Seeds | n | Variants | Runs |
|---|---|---|---|---|---|---|
| G1 head-to-head (§2.2.1/.3) | `wp22_headtohead_{muon,routed,dynmuon,adamuon,normuon}.yaml` — routed = **osc-only primary** (A1) | eval | eval 0–99 | 100 | 1 each | 500 |
| G2 fair-tuning stage A (§2.2.2) | `wp22_tuneA_{muon,routed}.yaml` — routed rows osc-only (A1) | dev | 1000–1024 | 25 | 9 each (3×3 LR×WD) | 450 |
| G3 fair-tuning stage B (§2.2.2) | `wp22_tuneB_{muon,routed}.yaml` (placeholders; routed osc-only) | eval | eval 0–99 | 100 | 1 each | 200 |
| G4 null ablations (§2.2.4) | `wp22_null_muon_wd.yaml` (placeholder), `wp22_null_muon_lrclip.yaml` (blocked), `wp22_null_routed_rhoignored.yaml` (**EXPLORATORY**, full-routing companion), `wp22_null_routed_randomgating.yaml` (osc-only placebo) | eval | eval 0–99 | 100 | 1 each | 400 |
| G5 constant-g_osc arms (A2) | `wp22_goscconst_{025,050,075}.yaml` (osc-only, fixed g_osc) | eval | eval 0–99 | 100 | 1 each | 300 |
| G6 LR stress (§2.2.5) | `wp22_stress_{muon,routed}.yaml` (routed osc-only) | dev | 1000–1009 | 10 | 3 each (LR ×{1,1.5,2}) | 60 |
| G7 exploratory arms (A1) | `wp22_exploratory_fullrouted.yaml`, `wp22_channel_noise_only.yaml` — **EXPLORATORY**, no pre-registered claim | eval | eval 0–99 | 100 | 1 each | 200 |
| G8 β sensitivity (A5) | `wp22_beta09_oscarm.yaml` (osc-only, β=0.9) | dev | 1000–1009 | 10 | 1 | 10 |
| G9 λ-tracking (§2.2.6, WP2.3) | `wp23_lambda_tracking.yaml` (blocked) | dev | 1000–1002 | 3 | 1 | 3 |

Dropped vs the pre-amendment matrix: `wp22_channel_osc_only.yaml` — it would
be byte-identical to the amended osc-only `wp22_headtohead_routed.yaml` (it
existed precisely so the osc-only branch needed no new configs).

Off-manifest (A4, instrumented measurement, not comparison runs):
`configs/dev/instrumented_airbench_{mom0,lrhalf,lrquarter}.yaml` — 2 dev
seeds each (1300–1305), stock WP1.2 instrumented recipe with a restricted
`probe_overrides:` block (momentum=0 / lr×0.5 / lr×0.25); analysis via
`scripts/analyze_mechanism.py` (descriptive). ~6 × ~3 min uncompiled
instrumented runs ≈ 0.3 h ≈ $0.12–0.30 on the A6000 class.

Grid details:

- **G2 LR axis** {0.12, 0.24, 0.48} (record 0.24 centered, ×/÷2).
  **WD axis** {0.0, 0.004, 0.008} on `optimizer.weight_decay` — see
  Deviation 4 for why WD rides the optimizer, and the value provenance
  (0.004 = harness bias/head default 2e-6 × batch 2000; 0.008 = 2×).
- **G6 LR axis** {0.24, 0.36, 0.48} = record ×{1.0, 1.5, 2.0}.

## Budget

Assumption: ~22 s/run amortized on one A6000 class (WP0.1/WP0.4 timing;
includes data-load amortization, torch compile off). Routed runs carry the
routing overhead (< 10% per the WP1.1 target) — covered by the headroom
below. Price: $0.4067/h (A6000 spot class).

| Group | Runs | GPU-time | Cost |
|---|---|---|---|
| G1 head-to-head | 500 | 3.06 h | $1.24 |
| G2 stage A | 450 | 2.75 h | $1.12 |
| G3 stage B | 200 | 1.22 h | $0.50 |
| G4 null ablations | 400 | 2.44 h | $0.99 |
| G5 constant-g_osc arms (A2) | 300 | 1.83 h | $0.75 |
| G6 LR stress | 60 | 0.37 h | $0.15 |
| G7 exploratory arms | 200 | 1.22 h | $0.50 |
| G8 β sensitivity | 10 | 0.06 h | $0.02 |
| **Total (G1–G8)** | **2120** | **12.96 h** | **$5.27** |
| G9 λ-tracking (pending wiring) | 3 | ~0.03 h | ~$0.01 |
| A4 mechanism probes (off-manifest, dev) | 6 | ~0.3 h | ~$0.12 |

Headroom to the ~15 GPU-h envelope: ~2.0 h (~14%) — absorbs routing
overhead, retries, and stress-run tails. If the envelope tightens, the
exploratory G7 arms are the pre-declared first cut (they carry no
pre-registered claim). Cost fields in results JSONs are human-filled per
CLAUDE.md rule 5.

### Machine-readable manifest (parsed by tests/test_wp22_configs.py)

```json
{
 "gpu_type_class": "A6000",
 "sec_per_run": 22,
 "price_per_hour_usd": 0.4067,
 "groups": [
  {
   "group": "G1-headtohead",
   "seed_policy": "eval",
   "n_seeds": 100,
   "configs": {
    "configs/wp22_headtohead_muon.yaml": 1,
    "configs/wp22_headtohead_routed.yaml": 1,
    "configs/wp22_headtohead_dynmuon.yaml": 1,
    "configs/wp22_headtohead_adamuon.yaml": 1,
    "configs/wp22_headtohead_normuon.yaml": 1
   },
   "runs": 500
  },
  {
   "group": "G2-tuneA",
   "seed_policy": "dev",
   "n_seeds": 25,
   "configs": {
    "configs/wp22_tuneA_muon.yaml": 9,
    "configs/wp22_tuneA_routed.yaml": 9
   },
   "runs": 450
  },
  {
   "group": "G3-tuneB",
   "seed_policy": "eval",
   "n_seeds": 100,
   "configs": {
    "configs/wp22_tuneB_muon.yaml": 1,
    "configs/wp22_tuneB_routed.yaml": 1
   },
   "runs": 200
  },
  {
   "group": "G4-null",
   "seed_policy": "eval",
   "n_seeds": 100,
   "configs": {
    "configs/wp22_null_muon_wd.yaml": 1,
    "configs/wp22_null_muon_lrclip.yaml": 1,
    "configs/wp22_null_routed_rhoignored.yaml": 1,
    "configs/wp22_null_routed_randomgating.yaml": 1
   },
   "runs": 400
  },
  {
   "group": "G5-goscconst",
   "seed_policy": "eval",
   "n_seeds": 100,
   "configs": {
    "configs/wp22_goscconst_025.yaml": 1,
    "configs/wp22_goscconst_050.yaml": 1,
    "configs/wp22_goscconst_075.yaml": 1
   },
   "runs": 300
  },
  {
   "group": "G6-stress",
   "seed_policy": "dev",
   "n_seeds": 10,
   "configs": {
    "configs/wp22_stress_muon.yaml": 3,
    "configs/wp22_stress_routed.yaml": 3
   },
   "runs": 60
  },
  {
   "group": "G7-exploratory",
   "seed_policy": "eval",
   "n_seeds": 100,
   "configs": {
    "configs/wp22_exploratory_fullrouted.yaml": 1,
    "configs/wp22_channel_noise_only.yaml": 1
   },
   "runs": 200
  },
  {
   "group": "G8-beta-sensitivity",
   "seed_policy": "dev",
   "n_seeds": 10,
   "configs": {
    "configs/wp22_beta09_oscarm.yaml": 1
   },
   "runs": 10
  },
  {
   "group": "G10a-gate2-ladder",
   "title": "Gate-2 extended LR ladder (reports/gate2-decision.md)",
   "configs": {
    "configs/wp22_stress2_muon.yaml": 3,
    "configs/wp22_stress2_routed.yaml": 3
   },
   "seed_policy": "dev",
   "n_seeds": 10,
   "runs": 60
  },
  {
   "group": "G10b-gate2-lr2x",
   "title": "Gate-2 2x-LR eval confirmation (reports/gate2-decision.md)",
   "configs": {
    "configs/wp22_lr2x_muon.yaml": 1,
    "configs/wp22_lr2x_routed.yaml": 1
   },
   "seed_policy": "eval",
   "n_seeds": 100,
   "runs": 200
  }
 ],
 "total_runs": 2380,
 "total_gpu_hours": 14.54,
 "total_cost_usd": 5.92,
 "pending": {
  "group": "G9-lambda-tracking",
  "seed_policy": "dev",
  "n_seeds": 3,
  "configs": {
   "configs/wp23_lambda_tracking.yaml": 1
  },
  "runs": 3,
  "blocked_on": "instrumentation-on-routed wiring"
 }
}
```

## Execution order

1. **Preflight (human unless delegated):** confirm the Gate-1 threshold sync
   (`TBD-CHECK-GATE1` — the Gate-1 record kept the Phase-1 dev thresholds,
   so no edit is currently required); settle DynMuon's `lr` (TBD-TUNABLE,
   open WP0.4 tuning issue) — dev-seed probes only if spent.
2. **A4 mechanism probes** (dev, 6 instrumented runs, off-manifest) —
   before/alongside stage A per the Gate-1 record; analyze with
   `scripts/analyze_mechanism.py` (descriptive; the bulk-vs-top anomaly is
   reported either way).
3. **G2 stage A** (dev, 450 runs). Aggregate with
   `scripts/aggregate.py --out-csv`, then
   `scripts/plan_wp22.py fill-tuneB` to materialize
   `wp22_tuneB_{muon,routed}.yaml` and `wp22_null_muon_wd.yaml`
   (WD argmax at lr 0.24). Human eyeballs the filled values before launch.
4. **G8 β sensitivity + G6 stress** (dev, 70 runs) — early: a stock-Muon
   divergence at 1.5× and a β-fragile oscillating set are both
   Gate-2-relevant findings and cheap to get.
5. **G1 head-to-head + G5 constant-g_osc** (eval, 800 runs) — the amended
   primary table: muon vs routed(osc-only, adaptive) vs constant-g_osc
   {0.25, 0.5, 0.75} vs baselines.
6. **G3 stage B + G4** (eval, 600 runs; `wp22_null_muon_lrclip.yaml`
   only after the harness `grad_clip` addition lands — otherwise it runs
   without clipping and looks valid; see Deviation 6).
7. **G7 exploratory arms** (eval, 200 runs) — last; first cut on budget
   pressure; reported in the appendix only.
8. **G9 λ-tracking** once instrumentation-on-routed is wired (WP2.3).
9. Aggregate everything into the comparison table
   (`aggregate.py --gpu-type <A6000 string>`); descriptive report vs
   `criteria/phase2_success.yaml` (to be drafted for the osc-only scope per
   amendment A1 **before** the eval-seed comparison runs); **stop at the
   Gate-2 human checkpoint.**

All eval sweeps are cloud runs launched by the human via
`scripts/launch_cloud.sh sweep <config>` per the compute boundary; the agent
consumes synced `results/` JSONs. All routed comparison runs record
`metrics["routing_stats"]` and `metrics["routing_timeseries"]` (A5) — a
null result without treated-fraction / gain-distribution logs would be
uninterpretable (n_min=50 leaves the first 50 of 200 steps untreated and
~67% of directions reset per refresh at align_min=0.9).

## Gate-1 scope (resolved)

The Gate-1 decision (`reports/gate1-decision.md`) selected the
**oscillation-only** branch; the section that previously enumerated both
scopes is resolved as follows, and the config edits are DONE (agent, under
the delegation recorded in the gate record):

- Osc-only (`enable_noise_channel: false`) in: `wp22_headtohead_routed.yaml`
  (primary), `wp22_tuneA_routed.yaml`, `wp22_tuneB_routed.yaml`,
  `wp22_null_routed_randomgating.yaml`, `wp22_stress_routed.yaml`,
  `wp22_goscconst_{025,050,075}.yaml`, `wp22_beta09_oscarm.yaml`.
- `wp22_null_routed_rhoignored.yaml` **keeps both channels** and is marked
  EXPLORATORY: rho_ignored disables the oscillation label, so osc-only +
  rho_ignored would be bit-for-bit stock Muon (a meaningless null); it is
  only interpretable as the companion null of the exploratory full-routing
  arm.
- `wp22_channel_osc_only.yaml` deleted (byte-duplicate of the amended
  primary); `wp22_channel_noise_only.yaml` and the former full-routing
  head-to-head (now `wp22_exploratory_fullrouted.yaml`) kept as EXPLORATORY
  appendix arms at eval n=100.
- `wp22_null_muon_wd.yaml` kept (bounds "WD alone" on this harness) but,
  like the noise channel it nulls, carries no pre-registered claim.

## What stage B fills in

`scripts/plan_wp22.py fill-tuneB --csv <stageA.csv> --manifest
<stageA_manifest.json> --target <placeholder.yaml> [--fix optimizer.lr=0.24]`:

- maps each stage-A variant config name → its grid overrides via the sweep
  `manifest.json`;
- picks the argmax `metric_mean` row (optionally restricted by `--fix`
  constraints, used for `wp22_null_muon_wd.yaml` to hold lr at 0.24);
- rewrites the placeholder's `TBD-STAGE-A` values and removes the `status`
  marker, printing the chosen cell and its mean ± CI for the human to
  confirm before launch.

## Deviations from plan §2.2 (gate record)

1. **Two-stage tuning** (selection dev n=25 → evaluation eval n=100 at the
   argmax only) instead of n=100 everywhere. Rationale: budget (§Two-stage
   tuning above). Consequence: selection noise not propagated; symmetric
   across optimizers.
2. **No matched LR×WD sweep for DynMuon / AdaMuon / NorMuon** (plan §2.2.3
   "baselines at matched sweep"). They run at WP0.4-carried settings only.
   This *weakens the killer table*: a baseline loss can be attributed to
   tuning. DynMuon is worst off (open WP0.4 tuning issue; its
   `adjust_lr=spectral_norm` semantics make the record lr scale
   inapplicable); its lr is flagged TBD-TUNABLE for a human decision.
   AdaMuon/NorMuon lrs are WP0.4 dev starting points, equally flagged in
   their config headers.
3. **Routing hyperparameters frozen** at constructor defaults; the plan-§2.1
   (g_noise, ρ_osc, k) sweep is not scheduled. Only LR×WD is swept for
   routed (matching muon's axes for fairness). A routing-hparam sweep is a
   natural post-Gate-2 follow-up if the frozen defaults underperform.
4. **WD grid rides `optimizer.weight_decay`** (filter params): the harness
   has no harness-level WD for filter params — `recipe.sgd_weight_decay`
   covers only biases/head (SGD side). Values {0, 0.004, 0.008} = {record 0,
   bias/head-default equivalent 2e-6×2000, 2×}. The momentum-grid fallback
   from the task brief is NOT needed (both muon.py and routed.py accept
   `weight_decay`).
5. **LR stress at dev n=10 per point** (not n=100): divergence-rate
   resolution is ±10%; accuracy at surviving seeds is indicative only. The
   secondary criterion (max stable LR ratio ≥ 1.3) is assessed on divergence
   pattern, not fine accuracy differences.
6. **`wp22_null_muon_lrclip.yaml` requires a ~5-line harness addition**
   (`recipe.grad_clip` → `clip_grad_norm_` on filter params in
   `src/optim/airbench_zoo.py`, outside this task's file ownership). The
   harness currently *silently ignores* unknown recipe keys, so the config
   is marked NOT-RUNNABLE-YET; launching it before the addition would
   produce a wrong-but-valid-looking run. Clip value 1.0 is provisional,
   TBD-tunable on dev seeds after the addition.
7. **λ-tracking (§2.2.6) pending wiring:** instrumentation on a routed run
   (and export of routed's own gating decisions into the sidecar) is not
   wired; `wp23_lambda_tracking.yaml` is a full spec marked NOT-RUNNABLE-YET
   with the wiring checklist in its header.
8. **Optimizer-internal RNG fixed:** `optimizer.seed` (routed's subspace /
   placebo-gating RNG) is a config literal (2600) and does not vary with the
   eval seed passed via `run.py --seed` (the harness does not forward the run
   seed into the optimizer config). Model init / data order / augmentation
   still vary per seed. Main impact: the `random_gating` placebo's RNG
   stream is common across seeds (gating decisions still interact with
   per-seed training dynamics). Wiring the run seed into `optimizer.seed`
   would be a small `run.py`/harness change — flagged, not assumed.

## Risks

- **Amortized 22 s/run** may be optimistic for routed (overhead target
  < 10%, unverified on A6000) and for stress runs that diverge slowly.
  Mitigation: 35% budget headroom; re-estimate after G2.
- **Divergent stress runs** may produce non-finite metrics or crash;
  `aggregate.py` refuses mixed present/missing metrics within a config
  group. Plan: count divergences from raw results JSONs (a run that crashed
  writes no results file — count shortfall vs the manifest), aggregate
  survivors with `--skip-invalid`, and report both numbers explicitly.
- **Stage-A argmax overfit** to 25 dev seeds (Deviation 1); ties within CI
  are broken toward the record value (plan_wp22.py reports ties for the
  human to adjudicate).
- **DynMuon under-tuning** (Deviation 2) invites a "weak baseline"
  criticism at Gate 2; the config header and this doc flag it explicitly.
- **Gate-1 threshold sync:** all routed configs carry constructor-default
  thresholds marked TBD-CHECK-GATE1; if the Gate-1 record set different
  Phase-1-informed values, the human must sync them *before* any G1–G6
  routed run, else the whole matrix runs at unregistered thresholds.
