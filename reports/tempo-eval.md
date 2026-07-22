# Program #8 — eval-seed confirmation (seeds 0–99, n=100 seed-paired)

2026-07-22. Comparison-table run per CLAUDE.md rule 2: the TempoMuon
configuration (global pool, kappa −0.25, rho\* −0.48, window [25, 60],
gain ∈ [0.2, 1]) was frozen on dev seeds 1420–1439 (`reports/tempo-phase-b.md`)
before this launch; no tuning touched eval seeds. Configs
`configs/tempo_eval_muon.yaml` / `configs/tempo_eval_global.yaml`
(commit 30a32ee, pre-launch). 800 runs, single GPU type
(NVIDIA GeForce RTX 5090), airbench_smoke record recipe, endpoint
`tta_val_acc`.

| lr | stock Muon | TempoMuon-global | Δ paired (mean ± SE) |
|---|---|---|---|
| 0.24 (1×, record) | 0.9399 ± 0.0013 | 0.9399 ± 0.0013 | **+0.001pp ± 0.016** |
| 0.48 (2×) | 0.9346 ± 0.0012 | 0.9370 ± 0.0014 | **+0.245pp ± 0.017** |
| 0.72 (3×) | 0.9262 ± 0.0016 | 0.9346 ± 0.0016 | **+0.842pp ± 0.022** |
| 0.96 (4×) | 0.9177 ± 0.0020 | 0.9330 ± 0.0017 | **+1.536pp ± 0.024** |

Descriptive reading (gate judgments are the human's; no `criteria/` file
exists for this comparison):

- At the record LR the controller is statistically exactly free
  (95% CI on Δ ≈ [−0.03, +0.03]pp) — the dev-phase P1 failure is fully
  repaired by the window-60 fix, now at n=100.
- The rescue grows monotonically with LR mis-set, recovering at 4× about
  69% of stock's deficit vs its own 1× (−2.22pp → −0.69pp).
- Same qualitative structure as the dev result (`tempo-phase-b.md` §3);
  no sign of dev-seed overfitting (eval deltas at 3–4× are, if anything,
  slightly larger than dev's B′ values).

Context for interpretation: the placebo decomposition
(`tempo-phase-b.md` §2) showed the mechanism is discovered-schedule, not
per-step feedback; this table establishes the discovery is reliable at
table grade on this testbed. Scale transfer (nanogpt) is being measured
separately (passive probe, PORT CHANGE P6).
