# Anneal dissection: the anneal is the optimization; the late constant-LR phase is nearly inert

Date: 2026-08-03. Flow-first program item 1 (docs/litreview/j-theory-theorem-sweep.md §6).
Data: `results/airbench_anneal_branch_*` (n=21 dev seeds incl. smoke, NVIDIA L40,
git 3691460), paired against the T1 stock-schedule finals
(`airbench_ema`, lr_schedule=linear, same seeds). Generated table:
`reports/wpj-flow-anneal-branch-table.md` / `.json`
(`scripts/analyze_anneal_branch.py`, deterministic).

## Design recap

Constant-LR (peak 0.24) base trajectory; at branch steps T_b ∈ {100, 150, 200}
the full training state is snapshotted and, per anneal length
k ∈ {0, 5, 10, 25, 50}, a branch decays LR linearly to zero over k steps —
all branches at a branch point share the snapshot and an identical cached
batch stream, so differences are attributable to k alone. Baseline: each
seed's own tuned linear-decay-over-200 final.

## Results (paired deltas vs stock final, 95% CI)

| k | T_b=100 | T_b=150 | T_b=200 |
|---|---|---|---|
| 0 | −0.278 ± 0.029 | −0.277 ± 0.039 | −0.258 ± 0.034 |
| 5 | −0.0657 ± 0.0021 | −0.0591 ± 0.0021 | −0.0557 ± 0.0023 |
| 10 | −0.0440 ± 0.0014 | −0.0411 ± 0.0020 | −0.0386 ± 0.0015 |
| 25 | −0.0257 ± 0.0013 | −0.0240 ± 0.0012 | −0.0220 ± 0.0011 |
| 50 | −0.0141 ± 0.0011 | −0.0113 ± 0.0010 | −0.0108 ± 0.0012 |

k* (within 0.2pp of stock): **not reached at any branch point** — the k-curve
is still climbing at k=50, gap ≈ halving per doubling of k (roughly
gap ∝ k^−0.7 over 5→50).

## The two findings

1. **Anneal length dominates; no fast-relaxation saturation.** Accuracy is a
   smooth, strong function of k with no plateau by 50 steps. The anneal is
   not a short "cash-out" of accumulated progress — it is an extended
   optimization process in its own right. (Together with the T1 refutation —
   EMA cannot substitute for it — the anneal is neither noise-removal nor
   fast relaxation: it is where the accuracy is *formed*.)
2. **The constant-LR phase past ~step 100 is nearly inert.** At fixed k, 100
   extra constant-LR steps (T_b 100→200) buy only +0.33pp at k=50 (+1.0pp at
   k=5, shrinking as k grows). The "river position" advanced during
   steps 100–200 of hot training is worth almost nothing to the final
   result. This *refutes the cash-out reading of the river-valley picture*
   for this workload: if the hot phase accumulated hidden progress, later
   branches would anneal to better finals; they barely do.

Corollary: the two findings jointly explain why airbench's tuned recipe
decays from step 0 — if value ∝ anneal length and the hot phase is inert past
early training, the optimal shape under a fixed budget is "spend everything
on the decay," which is exactly the record's linear-to-zero schedule. At
matched 200-step budget, constant-150 + anneal-50 is −1.1pp vs the tuned
shape; the tuned shape is effectively anneal-length 200.

## What this opens (next config, same harness, ~$0.5)

The branch-point independence must break somewhere below step 100 (a k=5
anneal from init cannot give 87%). Branch steps {15, 25, 50, 75, 100} map
where the essential high-LR buildup ends and the inert wandering begins —
i.e., the flow's phase boundary. If the buildup ends early (say ~50 steps),
the speed recipe on this benchmark is "short hot phase + long anneal," and
the phase boundary becomes the quantity an occupancy-style trigger should
detect (its collapse timing during the anneal is already instrumented).

Also natural: extend k beyond 50 at T_b=100 to find where the k-curve meets
the stock final (predicted k ≈ 100–200 by the observed power law, i.e. the
stock shape is near-optimal — worth confirming rather than assuming).

## Caveats

- Same caveats as T1 on regime (200-step recipe, per-step weight renorm,
  peak-LR constant phase); the base phase at a gentler constant LR might not
  be inert. Untested here.
- k* tolerance (0.2pp) is a descriptive convention, not a pre-registered
  criterion; no gate is evaluated by this report.
- Branch evals share the base trajectory's batch stream; branch-vs-stock
  comparisons pair by seed but the stock runs are separate trajectories
  (standard for this repo's seed-paired designs).

## Provenance

Smoke run caught a real harness bug before the sweep (post-eval training ran
with BatchNorm in eval mode and collapsed to chance; fixed in 3691460 by
re-asserting train mode per step — the k=0 branch's exactness isolated the
bug to the train path, validating the snapshot/restore design). VM episode:
rm-flow-l40, ~8.4 h × $0.80 ≈ $6.72 (≈5 h of that was an idle stall from a
watcher bug — pgrep matching its own command line — noted for ops hygiene;
per-run share $0.32 × 21 stamped). VM deleted after pull; 0 active VMs.
