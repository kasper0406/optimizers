# Program #10 — branched-probe meta-control: Phase A measurement plan

2026-07-22. Pre-registered measurement (not a gate); committed before any
grid run. Code: `recipe.lr_fork` hook in `src/optim/airbench_zoo.py`
(same-seed runs share a trajectory prefix; a mid-run LR multiplier makes
a pair of runs an approximately common-randomness-paired branch pair at
full-run cost, ~15 s here).

## 0. Decision record

Novelty sweep (2026-07-22): the full loop — CRN-paired short probe
branches from a single main run, paired-difference response estimates,
online controller of a low-dim hyperparameter vector — is unpublished.
Nearest neighbors: AutoLRS (sequential rollback probes + BO, absolute
loss, LR-only, ~2× search overhead), Ai2 CBS branched training
(branches as measurement, loop not closed, informs the *next* run),
NRES/PES (antithetic perturbed particles for meta-gradients; no
protected main run), HOOF (same-data candidate evaluation, RL one-step),
PBT family (population of full runs). Cheap main-trajectory controllers
(AdaLRS/GreedyLR/GNS-based batch control) are the zero-overhead
baselines any Phase B must beat. The sweep also flagged that *the
pairing-benefit measurement itself is absent from the literature* —
that measurement is this Phase A.

Motivating synthesis: `reports/steep-flat-taxonomy.md` — the steep knobs
are low-dimensional; the three failure modes any online meta-learning
must clear are short-horizon bias, closed-loop endogeneity (program #9),
and seed noise. Paired probes are the design answer; Phase A measures
whether the estimator they provide is actually informative, before any
controller exists.

## 1. Incidental findings already in hand (smoke, disclosed)

- **The airbench recipe's training loss silently overflows to inf**
  (fp16 sum-reduction CE, from ~step 40) — harmless to training because
  CE's backward never consumes the forward value; fatal for telemetry.
  The probe objective is therefore an fp32 recompute (logged only when
  `lr_fork` is set; the training path is untouched).
- **Same-seed runs are not bitwise deterministic** — first loss
  divergence at step ~7 (cudnn autotune/atomics). "Same seed = perfect
  CRN" is approximate; the pairing benefit is exactly the same-seed
  correlation that survives, and measuring its decay with horizon is
  the point of Phase A (and realistic: LM-scale runs are not bitwise
  deterministic either).

## 2. Grid (dev seeds 1452–1461, n = 10)

fork step t ∈ {25, 50, 100, 150} × mult m ∈ {0.7, 0.85, 1.0, 1.15, 1.4}
× 10 seeds = 200 runs, airbench_smoke record recipe. The m = 1.0 cells
are identical runs across t (the fork is a no-op) — retained as 4×
replicate determinism/noise calibrations per seed. Endpoints: fp32
per-step train loss series; final tta_val_acc.

## 3. Pre-registered measurements

- **M1 (pairing benefit):** Var of same-seed paired probe differences
  vs cross-seed unpaired differences, for probe objective = mean loss
  over window [t, t+Δ], Δ ∈ {5, 10, 25, 50}. Deliverable: the variance-
  reduction factor vs Δ (the CRN decay curve) — per the sweep, this
  curve is unpublished in any setting.
- **M2 (probe predictiveness):** correlation between the Δ-step paired
  loss difference and (a) the same pair's end-of-run acc difference,
  (b) the ground-truth d(acc)/d(lr) known from this repo's exhaustive
  LR ladders. Deliverable: minimum Δ\* at which probe sign agreement is
  reliable (the controller's minimum probe length).
- **M3 (short-horizon bias, direct):** per fork step t, does the
  Δ-window probe prefer a different multiplier than the final outcome
  does? (The literature's failure mode, measured against ground truth.)
- **M4 (noise floor):** replicate (m = 1.0) run spread: distribution of
  divergence onset and of final-acc spread among identical configs —
  the irreducible noise a controller must live with.

Phase B (the controller) is only specified after M1–M4 are read; its
prereg will cite the measured Δ\*, variance factors, and the AdaLRS-class
zero-overhead baselines.
