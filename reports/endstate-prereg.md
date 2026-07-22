<!-- Produced by the research-guide workflow (wf_cb6d46db-53e),
adversarially reviewed 2026-07-22; finalized with the dated novelty-sweep
verdict below. Committed BEFORE any run, checkpoint, or Track-S fit. -->

# Program #13 — end-state anatomy of the useful-LR shoulder: pre-registration

2026-07-22. Committed before any run, any checkpoint, and any Track-S fit.
Dev-phase measurement program; nothing here is an evaluation-gate claim.
Adjudication of every branch below is HUMAN; the agent reports numbers.

## 0. Decision record

Program #11 excluded the entire instrumented per-step-scalar span (singly
and in log-linear pairs, held-out, permutation-controlled) as the carrier
of the B^0.35 frontier invariant, and named end-state properties of the
learned solution as the first surviving candidate class. Programs #1
(equivalent destinations), #10 (a tuned trajectory absorbs 0.7–1.4× LR
perturbations from step 50 on at no measurable cost), and #11's
structural corollary (every instrumented scalar lr-flat while accuracy
falls ~1pp+) jointly locate the shoulder damage in the solution, not the
path. No endpoint of any airbench run has ever been saved or examined in
this project (verified: no checkpoint code in src/optim/airbench_zoo.py;
frontier sidecars are scalars only) — there is no zero-GPU route to the
primary question, so new runs are justified. The per-example probe
retained from program #12 serves the example-level anatomy.

Novelty sweep verdict (executed 2026-07-22, same-day, arXiv/Semantic
Scholar; full report in the session record): **the claimed-open cell is
open** — no found work combines (i) an LR ladder spanning the useful-LR
shoulder at several batch sizes, (ii) a multi-observable endpoint-anatomy
panel, (iii) a pre-registered (lr × batch) frontier tracking test, and
(iv) an accuracy-matched fully-annealed undertraining placebo, on Muon or
any optimizer; the placebo arm and the difficulty-quintile decomposition
of the LR effect have no direct precedent found. Claims DROPPED per the
sweep: "first endpoint λ1 across LR and batch" (Kaur, Cohen & Lipton,
arXiv:2206.10654 — SGD end-of-schedule, incl. an LR×batch co-scaling
equalization observation; O4 is hereby repositioned as the *annealed
residue of generalized EoS for spectral descent* extending them to
Muon/post-anneal); "first margins/linear probes under Muon"
(arXiv:2606.09658); any novelty on the difficulty metric (C-scores /
Feldman-Zhang / Toneva lineage, as already stated); "first LR-ladder
endpoint anatomy" in general (Sadrtdinov et al., arXiv:2410.22113 — SGD,
single batch, no placebo). Pre-registered expectation inherited from
arXiv:2302.07011 + 2206.10654: λ1 is *expected* to track LR — for O4 the
informative outcome is the cross-batch equalization test and any
violation, not the LR trend itself. Must-cite list recorded in the sweep
report (incl. 2306.04815, 2604.13627, 2606.04662, 2606.21514,
2605.13079, 2505.02222, 2309.10688, 2006.15081, 1905.13277, 1711.08856,
2605.29152, 2410.05192, 2311.04163).

## 1. Design overview and staging

Track S (zero-GPU, grafted from the LoopID design's review): passive
spectral screen over existing sidecars — runs before any new GPU work.
Stage 1 (B = 1000 only, ~110 runs): adjudicates P1a/P1b/P2/P3.
Stage 2 (B ∈ {2000, 4000}, 140 runs): launched ONLY if P3 passes;
adjudicates P4/P6 cross-batch. This staging makes K1/K2 fire before the
cross-batch budget is spent.
Phase B (coarse grid B ∈ {500..8000} incl. step-matched B = 8000; any LM
endpoint check at 75 min/run): requires a fresh human-reviewed
pre-registration; not part of this program.

## 2. Instrumentation (new, measurement-only)

Two recipe hooks in src/optim/airbench_zoo.py, following the
example_probe precedent (training path untouched):
- recipe.save_endpoint: torch.save of the final filter+head state_dict
  (~5–10 MB fp16) to the SCRATCH checkpoint directory (see §7 —
  checkpoints never enter results/, which is append-only; only the
  derived-features JSON is durable).
- recipe.save_test_logits: one eval-mode forward over the 10k test set
  and the fixed 2000-example train probe slab, saving BOTH plain and
  TTA-averaged (flip+translate, matching tta_level 2) fp32 logits.
  tta_val_acc remains the accuracy endpoint; every per-example
  correctness statement below (O1, O2, O6) is computed from the
  TTA-averaged logits so the "1pp shoulder loss" and the anatomy refer
  to the same endpoint. The plain-logit variants are reported as a
  robustness appendix.
Non-perturbation check (before Stage 1): 5 paired smoke runs hooks
on/off, dev seed 1466 + 4; require |Δ tta_val_acc| < 0.05pp (paired
mean) and wall-time delta < 5%.
Endpoint λ1 tooling: src/instrument/hvp.py currently computes Rayleigh
quotients along supplied directions, not eigenvalues; a power-iteration
matvec loop must be written (budgeted). Pre-registered definition: O4 is
the FULL-NETWORK top Hessian eigenvalue at the endpoint — 20 power
iterations on the fp32 functional loss (fp32_overrides pattern; the
documented fp16 trap is covered), fixed 2000-example batch, fixed probe
seed. Per-matrix Rayleigh quotients are computed but exploratory.

## 3. Runs and seeds

Dev seeds 1466–1475 (n = 10; next fresh block per the ledger; eval seeds
0–99 untouched everywhere). All arms uninstrumented (compile on, no
tracked-pair machinery) + the two hooks. Measured stock run time ~5 s;
budgeted ~8 s with hooks.
Stage 1 (B = 1000): (a) main ladder: the 7 dense rungs of
configs/dev/frontier_dense_b1000.yaml (lr ∈ {0.24 … 0.64}, rung 1 = the
#6 peak anchor) × 10 seeds = 70 runs (new configs endstate_b1000.yaml,
instrumentation off); (b) accuracy-matched undertraining placebo: lr
0.24, epochs ∈ {6, 7} × 10 seeds = 20 runs — NOTE (disclosed): the
airbench schedule is epoch-anchored, so these are fully-annealed
shorter-schedule runs, the intended control (avoids the truncated-
schedule nonzero-final-lr sharpness artifact); if neither budget's arm
mean lands within 0.15pp of the B = 1000 shoulder-rung mean, one
interpolated budget is added (+10 runs); (c) noise floor: 10 same-seed
replicates of the peak rung (bitwise nondeterminism from step ~1–7 makes
them independent draws; anchors below). Stage 2 (conditional on P3):
dense rungs of frontier_dense_b2000.yaml (0.24–0.84) and
frontier_dense_b4000.yaml (0.36–1.11) × 10 seeds = 140 runs. The
sub-peak low-lr rungs present in the B ∈ {2000, 4000} ladders serve as a
free second control: accuracy loss without high-LR exposure.

## 4. Observables (one pre-registered scalar each; per run, then
seed-averaged per cell)

O1 margin tail: q10 of (logit_true − max_other) over the 10k test set
   (TTA-averaged logits).
O2 difficulty profile: per batch, LOSO difficulty = leave-one-seed-out
   fraction-correct across that batch's 10 peak-rung runs (defined per
   batch — every cell has n = 10, so LOSO is available at every B);
   scalar = accuracy on the hardest LOSO quintile.
O3 linear-probe accuracy: ridge probe on frozen pooled penultimate
   features, fit on the train probe slab, evaluated on test.
O4 endpoint sharpness λ1 (full-network, §2) ; the composite lr·λ1 is
   also computed — see the §8 mechanicality disclosure.
O5 representation drift: linear CKA between penultimate features of a
   run and its SAME-SEED peak-rung run, on a fixed 2000-example test
   subset.
O6 generalization gap: mean per-example fp32 eval-mode loss, train probe
   slab minus test.
O7 (negative control) same-seed ‖ΔW‖/‖W‖ between rung pairs, total and
   per-layer, with the filter-normalization constraint noted.
Class labels (load-bearing for P1): O1, O2, O3, O6 are PREDICTION-SPACE
— computed from model outputs, and therefore partially re-encode the
accuracy change by construction (the 1pp shoulder loss IS ~100 flipped
test predictions); they can only describe the anatomy of the loss, never
establish that an end-state signal "exists". O4, O5 are GEOMETRY-SPACE —
the only observables on which an existence claim may rest. O7 is a
pre-committed non-finding.
Multiplicity: each observable contributes exactly the one scalar above.
P1b's family is {O4, O5} (2 tests; both bars must be read with that in
mind). P3's family is 6 observables; under a Gaussian null the chance of
any |separation| ≥ 3× pooled seed sd is ≈ 2% — reported next to the
result. P4's multiplicity is handled by the 1,000-draw permutation
control inherited from #11.

## 5. Pre-registered anchors (zero-GPU, committed as priors)

- Replicate noise floor: within-seed 0.061pp, across-seed 0.099pp
  (program #10); Stage-1 (c) must reproduce the same order or the run
  environment has changed (report, halt).
- Seed sd of tta_val_acc at record config: ≈ 0.14pp.
- B = 1000 shoulder cells and rung means: as recorded in
  reports/frontier-sharpening-tables.md; shoulders recomputed by the
  #6b interpolated-crossing convention, nearest measured rung — the #11
  convention, unchanged.
- lr spans of the dense ladders: B = 1000: 2.67; B = 2000: 3.50;
  B = 4000: 3.08. Consequence (disclosed): any lr-weighted composite
  mechanically clears the ≥ 3.0 within-B bar at B ∈ {2000, 4000} even
  for X flat in lr; see §8.

## 6. Pre-registered measurements and bars

Track S (zero-GPU spectral screen; from the 218 frontier sidecars, seeds
1400–1414). Features per (batch, lr) cell, per matrix, from the pooled
mature per-direction projection series (≥ 5-step post-refresh burn-in;
refresh-harmonic bins at k/t_refresh notched out — disclosed cadence
artifact): fS1 normalized spectral centroid, fS2 fraction of power at
f > 1/4 cyc/step, fS3 spectral flatness. Test: the #11 tracking
signature VERBATIM (fit/validation split as in #11; validation shoulder
ratio ≤ 1.5 AND within-B lr-axis max/min ≥ 3.0 at every validation
batch; 1,000-draw permutation control; expected false passes reported).
Series lengths vary with B (100–800 steps) — disclosed. Outcome: a pass
promotes the summary to a named invariant candidate alongside P4's; a
0/3 null extends the #11 exclusion from scalars to per-direction
spectral summaries and is reported as such.

P1a (anatomy — DESCRIPTIVE, expected, not a finding): the
prediction-space observables (O1, O2, O3, O6) trend across the B = 1000
rungs. Reported as the anatomy of the shoulder loss; by construction
this cannot fail informatively and no claim rests on it.
P1b (existence — the real test): at B = 1000 across the 7 rungs, at
least one GEOMETRY observable (O4, O5) shows Spearman |ρ| ≥ 0.8 over
rung means AND total swing ≥ 3× its pooled per-cell seed sd. Family
size 2, stated.
P2 (memorization tail): the peak→shoulder accuracy loss at B = 1000
(TTA-consistent) is tail-concentrated — hardest LOSO quintile carries
≥ 40% of total per-example accuracy loss (uniform null 20%). Refuted
≤ 25%. Middle band 25–40% (pre-committed reading): suggestive,
reported, does NOT count toward K3/escalation and earns no Phase B on
its own.
P3 (distinct pathology — the placebo test): shoulder-LR endpoints vs
accuracy-matched undertrained endpoints separate on ≥ 1 of O1–O6 by
≥ 3× that observable's pooled seed sd, AND the separation survives
regressing out per-run tta_val_acc (robustness bar, pre-registered —
guards against separation-through-residual-accuracy-mismatch; per-seed
accuracy-matched pairing reported alongside arm means). Refuted if every
observable < 2× after the regression. Middle band 2–3× (pre-committed
reading): suggestive; reported; human decides whether one further
placebo budget is worth ~10 runs; no automatic Stage 2 consequence —
Stage 2 launches only on a full P3 pass.
P4 (invariant candidacy; Stage 2): at least one of O1–O6 (or a
pre-declared composite, see §8) passes the #11 tracking signature —
shoulder ratio ≤ 1.5 across B ∈ {1000, 2000, 4000} AND within-B lr-axis
max/min ≥ 3.0 at every batch — with the 1,000-draw permutation null
reporting ~0 expected false passes. Named prior candidate: lr·λ1_end,
which counts ONLY if unweighted λ1_end itself shows a monotone lr trend
(Spearman |ρ| ≥ 0.8) at every batch (§8). Triviality diagnostic:
within-cell across-seed correlation of the passing observable (and any
composite) with tta_val_acc; |r| > 0.8 flags accuracy re-encoding —
reported, adjudication HUMAN.
P5 (pre-registered negative control): same-seed ‖ΔW‖/‖W‖ between rung
pairs stays in [0.9, 1.1] with no lr trend (Spearman |ρ| < 0.5) —
confirming #1's equivalent-destinations geometry; its confirmation is
pre-committed as a non-finding.
P6 (representation drift; Stage 2 for cross-batch): same-seed CKA to the
peak rung declines monotonically with rung at B = 1000, and the shoulder
rung sits below the replicate-pair CKA floor by ≥ 3× the floor's sd.

## 7. Kill / escalate (decision rules pre-stated; adjudication HUMAN)

K1 (geometry blindness — STOP): P1b fails (both O4 and O5 flat) while
accuracy falls ~1pp+ across the same rungs. Reading: end-state
geometry scalars join trajectory scalars as blind; the surviving
invariant class is genuinely non-scalar (distributions over directions;
the #9 loop transfer function) — plus whatever Track S returned. P1a's
anatomy is still reported, labeled descriptive.
K2 (generic damage — STOP with a paragraph): P3 fails after the
robustness regression. Reading: high-LR damage is indistinguishable
from lost progress at endpoint level; strengthens the noise-temperature
mechanism; Stage 2 does not launch.
K3 (pathology without invariant — HUMAN chooses scope): P1b/P3 pass,
P4 fails. The decomposition becomes a paper subsection extending
finding 6; no escalation of the invariant hunt.
ESCALATE (HUMAN decision only): P4 passes with permutation null ~0 and
the triviality diagnostic on the table. Phase B (fresh human-reviewed
prereg): extend the passing observable to the coarse grid
B ∈ {500..8000} incl. the step-matched B = 8000 arm (~1–2 GPU-h); only
if it survives, an LM endpoint check on the nanogpt harness (75 min/run
— expensive, explicitly human-gated). Any pass is scoped "airbench
only" until measured at LM scale (two airbench→nanogpt transfers have
already failed: #7, #8A).

## 8. Disclosures and risks

- Prediction-space observables re-encode the outcome by construction;
  that is why the existence claim (P1b, K1) rests on geometry
  observables only. This asymmetry is the program's load-bearing repair
  and is stated here so no P1a "pass" is ever cited as a finding.
- lr-weighted composites: the dense-ladder lr spans (2.67/3.50/3.08)
  make criterion 2 of the #11 signature near-mechanical for lr·X at
  B ∈ {2000, 4000}; the burden then rests on the shoulder-ratio
  criterion. Precedent: #11's f2–f4 were lr-weighted, disclosed. Hence
  the unweighted-trend requirement on any lr·X pass, and the triviality
  diagnostic computed on composites too.
- TTA convention: all per-example anatomy uses TTA-averaged logits so
  the anatomy decomposes the same accuracy that defines the shoulder;
  plain-logit robustness appendix disclosed.
- Placebo arm: epoch change alters the epoch-anchored schedule shape;
  the undertrained arm is a fully-annealed shorter schedule (intended);
  arm-mean matching tolerance 0.15pp against within-arm seed sd
  ≈ 0.14pp is why the P3 robustness regression is mandatory, not
  optional.
- Storage/governance: checkpoints (~2–3 GB) live in the session scratch
  directory, are deleted after feature extraction, and never enter
  results/ (append-only); the durable record is
  reports/endstate-features.json (per the #11 feature-cache pattern)
  plus results JSONs with the standard provenance fields.
- Tooling risk: the power-iteration λ1 loop is new code; budgeted, unit
  tested against a fixed tiny quadratic before use.
- Tier: dev-phase, n = 10, agent-committed prereg; no criteria/ file;
  effect sizes not comparable to the n = 100 tables; novelty claims
  dated by the §0 sweep.

## 9. Cost

Track S: zero GPU, ~half an agent-day. Stage 1: ~110 runs × ~8 s ≈ 15
GPU-min + smokes; Stage 2 (conditional): 140 runs ≈ 20 GPU-min.
Offline analysis (forwards, ridge probes, CKA, power-iteration HVP over
~240 checkpoints): ~0.5–1 GPU-h. Tooling + prereg + configs + analysis
script: ~1 agent-day. Wall ~4–6 h end-to-end; honest buffer +1–2 h if
the placebo needs a third budget or the λ1 probe needs batch tuning.
Zero cloud spend (the $13.60 pinned total unchanged). Seeds consumed:
1466–1475; next fresh block after this program: 1476+.
