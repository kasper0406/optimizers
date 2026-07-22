# Routed Muon — Project State & Findings

Consolidated 2026-07-20. Companion to `reports/paper-draft.md` (the outward
writeup) and the gate records. This document is the internal state of the
whole effort: what was tested, what was found, what is publishable, and what
the data leaves open.

---

## 1. What this set out to test

**Hypothesis (`routed-muon-research-plan.md`):** the variance Muon amplifies
decomposes per singular direction into three regimes — persistent signal,
sampling noise, edge-of-stability oscillation — that are separable online via
cheap temporal statistics (mean, variance, lag-1 autocorrelation of projected
gradients), and the correct response differs per regime (keep / shrink /
damp-and-raise-LR). Primary claim: online per-direction regime-routed spectral
shaping beats stock Muon and globally-scheduled shaping (DynMuon) at matched
tuning — or produces a measurement result that adjudicates why not.

It produced the measurement result. The routing idea is a well-powered,
placebo-controlled null on the tested workload; the measurement infrastructure
built to test it yielded the actual contributions.

## 2. Execution summary

Phase 0 (substrate) → Phase 1 (measurement) → **Gate 1** (adversarially
reviewed: proceed, oscillation-focused) → Phase 2 (the routed optimizer,
14-arm comparison) → **Gate 2** (adversarially reviewed: FAIL both criteria,
final) → five follow-up "brainstorm" programs (all four testable predictions
refuted) → a nanogpt-scale benchmark port and reproduction diagnosis.

Every gate and every consequential decision was drafted, then attacked by an
independent adversarial agent, then amended or overturned on the record. Two
of the biggest course-corrections (Gate 2 secondary criterion; the nanogpt
amend-and-proceed) came from those reviews overturning the primary draft.

Total cloud spend: **$13.60**. 639 tests. ~2,700 result JSONs with full
provenance. All committed. (2026-07-21: program #6 ran locally on the new
2× RTX 5090 box — the first non-cloud GPU capacity in the project.)

## 3. Findings

### 3.1 Affirmative measurements (novel; nothing equivalent in the 2026-07-19/20 literature sweeps, `docs/litreview/`)

1. **A structured negative-autocorrelation population in per-direction gradient
   projections under Muon.** 60–89% of tracked-direction snapshots have lag-1
   ρ < −0.2 vs a ~22–29% white-noise floor (airbench94, n=20). Phase-structured
   (peaks mid-training, collapses in the LR anneal), LR-driven (monotone across
   configs and within-run), momentum-independent (momentum=0 leaves it intact),
   robust to batch-sampling artefacts (with-replacement ablation). Bulk
   directions participate as much as top directions.
2. **Curvature does not govern Muon's stability.** Stable at HVP-measured
   η·λ ≈ 65 (GD's bound is 2); no divergence cliff even at 6× the record LR —
   graceful degradation, useful-LR shoulder at 2–3×. This is momentum+minibatch
   data on the regime the non-Euclidean edge-of-stability line
   (arXiv:2603.05002, ICML'26 oral) explicitly names as open.
3. **Amplitude ratios do not measure curvature under Muon.** Implied η·λ from
   oscillation amplitude saturates at the noise floor (~2–4) while HVP-measured
   η·λ spans 0–65 (Pearson ≈ 0). "Curvature-for-free from amplitude" is dead
   for normalized optimizers.
4. **Per-direction persistent signal is structurally unmeasurable at normal
   batch SNR.** t-ceiling |t| ≤ SNR·√ess with observed SNR q90 ≈ 0.26; and
   removing the integration-window ceiling (frozen probes, full-run
   accumulation, program #4) did **not** reveal hidden signal (median final
   |t| 0.61 naive / 0.87 Newey–West; growth slope ~0.05 vs √t's 0.5).

### 3.2 The intervention null (Gate 2, `reports/gate2-decision.md`)

Per-direction oscillation damping does nothing on airbench. Across adaptive
gains, constant gains {0.25, 0.5, 0.75}, full three-channel routing, and a
random-gating placebo (n=100 eval seeds each, seed-paired), any effect above
**0.042pp is excluded at 97.5% confidence** (pre-registered bar 0.27pp).
No convergence benefit at any epoch either — per-epoch curves overlie stock
and the placebo shows the same early jitter. Stability-margin ratio exactly
1.0 (both stock and routed cross the accuracy floor between 2× and 3× LR).
The one promising dev-seed signal (2× LR, +0.144pp, p=0.006) was killed by
its own pre-registered n=100 confirmation (−0.024pp, sign flipped).

### 3.3 The mechanism — equivalent destinations (program #1, `reports/brainstorm-programs.md`)

The null is **not** inertness. Twin-trajectory probes (identical init and batch
sequence; stock-vs-stock control diverges exactly 0.000 at every step) show
the intervention drives training to **essentially unrelated weights**
(‖ΔW‖/‖W‖ ≈ 0.95–0.99) that reach identical accuracy. The knob steers hard;
the landscape supplies an abundance of equally-good destinations. The
hypothesised output-vs-state "compounding" asymmetry does not exist (matched-
gain state/output divergence ratio 0.95× vs a predicted ≥10×). This is likely
general (cf. Song et al. ICLR'25: projecting out the entire dominant Hessian
subspace trains just as well); its *consequence* — no per-direction spectral
intervention buys anything — is probably task/headroom-dependent.

### 3.4 Follow-up refutations (all pre-registered, all refuted — informative)

- **Occupancy is not a state function of LR** (program #2): it tracks
  *schedule position* (lr/lr₀), not instantaneous lr; the record schedule does
  not hold it constant (0.75 → 0.41 across phases). Kills the naive
  "constant-occupancy schedule" idea; reframes occupancy as a progress signal.
- **Spectral directional smoothness does not equilibrate at c/lr** (program #3):
  lr·D_smooth_spectral scales 123.6/222.6/317.6 across the LR ladder (ratio
  2.57), no more lr-invariant than the Euclidean version (2.26). Whatever sets
  Muon's stable-LR ceiling is not a smoothness plateau in either norm.
- **No hidden persistent signal at long integration** (program #4, above).

### 3.5 Stability frontier in (lr × batch) (program #6, 2026-07-21, local 2×RTX 5090)

Pre-registered before any run (`reports/stability-frontier-preregistration.md`);
80-cell grid (5 batch × 8 lr rungs × 2 dev seeds), full report
`reports/stability-frontier.md`.

- **The useful-lr frontier is batch-coupled at lr\* ∝ B^0.35, CI95
  [0.30, 0.42]** (program #6b dense-ladder sharpening, n=5, interpolated
  crossings, 2026-07-21): the deterministic (batch-independent) account is
  decisively excluded, and so is the √B point prediction — program #6's
  coarse α = 0.50 was rung-quantization luck, as #6b was pre-registered to
  adjudicate. At B ≥ 4000 low lr becomes the losing side (acc non-monotone
  in lr), a signature that survives step-matching.
- **The B=8000 trend-break was undertraining** (program #6b Part 2):
  step-matched (epochs 32, ~200 steps) B=8000 recovers 93.75% and its
  peak-referenced shoulder lands at 0.96 ≥ B=4000's 0.72 — the rightward
  shift continues, it does not saturate.
- **The law does NOT transfer to the nanogpt record recipe** (program #7,
  2026-07-22, pre-registered, 48 cells on the validated local testbed):
  across an 8× token-batch range (98K–786K tokens/step, fixed 346M-token
  budget) the useful Muon-lr band is batch-invariant — α = −0.29, CI95
  [−0.35, +0.30]; the airbench point 0.35 is excluded; valley pinned at
  ~0.7× record lr at every batch; no cliff anywhere. Domain-bounding
  result: batch-coupled at CNN-scale batches, batch-invariant at LM-scale
  token batches — consistent with (untested) critical-batch-size
  saturation. `reports/frontier-nanogpt.md`.
- **No instrumented quantity is the frontier invariant**: occupancy, spectral
  and Euclidean directional smoothness, and HVP η·λ q90 all fail the
  pre-registered tracking signature; curvature is the *least* equalized
  (~5–6× across batch) — extending "curvature does not govern" to the batch
  axis.
- **Cliff-free degradation persists across a 16× batch range** (no collapse
  to random anywhere on the ladder).
- Infrastructure: the new local 5090/torch-2.13 stack reproduces the historic
  baseline at the bridge cell (94.14% + P2 quantities inside prior ranges);
  HVP probe OOMs at B=8000 on 32 GB (amendment A1, endpoint-neutral).

### 3.6 Infrastructure findings (nanogpt, WP0.2)

A 1×H100 port of the pinned modded-nanogpt record (2025-07-12_BosAlign) with
exact token-batch matching (8× grad accumulation, 393,216 tokens/step). Two
methodological findings fell out:
- **The record's published steps-to-target metric is 30%-censored on its own
  runs** — 6 of 20 never reach 3.28 — so its "1740.5 ± 4.5 steps" is an n=14
  survivor statistic, between-arm biased, unsuitable as a primary endpoint.
- **The record does not reproduce within its seed variance off its native
  hardware/nightly**, and the dominant cause was isolated by a pre-registered
  diagnostic: bf16 sequential embedding-gradient accumulation at D<8 (vs the
  record's 8-way ReduceOp.AVG) accounts for ~half the gap (+0.0112 → +0.0062
  with fp32 accumulation). Residual is torch-nightly + PCIe-vs-SXM5 numerics.
  The deficit is cooldown-phase-concentrated (≈10–15× denser per unit of loss
  removed than in the stable phase).

## 4. What is publishable now

A measurement paper (`reports/paper-draft.md`) whose contributions are the
§3.1 measurements, the §3.2 placebo-controlled null, the §3.3 equivalent-
destinations mechanism, the §3.4 negative results, the §3.5 frontier
measurement, and the §3.6 methodology notes — none of which required the
routing method to work, and
all of which are absent from the 2024–2026 literature we swept. The paper's
differentiator from the "yet another Muon variant" pile is precisely that it
is measurement-first with a clean null, not a method claim.

## 5. Open questions the data raises

1. **What actually sets Muon's maximum stable LR?** Now sharpened by §3.5:
   the frontier is batch-coupled at ≈ √B (noise-side), but none of the four
   instrumented quantities (occupancy, either smoothness norm, HVP η·λ) is
   the equalized invariant along it — the frontier-setting quantity remains
   unidentified. Tightening α past the rung-quantization envelope needs a
   denser ladder with more seeds (see the follow-ups in
   `reports/stability-frontier.md`).
2. **Does the negative-ρ population's "equivalent destinations" property break
   at higher headroom / larger batch / longer horizon?** The consequence
   (nothing to gain) is the part most likely to be task-specific.
3. **Is oscillation occupancy a usable progress signal** (schedule timing,
   early-stopping) even though it is not a control setpoint? §3.4 established
   what it is *not*; what it *is* is untested.
4. **Per-direction critical batch size** (`docs/litreview/i,e`): does §3.1's
   structurally-unmeasurable per-direction signal become measurable as batch
   grows — connecting to Muon's unexplained large-batch advantage? Unpublished
   either way.

## 6. Next steps (ranked by expected payoff)

1. ~~Muon stability-frontier measurement~~ **DONE 2026-07-21 (§3.5,
   programs #6 + #6b; sharpening pass also done — α = 0.35 [0.30, 0.42],
   step-matched B=8000 confirms the shift continues).** What remains open
   from this line: the frontier-setting invariant (all four instrumented
   scalars ruled out) and why the exponent is ~B^1/3 rather than √B —
   both are analysis/theory questions before they are new-compute
   questions.
2. **Occupancy-triggered cooldown** (`criteria/occupancy_cooldown_preregistration.md`,
   pre-registered). Requires first: (a) the occupancy instrumentation ported to
   the nanogpt harness — unwritten, with a real torch.compile/FP8 graph-break
   risk; (b) our own harness seed-variance estimate (the record's cannot serve,
   §3.6); (c) accounting for the cooldown-phase confound (§3.6) since the
   intervention manipulates exactly that phase; (d) the pre-registered metric
   repaired away from the censored steps-to-target. The path to a "removed a
   hyperparameter" result, but gated on real setup work.
3. **Per-direction SNR vs batch size** (open question #4): test whether
   per-direction signal becomes measurable at large batch; the frozen-probe
   tier (program #4) is the instrument.
4. ~~Full nanogpt validation~~ **DONE 2026-07-21
   (`reports/nanogpt-local-baseline.md`).** The port runs locally on the
   2×5090 box (chunked-head memory path, PORT CHANGE P5; fp8+flex+compile
   intact); n=10 baseline: final val 3.28888 ± 0.00125 — seed noise equal
   to the record's native sd (0.0013), no variance inflation. Endpoint
   settled: final-val-at-1750 (steps-to-3.28 censored 10/10 here). Power:
   the plan's 3–5% effects need 2 seeds/arm; 0.0025-loss effects need 4.
   Controls (seeds 1710–1719) are in `results/`; babysitter supervision
   (`scripts/babysit_nanogpt.sh`) handles the box's flaky-card risk. The
   run.py seed-injection bug this surfaced is fixed (ecd48f1) with
   regression tests.
5. **Finish and submit the measurement paper** (§4) — the actual deliverable
   to the field; the flag-plant the plan's risk analysis calls for while the
   niche's ~6-month idea half-life runs.

### Added 2026-07-22 (program #8 + intermittency scan)

6. **Program #8 (TempoMuon temporal trust ratio) — active.** Eval-seed
   table done (`reports/tempo-eval.md`: controller exactly free at record
   LR, +1.54pp at 4×, n=100 paired; placebo decomposition in
   `reports/tempo-phase-b.md`). Open, in order:
   (a) **HUMAN: gate judgment** on the eval table (no `criteria/` file
   exists for it); decide whether this becomes a method section of the
   paper or a separate note.
   (b) **nanogpt transfer, Phase A passive** — probe runs in flight
   (PORT CHANGE P6; 4-rung muon_lr ladder, seeds 1440–1441); analysis
   `analyze_tempo.py nanogpt-passive`. If the early-training dial exists
   at LM scale, controller-on runs are the follow-up (power: 2–4
   seeds/arm per the local baseline sigma).
   (c) **Spike-gate for the controller** before any nanogpt controller
   run: don't advance the gain on spike steps (motivated by
   `reports/intermittency-scan.md` §3 — LM training has loss spikes at
   normal LR; a spike pushes cos toward 0 and would cause spurious
   shrink).
   (d) 2×-LR under-rescue and a self-calibrating setpoint (short
   known-safe probe → own reference band) — Phase B″ material, dev seeds
   1440+ (1440–1441 consumed by (b)).
7. **Spike-rate as a frontier-invariant candidate** (from
   `reports/intermittency-scan.md`): the burn-in-corrected heavy-tail
   population is LR-monotone — the one instrumented observable never
   tested against the pre-registered frontier tracking signature.
   Offline first (existing sidecars, B ≤ 1000); a proper batch-axis test
   needs longer high-B instrumented runs.
8. **Paper §3.1 addition candidate**: the anchoring-artifact methodology
   note (93% of naive top-direction kurtosis spikes are the subspace
   re-anchoring transient; burn-in ≥5 required) + the corrected
   LR-dependent intermittency measurement. Cheap to fold in; guards
   future instrumentation work against the same trap.

## 7. Reproducibility / repo state

`docs/litreview/` — nine literature reports with novelty verdicts. `criteria/`
— human-authored/audited pre-registrations (phase-1, occupancy-cooldown).
`results/` — append-only, full provenance (git SHA, seed, gpu_type, wall time,
cost). Every gate record carries its adversarial review. 621 tests green
(the suite runs ~5 min; nanogpt model construction dominates). Instrumentation
tiers: tracked top-k, bulk probes, frozen probes, HVP, spectral smoothness —
all synthetic-validated. Optimizer zoo (muon/adamw/dynmuon/adamuon/normuon/
routed) behind one interface with unit-tested update rules.
