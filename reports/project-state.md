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

Total cloud spend: **$13.60**. 621 tests. ~2,600 result JSONs with full
provenance. All committed.

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

### 3.5 Infrastructure findings (nanogpt, WP0.2)

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
destinations mechanism, the §3.4 negative results, and the §3.5
methodology notes — none of which required the routing method to work, and
all of which are absent from the 2024–2026 literature we swept. The paper's
differentiator from the "yet another Muon variant" pile is precisely that it
is measurement-first with a clean null, not a method claim.

## 5. Open questions the data raises

1. **What actually sets Muon's maximum stable LR?** §3.2 shows no divergence
   cliff to 6×; §3.4 rules out directional-smoothness plateaus; the ICML'26
   theory covers only full-batch momentum-free. The instrument to answer this
   (simultaneous Euclidean-ηλ HVP + spectral directional smoothness along the
   trajectory) is built and validated (`src/instrument/smoothness.py`,
   `src/instrument/hvp.py`).
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

1. **Muon stability-frontier measurement (LR × batch size, instrument
   attached).** Directly attacks open question #1 and the ICML'26 open problem;
   uses already-built, already-validated instrumentation; the most robust
   positive finding (§3.2) is the seed. Best science-per-effort. Deliverable:
   the momentum+minibatch stability law the field lacks, plus a mechanistic
   link to the large-batch advantage.
2. **Occupancy-triggered cooldown** (`criteria/occupancy_cooldown_preregistration.md`,
   pre-registered). Requires first: (a) the occupancy instrumentation ported to
   the nanogpt harness — unwritten, with a real torch.compile/FP8 graph-break
   risk; (b) our own harness seed-variance estimate (the record's cannot serve,
   §3.5); (c) accounting for the cooldown-phase confound (§3.5) since the
   intervention manipulates exactly that phase; (d) the pre-registered metric
   repaired away from the censored steps-to-target. The path to a "removed a
   hyperparameter" result, but gated on real setup work.
3. **Per-direction SNR vs batch size** (open question #4): test whether
   per-direction signal becomes measurable at large batch; the frozen-probe
   tier (program #4) is the instrument.
4. **Full nanogpt validation**: fp32-embed-fixed harness + a seed set to
   establish σ and confirm a self-consistent testbed. Prerequisite for any
   nanogpt method claim (the value is a self-consistent A/B testbed, not
   matching the record — record-faithfulness off native hardware is not
   attainable and is now a documented finding, not a goal).
5. **Finish and submit the measurement paper** (§4) — the actual deliverable
   to the field; the flag-plant the plan's risk analysis calls for while the
   niche's ~6-month idea half-life runs.

## 7. Reproducibility / repo state

`docs/litreview/` — nine literature reports with novelty verdicts. `criteria/`
— human-authored/audited pre-registrations (phase-1, occupancy-cooldown).
`results/` — append-only, full provenance (git SHA, seed, gpu_type, wall time,
cost). Every gate record carries its adversarial review. 621 tests green
(the suite runs ~5 min; nanogpt model construction dominates). Instrumentation
tiers: tracked top-k, bulk probes, frozen probes, HVP, spectral smoothness —
all synthetic-validated. Optimizer zoo (muon/adamw/dynmuon/adamuon/normuon/
routed) behind one interface with unit-tested update rules.
