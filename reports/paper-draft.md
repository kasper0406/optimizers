# Per-Direction Gradient Statistics Under Muon: Measurement, Stability, a Placebo-Controlled Null for Regime Routing, and a Serial-Correlation LR Controller

*Draft (agent-written, 2026-07-20; program #8 folded in 2026-07-22). Status:
internal draft for human review; all gate-relevant claims follow the final
decision records `reports/gate1-decision.md` and `reports/gate2-decision.md`
verbatim in scope and framing. Nothing in this draft re-adjudicates a gate;
the program-#8 results (§5) carry their own disclosure of protocol deviations
from the Gate-2 conditional approval and await human judgment.*

---

## 1. Abstract

Muon and its variants (DynMuon, AdaMuon, NorMuon, Muon^p) adapt the spectral
shape of matrix updates by magnitude statistics or by globally scheduled
exponents. Whether individual singular directions of the momentum matrix carry
*temporal* structure that could be classified and routed online has, to our
knowledge, not been measured. We instrument stock Muon on the CIFAR-10
airbench94 record configuration with a synthetically validated statistics
pipeline (per-direction projections of the raw gradient onto tracked singular
pairs of momentum; EMAs, lag-1 autocorrelation, t-statistics, implied
step-curvature) at 7.7% median step-time overhead and no measurable effect on
training, and report eight measurement findings and two intervention results —
one placebo-controlled null and one positive-but-bounded (findings 1–5 and 7
on airbench94; findings 6 and 8 span a second substrate as well — a
numerics-audited port of the modded-nanogpt speedrun record on FineWeb):

1. **A large, phase-structured negative-autocorrelation population.** 60–89%
   of per-direction snapshots have lag-1 autocorrelation ρ < −0.2 during the
   first three quarters of training, versus 0.22–0.29 for white noise passed
   through the identical pipeline; the excess is consistent with an AR(1)
   population at ρ ≈ −0.3…−0.5, falls to 0.37–0.46 in the anneal tail, and is
   unchanged under with-replacement sampling (max |diff| 0.05). We also
   disclose that our original pre-registered cluster criteria are satisfiable
   under the white-noise null and were therefore too weak as registered
   (§3.1).
2. **The population is LR-driven (late phases), momentum-independent, and
   broadband.** Removing momentum entirely *increases* the ρ < −0.2 fractions
   (+0.03 to +0.10); halving and quartering the LR reduces them monotonically
   in phases 2–4 (up to −0.24 at 0.25×); bulk probe directions out-oscillate
   top singular directions early in training (0.80–0.83 vs 0.39–0.56),
   inverting the sharp-direction edge-of-stability prediction.
3. **A negative calibration result:** the trajectory-implied η·λ read from
   per-direction amplitude ratios is statistically indistinguishable from its
   pure-noise null (~2.05) and does not track HVP-measured lr·λ
   (Pearson −0.06/−0.14), while training remains stable with per-direction
   lr·λ_HVP up to ≈ 65. GD eigendynamics demonstrably do not govern Muon's
   per-direction stability — the momentum+minibatch measurement that current
   non-Euclidean edge-of-stability theory explicitly lacks.
4. **No divergence regime found:** stock Muon and routed Muon both degrade
   gracefully to 6× the record learning rate (90.5% accuracy at 6×, zero
   divergences across the extended ladder), crossing a pre-declared 93.0%
   floor between 2× and 3×.
5. **Structural measurement limits:** the instrument's t-statistic ceiling
   (|t| ≤ SNR·√ess) makes a persistent-signal population undetectable at the
   observed SNR and EMA timescales, and the "oscillating" label is a
   (direction, window) property (Jaccard 0.36 between β = 0.9 and 0.99 sets).
6. **The useful-LR frontier is batch-coupled at lr\* ∝ B^0.35 on airbench —
   and batch-invariant on the LM record recipe.** Dense-ladder interpolated
   crossings (n = 5 seeds/cell) give α = 0.350, CI95 [0.297, 0.425] at fixed
   sample budget over B = 1000–4000 — decisively batch-coupled
   (noise-side), and decisively *not* the √B reference; a step-matched
   B = 8000 arm shows the shift continues rather than saturating, and at
   B ≥ 4000 *low* LR becomes the losing side. None of four instrumented
   scalars (oscillation occupancy, spectral or Euclidean directional
   smoothness — neither of which equilibrates at c/lr in its own right —
   or HVP η·λ) is conserved along the frontier. On the LM port, the same
   pre-registered design over an 8× token-batch range (98K–786K
   tokens/step, fixed 346M-token budget) refutes transfer: α = −0.29, CI95
   [−0.35, +0.30], the airbench exponent excluded, the loss valley pinned
   at ≈ 0.7× the record LR at every batch, no divergence cliff anywhere.
   The pair bounds the law's domain: batch-coupled at CNN-scale batches,
   batch-invariant at LM-scale token batches.
7. **A placebo-controlled null for per-direction regime routing.** A minimal
   routed optimizer (oscillation-channel damping on tracked singular pairs,
   activity-verified at 13.6% treated direction-steps) is seed-paired
   equivalent to stock Muon over 100 evaluation seeds: routed − muon =
   +0.011pp, 95% CI [−0.020, +0.042]pp — effects above 0.042pp excluded at
   97.5% confidence against a 0.272pp (2σ) success bar — with the random-gating
   placebo, constant-attenuation arms, and a retuned-weight-decay control all
   statistically indistinguishable, while the same pipeline resolves
   0.29–0.82pp deficits in lightly-tuned baseline arms. A +0.144pp dev-seed
   stress-test signal (p = 0.006, n = 10) failed its pre-registered n = 100
   confirmation with inverted sign — a small case study in why the
   confirmation step exists.

8. **A matrix-level serial-correlation signal that rescues mis-set learning
   rates — with a schedule-discovery mechanism and a bounded domain.** The
   near-free statistic ρ̂ = EMA[cos(G_t, G_{t−1})] per weight matrix is a
   ~20σ LR dial early in airbench training with an *inverted* sign (the
   healthy record run is deepest negative, ρ̂ ≈ −0.51; excess LR
   *decorrelates* consecutive gradients toward 0) and a late-anneal reversal.
   A bounded multiplicative gain regulated against this signal (globally
   pooled, active only before the reversal) is, over 100 seed-paired
   evaluation seeds, exactly free at the record LR (+0.001pp ± 0.016) and
   recovers most of the accuracy stock Muon loses at mis-set LR (+0.245pp at
   2×, +0.842pp at 3×, +1.536pp ± 0.024 at 4×). Three qualifications are as
   informative as the effect: an open-loop replay of the controller's mean
   gain trajectory reproduces the closed loop within noise at every LR (the
   mechanism is online *schedule discovery*, not per-step feedback —
   consistent with finding 7's equivalent-destinations reading); per-matrix
   gains *lose* to a single pooled gain (per-matrix baseline ρ̂ levels differ
   by more than the LR effect); and the signal does not transfer to the LM
   record recipe (serial anti-correlation is even stronger there, −0.55…−0.74,
   but LR-flat over a 4.3× range) — extending finding 6's pattern that
   airbench-derived LR laws bound, rather than predict, LM behavior.

The null is scoped: airbench-8-epoch record config, 200-step horizon, and the
tracked-subspace intervention class. The finding-8 controller is likewise
scoped: airbench substrate, task-calibrated setpoint, no external-baseline
comparison yet (§5.6).

### Contributions

- The first per-direction *temporal* characterization of gradient projections
  under a practical Muon configuration (finding 1–2), with an artifact-tested
  excess over an explicit white-noise null and a disclosed critique of our own
  pre-registered criteria.
- Per-direction stability measurements for stochastic, momentum Muon
  (findings 3–4): Euclidean lr·λ up to ≈ 65 during stable training, amplitude
  ratios at their noise floor, graceful degradation to 6× record LR — the
  empirical regime named as open by the non-Euclidean edge-of-stability line
  (§7.3).
- A validated, cheap measurement instrument: synthetic recovery guarantees for
  every statistic used (§2.2), non-perturbation evidence, 7.7% overhead, and
  an honest account of what the instrument structurally cannot see
  (finding 5).
- A two-substrate, pre-registered measurement of the Muon useful-LR frontier
  in batch (finding 6): lr\* ∝ B^0.35 [0.30, 0.42] on airbench with no
  conserved scalar found along the frontier, versus batch-invariance across
  98K–786K tokens/step on the LM record recipe — a domain boundary for
  batch-aware LR scaling rules, with the practical corollary that the record's
  Muon LR needs no retuning under grad-accumulation token-batch changes in
  that range.
- A pre-registered, seed-paired, placebo-controlled equivalence result for
  online per-direction regime routing at short horizon (finding 7), including
  activity telemetry proving the intervention was live, and full disclosure of
  every deviation from the pre-registered protocol.
- The temporal trust ratio executed (finding 8): the first LR gain in any
  optimizer family driven by measured *serial* structure of the update stream
  (the signal family that spatial-norm and noise-magnitude trust ratios
  provably cannot see), with a two-sided verdict — an n = 100 seed-paired
  LR-robustness result on airbench, a placebo decomposition attributing the
  entire effect to the discovered gain schedule, a granularity result
  (global pooling beats per-matrix), and a pre-registered transfer test that
  bounds the signal's domain at CNN scale. Plus a methodology hazard we hit
  and documented: subspace re-anchoring fabricates a heavy-tail (kurtosis
  ≈ +47) population in tracked-direction statistics unless a post-refresh
  burn-in is applied (§5.7).

---

## 2. Setup

### 2.1 Substrate

All experiments run on the CIFAR-10 **airbench94** record recipe (vendored
`KellerJordan/cifar10-airbench` at pinned SHA `4c1b6d1`): 8 epochs, 200
optimizer steps, batch 2000, TTA evaluation, with the six convolutional filter
matrices managed by Muon (lr 0.24, momentum 0.6, Nesterov, 3 Newton-Schulz
steps, weight decay 0). Our reproduction on a single NVIDIA RTX A6000, n = 100
evaluation seeds (0–99): **tta_val_acc mean 0.94003, std 0.00141, 95% CI
±0.00028** (`reports/baseline_airbench_aggregate.md`), consistent with the
published A100 reference (≈ 0.9401). Every comparison table in this paper uses
a single GPU type (RTX A6000).

Seed discipline throughout: seeds 0–99 are reserved for evaluation
comparisons; all development, tuning, and instrumentation-only runs use seeds
≥ 1000.

### 2.2 Statistics pipeline with synthetic recovery guarantees

Per tracked direction i we log the scalar projection s_i(t) = uᵢᵀG_t vᵢ of the
**raw pre-momentum gradient** onto singular pairs (uᵢ, vᵢ) of the momentum
matrix — top-k₁ = 16 by warm-started subspace iteration plus k₂ = 16 bulk
probes (random directions orthogonalized against the top block), refreshed
every T_refresh = 50 steps. From each stream we maintain bias-corrected EMAs
at β ∈ {0.9, 0.99}: mean μ, second moment (→ variance), lag-1 autocovariance
(→ autocorrelation ρ with small-sample bias correction), an
autocorrelation-adjusted t-statistic of the mean, and an implied η·λ from the
EMA of amplitude ratios |s_t/s_{t−1}| (for s(t) = A·(−r)^t, η·λ = 1 + r). A
regime classifier (SIGNAL / NOISE / OSCILLATING) sits on top with a
start-in-signal confidence prior, an effective-sample-size gate n_min, and
innovation-reset detectors.

Before any instrumented run, the pipeline had to pass a synthetic validation
suite (`reports/wp05-stats-validation.md`; 127 tests) with known ground truth:

- AR(1) with ρ ∈ {−0.8, −0.4, 0, 0.4, 0.8}: recovered ρ within ±0.05 in all
  40 (β, ρ, scale, seed) cells (worst 0.040); estimator exactly
  scale-invariant.
- Drifting mean at SNR ∈ {0.1, 1, 10}: measured t within [0.6, 1.4]× the
  analytic E[t] ≈ SNR·√ESS in all six cells; threshold crossing iff the
  analytic model predicts it.
- Pure oscillation A·(−r)^t, r ∈ {0.8, 1.0, 1.1}: implied η·λ exact to display
  precision (DoD tolerance ±0.1); decay flags correct in all six cases.
- Mid-stream regime switches: re-classification within n_min = 15 steps in all
  eight (β, segment) cells, including a variance-collapse switch invisible to
  the jump detector.
- Bias-corrected estimates verified at small t against analytic expectations
  over 50,000 Monte-Carlo streams.

All classifier thresholds are constructor parameters; the scientific values
used in Phase 1 are recorded in the run configs.

### 2.3 Non-perturbation and overhead

Instrumentation is read-only (a hooks-based `InstrumentationHub`; the
optimizer's update path is untouched, verified structurally by test and by a
no-op-optimizer smoke run with `max_param_delta == 0.0`). Empirically,
instrumented-but-stock Muon achieves tta_val_acc **0.94036 ± 0.00152** (n = 20
dev seeds) versus the stock baseline **0.94003 ± 0.00141** (n = 100 eval
seeds) — indistinguishable. Overhead at the Phase-1 settings: **7.7% median
step time** (31.19 ms vs 28.97 ms stock; `results/bench_overhead_airbench_v2.json`).

HVP measurements (one Hessian-vector product per tracked pair per refresh) are
a Phase-1 validation feature only and were enabled in three dedicated runs;
they are excluded from the routing update path by design (grep-enforced).

---

## 3. Measurement findings

Twenty instrumented seeds (1000–1019) on the record config; snapshots every 5
steps; phases = step quartiles (0,50], (50,100], (100,150], (150,200] under
airbench's linearly decaying LR (phase 1 = highest LR). Full tables:
`reports/wp12-phase1-measurement.md`, `reports/wp12-phase1-preregistered-stats.md`,
`reports/wp12-disambiguation.md`, `reports/wp22-mechanism-probes.md`.

### 3.1 The negative-ρ population — and the null-satisfiability of our own pre-registered criteria

The (SNR, ρ) scatter (fig. `figures/wp12/regime_scatter.png`) is a broad
connected cloud with its ρ mass centered clearly below zero and a long tail to
ρ ≈ −1. Both pre-registered criteria (`criteria/phase1_preregistration.md`,
committed at `b9aab63`, predating all runs) were met in form: GMM/BIC
preferred ≥ 2 components in 100% of the 24 (matrix, phase) cells (bar: ≥ 70%),
and the ρ < −0.2 population in the highest-LR phase was 59.8–68.2% of
snapshots (bar: non-empty).

But the adversarial review of the Gate-1 record demonstrated that these
criteria, as literally worded, do not discriminate. Quoting the decision
record (`reports/gate1-decision.md`) verbatim:

> **Excess over null, not the formal criteria.** The adversarial review
> demonstrated (via white-noise streams through the identical pipeline) that
> both pre-registered criteria as literally worded are satisfied under the
> null: BIC prefers k≥2 on white noise at these sample sizes, and the null's
> ρ<−0.2 fraction is 0.22–0.29. The gate therefore rests on the
> **non-pre-registered excess**: observed ρ<−0.2 fractions 0.60–0.89 in
> phases 1–3 (vs 0.22–0.29 null), consistent with an AR(1) population at
> ρ ≈ −0.3…−0.5, phase-structured (falls to 0.37–0.46 in the anneal tail),
> and unchanged under with-replacement sampling (max |diff| 0.05, most ≤0.03).
> This is recorded as a deviation-in-spirit from the pre-registration: the
> registered statistics were too weak, and the decision uses stronger,
> post-hoc-but-adversarially-audited statistics. Both are reported.

The finding that survives is therefore the **excess over an explicit null**,
not the registered statistics: phase-wise ρ < −0.2 fractions (β = 0.99) of
0.598 / 0.886 / 0.768 / 0.458 across the four phases (β = 0.9:
0.682 / 0.804 / 0.676 / 0.382), against a white-noise pipeline null of
0.22–0.29. The phase structure — peak in phase 2, collapse in the anneal
tail — is the direction of DynMuon's "positive p early, negative p late"
narrative, observed here per-direction (fig.
`figures/wp12/regime_occupancy.png`).

**Sampling robustness.** Airbench samples batches without replacement within
an epoch, which could induce negative lag-1 correlation in the noise component
independent of any dynamics. Three with-replacement runs through the identical
pipeline leave every phase-wise fraction essentially unchanged (max |diff|
0.05, most ≤ 0.03; bootstrap CIs in `reports/wp12-disambiguation.md`).

### 3.2 Mechanism probes: LR-driven, momentum-independent, broadband

Gate-1 amendment A4 mandated two cheap causal probes (dev seeds; bootstrap
over runs, B = 2000; `reports/wp22-mechanism-probes.md`):

- **Momentum = 0** (2 runs): the ρ < −0.2 fractions do **not** drop — they
  rise slightly but significantly in every phase (Δ +0.026 to +0.101,
  β = 0.99). The population is not a momentum-overshoot artifact; if
  anything, momentum smooths it.
- **LR ladder ×0.5, ×0.25** (2 runs each): fractions fall monotonically with
  LR in phases 2–4 (β = 0.99, ×0.25: Δ −0.127 / −0.243 / −0.237 in phases
  2/3/4), i.e., the negative-ρ population is LR-driven exactly where LR is the
  operative knob. Phase 1 is the exception: the all-directions fraction is
  ≈ flat at reduced LR, and the *top*-direction fraction actually increases at
  ×0.25 (+0.087, β = 0.9), so the earliest-training population is not a simple
  LR effect.
- **Bulk vs top anomaly:** in phase 1, bulk probes out-oscillate top singular
  directions (0.80–0.83 vs 0.39–0.56 across β); the ordering inverts from
  phase 2 onward (top 0.84–0.98). A sharp-direction-only edge-of-stability
  story predicts the opposite sign in phase 1; the breadth is consistent with
  a broadband mechanism. This anomaly is reported as open.

### 3.3 Implied η·λ vs HVP: a negative calibration result

The Phase-1 plan's "curvature for free" conjecture — that amplitude ratios of
oscillating directions reveal η·λ — fails under Muon. On
oscillating-classified snapshots matched to HVP records (3 HVP-enabled runs,
seeds 1200–1202; 622/429 pairs at β = 0.9/0.99):

| β | n pairs | Pearson r | Spearman ρ | median rel. err | median implied η·λ | median lr·λ_HVP |
|---|---|---|---|---|---|---|
| 0.9 | 622 | −0.055 | −0.093 | 0.985 | 2.850 | 1.451 |
| 0.99 | 429 | −0.144 | −0.256 | 0.884 | 2.876 | 1.772 |

The implied values saturate at ~2–4 — statistically indistinguishable from the
pure-noise null of the amplitude-ratio statistic (~2.05; Gate-1 record) — while
HVP-measured lr·λ across all 2,304 HVP records spans from negative curvature
(min −23.4) up to **65.6** (median 0.15, q90 4.4), with 21.9% of records above
the Euclidean GD stability threshold of 2 — during entirely stable training.
Two consequences: (i) GD eigendynamics (the basis of the amplitude model) do
not govern Muon's per-direction stability — Euclidean lr·λ is not the
operative multiplier; (ii) the adaptive oscillation gain g_osc =
clip(1/(η·λ_implied − 1), 0.1, 1) as coded in the v0 router is a ≈ 0.53
near-constant attenuator, not adaptive damping, and its decay-escape gate
fired on 0 of 12,851 snapshots (dead code as configured). Both facts were
recorded at Gate 1 *before* the comparison runs and motivated the
constant-gain control arms of §4. (Calibration plot:
`figures/wp12_hvp/eta_lambda_calibration.png`.)

### 3.4 Stability observations: lr·λ ≈ 65, graceful degradation, no divergence regime

Three observations jointly characterize practical (stochastic, momentum,
Nesterov) Muon stability on this substrate:

1. Per-direction Euclidean lr·λ up to ≈ 65 coexists with stable training
   (§3.3).
2. The oscillation signature is momentum-independent and LR-monotone in late
   phases (§3.2).
3. An extended LR ladder (dev n = 10 per point, both stock and routed Muon)
   found **zero divergences at any multiplier up to 6× the record LR**, with
   smooth accuracy degradation:

| LR × record (lr) | stock Muon mean | routed mean | divergences |
|---|---|---|---|
| 1× (0.24) | 0.9402 | 0.9399 | 0 |
| 1.5× (0.36) | 0.9380 | 0.9380 | 0 |
| 2× (0.48) | 0.9341 | 0.9349 | 0 |
| 3× (0.72) | 0.9268 | 0.9255 | 0 |
| 4× (0.96) | 0.9181 | 0.9175 | 0 |
| 6× (1.44) | 0.9047 | 0.9049 | 0 |

Both optimizers cross the pre-declared 93.0% stability floor between 2× and
3×; neither ever diverges (all runs finite). "Max stable LR" under the
pre-registered operationalization is 0.48 (2×) for both — ratio 1.0.

**Positioning.** Islamov, Crawshaw, Cohen & Gower ("Non-Euclidean Gradient
Descent Operates at the Edge of Stability", arXiv:2603.05002, ICML 2026 oral)
prove that for steepest descent under arbitrary norms — including
spectral/Muon-style updates — stability is governed by directional smoothness
in the update's own geometry, with the Euclidean ℓ₂ sharpness *decoupled from
stability entirely*, and they report a pre-EoS broadband oscillatory regime
unique to ℓ∞/spectral geometries. Their analysis is full-batch and
momentum-free, and they name the momentum/stochastic extension as open. Our
data are precisely the momentum+minibatch measurements that theory lacks: the
lr·λ ≈ 65 decoupling is a quantitative instance of their qualitative claim;
the broadband negative lag-1 autocorrelation (period-2 bouncing appears as
negative lag-1 autocorrelation in per-direction projections) matches their
pre-EoS oscillation; momentum-independence and the absence of any divergence
regime to 6× are, to our knowledge, unreported. We subsequently added the
matching trajectory measurement of generalized (spectral) directional
smoothness and found that the Muon analog of GD's 2/η and Adam's ≈ 38/η
constant (Cohen et al., arXiv:2207.14484) **does not exist as an
LR-invariant plateau on this substrate**: across a 4× LR ladder the
dimensionless product lr·D_smooth varies 2.57× in the spectral norm and
2.26× in the Euclidean norm — the spectral quantity is no more LR-invariant
than the Euclidean one (§3.6).

### 3.5 Timescale sensitivity and the t-ceiling

- **Timescale:** phase-wise ρ < −0.2 fractions at β = 0.9 vs 0.99 differ by up
  to ~9 points while agreeing qualitatively; more sharply, the sets of
  directions labeled oscillating at the two β overlap at **Jaccard 0.36**
  (Gate-1 record): "oscillating" is a property of a (direction, window) pair,
  not of a direction.
- **No detectable signal population:** at mature classification
  (n_since_reset ≥ 50), frac(|t| ≥ 4) = 0.0000. This is structural, not
  merely empirical: the t-statistic obeys |t| ≤ SNR·√ess, and at the observed
  SNR (q90 ≈ 0.26) and ESS asymptote (199 at β = 0.99) the ceiling sits below
  the τ_sig = 4 threshold. The intuitive occupancy figure from the labeled
  plot (~68% signal / 27% noise / 5% oscillating) is an artifact of the
  start-in-signal confidence window; the truly classified mature population is
  ~85% noise / ~10–16% oscillating / ~0.3% signal. **Signal/noise separation
  was not observed**, which removed any empirical basis for noise-channel
  routing (maximal misclassification-asymmetry exposure) and scoped the
  Phase-2 intervention to the oscillation channel only.

---

### 3.6 The useful-LR frontier in batch: B^0.35 on airbench, batch-invariant at LM token batches

Three pre-registered programs (each with predictions committed before any
run; `reports/stability-frontier-preregistration.md`,
`frontier-sharpening-preregistration.md`,
`frontier-nanogpt-preregistration.md`) map how the useful-LR band moves
with batch size, and what is — and is not — conserved along it.

**Airbench: batch-coupled, sub-√B.** At fixed sample budget (8 epochs), the
largest LR keeping mean accuracy within 1.0pp of its per-batch reference
shifts right with batch. A coarse 5-batch × 8-rung grid (n = 2) lands the
OLS exponent at exactly 0.50 — but with a ±0.5 rung-quantization envelope,
which the sharpening pass was pre-registered to adjudicate: dense
×1.15-spaced ladders (n = 5/cell) with a log-linear interpolated floor
crossing give **α = 0.350, seed-bootstrap CI95 [0.297, 0.425]** over
B = 1000–4000. Both pre-registered nulls die: batch-independence
(|α| < 0.1) and the √B point prediction are excluded. Two further
structural observations: (i) at B ≥ 4000 the accuracy curve becomes
non-monotone in LR — *low* LR becomes the losing side (at step-matched
B = 8000, halving the record LR costs 3.1pp while tripling it costs
0.4pp), a signature that survives step-matching; (ii) a step-matched
B = 8000 arm (matching B = 2000's step count) recovers full accuracy and
extends the shift (peak-referenced shoulder 0.96 ≥ B = 4000's 0.72) — the
fixed-budget B = 8000 trend-break was undertraining, not saturation.

**No conserved scalar along the frontier.** At each batch's shoulder we
evaluated four instrumented candidates under a pre-registered
"frontier-tracking" signature (equalized across batch at the shoulder,
varying at fixed LR): oscillation occupancy, spectral and Euclidean
trajectory directional smoothness, and HVP η·λ (q90). **All four fail.**
Curvature is the *least* equalized (5–6× across batch at matched rungs),
extending §3.3–3.4's decoupling to the batch axis; and the directional
smoothness measurement doubles as the §3.4 plateau test — no c/η constant
in either norm. Whatever quantity sets Muon's stochastic frontier, it is
none of these as a scalar.

**The LM record recipe: no transfer.** On a numerics-audited single-GPU
port of the modded-nanogpt speedrun record (2025-07-12_BosAlign; §9.1
notes the port audit and our own n = 10 baseline σ = 0.00125, equal to
the record's native 0.0013), the same design — 4 token batches
(98,304–786,432 tokens/step via record-chunk count) × 6 √2-spaced Muon-LR
rungs × 2 seeds at a fixed 346M-token budget, loss-valley reference,
interpolated floor crossing at valley + 0.010 — gives **α = −0.29, CI95
[−0.35, +0.30]**: the airbench exponent is excluded, batch-independence is
not, and the loss valley pins at ≈ 0.7× the record LR at every batch. No
divergence cliff appears at any batch (worst arm mean 3.574 vs valley
3.510). The pre-registered valley-shifts-right prediction is refuted.

**Reading.** The two measurements bound the domain of any batch-aware Muon
LR rule: batch-coupled (≈ B^1/3, not √B) at CNN-scale batches of 10²–10³
samples; batch-invariant at LM token batches of 10⁵–10⁶. A single
noise-governed mechanism whose coupling saturates past a critical batch
size is consistent with both — the LM grid would then sit entirely above
the crossover that the airbench range straddles, and the LM's smallest
batch arm indeed shows 15–20× inflated seed noise and a 30× low-LR
penalty — but that unification is untested here; it predicts re-emergent
coupling at much smaller LM token batches. Practically, on this testbed
the record's Muon LR requires no rescaling when the token batch is changed
by grad-accumulation count anywhere in the measured range.

## 4. The routing experiment

### 4.1 Method summary (routed.py v0)

Two-tier, COSMOS-shaped (`src/optim/routed.py`). Bulk tier: stock Muon
(momentum → Newton-Schulz → O_t), bit-identical when routing is disabled.
Tracked tier: k = 16 singular pairs per matrix (warm-started subspace
iteration, T_refresh = 50); each raw-gradient projection stream feeds the
WP0.5-validated classifier; after Newton-Schulz the update receives a rank-≤k
correction O_t ← O_t + Σᵢ (g(i) − 1)(uᵢᵀO_t vᵢ)uᵢvᵢᵀ. Gains: oscillating and
amplitude-non-decaying directions get g = clip(1/(η·λ_implied − 1), 0.1, 1)
(adaptive mode) or a fixed g_osc_const (Gate-1 amendment A2); everything else
g = 1 in the oscillation-only primary arm. Distributed invariants are
structural: per-matrix owner-rank statistics, no full-gradient gathers, no HVP
anywhere in the update path (CI-enforced).

Per the Gate-1 scoping, the pre-registered Phase-2 claim attaches to the
oscillation channel only; full three-channel routing and noise-only arms ran
as exploratory with no pre-registered claim.

**Routing-activity fix (disclosed).** The first routed run's telemetry
confirmed the review's dead-zone concern: at the Phase-1 defaults
(n_min = 50, align_min = 0.9) every direction was reset at every refresh
(288/288) and 98% of direction-steps sat inside the confidence window —
routing effectively OFF (cumulative treated fraction ~0.7%). All routed arms
were switched to n_min = 25, align_min = 0.3, documented in the Gate-1
addendum *before* any evaluation-seed comparison run; the stage-A grid that
ran with the old defaults is retained as an incidental routing-inactive
placebo grid, and aggregations filter it out by git SHA. On the evaluation
head-to-head arm, cumulative treated fraction is **13.6%** of direction-steps
(mean over 100 seeds), so the evaluated intervention was demonstrably live.

### 4.2 Results (n = 100 evaluation seeds per arm, seed-paired, single GPU type)

Full table (`reports/wp22-comparison-table.md`; paired statistics recomputed
from per-seed results in `results/`):

| arm (config) | mean tta_val_acc | std | paired Δ vs stock (pp) | paired t | wall (s) |
|---|---|---|---|---|---|
| Muon, record config | 0.94014 | 0.00136 | — | — | 10.29 |
| **Routed (osc-only, primary)** | 0.94025 | 0.00148 | **+0.011** | 0.69 | 11.31 |
| Routed, random-gating placebo (4d) | 0.94006 | 0.00132 | −0.008 | −0.51 | 11.13 |
| Routed, g_osc = 0.25 const | 0.94016 | 0.00140 | +0.002 | 0.08 | 11.27 |
| Routed, g_osc = 0.50 const | 0.94021 | 0.00133 | +0.007 | 0.43 | 11.12 |
| Routed, g_osc = 0.75 const | 0.94015 | 0.00145 | +0.001 | 0.08 | 11.10 |
| Routed, full 3-channel (exploratory) | 0.94036 | 0.00139 | +0.022 | 1.36 | 11.17 |
| Muon + retuned WD (4a) | 0.94013 | 0.00123 | −0.001 | −0.04 | 10.27 |
| Muon, fair-tuned (tuneB) | 0.94030 | 0.00131 | — | — | 10.32 |
| Routed, fair-tuned (tuneB) | 0.94010 | 0.00138 | −0.020 vs tuned Muon | −1.25 | 11.10 |
| DynMuon (light tuning) | 0.93297 | 0.00160 | −0.717 | −35.1 | 10.51 |
| AdaMuon (light tuning) | 0.93199 | 0.00147 | −0.815 | −41.4 | 10.32 |
| NorMuon (light tuning) | 0.93729 | 0.00129 | −0.285 | −16.5 | 10.82 |

**Primary criterion (pre-registered): FAIL — final.** The registered claim was
a ≥ 2σ improvement in time/steps-to-94% at fair tuning, not matched by
ablations. **Metric substitution disclosed:** the harness runs a fixed
200-step budget and non-TTA validation accuracy never crosses 94%, so the
evaluated surrogate is accuracy-at-fixed-budget (tta_val_acc); on the time
axis routed is strictly worse (+10.0% wall time at equal accuracy), so the
substitution is conservative in routed's favor. Result: routed − muon =
+0.011pp, paired t = 0.69, p = 0.49; 95% CI [−0.020, +0.042]pp — **effects
above 0.042pp are excluded at 97.5% confidence versus the 0.272pp (2σ) bar**.
0 of 45 within-tier pairwise contrasts reach nominal p < 0.05 (min p = 0.109,
below the expected family-max under the global null). The fair-tuned contrast
is sign-negative (−0.020pp). This is an *equivalence* result, not an absence
of power: the identical pipeline resolves the baseline arms at paired
|t| ≥ 16.

**Placebo and constant-gain arms.** The random-gating placebo (same gating
machinery, gains assigned at random) and all three constant-attenuation arms
are indistinguishable from stock and from adaptive routing — consistent with
§3.3's finding that the "adaptive" gain was in fact near-constant. The
adaptive-vs-constant question posed by Gate-1 amendment A2 is answered:
neither does anything at this horizon.

**Ablation coverage (disclosed).** 4a (retuned weight decay) and 4d (placebo)
ran and are within the tier. 4b (LR bump + gradient clip) **never ran**: the
config required a small harness hook that was not implemented
(`docs/wp22-run-plan.md`, deviation 6) — moot under the null but recorded as a
deviation. The ρ-ignored gate (4c) is registered as a config
(`configs/wp22_null_routed_rhoignored.yaml`); under the Gate-1 oscillation-only
scoping it is meaningful only with the noise channel enabled and carries no
pre-registered claim; no evaluation-seed runs of it appear in `results/`. The
noise-only exploratory arm was likewise not executed. The λ-tracking plot
(plan §2.2.6) was never wired (deviation 7).

**Secondary criterion (stability margin ≥ 1.3): FAIL — final.** §3.4's ladder
gives max-stable-LR ratio 1.0. The n = 100 seed-paired confirmation at 2× LR
gives routed − muon = **−0.024pp** (sd 0.176pp, t = −1.34). Gate 2 is
therefore FAIL overall, fully adjudicated on both registered criteria.

### 4.3 A case study in pre-registered confirmation: the 2×-LR false positive

The adversarial review of the provisional Gate-2 record surfaced an omitted
signal: in the dev-seed stress grid, at 2× record LR — the only point tested
deep in degradation — active-routed led stock by **+0.144pp paired (t = 3.52,
p = 0.006, n = 10 dev seeds)**, exactly the plan-§2.2.5 prediction. Rather
than report it, the record pre-registered a completion protocol *before
looking at any new data*: an extended ladder {3, 4, 6}× and an n = 100
seed-paired confirmation at 2×, with the stability operationalization
pre-declared. The confirmation came back **−0.024pp (t = −1.34): the sign
inverted.** The dev signal is recorded as a multiple-comparisons false
positive caught by pre-registered confirmation. We highlight this because it
is the modal way a spurious "stability margin" claim would have entered the
literature: a nominally significant dev-scale effect at the single
hypothesis-confirming grid point, published without a confirmation set.

### 4.4 The baselines observation (tuning-effort-qualified)

The comparison arms for DynMuon, AdaMuon, and NorMuon are informative only
under their exact tuning protocol, and we state the observation exactly as
scoped in the Gate-2 record: at documented light tuning (LR-only 5-seed dev
probes, grids published) on the muon-co-adapted record config, none of
DynMuon/AdaMuon/NorMuon matched stock Muon (gaps 0.29–0.82pp) — a
tuning-effort-qualified observation, never an optimizer ranking; it says
nothing about home-scale/home-metric claims of those methods. The registered
"≥ the DynMuon gap on the same harness" success clause is separately
**degenerate**: WP0.3 (DynMuon reproduction at its home scale) **was never
executed**, so no reference gap exists, and the measured on-harness gap is
negative, making the clause trivially satisfied; the primary FAIL rests solely
on the 2σ clause. WP0.3 non-execution is recorded as a deviation.

---

## 5. The temporal trust ratio (program #8): serial-correlation LR control

The §8 future-directions entry of the 2026-07-20 draft is now executed
(2026-07-22, one day of local compute). Evidence trail:
`reports/tempo-phase-a.md` (signal measurement + Phase-B pre-registration,
commit `082a09d`, predating every Phase-B run), `tempo-phase-b.md` (dev
results, prediction scorecard, placebo), `tempo-eval.md` (n = 100
evaluation table), `tempo-nanogpt-phase-a.md` (transfer test),
`intermittency-scan.md` (the §5.7 methodology note). Optimizer:
`src/optim/tempomuon.py`, bit-identical to stock Muon at κ = 0
(unit-tested).

### 5.1 Position

A same-day three-sweep literature re-check confirmed the slot the §8
proposal targeted remains open: no published method in any optimizer family
modulates an LR gain by measured temporal/serial statistics of the
gradient/update stream (hard arXiv abstract queries for sign-flip × LR and
oscillation × layer-wise LR return zero papers). The wall is closing:
Greedy Alignment (arXiv:2512.06370; global scalar, adapts *momentum*, not
LR), MGUP-Muon (arXiv:2606.17526; instantaneous per-parameter sign
agreement), and CLARA (arXiv:2508.05408; global path-length LR, an implicit
serial statistic) each occupy one adjacent cell.

### 5.2 The signal, and why the naive controller design is wrong

Per matrix: ρ̂ = bias-corrected EMA (β = 0.9) of cos(G_t, G_{t−1}) on raw
pre-momentum gradients — one dot product and one prev-grad buffer per
matrix per step, computed in fp32 (in fp16 the elementwise products
overflow; the resulting `exp(0·nan)` poisoned early passive runs at *low*
LR, where large early gradients persist longest — disclosed as the
program's first incident, regression-tested). Passive measurement
(κ = 0 ≡ stock Muon) on dev seeds across LR ∈ {1, 2, 3, 4}× record:

- The **window-averaged level is useless** — non-monotone in LR (−0.365,
  −0.401, −0.386, −0.358), 4× indistinguishable from 1×.
- The **fixed-step level is a clean dial with an inverted sign**: at step
  20, ρ̂ = −0.512 / −0.455 / −0.361 / −0.305 across 1–4× (seed sd ≈ 0.01,
  ~20σ separation) — the healthy run is *deepest* negative; excess LR
  destroys serial structure. The ordering **fully reverses late in the
  anneal** (step 160: −0.179 / −0.297 / −0.373 / −0.398), reproducing the
  §3.1 phase structure at matrix level with a near-free statistic.
- Per-matrix baseline levels differ by more than the LR effect (−0.30 to
  −0.43 at 1×) — foreshadowing §5.4's granularity result.

### 5.3 Controller and dev-phase results (pre-registered predictions P1–P5)

gain ← clip(gain · exp(κ(ρ̂ − ρ\*)), [0.2, 1]) with κ = −0.25 (negative:
shrink when ρ̂ is *above* the setpoint), ρ\* = −0.48 (the healthy early
band), warm-up 25 steps, active window ending before the reversal, gain
frozen thereafter; applied as W −= lr·gain·O. Dev Phase B (n = 10
seed-paired, arms stock / per-matrix / globally-pooled × 4 LRs) with the
window at step 100: rescue confirmed (**P2**: +1.11pp at 3×, +1.92pp at 4×,
global arm) but **P1 failed** — at 1× the controller cost −0.26pp because
the healthy run's ρ̂ relaxes past the fixed setpoint after ~step 60 and the
[25, 100] window guaranteed late engagement (telemetry: mean 1× gain fell
to 0.36 by step 100 — the same diagnosis explains the **P3** failure).
Disclosed, not re-tuned in place: the one-variable fix (window → step 60)
ran as a labeled exploration on fresh dev seeds.

### 5.4 Placebo decomposition and granularity

Two results qualify the mechanism before the headline number:

- **Open-loop replay ≡ closed loop.** Replaying the global arm's mean gain
  trajectory as a fixed schedule (feedback off) matches the closed-loop
  controller within noise at every LR (e.g., +1.807 vs +1.918pp at 4×;
  −0.185 vs −0.258pp at 1×). The controller's value is *discovering* the
  LR-appropriate schedule online from the serial signal — within-run
  feedback beyond the mean trajectory contributes nothing, consistent with
  §4's equivalent-destinations reading. We state this plainly: this is an
  adaptive method whose entire effect is schedule discovery.
- **Global pooling beats per-matrix at every LR** (e.g., +1.92 vs +1.49pp
  at 4×; −0.26 vs −0.58pp at 1×): heterogeneous per-matrix ρ̂ baselines
  against a shared setpoint mis-treat matrices; pooling matches the
  calibration. The novelty cell as originally framed ("per-matrix ×
  temporal") is thus the part our own data argues against on this testbed —
  continuing the project's monotone pattern: per-direction null (§4),
  per-matrix worse, global works.

### 5.5 Evaluation-seed result (n = 100, seed-paired, frozen config)

Configuration frozen on dev seeds (window-60 variant), committed before
launch (`30a32ee`); seeds 0–99; single GPU type (RTX 5090); endpoint
tta_val_acc:

| lr | stock Muon | TempoMuon (global) | Δ paired (mean ± SE) |
|---|---|---|---|
| 0.24 (1×) | 0.9399 ± 0.0013 | 0.9399 ± 0.0013 | **+0.001pp ± 0.016** |
| 0.48 (2×) | 0.9346 ± 0.0012 | 0.9370 ± 0.0014 | +0.245pp ± 0.017 |
| 0.72 (3×) | 0.9262 ± 0.0016 | 0.9346 ± 0.0016 | +0.842pp ± 0.022 |
| 0.96 (4×) | 0.9177 ± 0.0020 | 0.9330 ± 0.0017 | **+1.536pp ± 0.024** |

Exactly free at the record LR (95% CI ≈ [−0.03, +0.03]pp); recovers ~69% of
stock's 4× deficit. The dev-phase failure mode did not recur at n = 100.

### 5.6 Transfer to the LM record recipe: negative, with one live thread

A passive probe inside the record's Muon step (measurement-only; update
path untouched, unit-tested) on the §3.6 LM port — 4 Muon-LR rungs
{0.7, 1, 2, 3}× record × 2 seeds, 600-step stable-phase truncation:
serial anti-correlation is *stronger* at LM scale (cos_gg −0.55…−0.74)
but **LR-flat** (≤ 0.06 spread over 4.3×; per-matrix orderings split both
ways in every window). The airbench-calibrated dial does not transfer —
the same shape of result as §3.6's frontier non-transfer, and together
they make the pattern explicit: airbench LR laws have repeatedly bounded,
never predicted, this LM recipe. One thread stays live: the zero-memory
variant cos(G_t, momentum buffer) shows an airbench-sign dial in steps
25–100 only (median per-matrix Spearman(lr, ·) = +0.80, 52% of matrices
> +0.5 vs 9% < −0.5; n = 2 seeds — suggestive, unconfirmed).

**Protocol deviations from the Gate-2 conditional approval (disclosed).**
The §8 direction was approved conditional on (i) a pre-registration
committed to `criteria/` (human-audited) and (ii) mandatory baselines
(OrScale, NAMO/LANTON-style noise scaling, GALA, Prodigy, hand-tuned
schedule). As executed: the pre-registration lived in `reports/`
(agent-committed; commit order verifiably predates the runs, but without
the human audit step), and **no external-baseline arm has run** — the only
comparators are stock Muon and the internal placebo/granularity arms. Both
gaps must close before any comparative claim against those methods; the
natural hand-tuned-schedule baseline (an earlier LR anneal tuned per
mis-set LR) is exactly what the placebo replay approximates from the
discovered side, but a properly tuned version has not been run. The
setpoint ρ\* is calibrated on this task's healthy runs; setpoint transfer
is untested.

### 5.7 A methodology hazard: subspace re-anchoring fabricates heavy tails

An offline scan of the §3.6 programs' stored per-direction series (25,920
directions; zero new compute) for excess kurtosis / spike-rate — motivated
by the observation that findings 5's t-ceiling and the frozen-probe null
are both *mean*-based, so an intermittent sign-varying signal could hide
from them — produced a spectacular false positive: median excess kurtosis
+47 in top-tracked directions, 56% of all directions above an
AR(1)-Gaussian pipeline null's 99th percentile, split-half spike stability
+0.80. The offset diagnostic attributes it: **93.4% of top-direction
spikes sit in the first 5 steps after subspace re-anchoring** (the tracked
direction is aligned to the momentum top at refresh; the projection starts
large and decays). Any 4th-moment statistic on subspace-tracked streams
requires a post-refresh burn-in. After burn-in, the surviving population
is modest (~10% of directions above the null q99 vs 1% expected,
split-half +0.29, robust to doubling the burn-in) and **LR-monotone**
(4.3% above null at record LR → 18.9% above the useful-LR shoulder) —
instability-flavored, not hidden-signal-flavored, and near-null exactly
where hidden signal would have mattered. The LR-monotone spike-rate is
the one instrumented observable not yet tested against §3.6's
frontier-tracking signature.

---

## 6. Scoping and limitations

The null result is precisely scoped; we claim nothing beyond it.

1. **Horizon.** 200 optimizer steps / 8 epochs. Short-horizon evaluation is
   documented to be unreliable in both directions: optimizer rankings flip
   with training budget (Wen et al., arXiv:2509.02046), and efficiency gains
   measured at short compute budgets often vanish at longer ones (Kaddour et
   al., arXiv:2307.06440). The short-horizon-bias literature cuts *against*
   rescuing this null at longer horizons: greedy short-horizon objectives
   systematically over-reward damping interventions (Wu et al.,
   arXiv:1803.02021), so a damping method that cannot even win at 200 steps
   receives, if anything, mild additional discouragement. On the other side,
   "Stability of Singular Distribution" (arXiv:2605.26489) argues spectral
   constraints bind only in the prolonged slow phase, which a 200-step run
   never reaches — a principled reason the intervention *class* remains
   untested at scale. Both readings are consistent with our scoping; neither
   licenses a long-horizon claim.
2. **Config co-adaptation.** The record config is tuned for stock Muon; our
   fair-tuning sweep was LR × WD only, two-stage (dev-seed selection n = 25 →
   eval n = 100 at the argmax), and routing hyperparameters were frozen at
   defaults rather than swept (deviations 1 and 3).
3. **Intervention class.** The null covers the tracked-subspace intervention
   class: rank-≤16 multiplicative corrections on singular pairs of the
   momentum matrix, gains in [0.1, 1], with this classifier and these
   timescales. It says nothing about other response functions or other substrates —
   and the per-matrix/global LR actuator driven by the same statistic
   family has now been tested, with a positive-but-bounded result (§5)
   that leaves this null intact. Song et al. (arXiv:2405.16002) found
   projecting out the dominant subspace entirely leaves training unharmed at
   short horizons, and Damian et al.'s self-stabilization result gives a
   principled reason externally damping oscillating directions may be
   redundant — our null is consistent with both.
4. **Measurement limits.** The t-ceiling (§3.5) means "no signal population"
   is a statement about this instrument at these β, not about the absence of
   persistent signal; the Jaccard 0.36 timescale sensitivity means any
   deployment of the oscillation label inherits a window choice; the HVP
   calibration ran on 3 seeds; the mechanism probes on 2 runs per condition
   (bootstrap CIs reported).
5. **Substrate breadth.** One dataset, one architecture, one GPU type, one
   record recipe. The nanogpt leg of the measurement plan (WP1.2) was not
   executed before the program's Phase-2 track closed.

---

## 7. Related work

### 7.1 DynMuon and scheduled/global spectral shaping

DynMuon (arXiv:2605.17109) replaces Muon's UVᵀ with UΣᵖVᵀ where p follows "a
simple decreasing logistic schedule … interpolating from positive values early
in training to mildly negative values later"; the schedule's only inputs are
(step t, total steps T), and in the released code p is a process-global scalar
set once per step for all matrices (verbatim quotes and code locations:
`reports/dynmuon-quotes.md`). Their theory is mode-wise — "the spectral
exponent p controls a mode-wise signal–noise tradeoff" — but the *method*
applies one exponent to every direction of every matrix. Their conclusion
names our direction explicitly as future work: "selecting p *online* based on
observed optimization statistics." Our measurement section is the empirical
input such a method would need, and our null is direct evidence that at short
horizon the per-direction-gain version of that program (in the damping
response class we tested) has no effect, while their global schedule's home
claims (10.6–26.5% fewer steps at 127M–1.1B/10–20B tokens) live at a scale and
metric we did not test (§4.4).

### 7.2 Per-direction and subspace-selective optimizers

COSMOS (arXiv:2502.17410) splits by eigenvalue rank and applies different
optimizers per subspace — subspace-selective treatment without behavioral
statistics; our two-tier architecture is COSMOS-shaped by design. SOAP/Shampoo
(Vyas et al. 2024) is per-eigendirection gain modulation driven by
second-moment magnitude — the closest in spirit at LLM scale, but variance
normalization, not temporal classification. Dion (arXiv:2504.05295)
rank-truncates with error feedback, showing bulk directions can be deferred at
scale. Muon^p (arXiv:2606.13867) applies fixed monotone per-σ gains. SpecMuon
(arXiv:2602.16167) modulates per-mode step sizes online, in SciML only.
Aurora (arXiv:2606.27715), NorMuon (arXiv:2510.05491), and AdaMuon
(arXiv:2507.11005) reweight per-row or per-coordinate, never in the singular
basis; the cautious-optimizer family (arXiv:2411.16085) gates per-coordinate
on a sign-agreement statistic — crude per-coordinate oscillation gating that
does help at LLM scale. To the best of our literature review
(`docs/litreview/c-perdirection-horizons.md`), per-direction *online temporal*
classification with routed responses in a spectral optimizer was unoccupied
territory; this paper populates it with a measurement study and a null.

### 7.3 Edge-of-stability theory for normalized updates

The theory line runs from normalized-GD EoS (Arora, Li & Panigrahi,
arXiv:2205.09745) through preconditioned sharpness for Adam (≈ 38/η; Cohen et
al., arXiv:2207.14484), central flows (arXiv:2410.24206 — explicitly not
covering Muon or heavy-ball momentum), to non-Euclidean EoS (Islamov et al.,
arXiv:2603.05002; §3.4). Muon-specific stability analyses give an η_max
governed by the average singular value under a K-FAC model ("Spectral
Flattening Is All Muon Needs", arXiv:2605.13079), document a broad LR plateau
(arXiv:2606.08388), attribute Muon's advantage to lower normalized directional
sharpness (arXiv:2606.04662), and predict intrinsic bounded oscillation near
minima (river-valley view, arXiv:2606.21514); Lion's Lyapunov analysis
(arXiv:2310.05898) shows sign-momentum iterates are structurally bounded. Our
contribution to this line is data, not theory: simultaneous per-direction
Euclidean lr·λ (to ≈ 65) and temporal oscillation statistics along practical
momentum+minibatch Muon trajectories, plus the no-divergence-to-6× ladder —
the regime every one of these papers either assumes away (full batch, no
momentum) or names as open.

---

## 8. Future directions

**Temporal trust ratios — executed; what remains.** The direction this
entry proposed on 2026-07-20 is now §5 (executed 2026-07-22): the setpoint
controller exists, the LR-recovery prediction held at n = 100 on airbench,
and the mechanism decomposed into schedule discovery. What remains open, in
order of bindingness: (i) the **mandatory external baselines** from the
Gate-2 conditional protocol (OrScale, NAMO/LANTON-style noise scaling,
GALA, Prodigy, and — most pointedly — a hand-tuned earlier-anneal schedule,
which the placebo replay approximates from the discovered side but which
has not been independently tuned); (ii) **setpoint transfer** — ρ\* is
read off this task's healthy runs; a self-calibrating variant (short
known-safe probe → own reference band) is designed but unrun; (iii) the
**LM thread**: the cos(G, momentum) early-window dial (§5.6; n = 2,
suggestive) plus a spike-gate on the controller (motivated by §5.7 — LM
training spikes at healthy LR would read as spurious "too hot"); (iv) the
2×-LR region is under-rescued by the window-60 configuration (+0.25pp of a
−0.53pp deficit). No results are claimed for these here.

**The Muon stability law (measured; invariant unidentified).** We built the
dual-norm directional-smoothness probe this section previously called for,
and the answer is negative twice over: generalized spectral sharpness along
practical Muon trajectories does **not** equilibrate at a c/η constant
(§3.4), and none of four instrumented scalars is conserved along the
measured B^0.35 frontier (§3.6). What remains open sharpens accordingly:
(i) what quantity *is* equalized along the airbench frontier — the
candidates that magnitude, curvature, and serial-structure statistics
supply are all ruled out as scalars; (ii) why the exponent sits near B^1/3
rather than the √B that simple noise-averaging predicts; (iii) where the
LM batch-coupling crossover sits — the critical-batch-size unification of
§3.6 is falsifiable with small-token-batch LM arms on the released
harness. No results are claimed for these here.

---

## 9. Reproducibility appendix

### 9.1 Pipeline and provenance

Every run is driven by a pinned YAML config (no CLI-arg science) through one
entrypoint; every results JSON carries config hash, git SHA, seed, GPU type
string, wall time, and a cost field (human-filled for cloud runs per the
compute-boundary protocol); `results/` is append-only (structurally enforced:
the writer refuses overwrites). Vendored substrates at pinned SHAs:
`vendor/airbench` @ `4c1b6d1e`, `vendor/modded-nanogpt` @ `edf47a05`,
`vendor/DynMuon` @ `89baa666`.

Key evidence commits: statistics suite at `d777d22`; Phase-1 pre-registration
at `b9aab63` (verified to predate the first instrumented run); Phase-1
measurement evidence at `a8e40dc`; disambiguation at `ec6cf7b`; all
evaluation-seed comparison runs at binary `411423f` (aggregations over routed
dev grids filter by this SHA to exclude the retained routing-inactive
duplicates); comparison table at `b323ed6`; overhead benchmark at `87a997f`.
Deterministic figure/table scripts: `scripts/analyze_phase1.py`,
`scripts/analyze_disambiguation.py`, `scripts/analyze_mechanism.py`,
`scripts/aggregate.py`, `reports/figures/wp05/make_figures.py`; frontier
programs: `scripts/analyze_frontier.py`, `analyze_frontier_dense.py`,
`analyze_frontier_nanogpt.py`, `analyze_local_baseline.py`; program #8:
`scripts/analyze_tempo.py` (passive / compare / nanogpt-passive modes),
`scripts/analyze_intermittency.py` (synthetic-validated null pipeline).

The frontier programs ran on a local 2× RTX 5090 (32 GB) workstation. The
LM substrate is a single-GPU port of the modded-nanogpt record
(2025-07-12_BosAlign) with exact token-batch emulation via
grad-accumulation, an audited deviation ledger stamped into every run
(`metrics.deviations` / `record_faithful`), an fp32 embedding-gradient
accumulation fix isolated by a pre-registered one-variable diagnostic, and
a row-chunked head path (loss- and gradient-equivalence pinned by tests)
that fits the record's 49K-token train chunks and 262K-token validation
sequences in 32 GB. Own-harness seed noise: σ = 0.00125 over n = 10
(χ² CI [0.00086, 0.00228]), equal to the record's native 0.0013; the
record's steps-to-3.28 endpoint is censored 10/10 on this harness, so all
LM comparisons use final val loss at fixed steps. Long local runs are
supervised by a retry-with-resume babysitter (hang timeout, escalating
backoff); checkpoints are keyed by a config fingerprint and deleted on
successful completion after an incident in which sweep variants sharing a
(seed, iterations) key replayed a sibling's finished trajectory — the
three affected result files are tombstoned in `results/INVALID_RUNS.json`
(results are append-only) and all analyzers honor the tombstone list.

### 9.2 Seed discipline

Evaluation seeds 0–99 appear only in comparison tables (§4); development,
debugging, tuning, and measurement-only runs use seeds ≥ 1000 (Phase-1
instrumentation: 1000–1019; HVP: 1200–1202; stress ladders: dev seeds). A
standing grep-check enforces that no config under development references eval
seeds; two-stage tuning selected on dev n = 25 and evaluated on eval n = 100
at the argmax only. One recorded caveat: the routed optimizer's internal RNG
(subspace init, placebo gating) is a config literal (2600) common across
seeds; model init, data order, and augmentation vary per seed (deviation 8).
The frontier programs use later dev blocks: airbench 1400–1414 (programs
#6/#6b), nanogpt 1700–1721 (port bring-up, 10-seed baseline, transfer
grid); program #8 uses 1420–1439 (airbench phases A/B/B′ — 1430–1439 the
labeled B′ exploration after the disclosed window re-tune) and 1440–1441
(LM passive probe) — all disjoint from every earlier block and from eval
seeds. The §5.5 evaluation table's controller configuration was frozen and
committed before any eval-seed run.

### 9.3 Cost

All cloud GPU work ran on spot-priced VMs (Hyperstack; RTX A6000 for the
airbench phases, 1× H100 PCIe for the nanogpt port audit). Documented phase
totals: WP0 baseline buildout $1.24; Phase-1 instrumentation VM $3.23 (98%
of it environment setup and benchmarks; the 20-seed instrumented sweep
itself was ~9 min ≈ $0.06); Phase-2 comparison matrix, mechanism probes, and
Gate-2 completion runs ≈ $5.25; nanogpt record-port reproduction and
fp32-embed diagnostic ≈ $4.80. **Total project cloud spend $13.60**
(reconciled against per-run cost fields; the total is pinned by a repo
test). Per-run attributed cost fields are stamped in all cloud run JSONs
(billed sweep window amortized evenly per run) — e.g., $0.003/run for the
100-seed comparison arms. The frontier programs (#6/#6b/#7: 249 training
runs plus the 10-seed LM baseline) and program #8 (~1,170 airbench runs
incl. the 800-run evaluation table, 8 LM probe runs, and the zero-compute
offline intermittency scan) ran on a local 2× RTX 5090 workstation at zero
marginal cloud cost. The headline scientific results — the Phase-1
characterization, the placebo-controlled null, the two-substrate frontier,
and the §5 controller with its n = 100 table — cost under $15 of cloud
compute combined, which we note as evidence that measurement-first
optimizer research has an extremely favorable cost profile.

### 9.4 Gates and adversarial review

The project ran under pre-registered gates with hard stops. Gate 1
(measurement → intervention) and Gate 2 (intervention verdict) were each
decided under explicit full delegation with a mandatory adversarial review;
both reviews materially changed the record: Gate 1's review discovered the
null-satisfiability of the pre-registered criteria (§3.1), forced the
constant-gain control arms and mechanism probes, and mandated activity
telemetry without which the null would have been uninterpretable (§4.1);
Gate 2's review caught the metric substitution disclosure, the degenerate
DynMuon clause, and the omitted 2×-LR dev signal, and converted the latter
into the pre-registered confirmation of §4.3. Both decision records, including
every accepted amendment and every disclosed deviation, are in the repository
(`reports/gate1-decision.md`, `reports/gate2-decision.md`) and are the
authoritative statements of what this paper may claim.

### 9.5 Figures

- `figures/wp12/regime_scatter.png` — (SNR, ρ) scatter, 20 seeds.
- `figures/wp12/regime_occupancy.png` — regime occupancy vs step (labels
  subject to the §3.5 confidence-window artifact).
- `figures/wp12_hvp/eta_lambda_calibration.png` — implied η·λ vs HVP lr·λ.
- `figures/wp05/rho_recovery.png`, `figures/wp05/eta_lambda_recovery.png`,
  `figures/wp05/switch_timeline.png` — synthetic validation of the pipeline.
- `figures/frontier_acc_vs_lr.png`, `figures/frontier_shoulder_vs_batch.png`,
  `figures/frontier_p2_candidates.png` — program #6: airbench accuracy-vs-LR
  ladders per batch, shoulder scaling, invariant candidates.
- `figures/sharpening_dense_ladders.png`, `figures/sharpening_alpha_fit.png`
  — program #6b: dense-ladder crossings and the α = 0.35 fit with CI.
- `figures/frontier_nanogpt_transfer.png` — program #7: LM loss-vs-LR per
  token batch and the flat (α = −0.29) crossing fit vs the airbench 0.35
  reference.
