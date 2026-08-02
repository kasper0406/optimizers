# Theory/Theorem Sweep — Adapting Recent Mathematics to Our Findings (2026-08-02)

Four parallel literature sweeps (spectral/modular-norm geometry; edge-of-stability
and oscillation theory; loss-landscape differential geometry and symmetry;
stochastic-process/schedule/batch theory) searching 2024–2026 theorems that could
be adapted, using this repo's findings, to improve convergence times. ~55 searches
+ ~28 primary-source fetches total. All arXiv ids transcribed verbatim from search
results; unverified items flagged in §4.

Repo findings referenced throughout (from `reports/project-state.md`):
- **F1** negative-ρ population (60–89% of per-direction projections, LR-driven,
  momentum-independent, collapses in anneal)
- **F2** η·λ ≈ 65 stability, no divergence cliff to 6× LR
- **F3** amplitude ratios don't recover curvature (implied η·λ saturates 2–4)
- **F4** per-direction persistent signal unmeasurable (SNR ceiling ~0.26)
- **F5** intervention null + equivalent destinations (‖ΔW‖/‖W‖≈0.97, identical acc)
- **F6** occupancy tracks lr/lr₀; progress cooldown-concentrated (10–15× denser)

---

## 1. Convergent themes (found independently by ≥2 sweeps)

### T1. Weight-EMA as a free, anytime anneal — the direct exploitation of F1+F6
- Sandler et al. arXiv:2301.02312: exact equivalence map between iterate averaging
  and LR-decay schedules under a verified per-minibatch quadratic model.
- Defazio et al. "The Road Less Scheduled" arXiv:2405.15682 (schedule-free optimal
  worst-case rates, no horizon needed); "Anytime Pretraining" arXiv:2602.03702
  (minimax at *every* stopping time via averaging); Hägele et al. arXiv:2405.18392
  (SWA on the constant branch recovers most of the cooldown gain).
- EMA dynamics: arXiv:2411.18704, arXiv:2502.06761, arXiv:2508.00180.

Why it fits us specifically: averaging cancels the *anticorrelated* component of
successive iterates — exactly the ρ<−0.2 population of F1. With lag-1 anticorrelation,
even a 2-iterate average cancels oscillation variance faster than for white noise.
F5 says the oscillation carries no performance-relevant information, so averaging
it away is free. Prediction: EMA checkpoint mid-stable-phase ≈ post-anneal accuracy
— harvesting the 10–15×-dense cooldown progress (F6) continuously, without
committing to a schedule position. Open caveat we can measure: whether the
Sandler equivalence survives Muon's orthogonalization nonlinearity.

### T2. The correct stability functional for Muon (resolves "what sets max LR")
- arXiv:2603.05002 (ICML'26 oral, already our anchor): generalized sharpness in
  the *update's own norm* pins at 2/η while Euclidean sharpness roams free —
  our HVP η·λ≈65 (F2) is exactly the quantity this theory calls diagnostic-irrelevant.
  Reframes F3: amplitude ratios may be correctly measuring *generalized* sharpness
  at its threshold, not failing to measure Euclidean curvature.
- arXiv:2604.14108 (2026): with momentum, *batch sharpness* (expected minibatch
  directional curvature along the realized update) plateaus at 2(1−β)/η (small
  batch) or 2(1+β)/η (large batch). **Numerical coincidence: 2(1+β) = 3.9 at
  Muon's β=0.95 — the top of our observed implied-η·λ saturation band 2–4.**
- Predecessor: Edge of Stochastic Stability arXiv:2412.20553; stochastic
  self-stabilization arXiv:2606.30930, arXiv:2604.21016.

Adaptation: one extra minibatch-HVP per refresh gives batch spectral sharpness
along the realized polar update. A β-sweep (dev seeds) tests whether the
saturation band tracks 2(1±β). If it does, we have the momentum+minibatch
stability law the ICML'26 paper names as open — and a feedback signal for a
CDAT-style (arXiv:2407.06183) controller that drives LR to the true edge.

### T3. Bounded-update limit cycles — an open, provable theorem explaining F2+F3
Convergent mechanism from four independent angles:
- Trust-region: Muon = spectral-norm trust region, update norm exactly η
  regardless of gradient/curvature (Kovalev arXiv:2503.12645) → instability
  cannot self-amplify.
- Constraint set: Muon+WD = Frank-Wolfe over a spectral-norm ball / implicit
  spectral-norm constraint (arXiv:2506.15054); a.s. bounded iterates with WD
  (Sato et al. arXiv:2507.01598).
- Angular dynamics: WD equilibrium ‖W‖ ≈ 1/λ_wd, effective *angular* step ≈ ηλ_wd
  (Kosson arXiv:2305.17212; Hübler et al. arXiv:2606.23637) — η·λ≈65 is an
  angular learning rate on a norm-pinned sphere; large angular steps degrade
  gracefully instead of exploding.
- SDE view: Muon's continuous limit is a differential inclusion with *saturated
  drift* (arXiv:2605.23871); saturating-drift SDEs generically have no divergence
  cliff, just a stationary distribution that widens with η (cf. arXiv:2411.15958).

Synthesis (no paper states this for Muon — within reach as a theory contribution):
because per-direction step magnitude is capped at η (singular values flattened),
any-λ directions produce bounded limit cycles of amplitude ~η, not exponential
blowup → no cliff (F2), amplitude reflects η not λ (F3), and the stable-LR
ceiling is set by the aggregate oscillation loss penalty ~η²·Σ_occupied λ, not a
smoothness plateau (consistent with our program-#3 refutation). The penalty
scaling is checkable against existing HVP logs. Two zero-compute log checks:
‖W‖ ≈ 1/λ_wd equilibrium and angular-step ≈ ηλ.

### T4. The occupancy trigger now has a theoretical lineage and a value function
- Pflug diagnostics: the scalar lag-1 sign test *provably fires prematurely*
  (Pesme et al. arXiv:2007.00534); coupled-chain successor arXiv:2412.11341.
  Occupancy is a direction-resolved, thresholded, aggregated Pflug statistic —
  plausibly the variance reduction that rescues the diagnostic; binomial
  concentration over ~k per-direction tests gives a detection-delay guarantee
  the scalar test lacks.
- Value function: annealing-area / multi-power scaling laws (arXiv:2408.11029,
  arXiv:2503.12811) predict the loss released by "anneal now vs later" → the
  trigger becomes an optimal-stopping rule; convex last-iterate bound theory
  (Schaipp et al. arXiv:2501.18965) independently reproduces the cooldown drop
  as a vanishing noise term.
- Oracle check: WSD-optimality phase transition (arXiv:2602.06797) predicts the
  optimal decay fraction from estimable exponents (s, β) — pre-register it and
  test whether occupancy triggers near the oracle point *without knowing s, β*.
- Backbone: central flows (arXiv:2410.24206) — occupancy = # directions pinned
  at threshold, a state variable of the flow; river-valley (arXiv:2410.05192,
  arXiv:2508.01483) — occupancy = transverse hill directions, anneal = collapse
  onto the river. Falsifiable ordering prediction: during cooldown, per-direction
  ρ flips to ≈0 in curvature order (largest λ first).
- Independent stationarity cross-check: Yaida FDR residual (arXiv:1810.00004) on
  pre-orthogonalization momentum buffers; **no 2024–2026 FDR extension to
  normalized/Muon-type optimizers exists — open niche.**

### T5. Batch is the lever the SNR ceiling points at (F4)
- Spectral (S∞) gradient-noise scale for Muon-geometry methods
  (arXiv:2602.03001): Euclidean GNS is the wrong object (independently
  discredited as CBS proxy — arXiv:2505.23971); adaptive batch schedules driven
  by the dual-norm GNS report up to 66% fewer steps for Signum/Muon at 160M.
  Our per-direction projection variances ≈ a sketch of the spectral GNS —
  computable from existing logs.
- Muon critical batch size b* ∝ σ² with β, λ_wd scaling (arXiv:2507.01598);
  our instrumentation measures the unobservables in their bound.
- Optimal batch schedules: "fast catch-up, late switching" (arXiv:2602.14208) —
  defer large batch to the end at no final-loss penalty. Suggests an
  occupancy-triggered *batch ramp* (hold η·λ, ramp grad-accum) as an alternative
  arm to occupancy-triggered LR decay; F4 predicts per-direction t-statistics
  finally grow post-ramp — a sharp instrumented check.

### T6. Landscape symmetry: why the null happened, and which levers survive it
- Manifold-drift (Katzenberger) theory: Li-Wang-Arora arXiv:2110.06914; Adam's
  manifold sharpness drift arXiv:2511.02773; steepest-descent implicit bias in
  Muon geometry arXiv:2602.11557. Late training = motion *on* the minima
  manifold; per-direction gains perturb the normal dynamics, which the quotient
  forgets → predicts our F5 null. Solution quality along the reachable set is
  set by the drift's regularizer (optimizer geometry), not per-direction gains.
- Symmetry teleportation: Zhao et al. arXiv:2205.10637, ICLR'24 arXiv:2305.13404;
  sharpest conditions in Mishkin et al. arXiv:2403.03362 — teleportation
  provably accelerates iff gradient norm varies along level sets. **One-day
  go/no-go gate: measure ‖∇L‖ variation along closed-form symmetry orbits at
  fixed loss (dev seeds). O(1)% variation kills the direction; large variation
  funds a teleport-inside-Muon step.** Most implementable variant: null-space
  projection teleportation arXiv:2502.11362.
- Proven drift-steering levers are *timescale couplings*, not per-direction
  gains: 1−β ∝ η^{2/3} momentum-LR coupling for maximal on-manifold speedup
  (Cowsik et al. arXiv:2210.16400) — a two-line change coupling β to the
  decaying LR in cooldown; QSR sync-interval scaling for distributed
  (arXiv:2310.14423).
- Conservation laws: Marcotte et al. arXiv:2307.00144, arXiv:2405.12888; Muon
  (discrete+momentum+WD+orthogonalized) conserves essentially none — every step
  implicitly teleports across gradient-flow orbits; balancedness logging is a
  trivial instrumentation add with two testable predictions (drift insensitive
  to our interventions → mechanism for F5; balanced init → early-convergence test).
- Anti-PGD (arXiv:2202.02831): anticorrelated perturbations ≡ Tr(H)
  regularization — F1's endogenous anticorrelation is a built-in, LR-controlled
  Anti-PGD term; damping it should *raise* spectral mass at unchanged accuracy
  (retroactively checkable in F5 run HVP data).

### T7. Provably faster nonmonotone/supercritical schedules
- Silver stepsizes: κ^0.7864 acceleration from a fractal schedule whose long
  steps exceed the classical divergence threshold (Altschuler-Parrilo
  arXiv:2309.07879/2309.16530; stochastic extension arXiv:2511.21917).
- Large-stepsize GD on logistic loss: Õ(1/T²) *without momentum*, from an
  EoS-like oscillatory phase cashed in later (Wu et al. COLT'24 arXiv:2402.15926).
- Given no cliff to 6× (F2), test a supercritical stable phase + longer cooldown
  at equal step budget; and a two-scale alternating-LR pattern at fixed mean LR
  as a cheap discriminator (hedging accessible in Muon geometry or not — the F5
  null argues not, worth stating in the paper).

---

## 2. Ranked adaptation shortlist (convergence-time payoff ÷ cost)

1. **EMA-as-anytime-anneal (T1)** — parallel weights-EMA in the stable phase;
   compare EMA checkpoint vs anneal-branch at matched annealing-area. Minutes on
   airbench, zero trajectory risk. If confirmed: shorten/skip cooldown, and the
   pre-registered occupancy experiment gains a third arm
   (constant-LR + EMA + occupancy-triggered *stop*).
2. **β-sweep + batch spectral sharpness (T2)** — test the 2(1±β)/η law; if it
   tracks, we own the momentum+minibatch stability law (the ICML'26 open
   problem) and get a principled max-LR controller.
3. **Zero-compute log checks for T3** — ‖W‖·λ_wd ≈ 1, angular step ≈ ηλ,
   η²·Σ_occupied λ penalty scaling; then η,λ_wd co-tuned as one angular knob.
   Feeds the bounded-limit-cycle theorem (paper-grade theory contribution).
4. **Occupancy trigger upgraded to optimal stopping (T4)** — fit the multi-power
   law on existing nanogpt logs (free), pre-register the predicted optimal
   anneal point + the curvature-ordered ρ-collapse prediction, add the Yaida-FDR
   residual as an independent cross-check.
5. **Spectral GNS + occupancy-triggered batch ramp (T5)** — compute from existing
   logs; batch-ramp arm vs LR-decay arm; F4's t-statistics as the mechanism probe.
6. **Teleportation go/no-go gate (T6)** — one day, dev seeds; funds or kills the
   only categorically-new intervention class our null doesn't cover.
7. **Momentum-LR coupling 1−β ∝ η^{2/3} in cooldown (T6)** — two-line config
   change with a theorem-shaped rationale; composes with any schedule arm.
8. **Muon-MVR2** (arXiv:2509.15816, arXiv:2512.16598) — only theorem-backed
   *rate-class* improvement found (optimal Õ(T^{−1/3}) via momentum variance
   reduction); add to the zoo behind `interface.py`, measure wall-clock cost of
   the extra batch. Also: Gluon per-layer (L0,L1) measured stepsizes
   (arXiv:2505.13416) and operator-norm-targeted LR transfer (arXiv:2510.03871)
   for WP3.x tune-once-transfer.

## 3. Theory niches this project could claim (measurement paper leverage)

- The bounded-update limit-cycle theorem for Muon (T3) — unstated in the
  literature; our data is the evidence base.
- Muon fluctuation-dissipation relation (T4) — no normalized-optimizer FDR
  exists as of this sweep.
- Muon SDE with saturated drift whose stationary law *is* the occupancy-vs-lr/lr₀
  curve (T3/T4).
- The η·λ ≈ 65 momentum+minibatch no-cliff measurement as the missing data point
  for arXiv:2603.05002 / arXiv:2604.14108 / arXiv:2606.30930.
- Kaon/random-spectra theory (arXiv:2605.11181: alignment + descent potential
  govern performance, precise spectral geometry "practically irrelevant") as the
  theoretical companion of our F5 null — compute their two quantities from our
  logs; if the damped runs preserved both, the null is *explained*.

## 4. Verification flags (carried from the sweeps)

- 26xx-range ids are post-training-window; claims are abstract-level from
  fetches, not proof-checked.
- Author lists unverified: arXiv:2604.07405, arXiv:2506.15054, arXiv:2602.16340.
- arXiv:2503.04046 vs 2503.04049 ambiguity (teleportation/MTL).
- arXiv:2606.23637 title rendered "Muown" in fetch — check PDF before citing.
- arXiv:2602.01480 (rod flow) authors not surfaced.
- Blog-tier (engineering guidance, not theorems): Modular Manifolds (Thinking
  Machines), Gram-Space Manifold Muon (Tilde), manifold-Muon ADMM (Buchanan).

Full per-item notes (theorem statements, per-finding mappings, adaptation
details, complete source URLs) are in the four sweep transcripts; this file is
the deduplicated synthesis. Related prior sweeps: `a`–`i` in this directory
(2026-07-19/20) — overlaps: 2603.05002 (d), Pflug/GALA lineage (a), OrScale/
NAMO trust-ratio columns (b), Song et al. bulk-subspace (c), CBS landscape (e),
schedule-as-control composite (f).

---

## 5. Repo connection map (added 2026-08-02, structuring pass)

Two structural facts established by code inspection during this pass, which
recontextualize T3 before any measurement:

- **airbench pins filter-weight norms by construction**: the vendored Muon
  renormalizes every step (`p.data.mul_(len(p)**0.5 / p.data.norm())`,
  `vendor/airbench/airbench94_muon.py:83`). The "η·λ is an angular step on a
  norm-pinned sphere" reading of T3 is *structurally enforced* on airbench —
  no equilibrium to verify. The F2 no-cliff finding was collected under pinned
  norms, consistent with the bounded-limit-cycle picture.
- **nanogpt Muon runs with `weight_decay: 0.0`** (`src/nanogpt/config.py:153`),
  so the WD-equilibrium (‖W‖ ≈ 1/λ_wd) form of T3 does not apply there; the
  applicable result is the no-WD variant (norm growth ⇒ implicit angular
  step-size decay, arXiv:2606.23637). Measurable signature: ‖W‖_F(t) growth —
  **not currently logged**; needs a per-matrix norm hook in a future
  instrumented nanogpt run.

| Theme | Finding | Repo asset today | Concrete next step | Status |
|---|---|---|---|---|
| T1 EMA-as-anneal | F1, F6 | `run_airbench_smoke` harness + hooks (`src/optim/airbench_zoo.py`); stock schedule is linear-decay-to-zero (the Defazio-optimal shape — clean comparison arm) | `airbench_ema` experiment: weight-EMA (multi-γ) + per-epoch EMA eval; arms stock-schedule vs constant-LR; dev seeds | **REFUTED at airbench scale** (2026-08-02, n=20 paired: best EMA −4.6pp TTA vs annealed, 0/20 harvest crossings, EMA-on-anneal adds 0.0 — `reports/wpj-t1-ema.md`; nanogpt regime untested) |
| T2 stability law (2(1±β)/η) | F2, F3 | β is probe-overridable (`PROBE_OVERRIDE_KEYS`); HVP machinery `src/instrument/hvp.py`; mom0 + LR-ladder configs in `configs/dev/` | β-sweep + minibatch-HVP along realized polar update ("batch spectral sharpness") — new instrument mode | follow-up (next cheap cloud batch) |
| T3 bounded limit cycles | F2, F3 | airbench: structural (renorm, above); per-direction λ_hvp series in HVP-enabled sidecars; LR ladder ½×–6× results | η²·Σ_occupied λ penalty-scaling analysis from existing HVP + stress-test JSONs (offline); nanogpt ‖W‖_F logging | analysis follow-up; partially structural |
| T4 occupancy trigger | F1, F6 | occupancy machinery `src/stats/` (WP0.5-validated); `criteria/occupancy_cooldown_preregistration.md`; nanogpt harness lacks occupancy port | multi-power-law fit on existing wp02 loss curves (free, offline); Yaida-FDR residual on momentum buffers = new cheap instrument | follow-up; pre-reg already exists |
| T5 spectral GNS / batch ramp | F4 | per-direction projection variances in instrumentation sidecars ≈ sketch of S∞ noise scale | compute spectral GNS from existing sidecars (offline); batch-ramp arm needs harness batch-size schedule support | analysis follow-up |
| T6 teleportation gate | F5 | twin-trajectory probe harness (`scripts/probe_divergence.py`) is the natural home for orbit moves | one-day dev experiment: ‖∇L‖ variation along closed-form symmetry orbits at fixed loss | follow-up, gated on T1 outcome |
| T7 supercritical schedules | F2 | stress configs (2×/3×/6× LR) already exist with results | two-scale alternating-LR config at fixed mean LR | follow-up |

Decision (delegation, this pass): T1 first — cheapest, clearest prediction,
directly convergence-time-relevant; its constant-LR arm also produces the
stable-phase EMA data T4's "un-cashed progress meter" needs. T2 is the next
cloud batch. T3/T4/T5 offline analyses queue behind T1's run.

Outcome (2026-08-02): T1 refuted in the airbench regime — the anneal's
contribution is not recoverable by iterate averaging (details and
interpretation in `reports/wpj-t1-ema.md`). Sharpens rather than weakens T4:
the occupancy trigger must drive a real anneal, and "anneal as mechanism, not
noise-removal" is now supported by a direct instrument.

## 6. Direction pivot (2026-08-02, user steer — memory `flow-insight-over-laws`)

Law-measurement (T2's 2(1±β)/η constant, T3's penalty-scaling fit) is
deprioritized: constants describe the regime but don't provide a lever.
Re-ranked program — interventional flow-mechanism experiments, each with a
convergence-speed payoff:

1. **Anneal dissection ("how short is the last mile?")** — direct follow-up
   to the T1 null. From a constant-LR trajectory, branch anneals of length
   k ∈ {0, 5, 10, 25, 50} at several branch points (shared batch stream and
   snapshot per branch point). The accuracy-vs-k saturation point k* measures
   how much of the anneal is fast dynamical relaxation vs slow walking. If
   k* is small, "constant LR + short anneal" beats the tuned schedule at
   matched accuracy with fewer steps — a direct, mechanistically-grounded
   speed recipe. Experiment: `airbench_anneal_branch`.
2. **Progress decomposition** — attribute per-window loss decrease to motion
   along tracked oscillating directions vs the bulk complement, phase by
   phase (the measurable aggregate form of the river coordinate; the
   per-direction SNR ceiling does not apply to the aggregate). Existing
   tracker + new analysis; answers "where does progress live in the flow?"
3. **Twin-trajectory flow-structure probes** — perturbation growth/decay
   transverse vs longitudinal along training (local stability structure of
   the flow; extends the program-#1 twin machinery).
4. **Teleportation go/no-go (T6)** — ‖∇L‖ variation along closed-form
   symmetry orbits at fixed loss; funds or kills the only intervention class
   the Phase-2 null doesn't cover.
5. Occupancy-triggered anneal (T4) — unchanged, but now framed as: the
   trigger decides when the (short, per #1) real anneal starts.
