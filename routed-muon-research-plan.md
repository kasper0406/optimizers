# Routed Muon: Research & Implementation Plan

**Working hypothesis.** The variance Muon amplifies decomposes into (at least) three regimes per singular direction — persistent signal, sampling noise, and edge-of-stability oscillation. These are separable online via cheap temporal statistics (mean, variance, lag-1 autocorrelation of projected gradients), and the correct responses differ: keep/boost, freeze/shrink, damp-and-raise-LR. Published Muon variants (AdaMuon, NorMuon, NAMO, Muon², DynMuon) adapt by *magnitude* or by a *global scheduled* shaping exponent; none classify regimes per direction, none use temporal structure, none route.

**Primary claim to test.** Online per-direction spectral shaping driven by regime classification beats (a) stock Muon and (b) globally-scheduled shaping (DynMuon) at matched tuning effort — or, failing that, produces a measurement result that adjudicates why not.

**Hardware.** Provisioned on demand via Hyperstack VMs, sized per phase: airbench experiments need a single mid-range GPU (an A6000/L40-class VM is plenty; airbench runs are seconds, so a cheap instance left running through a seed sweep is the economical shape); instrumented nanogpt runs want a single A100/H100 or a 2× GPU VM with DDP. Distributed scalability is addressed by analysis only (see the dedicated section) — no multi-node or FSDP experiments are in scope. The 2×RTX 5090 local box remains useful as a free dev/debug environment — iterate locally, sweep in the cloud. Operational consequences, baked into the plan: (a) every experiment must be resumable from checkpoint (spot/preemptible instances are the cheap tier); (b) all runs launch from a pinned container image + one entrypoint script, so a fresh VM is productive in minutes; (c) results (metrics JSONs, not checkpoints) sync to durable storage continuously, since VMs are disposable; (d) per-run cost gets logged next to per-run results — cost-normalized comparisons are part of the story for a method that adds overhead.

---

## Phase 0 — Baselines, harnesses, reproduction

Goal: a trusted experimental substrate. Nothing here is novel; everything here determines whether later comparisons mean anything.

### 0.1 CIFAR-10 airbench
- Clone `KellerJordan/cifar10-airbench`. Reproduce the 94% tier on whatever single GPU is cheapest/available (local 5090 or a Hyperstack instance). Record: wall time, accuracy distribution over **n = 100 seeds** (runs are seconds; do this properly from day one).
- Deliverable: `baseline_airbench.json` — per-seed accuracy + time; mean, std, 95% CI.
- Sanity target: accuracy distribution consistent with published record (deviations from hardware differences vs the A100 reference are fine; what matters is your own stable reference distribution). One rule: **all runs within any single comparison table use the same GPU type** — never mix instance types inside a comparison, since kernel/numerics differences can exceed the effect size you're resolving.

### 0.2 modded-nanogpt (single A100/H100-class VM or 2-GPU DDP)
- Clone `KellerJordan/modded-nanogpt`. Pick a **historical record config** (pattern used by Newton-Muon et al.: pin an exact record with its released config/log). Prefer a mid-2025 record — recent records are hyper-co-adapted engineering artifacts; older ones are cleaner optimizer testbeds.
- Port to the provisioned GPU count: reduce per-step batch via grad accumulation to match the record's token batch; verify loss-vs-**tokens** curve overlays the record's log (wall-clock will differ; token trajectory should not).
- Establish per-seed variance with 3 seeds; record per-run VM cost alongside.
- Deliverable: `baseline_nanogpt/` — loss curves, config, seed variance at fixed token checkpoints.

### 0.3 DynMuon reproduction (small scale)
- Clone `fzwark/DynMuon`. Reproduce their Muon-vs-DynMuon gap on the smallest config that shows it (their claim: 10.6–26.5% fewer steps to target loss). This is now the strongest published baseline in your niche; if you can't reproduce a gap at small scale, that's important information about the niche itself.
- Deliverable: reproduction note — did the gap appear, at what magnitude, with how much tuning.

### 0.4 Baseline zoo (implementation only, runs later)
Implement/vendor, behind one optimizer interface with per-matrix hooks:
- Muon (reference), AdamW, **DynMuon** (scheduled p), **AdaMuon** (element-wise 2nd moment on O_t + sign stabilization), **NorMuon** (neuron-wise post-orthogonalization normalization).
- These are the comparisons a reviewer will demand. Skip Muon²/NAMO initially; add only if results warrant a paper-grade table.

**Gate 0 → 1:** airbench and nanogpt baselines reproduce within tolerance; DynMuon gap reproduced (or its absence documented).

---

## Phase 1 — Measurement study: do the regimes exist and separate?

Goal: pure instrumentation, zero behavior change. This phase is valuable standalone (nobody has published this characterization) and is the cheap kill-switch for the whole idea.

**Precondition (added):** the statistics machinery (EMAs, autocorrelation, classifier, implied-ηλ estimator) must first pass a synthetic-data validation suite — AR(1) processes with known ρ, drifting means with known SNR, pure oscillations with known amplitude ratio, and mid-stream regime switches — recovering ground truth within tolerance. A sign error in the lag-1 autocovariance would produce a plausible-looking wrong regime scatter with no way to notice downstream; no instrumented run launches before this suite is green. (Spec: `claude-code-handoff.md`, WP0.5.)

### 1.1 Instrumentation
For each Muon-managed weight matrix, track **k singular pairs** of the momentum matrix M_t:
- Refresh (u_i, v_i) every **T_refresh = 50** steps via subspace iteration warm-started from previous vectors (Dion-style amortized power iteration; 2–3 iterations per refresh). Track top-k₁ = 16 and, separately, a random/bottom sample k₂ = 16 from the bulk (random Gaussian vectors orthogonalized against the top block — the bulk is where noise lives; you must observe it, not just the top).
- Per tracked pair, per step, log the scalar projection **s_i(t) = uᵢᵀ G_t vᵢ** (raw gradient, *pre*-momentum — momentum filtering would mask the autocorrelation structure you're trying to measure). Cost: k rank-1 contractions per matrix per step — negligible.
- Maintain online EMAs (β = 0.99 and β = 0.9, both — timescale sensitivity is itself a finding):
  - μ_i = EMA[s_i] (mean)
  - q_i = EMA[s_i²] (second moment) → var_i = q_i − μ_i²
  - a_i = EMA[s_i · s_i(t−1)] → lag-1 autocov ĉ_i = a_i − μ_i²; autocorr ρ_i = ĉ_i / var_i
- Also log per matrix: top singular value of M_t, ‖G_t‖_F, and (cheaply, every T_refresh) **λ_i ≈ uᵢᵀ H (vᵢ ...)** — concretely, curvature along the update direction uᵢvᵢᵀ via one HVP per tracked pair per refresh: `vᵀHv` with v = vec(uᵢvᵢᵀ) restricted to that matrix. This is the only nontrivial cost; at T_refresh = 50 it's a few % overhead.

### 1.2 The three plots that decide everything
Run instrumented-but-stock Muon on (a) airbench, (b) the nanogpt config, and produce per-run:
1. **Regime scatter:** each tracked direction as a point in (|μ|/√var, ρ) space, colored by λ, animated/faceted over training. **The hypothesis predicts three clusters:** high-SNR/ρ≈0⁺ (signal), low-SNR/ρ≈0 (noise), any-SNR/ρ<0 with |ρ| large (oscillation). The null predicts a single smear.
2. **Regime occupancy vs time:** fraction of tracked directions per regime across training, per LR phase. Predictions to check: oscillation appears at high LR / after sharpening; noise fraction grows in the anneal tail; DynMuon's "positive p early, negative p late" should be *visible* as a shift in which regime dominates.
3. **Oscillation ↔ stability link:** for directions classified oscillating, plot implied ηλ from amplitude ratio |s_i(t)/s_i(t−1)| against the HVP-measured ηλ. If these agree, you get curvature-for-free along sharp directions — a standalone mini-result.

### 1.3 Statistical hygiene
- Airbench: ≥20 instrumented seeds. Nanogpt: 2–3 seeds (expensive; regimes should be consistent across seeds if real).
- Pre-register (in the repo README, dated) the cluster-existence criterion before looking: e.g., GMM/BIC prefers ≥2 components in (SNR, ρ) space in ≥70% of (matrix, phase) cells, and the ρ<−0.2 population is non-empty during the high-LR phase.

### 1.4 Decision gate (the important one)
- **Regimes separate cleanly** → Phase 2 as planned.
- **Only 2 regimes separate** (e.g., oscillation exists but noise/signal smear) → Phase 2 ships the *oscillation channel only* (see 2.4) — smaller but cleaner claim.
- **No separation** → stop building the optimizer. Write the measurement study as a short paper/blog post: "per-direction gradient statistics under Muon: a null result for regime routing" + the ηλ-from-amplitude observation. This is still a contribution (it adjudicates AdaMuon/NorMuon/DynMuon's implicit assumptions) at a small fraction of the full project's cost.

---

## Phase 2 — Routed Muon prototype on airbench

Goal: minimal routing implementation; establish effect size with real error bars; ablate against the magnitude-based incumbents.

### 2.1 The optimizer (v0 spec)
Two-tier, COSMOS-shaped:
- **Bulk tier:** vanilla Newton-Schulz on M_t, unchanged.
- **Tracked tier:** k = 16–64 singular pairs (from Phase 1 machinery). After NS produces O_t, apply per-tracked-direction correction: O_t ← O_t + Σ_i (g(i) − 1) · uᵢvᵢᵀ · (uᵢᵀO_tvᵢ), where g(i) is the routing gain:
  - **Signal** (t-stat of μ_i above threshold τ_sig, ρ ≥ 0): g = 1 (Muon default). Optional v1: g > 1 if HVP-confirmed low λ.
  - **Noise** (t-stat below τ_noise, |ρ| small): g = g_noise ∈ [0, 0.5]. Start 0.25, not 0 — misclassification asymmetry (a starved weak-signal flat direction is exactly what Muon exists to feed) demands a conservative floor.
  - **Oscillation** (ρ ≤ −ρ_osc, amplitude non-decaying): g = g_osc chosen to target implied ηλ → 1 (critical damping): g_osc ≈ 1/(ηλ_implied − 1) clipped to [0.1, 1].
- **Confidence:** each direction starts in signal (= stock Muon behavior) and only leaves it when its classification t-statistic clears threshold with effective sample size ≥ N_min ≈ 50 steps. Subspace refresh resets confidence for rotated directions (innovation > threshold → reset).
- Hyperparameters introduced: k, T_refresh, β, τ_sig, τ_noise, ρ_osc, g_noise, N_min. **This is too many.** Freeze all but (g_noise, ρ_osc, k) at Phase-1-informed defaults; sweep only those three.

### 2.2 Experiments (airbench, n = 100 seeds per config)
1. Routed Muon vs stock Muon at the record config (unfair to you — config is tuned for stock; report anyway).
2. Both with a small LR × WD sweep (3×3) — the fair comparison.
3. **Baselines at matched sweep:** DynMuon, AdaMuon, NorMuon. The killer table is: does routing beat *magnitude adaptation* and *global scheduling*, not just stock Muon.
4. **Null-hypothesis ablations** (designed to kill the idea):
   - Muon + retuned weight decay (tests: noise channel ≈ decay?)
   - Muon + global LR bump + grad clip (tests: oscillation channel ≈ clipping?)
   - Routing with ρ *ignored* (magnitude-only gate) — isolates the autocorrelation channel specifically; if this matches full routing, the novel part is dead even if the method "works".
   - Random gating (same fraction of directions gated, chosen randomly) — placebo control.
5. Oscillation-channel stress test: raise LR 1.5–2× above record. Prediction: stock Muon degrades/diverges; routed Muon holds via per-direction damping. This is the cleanest possible demo of the multiplicative LR-ceiling argument.
6. Log λ along damped directions over time → does damping trigger runaway sharpening (Damian et al. self-stabilization concern)? Plateau = safe; runaway = need λ-tracking in the damper.

### 2.3 Success criteria (pre-registered)
- Primary: time/steps-to-94% improvement over stock Muon ≥ 2σ of seed noise at fair tuning, **and** ≥ the DynMuon gap on the same harness, **and** not matched by ablations 4a–4c.
- Secondary (independently publishable): stability margin — max stable LR ratio routed/stock ≥ 1.3.

### 2.4 Contingency branches
- Only oscillation channel wins → rebrand: "per-direction edge-of-stability damping for spectral optimizers" — drop noise gating, simplify to (ρ_osc, g_osc), re-run. Smaller, cleaner, likely easier to publish.
- Only noise channel wins → the interesting comparison becomes late-training/anneal-tail and small-batch regimes; pivot Phase 3 toward fine-tuning workloads (where the noise set is large from step 1), and note the connection to "Can Muon fine-tune Adam-pretrained models" line.
- Both wash but Phase 1 showed clean regimes → the routing *response* is wrong, not the *diagnosis*; iterate on g(·) (e.g., DynMuon-style per-direction exponent p_i instead of linear gain) before abandoning.

---

## Phase 3 — nanogpt scale (gated on Phase 2 primary or secondary success)

- Port routed Muon into the pinned modded-nanogpt config. 3–5 seeds baseline vs routed vs DynMuon at matched token budget; metric = val loss at fixed tokens + tokens-to-3.28.
- Add the large-batch axis: Essential AI showed Muon's edge grows with batch size; the oscillation channel predicts routed Muon extends that frontier. Run batch-size ladder (2–3 points) if budget allows — this doubles as the most industrially relevant claim.
- Resolve a 3–5% effect or report honestly that you can't at this budget; design seeds accordingly (power analysis from Phase 0 seed variance *before* running).
- If results hold: this is the paper. Structure: Phase 1 measurement (the empirical contribution nobody has) → routing method → airbench with 100-seed stats → nanogpt confirmation → ablation table vs DynMuon/AdaMuon/NorMuon. The measurement section is what differentiates it from the "yet another Muon variant" pile.

## Phase 4 — Optional extensions (only with strong Phase 3 results)

- **Dion substrate:** re-implement tracked tier on Dion's warm-start basis + error feedback (frozen directions' residuals flow into EF buffer naturally). Positions the method for distributed settings and gives a second platform datapoint.
- **HVP-boosted signal tier** (the g > 1 branch): the original "selective Hessian" idea — only worth it once routing itself is validated.
- Speedrun submission attempt if airbench numbers beat the record with clean methodology.

---

## Distributed scalability (design constraints, enforced from Phase 2)

Reference deployment model: Megatron/Moonlight-style layer-wise ownership — each matrix's momentum gathered on an owner rank, NS computed there, update broadcast.

**Invariants to build in now (free at 2 GPUs, expensive to retrofit):**
1. **Single-writer statistics.** All tracked-pair state, EMAs, and routing decisions for a matrix live on that matrix's owner rank. Decisions made once, corrections applied to O_t pre-broadcast. No replicated decision logic → no divergence risk, no extra communication for the correction itself.
2. **Projections must tolerate sharded G.** Implement s_i(t) as local partial contraction + one batched all-reduce of all matrices' k scalars per step (bytes, not megabytes). Under plain DDP this degenerates to a local op.
3. **Core routing must be HVP-free.** HVPs are a full extra pipeline pass at scale — unacceptable under PP. The core method doesn't need them: noise channel needs no curvature; oscillation channel reads ηλ from amplitude ratios (trajectory-derived, free). HVPs are confined to the optional Phase-4 g>1 boost tier and to Phase-1 *validation* of the amplitude-ratio estimator. Any Phase-2 routing logic that depends on an HVP is a design bug.
4. **Subspace refresh = sharded matvecs.** Amortized power iteration against sharded M is Dion's native primitive (comm ∝ rank r, not m·n). The Phase-4 Dion-substrate branch is therefore also the scale-out path, not just a convenience.

**Scale interactions (arguments, not just constraints):**
- Large batch ⇒ noise regime shrinks, oscillation channel dominates — and that's the channel with the best distributed profile and the Essential AI large-batch tailwind. The industrial claim at scale is "extends Muon's critical-batch-size frontier."
- Overhead vs DynMuon (which adds zero distributed cost) must be reported: ours = k scalars/matrix/step collective + amortized sharded power iteration. NorMuon's FSDP2 implementation is the precedent that per-direction optimizer state at scale is acceptable; use it as the reference implementation pattern.

**Feasibility deliverable (analysis only, no distributed runs):** a short cost-accounting section for the paper covering, per matrix of shape m×n with k tracked pairs: optimizer state added ((m+n)k + O(k) scalars vs mn momentum), per-step communication (one batched all-reduce of k scalars per matrix under sharded gradients; zero under DDP), amortized refresh cost (2–3 sharded matvecs per T_refresh, comm ∝ (m+n)k), and update-path cost (rank-k correction on the owner rank, zero communication). Present alongside the same accounting for Muon, DynMuon (zero added), AdaMuon/NorMuon (mn / m added state), and Dion. Claiming compatibility with the Megatron owner-rank pattern and NorMuon's FSDP2 precedent is sufficient; empirical distributed validation is explicitly out of scope. All experiments run single-node (DDP or single GPU) — code respects the four invariants above so nothing structural blocks a future distributed port, but no run exists to test it.

## Risks & honest priors

1. **DynMuon partially scooped the framing.** Differentiation must be explicit everywhere: per-direction + online statistics + oscillation channel vs global + scheduled. Read their theory section closely — their p-selection theory (curvature, noise, stage) is your best argument that per-direction is the "correct" version of their own analysis.
2. **Field velocity:** ~6-month half-life on unclaimed ideas in this niche. Phase 1 results are postable (blog/arXiv note) as a flag-plant even before the optimizer works.
3. **Speedrun configs are hostile to modifications** (co-adapted). Fair-tuning sweeps are non-negotiable; report both untuned and tuned.
4. **Misclassification asymmetry** is the method's Achilles heel — conservative gates capture less of the win. If forced to choose, ship the oscillation channel; its false positives (damping a direction that wasn't really oscillating) are far cheaper than the noise channel's (starving weak signal).
5. **Effect-size prior:** layers 2–3 of the original analysis cap realistic gains at tens of percent. If you need 10x to feel this is worth it, stop now; if a robust 10–20% with a clean mechanistic story is a win (it is, by field standards — DynMuon's 10–26% is a strong paper), proceed.

## Immediate next actions

1. Repo skeleton: one optimizer interface, instrumentation hooks, seed-sweep runner, plotting for the Phase-1 scatter.
2. Container image + entrypoint script (the Hyperstack bring-up path) — do this before any cloud run; it pays for itself immediately.
3. Phase 0.1 (airbench reproduction) — runnable on the local box today; it's minutes of compute.
4. Read DynMuon end-to-end; write a half-page "differentiation memo" for the future paper's related-work section while it's fresh.
5. Pre-register Phase 1 cluster criteria in the repo README before first instrumented run.
