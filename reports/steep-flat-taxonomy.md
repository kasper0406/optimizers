# The steep/flat taxonomy of gradient-guided design objects

2026-07-22. Synthesis note (agent-written, from the programs' combined
evidence; requested written down before the meta-control discussion).
This is the empirical answer to "which mathematical objects guided by
gradients have a non-flat response surface?" — sorted by measured
steepness, with provenance for every entry.

## The pattern

**Flat objects are shapes** (high-dimensional design directions):

| object | evidence | source |
|---|---|---|
| per-direction update gains / routing | placebo-controlled null, ≤0.042pp at 97.5% | Gate 2 |
| spectral profile of the update | random spectra tie Muon | Kaon 2605.11181 |
| orthogonalization accuracy | cubic ≈ quintic ≈ SVD | Huang 2606.00371 |
| momentum kernel shape (matched staleness) | min-variance kernel loses −0.5/−1.8pp | program #9 |
| adaptive gain trajectory beyond its own mean | open-loop replay ties closed loop | program #8 placebo |
| which equivalent destination is reached | ‖ΔW‖/‖W‖≈0.97 at equal accuracy | program #1 |

**Steep objects are scalars, allocations, and measures** (low-dimensional):

| object | measured steepness | source |
|---|---|---|
| effective LR scale | 0.5–3.5pp across ladders; the steepest knob everywhere | frontier programs, #8 |
| schedule timing (anneal placement) | cooldown-concentrated deficits (10–15× denser); WSD fraction | WP0.2 §3.6, lit |
| batch trajectory | lr\* ∝ B^0.35 (CNN scale); ramping worth ~36% wall-clock in lit | programs #6/#6b; Seesaw |
| momentum *scale* (not shape) at hot LR | β 0.6→0.45 at 2× = +0.17pp | program #9 control arm |
| data measure (selection/order) | the pinned record's own gain is a data-alignment fix (BosAlign); selection lit claims 30–40% | vendored record; lit |
| readout/averaging operator | consistent small positive (tail-EMA, schedule-free) | speedrun PRs, lit |

## Why (mechanism, measured)

Solutions are massively degenerate (program #1), so anything that only
*redirects* the trajectory lands somewhere equally good — shape-space is
flat. Final loss is governed by the noise-temperature trajectory (how
much stochastic energy is injected, where, and when it is removed) and
by what the gradient budget is spent on — scale/allocation/measure-space
is steep. Corollary observed three times: steep knobs are steep
*globally* but flat *locally at a tuned optimum* (records resist).

## The strategic trap this exposes

**Novelty is cheap where payoff is flat** (shape-space is
high-dimensional; unclaimed variants always exist), **and payoff is
steep where novelty is scarce** (the scalars are few and heavily mined).
Two exits:

1. **Find new stiff directions** — metric identification: fit the
   geometry in which the measured dynamics become canonical (the
   frontier data proves an unidentified stiff coordinate exists, since
   something sets the useful-LR ceiling and none of four instrumented
   scalars is it).
2. **The data measure** — the only object that is simultaneously steep,
   high-dimensional (novelty available), and not strip-mined by us.

## Corollary for meta-optimization (the "learn the knobs online" idea)

The steep set is ~4–6 dimensional (global LR gain, 2–3 per-role LR
splits, momentum scale, batch size, data-mixture temperature) — small
enough for conventional optimization *if* three known failure modes are
designed around: (i) short-horizon bias (greedy hypergradients collapse
LR — Wu et al. 1803.02021); (ii) closed-loop endogeneity (program #9:
statistics move when the knob moves; only *measured closed-loop
response* is trustworthy); (iii) seed noise (σ ≈ 0.14pp swamps small
effects — needs paired/common-randomness probes). Design implication:
perturbation-based response estimation (branched, common-randomness
probes feeding a slow controller) rather than gradient-based
meta-descent. See the meta-control discussion following this note.
