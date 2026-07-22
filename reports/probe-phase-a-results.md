# Program #10 Phase A results — the estimator works; the harvest isn't there

2026-07-22. Pre-registered measurements M1–M4 (`reports/probe-phase-a.md`,
commit 549e7f5). 200 runs (4 fork steps × 5 LR multipliers × seeds
1452–1461), single GPU, zero retries.

## M1 — pairing benefit: large, and the curve is new

Same-seed pairing reduces probe-difference variance by **2.6–6.6×** at
fork steps 25–100 and **18×** at step 150 (Δ=5–10), despite the
pipeline's bitwise nondeterminism (replicates diverge at step ~1–7).
The CRN estimator machinery works as designed; per the novelty sweep,
this variance-reduction-vs-horizon curve is unpublished in any setting.

## M2/M3 — but the probe signal is uninformative-to-inverted about outcomes

Correlation between the paired Δ-step probe-loss difference and the
paired final-accuracy difference is ≈ 0 everywhere (−0.13…+0.10) except
fork step 25, where it is **wrong-signed** (+0.38–0.41: lower probe loss
predicts *worse* final accuracy). And the probe's preference is
maximally greedy: at every fork step and every probe length, the probe
prefers mult 0.7 — lowering LR always lowers immediate loss — the
textbook short-horizon bias, here measured directly against ground
truth rather than argued.

## M3/M4 — the deeper finding: the trajectory is self-correcting

The ground-truth outcome response to a mid-run ±40% LR change is
essentially **flat**: paired final-acc deltas are −0.21…+0.08pp, and
outside the single early-undertraining cell (fork 25, mult 0.7:
−0.21pp) everything is within ~2× the replicate noise floor (M4:
within-seed replicate sd 0.061pp, across-seed 0.099pp). A tuned
airbench run absorbs a 0.7–1.4× LR perturbation applied anywhere from
step 50 onward with no measurable cost.

This is the strongest form of the steep/flat taxonomy's local-flatness
clause: not only is the static knob flat at the optimum — the
*trajectory* is insensitive to mid-run perturbations of the steepest
knob we know. It also matches program #8 from the other side (its
controller was exactly free at 1×) and the graceful-degradation
findings: the system self-corrects.

## Verdict for Phase B

A mid-run LR controller on airbench-at-record has **nothing to
harvest**: the estimator (M1) is excellent, the naive signal (M2/M3) is
greedy-biased exactly as the literature warns, and the true response it
would exploit (M3/M4) is flat. Phase B as originally imagined is
therefore not run — a decision made by the pre-registered measurements,
at a cost of 200 short runs.

Where the controller case still lives (not pursued without direction):
(a) mis-set configs — already served more cheaply by program #8's
signal-based controller; (b) quantities with real measured drift at LM
scale — batch-size warmup (Ai2's branched-CBS line, 43% step savings),
where the probe machinery's pairing (M1) would cut their branch cost
substantially, but adjacent to a fast-moving published program and
expensive on this box (75 min/run); (c) the M1 curve + M3
bias-vs-ground-truth measurement as a standalone methodology
contribution for the paper.

## Ledger

Program #10 Phase A: 200 runs, ~1 GPU-hour, two incidental instrument
findings (fp16 loss inf; nondeterminism onset) now documented in the
harness, and one methodology result (M1) plus one negative-with-teeth
(M3/M4). Dev seeds 1452–1461 consumed.
