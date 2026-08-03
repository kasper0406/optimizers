# Teleport-Muon round 2: moves work as designed; training doesn't care (−0.4pp)

Date: 2026-08-03. Follow-up to the GO gate of `reports/wpj-mech-round1.md`
§1. Data: `results/airbench_teleport_*` (ON/OFF arms, n=20 seed-paired dev
seeds + smoke, NVIDIA L40 spot, git 53c064e). Generated table:
`reports/wpj-teleport-arms-table.md` / `.json`
(`scripts/analyze_teleport.py`).

## What was built

Teleport moves inside the stock airbench Muon recipe: every 25 steps from
step 10, between backward and step, a closed-form **gauge-fixed** nuclear-
norm ascent (`src/optim/teleport.py`) finds per-channel conv→BN orbit scales
maximizing the *Muon-spendable* gradient nuclear norm (the uniform orbit
direction is projected out — it satisfies polar(cG)=polar(G) and the
recipe's renorm removes it; the ungauged ascent of the original spec
saturated the clamp box on exactly that invisible direction, a flaw caught
by the implementing subagent and fixed before running). The weights move
along the orbit and the full first-order state (gradient, Muon momentum) is
transported to the new gauge, so the imminent step is coherent. Ascent
validated by finite differences (2.4e-9) and a closed-loop test (predicted
vs realized post-teleport gradients agree to 6e-5).

## What happened

- The moves did what the gate promised, and more: mean spendable ratio
  1.22 (first teleport) → 1.32 (last), per-matrix up to 1.49 — every
  teleport handed the very next Muon step 20–30% more first-order descent
  potential, 7 times per run. Overhead +4.7 s/run (uncompiled SVD ascent on
  CPU-side matrices; optimizable, irrelevant here).
- **Final accuracy: ON − OFF = −0.43pp ± 0.13 (val), −0.39pp ± 0.08 (TTA).**
  A small but clearly resolved negative.
- Epoch profile: flat-to-positive mid-training (epoch 3: +1.5pp ± 1.7, n.s.),
  significantly negative from epoch 6 on. The teleports at steps 150/175
  land in the decay phase — exactly where the anneal-dissection experiment
  located the formation of final accuracy — and that is where the deficit
  appears.

## Reading

1. **One-step descent potential does not compound.** Gaining 20–30% more
   ⟨G, polar(G)⟩ at a point, seven times, nets *negative* by the end. The
   equivalent-destinations picture absorbs locally-better moves as readily
   as locally-worse ones.
2. This is the **fifth** intervention class on this recipe to return
   null-or-negative at matched budget (per-direction spectral gains, EMA
   substitution, anneal shortcuts, explicit central-flow term, teleport
   moves) — and the first with a *provably larger* first-order descent
   quantity at every intervention point. Together they say something
   sharp: stock Muon at the tuned schedule sits at a remarkably flat
   optimum of *training-procedure space*; local first-order improvements to
   the procedure are reabsorbed by the dynamics.
3. The theory condition (Mishkin et al.: teleportation helps under Hessian
   stability along the orbit) is evidently violated here in the phase that
   matters; the mid-training hint (+1.5pp at epoch 3, n.s.) is consistent
   with their convex-phase prediction and with the damage being
   decay-phase-specific.

## Obvious next knob (not run)

Restrict teleports to the hot phase (e.g. steps 10–100, before the decay
does its work): the epoch-3/4 bump suggests hot-phase teleports are at
worst free, and the endgame deficit is attributable to late moves. One
config knob (`stop_step`) + one 40-run sweep ≈ $1. Left for the next
session under the standing delegation.

## Provenance

Episode: VM rm-tp-l40, ~1.4 h × $0.80 ≈ $1.10; $0.027 × 41 runs stamped;
VM deleted after pull (0 active). Suite green (859+); `criteria/` untouched;
no gate evaluated by this report. Core math by an opus-5 subagent (verified
here, incl. the gauge-fix decision it correctly escalated); harness,
analysis, and interpretation by the session agent.
