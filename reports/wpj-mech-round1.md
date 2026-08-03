# Mechanistic program, round 1: teleport gate GO (+21% descent potential); central-flow v0 negative

Date: 2026-08-03. Flow-first program items 2 and 4 (docs/litreview/
j-theory-theorem-sweep.md §6; direction decision of 2026-08-03). Data:
`results/airbench_teleport_gate_*` (n=6 dev seeds) and
`results/airbench_centralflow_*` (2×2 arms × n=10 paired dev seeds), NVIDIA
L40 spot, commits d0686e0…412e500. Generated tables:
`reports/wpj-teleport-gate-table.md`, `reports/wpj-cf-arms-table.md` (+ .json).

## 1. Teleportation gate: GO

The conv→BN per-channel scale orbit is loss-invariant in practice (worst
|Δloss|/L = 1.2e-5 across 1,152 draws) and gradient size varies hugely along
it at fixed loss:

| training state | median random draw | best random (64 draws) | best refined |
|---|---|---|---|
| step 50 | +10.8% | +14.0% ± 0.6 | **+22.9% ± 1.1** |
| step 100 | +10.4% | +13.6% ± 0.6 | **+22.3% ± 1.4** |
| step 150 | +9.8% | +12.8% ± 0.4 | **+21.4% ± 1.4** |

(nuclear-norm sum over filter matrices = Muon's one-step descent potential
⟨G, polar(G)⟩; Euclidean total-gradient ratios are larger still, ~+32%.)

The kill criterion (O(1%) heterogeneity) is missed by an order of magnitude:
teleportation's precondition — gradient-norm variation along level sets — is
strongly present for Muon on this workload, at every probed training state.
**Next build:** actual teleport moves inside training (periodically move to
a higher-descent-potential orbit point before stepping), plus a cheaper
orbit search than random-refinement (the objective is differentiable in
log-scales; an analytic ascent should need ~2-3 evals, not 64+64). Whether
+20% per-step potential compounds into faster convergence is exactly what
that experiment will measure — the gate says it is worth building.

## 2. Central-flow v0: negative in both regimes

2×2 (LR scale × explicit term), n=10 seed-paired:

| arm | final val | reading |
|---|---|---|
| A stock, off | 0.9313 | reference |
| B cold (0.25×), off | 0.9118 | cold costs −1.95pp |
| C cold + term | 0.9108 | **term inert: C−B = −0.10pp ± 0.17** |
| D stock + term | 0.1000 | **collapse** (penalty norm 56 → 13,793) |

Recovery fraction −5%: the explicit Euclidean central-flow term recovers
none of what the cold LR gives up. At stock LR the open-loop term (10-step-
stale penalty, no self-consistency feedback) enters a positive feedback loop
with the curvature it penalizes and destroys training — the real oscillation
never does this, because self-stabilization is closed-loop.

Readings, in decreasing confidence:
1. **The v0 transplant fails**: Euclidean central-flow drift with η²/2
   weights on top-k momentum directions does not stand in for whatever
   high-LR Muon training does. The machinery (third-order autograd on fp32
   functional overrides, validated on the EoS toy at 2.5% error) is sound;
   the transfer is what failed.
2. Consistent with the project's accumulated picture: this is now the
   *fourth* independent intervention on Muon's oscillation channel with a
   null/negative result (per-direction damping, EMA-averaging, anneal
   branch-point independence, explicit CF term). The oscillation's implicit
   regularization is not load-bearing on this workload.
3. NOT concluded: that central flows are wrong for Muon. The correct
   non-Euclidean central flow (the open problem of arXiv:2603.05002) may
   have a different drift term entirely; our negative constrains transplants
   of the Euclidean form, and the D-arm explosion is partly an open-loop
   implementation artifact (noted, not over-read).

## 3. Engineering notes (hard-won, now encoded in code/tests)

- Second/third-order autograd through the fp16 training graph is NaN from
  step 0; curvature work must use the detached-fp32 functional pattern
  (`src/instrument/hvp.py`), which the CF refresh now does.
- A joint third-order refresh over 24 directions on a full-batch graph OOMs
  a 48 GB card; penalty grads are additive, so chunk per matrix on a
  256-sample probe slice (`CentralFlowTerm.refresh_from_chunks`).
- cuSOLVER SVD fails on early ill-conditioned momentum buffers, and those
  buffers transiently contain non-finite fp16 values (stock Muon never
  notices — the polar update is scale-invariant; anything reading momentum
  numerically must sanitize).
- A non-finite refresh must never reach the weights (poison guard).

## 4. Costs / provenance

VM rm-mech-l40, ~9.7 h × $0.80 ≈ $7.76 (five smoke iterations on the CF GPU
path are the bulk of the wall time); $0.165 × 47 result files stamped.
VM deleted after pull; 0 active. All arms verified seed-paired on one GPU
type; `criteria/` untouched; no gate evaluated by this report.
