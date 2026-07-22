# Program #13 Stage-2 results — no endpoint invariant under the registered criterion; two structural discoveries in the failure

2026-07-22. Stage 2 launched on explicit human direction ("go with the
route with highest potential") with P3 in the pre-committed middle band —
a disclosed deviation from the mechanical full-P3-pass rule; the launch
authorization is the human routing decision the prereg reserves. 150
runs (dense ladders B ∈ {2000, 4000} × 10 seeds + the third placebo
budget + the B = 1000 peak-arm restoration), zero retries. Extraction
merge policy added: scratch-deleted checkpoints keep their committed
features (disclosed; `analyze_endstate.py`).

## P4 — 0/7: the endpoint-scalar panel joins the excluded span

Under the registered tracking signature (shoulder ratio ≤ 1.5 across
B ∈ {1000, 2000, 4000} AND within-B lr-variation ≥ 3.0), **no observable
and no pre-declared composite passes** (permutation null: 0.054 expected
passes, q95 = 1 — the zero is meaningful):

| candidate | shoulder ratio | min lr-var |
|---|---|---|
| O1 margin q10 | 1.59 | 2.15 |
| O2 hard-quintile acc | 1.08 | 1.08 |
| O3 probe acc | 1.01 | 1.02 |
| **O4 endpoint λ1** | **1.90** | 1.94 |
| O5 CKA to peak | 1.01 | 1.02 |
| lr·λ1 (composite; eligibility failed anyway — see below) | 2.89 | 3.39 |

Headline number: λ1 at the three shoulders is **76.2k / 81.6k / 144.5k**
— endpoint sharpness is *not* equalized along the useful-LR frontier
(B = 4000 sits at ~1.9× the others). Combined with #11, the frontier
invariant is now excluded from: trajectory scalars, their log-linear
pairs, per-direction spectral summaries, and this endpoint-scalar panel.

Criterion-design disclosure: the ratio-based lr-variation bar (≥ 3.0) is
structurally unreachable for bounded-near-1 observables (O2/O3/O5 vary
by 3–5 seed-sds while their max/min ratios stay ≤ 1.08); the registered
criterion inherited from #11 is a magnitude-ratio test and O-panel
failures of that specific form should be read accordingly.

## Two structural discoveries inside the failure (exploratory, post hoc)

1. **The λ1 dial breaks at B = 4000.** At B ∈ {1000, 2000} endpoint λ1
   is a clean monotone LR dial (Spearman +1.00 both, similar scales:
   52→120k, 45→100k). At B = 4000 (≈100-step schedules) it shatters:
   168, 87, 108, 144, 137, 99, 145k across adjacent rungs — Spearman
   0.0, large scatter. Short-schedule endpoints have unstable sharpness;
   whatever λ1 measures at B ≤ 2000 endpoints is not yet formed (or not
   yet annealed away) at B = 4000's step counts.
2. **CKA drift superposes across batches.** The same-seed
   representation-drift curves are nearly identical at every batch
   (0.941→0.918 / 0.938→0.912 / 0.939→0.909 across the seven
   ×1.15-spaced rungs): drift is a function of the rung's *relative*
   position (lr / lr_peak(B)), independent of batch — the first
   quantity in this project observed to superpose across the batch
   axis. This is the endpoint-level analogue of program #2's
   "occupancy tracks schedule position", is NOT a pass of the
   registered magnitude-ratio test (see disclosure above), and is
   flagged as the natural pre-registrable target for any Phase B: test
   CKA-drift superposition as an *invariance in relative-lr form*
   rather than magnitude-ratio form.

## P3 (final, three-budget bracketing placebo, n = 30)

λ1 separation after the mandatory accuracy-regression: **2.47×**
(raw 76.2k shoulder vs 55.7k accuracy-matched undertrained); every
prediction-space observable ≤ 1.02×. The middle-band conclusion firms:
high-LR exposure leaves a real, moderate sharpness residue distinct from
lost progress — below the registered 3× pass, far from the < 2×
refutation, robust to placebo-budget composition {4}, {4,6}, {4,5,6}.

## P6

Monotone CKA decline holds at every batch (see superposition above);
the shoulder-below-replicate-floor clause is reported with a caveat —
the stored replicate-pair CKA floor mixes same-seed and cross-seed
pairs (disclosed) — and is not load-bearing for any claim here.

## Routing (adjudication HUMAN)

The outcome is K3-shaped: anatomy (P2), geometry-dial existence (P1b),
and a suggestive pathology signature (P3 ≈ 2.5×) without a registered-
form invariant (P4 0/7). Pre-registered K3 consequence: the
decomposition becomes a paper subsection extending findings 6/9; no
escalation of the invariant hunt in its current scalar-ratio form. The
one concrete Phase-B candidate this stage surfaced — CKA-drift
superposition in relative-lr form — awaits a fresh human-reviewed
pre-registration per §7.

Seeds: 1466–1475 (both stages); next fresh 1476+. Checkpoints deleted;
durable record `reports/endstate-features.json` (275 runs).
