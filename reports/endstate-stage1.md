# Program #13 Stage-1 results — the endpoint carries the signal the trajectory hides

2026-07-22. Pre-registration `reports/endstate-prereg.md` (finalized with
dated novelty sweep, commit 8908469, before any run/checkpoint/fit).
130 runs total (70 ladder + 30 placebo incl. the pre-registered
contingency budget + 10 replicates + 20 smoke/bracket), single GPU, zero
retries. Durable record: `reports/endstate-features.json`; checkpoints
were scratch-only and are deleted per §8. Adjudication of all branches
is HUMAN; this report states numbers and the pre-committed readings.

## Track S (zero-GPU spectral screen): 0/3 — the pre-registered null

All three per-direction spectral summaries (centroid, high-frequency
fraction, flatness) are near-constant across the entire (lr × batch)
grid (validation shoulder ratios 1.05–1.14, within-B lr-variation
1.02–1.03; permutation null 0.00). The #11 exclusion extends from
trajectory scalars to per-direction spectral shapes. Implementation
note (disclosed): the prereg's refresh-harmonic notch is inapplicable at
these window lengths (each periodogram lives within one refresh period;
the notch bands would blanket the spectrum) and was dropped.

## P1b — PASS on both geometry observables: the existence result

| observable | rung means (lr 0.24 → 0.64) | Spearman | swing vs 3× pooled sd |
|---|---|---|---|
| O4 endpoint λ1 | 52.0k → 120.3k | **+1.00** | 68.2k vs 54.4k — PASS |
| O5 CKA to same-seed peak | 0.941 → 0.918 | **−1.00** | 0.0224 vs 0.0065 — PASS |

After programs #10/#11 showed every *trajectory* statistic is locally
blind to LR within the frontier region, the **end state is not blind**:
the annealed residue of edge-of-stability sharpness survives lr → 0 and
encodes LR dose monotonically (extending Kaur–Cohen–Lipton 2206.10654
from SGD/end-of-schedule to Muon/post-anneal, as pre-positioned), and
representations drift monotonically away from the peak-rung solution.
K1 (geometry blindness) does not fire.

## P1a (descriptive anatomy — no claim)

All four prediction-space observables trend monotonically across the
ladder: margin-tail q10 +0.62 → +0.29, hardest-quintile accuracy
0.705 → 0.655, linear-probe accuracy 0.928 → 0.909, and the
(label-smoothed) train−test gap −0.087 → −0.065.

## P2 — PASS: the shoulder loss is memorization-tail-concentrated

The hardest LOSO quintile carries **52.9%** of the peak→shoulder
per-example accuracy loss (bar ≥ 40%; uniform null 20%). Half the
shoulder damage lives in the hardest fifth of examples.

## P3 — MIDDLE BAND (pre-committed reading applies)

Placebo matching (disclosed): integer-epoch schedule quantization floors
the achievable arm-mean match at ~0.2pp (epochs 4 → 92.70%, 6 → 93.55%,
target 92.91%); the pre-registered contingency budget was run and the
final comparison uses the *bracketing* {4, 6}-epoch set, making the
mandatory accuracy-regression an interpolation. Result: every
prediction-space observable collapses after the regression (0.24–0.84×
— they re-encode lost progress, exactly as the prereg's class-label
analysis anticipated), while **endpoint λ1 retains a 2.40× separation**
(raw: 76.2k at the shoulder vs 56.0k in undertrained runs of comparable
accuracy) — the 2–3× middle band. Pre-committed reading: suggestive,
reported, **Stage 2 does not launch**; the human decides whether one
further placebo budget is worth ~10 runs.

## P5 — violated, informatively

Same-seed relative weight distance to the peak rung is **0.53 → 0.72**,
rising smoothly with rung — far below the pre-registered [0.9, 1.1]
confirmation band taken from program #1's equivalent-destinations twins
(0.95–0.99). LR perturbation moves endpoints along a *graded,
dose-dependent path* in weight space; spectral interventions scattered
to unrelated points. The pre-committed non-finding became a finding:
the two intervention classes have qualitatively different endpoint
geometry.

## Anchors and instrument disclosures

- Replicate noise floor reproduced: 0.057pp (n = 11 same-seed) vs the
  #10 anchor 0.061pp — environment consistent.
- Non-perturbation smoke: the registered |Δacc| < 0.05pp bar is
  structurally mis-posed for these hooks (they execute after
  tta_val_acc is computed and cannot causally affect it — verified in
  code order); observed paired deltas (incl. one 0.43pp pair) are
  compile-mode run nondeterminism, consistent with the replicate floor
  under torch.compile. Steady-state wall overhead +1.9% (bar < 5%,
  pass); the first run pays torch.compile warm-up (cold 21s), excluded
  as a cache artifact.

## Status

P1b existence: PASS ×2. P2 anatomy: PASS. P3: middle band. P5: violated
(informative). K1/K2 do not fire; the situation is nearest K3
(anatomy + geometry signal without a completed placebo separation), and
per the prereg every routing decision from here — further placebo
budget, Stage 2 (B ∈ {2000, 4000}) toward P4's invariant candidacy, or
paper-fold as an extension of finding 6 — is the human's.
