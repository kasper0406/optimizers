# Program #15 results — the dose path is coherent, flat, and barriered

2026-07-22. Prereg `reports/dosepath-prereg.md` (committed before runs).
40 runs (4 rungs × seeds 1476–1485, B = 1000), zero retries; measurement
script `scripts/analyze_dosepath.py`; durable record
`reports/dosepath-features.json`; checkpoints scratch-deleted.

## Measurements vs pre-registered bars (n = 10 seeds, extremely consistent)

- **M1 mode connectivity: BARRIERED.** Linear interpolation between
  same-seed peak (lr 0.24) and shoulder (0.48) endpoints, with BN-stat
  repair (endpoints reproduce their true accuracies, validating the
  repair), dips to a **−14.9 ± 0.8pp barrier at the midpoint**
  (path ≈ 0.939 → 0.923 → 0.777 → 0.915 → 0.929). Far past the −5pp
  "barriered" bar; the graded dose distances of program #13's P5 do
  **not** imply a shared linear basin.
- **M2 straightness: partially coherent (cos ≈ +0.50).** Between the
  chord (≥ 0.7) and independent (≤ 0.3) bars — but hugely above the
  ~0.003 random-cosine baseline at this dimensionality: nested dose
  displacements share about half their direction. A persistent common
  dose component exists; the path curves.
- **M3 sharp-or-sloppy: FLAT (ratio ≈ 0.000).** The normalized dose
  direction has essentially zero Rayleigh quotient in the shoulder
  endpoint's Hessian (λ1 ≈ 76k for scale). The displacement lives in
  locally flat directions — yet M1's barrier shows the flatness is
  local only: two individually-flat endpoints separated by a ridge.
- **M4 layer profile:** recorded in the features JSON (descriptive).

## Pre-committed gate: the repair path is closed

The interpretation gate required M1-connected AND M3-sloppy before any
weight-space repair-operator proposal. M1 fails decisively, so **no
repair operator is proposed**: linear arithmetic on same-seed cross-LR
endpoints (interpolation, dose-vector subtraction, souping) crosses a
~15pp ridge and cannot work in this form. The geometric picture that
remains — coherent-but-curved dose paths between locally-flat,
ridge-separated minima — is itself a clean characterization: LR dose
moves solutions along a *consistent direction family* (M2) through
territory that linear operations cannot shortcut (M1).

## Scope and claims

**Novelty-sweep verdict (appended 2026-07-22):** cross-LR linear
connectivity is *partially* measured — Sadrtdinov et al. (2410.22113,
NeurIPS 2024) tested it for fine-tuning-LR ladders from a shared anchor
in a deliberately scale-invariant SGD-on-sphere setting (binary
connected/not, regime-dependent) — so the phenomenon may not be claimed
as unmeasured. Our deltas that survive: from-scratch full-run same-seed
ladders; Muon (the Muon-connectivity paper 2605.09991 is cross-seed /
cross-optimizer only); the quantitative graded-path structure
(monotone distance, cos ≈ 0.5 coherence, straightness); BN-repaired
barriers in a standard network; and the intervention-class contrast
(graded LR path vs program #1's scattering). The M3 flat-displacement
result should be positioned as the first direct Hessian-eigenspace test
of the river-valley prediction on Muon (Wen et al. 2410.05192;
Muon-specific 2606.21514): post-anneal, the wall-oscillation component
has been annealed away and the dose displacement is pure
river-position deficit — which is exactly what ratio ≈ 0.000 shows.
The strongest unclaimed operator in the neighborhood — a
hyperparameter-dose damage vector subtracted CUTS-style — is precisely
what our M1 barrier closes in linear form; nonlinear versions remain
open but unmotivated by our data. Must-cite: Keskar 1609.04836 (the
batch-size interpolation ur-precedent), Altintas 2312.09832 (LR matched
within pairs — the explicit gap), 2510.04686 (competing noise-scale
dose framing, across seeds), 2605.28751 (smooth frontier along a
training-condition axis). Dev-phase, one substrate, B = 1000,
same-seed pairs only; cross-seed weight comparisons deliberately
excluded (permutation confound, per prereg). Seeds 1476–1485 consumed;
next fresh 1486+.
