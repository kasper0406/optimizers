# Program #14 — hot training + tail repair: Phase A pre-registration

2026-07-22, committed before any run. Launched on the standing "proceed
with highest value initiatives" authorization. Dev-phase; adjudication
HUMAN. Novelty sweep (dated 2026-07-22, session record): the package —
measured tail-concentration of above-optimum-LR damage exploited by a
difficulty-targeted late phase, at matched budget — is unpublished;
nearest neighbors Blakeney 2406.03476 (end-of-run *quality* upsampling),
InfoBatch (restores *easy* examples at the end — the mirror image),
airbench96 (hard-selection uniform in time). Motivating measurement:
program #13 P2 — 52.9% of the peak→shoulder accuracy loss sits in the
hardest LOSO quintile.

## Intervention (recipe.tail_phase; training path untouched when absent)

In the final `tail_epochs` epochs, per batch: per-example CE losses;
the hardest `hard_frac` = 0.5 of the batch get weight 1/hard_frac = 2×,
the rest weight 0 (selection-with-rescale, InfoBatch-style expectation
preservation in reverse). Equal compute per step (no wall-clock claim
in Phase A — the mechanism test precedes any speed engineering).

## Arms (B = 1000, 8 epochs, seeds 1486–1495, n = 10 paired; 40 runs)

A stock (lr 0.24) · B hot (lr 0.48) · C hot + tail_phase(last 2 epochs)
· D stock + tail_phase(last 2 epochs).

## Pre-registered predictions

- **P1 (mechanism):** C − B ≥ +0.3pp paired (tail repair recovers part
  of the hot deficit; #13 anatomy predicts the recoverable pool is
  ~0.5pp of the ~0.95pp deficit). Refuted if C − B ≤ +0.1pp.
- **P2 (specificity):** the C-vs-B recovery is tail-concentrated —
  hardest-quintile examples (ranked by the #13 B=1000 LOSO difficulty
  where test-example overlap allows, else by within-run agreement)
  carry ≥ 40% of the recovered accuracy.
- **P3 (control):** D − A ∈ [−0.2, +0.2]pp — hard-upweighting late does
  not help an already-tuned run (if D > A materially, the effect is a
  generic curriculum, not hot-LR repair; disclosed reading).
- **P4 (escalation gate, HUMAN):** only if P1 holds and P3 shows
  specificity does the wall-clock version (shorter hot budgets,
  selection-for-speed) get proposed.
