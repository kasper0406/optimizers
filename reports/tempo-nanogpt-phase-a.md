# Program #8 — nanogpt Phase A: does the serial-correlation dial transfer?

2026-07-22. Passive probe (PORT CHANGE P6, measurement-only) on the local
nanogpt testbed: 4-rung muon_lr ladder {0.035, 0.05, 0.10, 0.15} =
{0.7×, 1×, 2×, 3×} record, 600-step stable-phase truncation (cooldown
starts at 962), seeds 1440–1441, 8 runs, babysat, zero retries.
Statistics per observed matrix per step: cos_gg = cos(G_t, G_{t-1})
(the airbench-calibrated signal) and cos_gm = cos(G_t, momentum_buffer)
(zero extra memory). Analysis `analyze_tempo.py nanogpt-passive` +
per-matrix Spearman slicing.

## Result

1. **The signal is strongly present at LM scale** — cos_gg sits at
   −0.55…−0.74 (deeper than airbench's −0.3…−0.5): heavy serial
   anti-correlation in the record recipe's gradient stream.
2. **But cos_gg is not an LR dial here.** Across a 4.3× lr range the
   aggregate moves ≤0.06 (airbench: 0.2 with 0.01 seed noise), with no
   monotone window; per-matrix orderings split both ways in every window
   (median Spearman −0.2…−0.4, large spread). The airbench dial does not
   transfer in its calibrated form. This continues the program-#7
   pattern: airbench-derived laws about the lr axis have not survived
   contact with the LM testbed.
3. **One live thread: cos_gm carries an early-window airbench-sign dial.**
   Steps 25–100: median per-matrix Spearman(lr, cos_gm) = **+0.80**,
   52% of matrices > +0.5 vs 9% < −0.5 (hotter → alignment toward zero,
   the airbench sign). It reverses/washes out by step ~100. Caveats:
   n=2 seeds, one 75-step window, 23 matrices — suggestive, not
   established.
4. Truncated-run losses order as expected (0.7×/1× ≈ 3.677/3.678 tied,
   2× 3.701, 3× 3.712) — consistent with program #7's valley at ~0.7×.

## Implications (descriptive; direction decision is the human's)

- A TempoMuon controller driven by cos_gg with the airbench calibration
  would be flying blind on nanogpt — do not launch controller-on runs
  with that signal.
- If the program continues at LM scale, the candidate design is:
  cos_gm signal, active window ≈ steps 25–100 only (~6% of the run),
  spike-gated (`reports/intermittency-scan.md` §3), global pool — and it
  first needs the early dial confirmed with more seeds (the truncation
  can be cut to ~150 steps, ~7 min/run, making a 4-lr × 6-seed
  confirmation ≈ 3 GPU-hours).
- The narrow actuation window bounds the achievable rescue: whatever a
  gain schedule discovered in the first 100 steps can fix must be damage
  that accrues early. Whether mis-set-LR damage is early-concentrated on
  this recipe is itself unmeasured.

Airbench remains the demonstrated domain of the controller
(`reports/tempo-eval.md`, n=100: free at 1×, +1.5pp at 4×).
