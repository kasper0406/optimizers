# Program #16 results — averaging settles oscillation; the anneal does more

2026-07-22. Prereg `reports/readout-prereg.md` (committed first). 10 runs
+ ~300 offline readout evaluations; anchors verified (last@400 matches
endpoints; T-grid corrected for the 400-step schedule — an initial
200-step-grid analysis probed mid-run by mistake, disclosed; its numbers
were valid for those T and are superseded).

Tail iterate anti-correlation confirmed in weight space (rho1 −0.43…−0.46).

**All three pre-registered bars FAIL:**
- M1: best readout at 85% budget trails the full run by 0.75pp
  (bar 0.15pp); 0.33pp at 95%.
- M2: process-matched kernels ≈ EMA ≈ uniform (ema0.95 best by ≤0.2pp);
  matching the measured autocovariance adds nothing — the #9
  family-vs-matching decomposition, a third time.
- M3: at matched steps (250), the best truncated-hot readout (91.98%)
  loses by 1.17pp to the plainly re-tuned 5-epoch schedule (93.15%).

**What is real:** readout averaging recovers +2.4pp over the raw hot
iterate at 62% budget (+1.3pp at 75%) — the oscillation-settling
component is large and free. **What is refuted:** that this substitutes
for the anneal on this substrate. The low-LR phase makes genuine
progress along the valley (consistent with the cooldown-concentrated
loss removal measured at LM scale in WP0.2, and with #15's
river-position reading of the deficit) that averaging hot iterates
cannot synthesize. Scope note: this airbench negative coexists with
LM-scale positives for checkpoint-merging-replacing-decay (WSM,
2403.19390) — long-stable-phase WSD regimes are a different setting
from an aggressive tuned anneal-to-zero; the negative is claimed only
for the latter.

Tail-of-training conclusions across programs #14/#16 (+#10): on a
tuned small recipe, the anneal is irreducible by data reweighting,
iterate averaging, or mid-run LR control — its work is real progress,
not settling. Seeds 1506–1510 consumed; next fresh 1511+.
