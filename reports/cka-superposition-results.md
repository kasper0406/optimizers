# CKA-superposition Phase B results — the law survives a held-out 2× smaller batch

2026-07-22. Prereg `reports/cka-superposition-prereg.md` (committed
first). 70 runs (B = 500 relative ladder, seeds 1496–1505), zero
retries.

**Analysis correction (disclosed):** the Stage-2 ladders are not
uniform relative-LR ladders, so the pre-registered per-rung comparison
was computed on the law's actual coordinate x = log(lr / lr_peak(B))
with lr_peak(B) = 0.24·(B/1000)^0.35, interpolating the
B ∈ {1000, 2000, 4000} curves onto the B = 500 grid. A first
index-aligned computation was discarded as coordinate-misaligned (and
its verdict logic mishandled an uncovered rung); both versions are in
the session record.

**P1: PASS on the entire testable range.** At x ∈ {0.42, 0.56, 0.70,
0.84} (four of six rungs; the two smallest-x rungs are uncovered by
the older ladders — disclosed), the held-out B = 500 drift lies within
2× (three of four within 1×) of the interpolated master curve:
deviations 0.0014 / 0.0012 / 0.0003 / 0.0022 CKA against spreads
0.0020–0.0033. **P2: PASS** — B = 500's accuracy peak falls at the
law-derived anchor rung (k = 0: 93.68%, k = 1 tied at 93.66%).

**Standing of the law:** same-seed endpoint representation drift is a
function of relative LR position, log(lr / lr_peak(B)) with
lr_peak ∝ B^0.35, batch-free across B = 500–4000 (8×) on this
substrate — the first quantity in the project confirmed to superpose
across the batch axis under a pre-registered held-out test. It unifies
the B^0.35 frontier (the anchor), the CKA dial (the slope), and the
relative-lr/schedule-position theme (programs #2, #8) into one
measured law. Scope: airbench, Muon, dev seeds, endpoint drift; the
small-x regime and any LM transfer are untested (two prior
airbench→LM transfers failed; stated per house rules).
