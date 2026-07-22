# Program #12 — momentum-referenced data selection: decision record + Phase A

2026-07-22. Committed before any instrumented run.

## 0. Decision record

The steep/flat taxonomy identified the data measure as the only object
that is simultaneously steep (published ~2× token-efficiency ceilings),
high-dimensional (novelty available), and unmined by us. Novelty sweep
(2026-07-22): "optimizer-aware selection" as a *metric* with external
validation/proxy references is claimed (OPUS 2602.05400 — AdamW+Muon
preconditioners, pretraining scale, 2× token efficiency, 4.7% overhead,
mandatory baseline; LESS, CoLM, Filter-then-Weight). Momentum as the
*reference* exists only at toy scale (GMC 2605.05856: elementwise
|g·m/v| curiosity bandit, MNIST/MiniGrid). **Unpublished: selection
referenced to the momentum matrix's singular structure; selection driven
by per-matrix temporal statistics; any Muon-state-referenced selection;
deliberate anti-alignment selection; analysis of the selection→momentum
feedback loop** (the field's anchors — InfoBatch rescaling, Boltzmann
sampling, periodic refresh — are all untested when the anchor is the
optimizer state itself; GMC's plateau artifact is the one empirical
glimpse of the loop).

Known risks, stated up front: (i) the feedback loop — aligning selection
with m biases the buffer being read (program #9's endogeneity lesson,
now in data space; this is both the main hazard and the main
contribution opportunity); (ii) OPUS's open flank cuts both ways — if
optimizer-awareness adds nothing over plain gradient alignment there,
the same may hold here; (iii) all prior effect sizes are LM-scale data
efficiency; airbench (fixed 8-epoch budget, 10 classes, clean data) may
have little selection headroom — Phase A is sized accordingly (signal
characterization, not efficiency claims).

## 1. Phase A: characterize the signal before touching selection

Instrument (`recipe.example_probe`, measurement-only): every `every`
steps, in eval mode (BatchNorm running stats untouched — probe forwards
must not perturb training; grads under eval-mode BN are the documented
convention), compute single-example gradients for (a) a **fixed probe
set** of `n_fixed` training examples (same examples all run — gives
per-example alignment *trajectories*) and (b) `n_fresh` examples drawn
from the current batch (population statistics). Per example z, per Muon
matrix: cos(g_z, M) (momentum alignment), the fraction of ‖g_z‖²
captured by M's top-k singular subspace (k = 8, one subspace iteration
from the tracked machinery's conventions), and the example loss.

### Pre-registered measurements (dev seeds 1462+, lr ∈ {1×, 3×}, n = 4)

- **M1 (structure):** distribution of per-example momentum alignment —
  spread vs the shuffled-pair null (cos of example i's gradient with a
  *different* step's momentum); heavy-tailed or compact?
- **M2 (persistence):** is alignment a stable property of the example
  (fixed-set trajectories: between-example variance vs within-example
  across steps) or transient? Selection only makes sense if examples
  differ persistently on the signal.
- **M3 (redundancy):** correlation of momentum alignment with example
  loss — if |corr| is high, the signal is loss-selection in disguise
  (the strip-mined baseline) and the program pivots or stops.
- **M4 (spectral concentration):** is per-example alignment carried by
  M's top singular directions or broadband? (Determines whether the
  unpublished spectral variant differs measurably from GMC-style scalar
  coupling.)
- **M5 (lr dependence):** all of the above at 3× LR — connects the
  selection signal to the frontier programs' open thread (does the
  high-lr degradation change *which examples* align?).

Phase B (any actual selection arm) is specified only after M1–M5; it
requires M2-persistence and M3-non-redundancy to hold, and will
pre-commit the feedback-loop control (fixed-reference placebo: same
gain schedule of selection pressure, reference frozen at a checkpoint).
