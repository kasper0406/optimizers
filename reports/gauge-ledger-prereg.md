# Pre-registration — Program #20: Gauge Ledger, Phase A

Registered 2026-07-24, before any radial/tangential computation on the Wave-1
artifacts and before any replay run. Source: `reports/ideation-geometry-
theory-2026-07-24.md` §1 program #1, human GO ("Please go ahead"). Committed
adaptations from the ideation draft are listed in §5 (seed counts to match
artifact availability; roster derived from source inspection).

**Disclosure of prior data contact:** the aggregate cos(v, D) = −0.68 (all
parameters pooled) and ‖D‖/‖v‖ ≈ 1.45 are known from the Wave-1 readout. No
radial/tangential decomposition, no per-matrix or per-block cosine, no
weight-norm trajectory, and no rescale evaluation has been computed.

## 1. Hypothesis

At weight decay 0, Muon's fixed-spectral-norm updates make per-matrix weight
norms grow, so the *effective* LR η_eff,m(t) = η(t)·‖ΔW_m(t)‖/‖W_m(t)‖ decays
even at constant η — a hidden schedule. On *scale-invariant* weight blocks
the radial component of any displacement is function-null. H_gauge: the
anomalous anti-alignment cos(v, D) = −0.68 between the constant-LR tail drift
(v = W2 − W1) and the anneal displacement (D = A_final − W2) is dominated by
this gauge (radial) component — the annealed and constant-LR runs disagree
mostly about *norm*, not *function* — and the tangential (function-relevant)
parts are near-orthogonal rather than anti-aligned.

## 2. Invariance roster (registered before computation)

From source inspection of `src/nanogpt/model.py` (the record architecture):

- **Primary roster (exactly scale-invariant, Muon-owned):** for each of the
  12 layers and each of the 6 heads, the Q head-block and the K head-block of
  the merged `qkv_w` parameter — `qkv_w[0][h·128:(h+1)·128, :]` and
  `qkv_w[1][h·128:(h+1)·128, :]` — 144 blocks of shape (128, 768). Exactness:
  `q, k = norm(q), norm(k)` (model.py:202) applies RMS norm over head_dim to
  the block's entire output; positive rescaling of the block is function-null.
- **Secondary (invariant but Adam-owned, excluded from all primary criteria,
  reported descriptively):** token-embedding rows (`x0 = norm(embed(...))`,
  model.py:400). Value embeddings are NOT invariant (added unnormalized).
- **Non-invariant (control group, reported alongside):** V slices, c_proj,
  MLP matrices.

## 3. Definitions

Per unit u (roster block or full matrix), base point W2_u (the arm-C tail
mean): radial direction r̂_u = W2_u/‖W2_u‖_F; radial part of X_u is
⟨X_u, r̂_u⟩_F · r̂_u; tangential part X_u − radial. v_u = W2_u − W1_u,
D_u = A_final,u − W2_u, from the stored Wave-1 fp32 artifacts. Aggregates
over a set S: cos_S(X_tan, Y_tan) = Σ_u⟨X_tan,u, Y_tan,u⟩ /
(√Σ_u‖X_tan,u‖² · √Σ_u‖Y_tan,u‖²); radial fraction of D on S =
Σ_u⟨D_u, r̂_u⟩² / Σ_u‖D_u‖².

Seeds with both required artifacts (arm A + arm C): **1511, 1512, 1513**
(n=3; Phase-B eval seeds have no arm-A weight artifacts).

## 4. Pre-registered criteria (Phase A PASSES iff all four hold)

(a) **Radial dominance on the roster:** radial fraction of ‖D‖² on the
primary roster ≥ 0.5 on **all 3 seeds**.

(b) **Tangential de-anti-alignment:** |cos(v_tan, D_tan)| < 0.2 on the
primary roster on **≥ 2 of 3 seeds**, with ‖D_tan‖/‖D‖ ≥ 0.3 (aggregate,
each seed) and a 10,000-resample bootstrap CI over roster blocks excluding
cos < −0.4 on every seed.

(c) **Perpendicular-update norm-growth law** (from replay runs, §6): median
over steps of |2⟨W_u,t, ΔW_u,t⟩| / ‖ΔW_u,t‖² ≤ 0.2 for ≥ 80% of primary
roster blocks (constant-LR replays, seeds 1511 and 1512), and the zero-fit
prediction ‖W_u(t)‖² = ‖W_u(963)‖² + Σ_{s<t} η_eff(s)²‖ΔW_u(s)‖²_F
reproduces the logged ‖W_u(t)‖² within 5% RMS relative error on ≥ 80% of
roster blocks.

(d) **Function-nullness in practice:** rescaling all primary-roster blocks
radially by ×1.1 (and, separately, ×0.9) changes full-val loss by < 0.0025,
evaluated on both the seed-1511 arm-C Polyak endpoint and the seed-1511
arm-A final endpoint (4 evaluations, forward passes only, window step 1750).

**FAIL** on (a), (b), or (d) closes the gauge-artifact route for the
anti-alignment (the −0.68 is then genuine functional opposition). (c) failing
alone degrades the hidden-schedule quantification but does not, by itself,
refute H_gauge; this asymmetry is registered now.

**Exploratory (labeled, no criteria):** gauge-transport evaluation — arm-C
Polyak endpoint with each roster block rescaled to the same-seed arm-A
per-block norm, full-val eval; per-layer and Q-vs-K breakdown of (a)/(b);
the same decomposition on the non-invariant control group (expected: low
radial fraction if H_gauge is specific); η_eff,m(t) reconstruction plots;
embed-row radial accounting.

## 5. Adaptations from the ideation draft (made at registration, before data)

- Seed thresholds restated for n=3 (draft assumed 4): (a) 3/3, (b) ≥ 2/3.
- Roster from direct source inspection: per-head Q/K blocks (144 units), not
  whole Q/K matrices; the draft's "train_gpt.py:1088" citation is replaced by
  the port's `model.py:202`.
- Base point for the radial direction fixed to W2 (both v and D are anchored
  there); the draft left it implicit.

## 6. Runs and instrumentation

- **Gauge probe (measurement-only):** a P6-style passive hook in `Muon.step`
  logging, per optimizer step and per Muon parameter (plus per primary-roster
  block), the scalars ‖W‖²_F, ⟨W, V⟩_F, ‖V‖²_F and eff_lr (V = the
  Newton-Schulz output before the −eff_lr·V application). Update path
  untouched; `gauge_probe` is excluded from both config fingerprints
  (measurement-only, trajectory-neutral — required so stored Wave-1 prefix
  checkpoints remain fork-compatible; unit-tested).
- **Replays (~1.7 GPU-h total, forked from stored step-963 prefixes):**
  constant-LR tail (min_lr_frac 1.0), gauge probe on, seeds **1511, 1512**;
  WSD tail (record schedule), gauge probe on, seed **1511** (descriptive
  η_eff comparison; not used in criteria).
- Everything else is CPU tensor arithmetic on stored artifacts plus 4 GPU
  forward-pass evaluations (§4d).

## 7. Costs and gates

Phase A ≈ 1.7 GPU-h + ~1 analyst-day (automated here). Phase B (open-loop
η_eff(t) replay as an explicit schedule; constant-η_eff arm) is **gated on
Phase-A PASS and a human gate**, with its own pre-registration; nothing in
Phase B launches from this document.
