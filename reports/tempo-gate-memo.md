# Program #8 (TempoMuon) — gate decision memo

2026-08-31. Decision material only: this memo states the measured record and
argues the three exits. **It contains no recommendation** — the gate judgment
is human-only (CLAUDE.md ground rule 1). The gate fields themselves are
blank in `criteria/tempo_gate.yaml.template` (human authors `criteria/`;
agents may only draft templates). No new runs were made for this memo; every
program-#8 number below was recomputed read-only from `results/` and agrees
with the committed reports to the digits they print. The one non-#8 result
cited (program #18, §4) is sourced the same way where `results/` allows and
otherwise quoted from its registered artifact, as noted at the point of use.

Status: parked since 2026-07-22 for want of this decision
(`reports/project-state.md` §6 item 6a).

## 1. The eval table (the thing being gated)

n = 100 **seed-paired** eval seeds 0–99 per cell: for each rung, the same seed
index runs both arms and Δ is formed within seed; SE = sd(paired Δ)/√100;
CI is a paired t interval, t₀.₉₇₅,₉₉ = 1.984. 800 runs, 1.17 GPU-h total
(≈5.2 s/run), single GPU type (RTX 5090), airbench_smoke record recipe
(8 epochs, batch 2000, 200 steps), endpoint `tta_val_acc`. Treatment
`configs/tempo_eval_global.yaml` (κ −0.25, ρ\* −0.48, window [25, 60],
gain ∈ [0.2, 1], global pool), frozen on dev seeds at `30a32ee` **before**
launch; control `configs/tempo_eval_muon.yaml`.

| lr | stock Muon (sd) | TempoMuon-global (sd) | Δ paired ± SE | 95% CI | t | sign wins | stock deficit vs own 1× | deficit recovered |
|---|---|---|---|---|---|---|---|---|
| 0.24 (1×, record) | 0.93990 (0.00133) | 0.93991 (0.00134) | **+0.001pp ± 0.016** | **[−0.031, +0.033]** | 0.06 | 51/100 | — | — |
| 0.48 (2×) | 0.93456 (0.00118) | 0.93701 (0.00137) | +0.245pp ± 0.017 | [+0.210, +0.279] | 14.07 | 90/100 | −0.534pp | 45.9% |
| 0.72 (3×) | 0.92616 (0.00157) | 0.93459 (0.00160) | +0.842pp ± 0.022 | [+0.799, +0.886] | 38.31 | 100/100 | −1.373pp | 61.3% |
| 0.96 (4×) | 0.91766 (0.00195) | 0.93302 (0.00173) | **+1.536pp ± 0.024** | **[+1.488, +1.585]** | 62.78 | 100/100 | −2.224pp | 69.1% |

- **Record LR:** the two-sided 95% CI is [−0.031, +0.033]pp, so under a
  CI-containment rule the smallest symmetric equivalence margin this table can
  clear is **0.033pp**; under TOST at α = 0.05 (90% CI [−0.026, +0.028]) it is
  **0.028pp**. Those bounds, not the point estimate, are what a chosen margin
  must be compared against.
- **4× LR:** +1.536pp [+1.488, +1.585]; the controller arm still sits
  −0.688pp ± 0.022 below stock's own 1× accuracy, i.e. rescue, not repair.
- **Step-time cost (recomputed from the same 800 files;
  `scripts/analyze_tempo_overhead.py`):** the controller is accuracy-free at
  1×, not compute-free. Paired train time
  (`metrics.time_seconds`) at lr 0.24 is 3.7621 s stock vs 3.8883 s TempoMuon,
  paired Δ **+0.126 s ± 0.014** (+3.36%, t = 8.8, 95% CI [+0.098, +0.155]),
  and the overhead holds at every rung (+3.36 / +3.30 / +3.18 / +2.73% at
  1×/2×/3×/4×) and in `wall_time_s` (+3.14% at 1×). It is the cost of *this*
  implementation — one `.item()` sync per matrix per step plus a prev-gradient
  buffer (`src/optim/tempomuon.py`), with `history_every: 10` telemetry on in
  the treatment arm and absent from the control — so it bounds what the measured
  configuration costs, not what the signal must cost. It matters because the
  testbed is a wall-clock speedrun tier and because the gate's (a) field
  (`criteria/tempo_gate.yaml.template`, `free_at_record_lr`) is an accuracy
  margin only.
- Provenance check: config sha256 is constant within every cell. The treatment
  arm ran entirely at code `b1a2973`; the control arm spans `30a32ee`,
  `ed79b9e`, `b1a2973`. Diffing those commits: they add only configs,
  `src/nanogpt/*`, `scripts/analyze_tempo.py` and tests — the airbench
  train/optimizer path is unchanged, so the control arm is code-identical
  across its three SHAs.

## 2. Mechanism decomposition (Phase B, dev seeds 1420–1429, n = 10 paired)

The pre-committed placebo replays the closed-loop arm's **mean gain
trajectory** as a fixed open-loop schedule (κ = 0 + `gain_schedule`), feedback
switched off, on the same seeds:

| lr | closed loop Δ vs stock | open-loop replay Δ vs stock | closed − replay (paired) | 95% CI |
|---|---|---|---|---|
| 0.24 | −0.258pp ± 0.052 | −0.185pp ± 0.035 | −0.073pp ± 0.054 | [−0.196, +0.050] |
| 0.48 | +0.427pp ± 0.048 | +0.457pp ± 0.041 | −0.030pp ± 0.050 | [−0.142, +0.082] |
| 0.72 | +1.110pp ± 0.048 | +1.131pp ± 0.075 | −0.021pp ± 0.054 | [−0.143, +0.101] |
| 0.96 | +1.918pp ± 0.065 | +1.807pp ± 0.079 | +0.111pp ± 0.063 | [−0.033, +0.255] |

**Finding, stated accurately:** the effect is carried by the *discovered gain
schedule*, not by per-step feedback. The feedback-attributable component is
statistically indistinguishable from zero at all four rungs; at 4× its CI
bounds it to ≤ +0.26pp — ≤ 13% of the +1.92pp total (point estimate 5.8%).
It is **bounded, not zeroed**: n = 10, dev seeds, and the replay arm was never
repeated at n = 100. Feedback's only demonstrated role is *finding* the
LR-appropriate schedule online without being told the LR is mis-set.

Two adjacent results bear on the framing: (i) **granularity** — global pooling
beat per-matrix at every rung (4×: +1.918 vs +1.487pp; 1×: −0.258 vs
−0.580pp), so the pre-registered novelty cell "per-matrix × temporal" is
refuted by our own data; (ii) **endogeneity** — program #9 measured ρ̂₁ *inside*
FIRMuon at +0.10 (range +0.06…+0.22) versus −0.3…−0.5 under stock Muon: the
signal is a closed-loop property of the (momentum kernel, landscape) loop, not
of the gradient stream. The window-60 fix that produced the frozen config was
labeled exploration on fresh dev seeds 1430–1439 (Δ = −0.064 / +0.240 / +0.797
/ +1.602 pp across the ladder), and the n = 100 table reproduced it.

## 3. Scale transfer (nanogpt Phase A, passive probe, `reports/tempo-nanogpt-phase-a.md`)

Largely negative. 4 muon_lr rungs {0.035, 0.05, 0.10, 0.15} = {0.7×, 1×, 2×,
3×} record, seeds 1440–1441, 8 runs, 600-step stable-phase truncation.

- The signal is *present and stronger*: cos_gg = −0.55…−0.74 (airbench:
  −0.3…−0.5).
- **It is not an LR dial there:** the aggregate moves ≤ 0.06 across a 4.3× LR
  range (airbench moved 0.2 with 0.01 seed noise), with no monotone window;
  per-matrix orderings split both ways in every window (median Spearman
  −0.2…−0.4, large spread). The airbench-calibrated controller would fly blind.
- **The one exception:** the zero-memory variant cos_gm = cos(G_t, momentum)
  carries an airbench-sign dial in the early window only — steps 25–100, median
  per-matrix Spearman(lr, cos_gm) = **+0.80**, 52% of 23 matrices > +0.5 vs 9%
  < −0.5 — which reverses/washes out by ~step 100. n = 2 seeds, one window:
  suggestive, not established.
- Truncated losses order as expected (3.677 / 3.678 / 3.701 / 3.712).

## 4. The three exits

One result outside program #8 bears on all three exits: **program #18**
(schedule-free tail graft, `reports/wave1-phaseb-results.md`, 2026-07-24;
human GO gate at `reports/wave1-phase-b-gate.md`). On the nanogpt record
recipe, n = 4 paired seeds 1710–1713, the schedule-free tail reaches **parity
with the tuned record anneal** — paired B − A = **−0.00053**, 95% CI
[−0.00141, +0.00035], so the registered KILL bar (≥ +0.0025) is excluded and
the WIN bar is not met — **while removing the decay schedule entirely** (no
cooldown_frac, no decay shape, no decay length), and it beats open-loop Polyak
averaging of the identical trajectory by **0.02208 val loss** (paired
B − C_polyak = −0.02208, t = −88). Provenance: B − A recomputes from
`results/` (per-seed `final_val_loss`, 3.28776 / 3.28730 / 3.28904 / 3.28695
vs baseline 3.28762 / 3.28789 / 3.28949 / 3.28815); C_polyak is quoted from
the registered artifact `reports/wave1-phaseb.json`, since it was produced by
forward passes over the arm-C tail artifact.

Two qualifications belong with it. Its registered ANNEAL-REPLACED branch did
**not** fire — the flatness rider failed on 4/4 seeds — so the graft is an
anneal equivalent, not an anneal transcendent, at this horizon. And the two
programs' open-loop controls are *not* the same test: §2's placebo replays #8's
own discovered gain trajectory with feedback off, whereas #18's C_polyak
averages the constant-LR iterates rather than replaying #18's discovered
readout trajectory, so #18's feedback ingredient has never faced a #8-style
placebo. What it does establish is that the paper's method slot is contested:
#18's own report proposes drafting it into `reports/paper-draft.md`
§method-note, and nothing of #18 is in the paper today (§5 is program #8's).

### Exit A — paper method section (keep as `paper-draft.md` §5)
- **Strongest support:** the section is already written, including the placebo,
  the granularity refutation and the negative-transfer disclosure, so the
  drafting cost is sunk. It is the project's only intervention with a positive
  eval-seed accuracy table against stock (routing Gate 2, FIRMuon #9, #11, #20,
  #22 are nulls/negatives; #14, #16, #17 and #19 closed on registered
  FAIL/PARTIAL branches at dev-seed scale) — program #18 above is the other
  positive, but its claim type is parity-plus-hyperparameter-removal, not an
  accuracy gain over stock. It is n = 100 seed-paired with a frozen
  pre-committed config, effect sizes are 60σ at 4×, and the temporal
  trust-ratio slot was re-verified open on 2026-07-22 with only adjacent
  neighbors (GALA, CLARA, MGUP-Muon).
- **Strongest opposition:** the Gate-2 conditional approval
  (`reports/gate2-decision.md`) made this direction contingent on a
  `criteria/`-committed, human-audited protocol **and** mandatory external
  baselines (OrScale, NAMO/LANTON-style noise scaling, GALA, Prodigy, hand-tuned
  schedule). Neither happened: the prereg lived in `reports/`, and zero external
  comparators have run. Shipping §5 puts a method claim with no external
  baseline inside a measurement paper, and changes how the whole paper is
  reviewed (method papers get benchmarked); the honest comparator for a
  "mis-set LR rescue" is "sweep the LR", which is cheap on this testbed.
  The novelty argument is also weaker than the support bullet's "slot open"
  suggests: what was re-verified open on 2026-07-22 is the *per-matrix* ×
  temporal cell (`docs/litreview/b-layer-temporal-trust-ratio.md`,
  `reports/tempo-phase-a.md` §0), and §2 records that #8's own data refutes
  that cell (global pooling beat per-matrix at every rung). What §5 would
  claim is a global temporal scalar — the cell GALA/CLARA already occupy — so
  the section's novelty rests on the mechanism decomposition and the
  equivalence/rescue table, not on an unoccupied slot. The sunk-cost framing
  above also covers the writing only: A's marginal cost is not ≈ 0, because
  choosing it *binds* the Gate-2 baseline set plus a hand-tuned-schedule arm as
  pre-submission work (§5), and a benchmarked method section has to report the
  +3.4% step-time overhead of §1. Finally the slot itself is contested —
  program #18 is a competing, pre-registered, LM-scale candidate for the same
  method-note space.

### Exit B — standalone note
- **Strongest support:** it isolates the positive result so the measurement
  paper keeps its identity, and it can be scoped honestly in its own title —
  airbench-only, rescue-only, mechanism = online schedule discovery. The
  placebo-controlled statement "this adaptive method's entire effect is
  schedule discovery" is itself the interesting content and reads better
  standing alone than as a caveat inside §5. Everything it needs is local and
  small in GPU-hours: of §5's five items, the two costed ones are the nanogpt
  cheap-confirm (≈ 3 GPU-h) and the eval-seed replay arm (≈ 0.6 GPU-h), and no
  cloud spend or external harness is involved.
- **Strongest opposition:** the same missing-baseline defect as Exit A, just at
  smaller scale; a standalone method note resting on a 94%-tier CIFAR speedrun
  at a 200-step horizon and a single GPU type is a weak venue fit; and the LM
  thread that would give it scale is n = 2 seeds, one 75-step window, and needs
  a controller design (cos_gm, spike-gated, global pool) that has never been run
  even once. It also splits one day of compute into two submissions. Its §5 cost
  is five items, not the two costed runs the support bullet names, and two of
  them are prerequisites rather than polish: no controller-on LM run is
  defensible without the spike-gate (`reports/intermittency-scan.md` §3 — a
  spike pushes cos toward 0, which the controller reads as "too hot"), and the
  placebo decomposition that *is* the
  note's headline content exists only at n = 10 on dev seeds until the 400-run
  eval-seed replay arm runs. The spike-gate, the hand-tuned-schedule arm and
  Phase B″ are uncosted. The same method-note slot is contested by program #18,
  which is already at LM scale.

### Exit C — drop (park permanently; code and reports stay as record)
- **Strongest support:** the program's own placebo reduces the claim to "a
  discovered LR schedule fixes a mis-set LR", and the natural baseline — a
  hand-tuned earlier anneal — has never been run, so the result may be
  reproducible by a one-line schedule change. The pre-registered novelty cell
  was refuted internally (global > per-matrix); what remains is a global
  temporal scalar, which is the cell GALA/CLARA already occupy. Transfer is
  negative on the calibrated signal at LM scale, and program #9 showed the
  signal is optimizer-endogenous (change the kernel and ρ̂ inverts) — a
  fragility argument against building on it. What is left (external baselines,
  setpoint transfer, spike-gate, LM confirm) is a program, not a finish — and
  the paper slot #8 would vacate has a second, fully pre-registered candidate
  (program #18) that does not depend on #8.
- **Strongest opposition:** C's load-bearing claim — that a hand-tuned earlier
  anneal is the natural substitute — is not available in the setting the result
  is about. §2's finding is that feedback's demonstrated role is *finding* the
  LR-appropriate schedule online without being told the LR is mis-set; the
  replay arm exists only because the closed loop discovered the LR-specific
  trajectory first. A hand-tuned schedule therefore presupposes exactly the
  knowledge the rescue scenario denies: it is the right comparator for the
  achievable-accuracy claim, not a replacement for the mechanism claim. Beyond
  that, C discards a well-powered, placebo-decomposed, frozen-config result that
  cost 1.17 GPU-h and is already written up; the 1× equivalence finding alone
  (accuracy cost bounded to ±0.033pp at n = 100) is a usable standalone
  statement that a temporal-statistic gain can be switched on at no measured
  accuracy cost at the record LR — though not for free in compute, at +3.4%
  step time in this implementation (§1); and the artifacts
  (`src/optim/tempomuon.py`, bit-identical to Muon at κ = 0, unit-tested) then
  go unread.

## 5. What each exit unblocks

- **A (method section):** unblocks paper submission on the §5 content, but
  *binds* the Gate-2 mandatory-baseline set (OrScale, NAMO/LANTON noise scaling,
  GALA, Prodigy) plus a hand-tuned-schedule arm as pre-submission work, and
  binds a scoping statement. It does not require the nanogpt work; the LM thread
  stays a future-directions line. GPU lane otherwise freed.
- **B (standalone note):** makes the **nanogpt cheap-confirm the first thing
  worth running** — 4 muon_lr × 6 seeds at 150-step truncation, ~7 min/run,
  **≈ 3 GPU-h**, on fresh dev seeds (the #8 airbench blocks 1420–1439 and the
  nanogpt block 1440–1441 are consumed) — plus, in order: the spike-gate
  (`reports/intermittency-scan.md` §3) before any controller-on LM run, the
  eval-seed replay confirmation (a 400-run replay arm ≈ 0.6 GPU-h; the stock and
  closed-loop arms already exist on seeds 0–99), the hand-tuned schedule
  baseline, and Phase B″ (self-calibrating setpoint, 2× under-rescue).
- **C (drop):** unblocks nothing new but closes `project-state.md` §6 items
  6(a)–(d) and returns the lane to the ranked next steps — occupancy-triggered
  cooldown setup, per-direction SNR vs batch size, and finishing the measurement
  paper.

## 6. How to record the decision

Fill `criteria/tempo_gate.yaml.template` → `criteria/tempo_gate.yaml` (human):
(a) the record-LR equivalence margin, (b) the 2×/3×/4× rescue minima, (c) the
yes/no on whether the schedule-discovery mechanism disqualifies the "controller"
framing, (d) the disposition rule mapping (a)–(c) to A/B/C. The template
discloses that this gate is post-hoc: the table in §1 already exists, so every
threshold is authored with the outcome visible.

One coverage gap to decide explicitly rather than by omission: the (a) block
(`free_at_record_lr`) is an accuracy margin only, and no field in the template
covers the +3.4% step-time cost measured in §1. If "free" is meant to include
compute on this wall-clock testbed, that has to be said in `notes` (or in a
field the human adds); as the template stands, a PASS on (a) is a statement
about accuracy alone.
