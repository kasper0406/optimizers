# Wave-1 Phase B results — program #18, schedule-free tail graft

2026-07-24. Gate record: `reports/wave1-phase-b-gate.md` (human GO).
Registered quantities: `reports/wave1-phaseb.json` via
`scripts/analyze_wave1_phaseb.py` (committed before the data, `0778022`).
12 runs (eval seeds 1710–1713 × prefix/C/B), zero retries, single lane.
Arm A = the pre-existing n=10 baseline runs (same seeds, hot fingerprint
identical to the prefix config — same seed, init, and data order).

## Registered outcome

| seed | B = SF x̄ (κ=1, ρ=0.7) | A = tuned anneal | B − A | C_polyak | B − C_polyak |
|---|---|---|---|---|---|
| 1710 | 3.28776 | 3.28762 | +0.00014 | 3.30976 | −0.02201 |
| 1711 | 3.28730 | 3.28789 | −0.00060 | 3.30872 | −0.02143 |
| 1712 | 3.28904 | 3.28949 | −0.00045 | 3.31130 | −0.02226 |
| 1713 | 3.28695 | 3.28815 | −0.00121 | 3.30957 | −0.02262 |

- **Paired B − A: mean −0.00053, sd 0.00055, t = −1.92, 95% CI
  [−0.00141, +0.00035]** (n=4).
- Paired B − C_polyak: mean −0.02208, t = −88 — the closed-loop tail beats
  open-loop averaging of the identical trajectory by ~24 sd.
- Flatness (x̄ at 1375/1500 within 0.01 of endpoint): **fails on all four
  seeds** (x̄ is ~0.057 / ~0.035 above its endpoint at those points and
  still descending at 1750).

**Branch evaluation (registered, human verdict pending):**
- WIN (mean ≤ −0.0025, CI excl. 0): **not met.**
- KILL (mean ≥ +0.0025): **not met** — the CI upper bound (+0.00035)
  excludes any degradation ≥ 0.0025; parity is established at that bar.
- ANNEAL-REPLACED: parity ✓ and B ≪ C_polyak ✓, but the **flatness rider
  fails** → the branch as registered does not fire.

## Honest reading

The schedule-free tail graft is **statistically indistinguishable from the
tuned record anneal** at matched tokens (point estimate slightly better;
3 of 4 paired deltas negative), while **removing the decay schedule
entirely** (no cooldown_frac, no decay shape, no decay length — κ and ρ
remain, with κ=1 meaning "just keep the plateau LR"). The feedback
ingredient is decisively real: +0.022 over open-loop averaging of the same
constant-LR trajectory, the controlled comparison the ideation vetting asked
for (and the cell WSM left open).

What did *not* materialize is the anytime property: x̄ approaches the anneal
endpoint on an anneal-like arc rather than converging early. At this horizon
the graft is an anneal *equivalent*, not an anneal *transcendent* — and dev
evidence (x̄ still descending at 1750) suggests the interesting open question
is whether it keeps improving *past* the point where the record's schedule
must stop.

## Proposed next steps (human gates)

1. **Longer-tail probe (dev seeds 1514+, ~2 GPU-h):** extend one SF tail to
   2×787 steps at constant LR. If x̄ drops below the anneal endpoint by
   > 0.0025 with the extra budget, the framing shifts from "matches the
   anneal" to "the anneal is what stopping early costs you" — a stronger
   claim no schedule can replicate without knowing the horizon.
2. **Paper section:** parity-with-no-schedule + the +0.022 feedback margin +
   the Wave-1 mechanism decomposition (76% averaging / feedback remainder /
   anti-aligned drift, cos = −0.68) is a coherent, fully pre-registered
   story. Recommend drafting it into `reports/paper-draft.md` §method-note.
3. Reboot the box (unkillable R-state FIR zombie from program #9 holds the
   bus-01 GPU; `kill -9` ineffective) before any further compute wave.
