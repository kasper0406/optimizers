# Wave-1 Phase-B gate record — program #18 (schedule-free tail graft)

2026-07-23. **Human decision: GO** ("Let's do 1 if that's the recommendation",
approving recommendation #1 of `reports/wave1-dev-results.md` §5). Programs
#17 (FAIL) and #19 (PARTIAL + guard fail, closed at Stage 1) do not proceed;
the #19 hybrid is shelved per the same recommendation.

## Registered Phase-B design (verbatim scope from the prereg §2 + dev report §2)

- **Arms**, per eval seed s ∈ {1710, 1711, 1712, 1713}, all sharing the
  regenerated hot prefix P(s) (checkpoint at T_c = 963):
  - **B** = schedule-free tail, winner cell κ = 1.0, ρ = 0.7
    (`configs/dev/wave1_sf_k10_r07.yaml`, `--seed s`), endpoint val(x̄) at 1750.
  - **C** = constant LR + accumulators
    (`configs/dev/wave1_constlr_acc.yaml`, `--seed s`); C_polyak evaluated
    from its artifact by forward passes.
  - **A** = the existing n=10 local baseline runs (seeds 1710–1719,
    `configs/dev/nanogpt_local_baseline.yaml`) — no new A runs. Hot
    fingerprint of the baseline config equals the prefix config (0df54b47),
    so B/C share seed, init, and data order with the same-seed A run.
- **Criteria (pre-registered, unchanged):**
  - **WIN** iff paired mean(B − A) ≤ −0.0025 with 95% CI excluding 0.
  - **ANNEAL-REPLACED** iff paired mean(B − A) within +0.0025 AND
    B < C_polyak − 0.005 (paired) AND val(x̄) at steps 1375 and 1500 within
    0.01 of its endpoint.
  - **KILL** iff paired mean(B − A) ≥ +0.0025.
- Endpoint: final val loss at step 1750 (matched tokens). n = 4 paired seeds
  resolves 0.0025 (power basis: `reports/nanogpt-local-baseline.md`).
- Compute: 4 × (prefix 41 + C 34 + B 34) min ≈ 7.3 GPU-h, single-lane
  (GPU bus-01 still refuses contexts despite re-enumerating).
- Analysis: paired t on B−A and B−C_polyak; script committed with results.
  The gate verdict on the outcome is human, per standing rules.
