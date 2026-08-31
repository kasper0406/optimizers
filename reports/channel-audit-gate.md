# Program #23 (channel audit) — gate record

Date: 2026-08-31. Verdict: **HUMAN**, recorded from the session decision
"Adjudicate as read" (same session that froze the thresholds; both decisions
logged in the session transcript). The agent's role in this document is
clerical: it transcribes the mechanical outcome rows of
`reports/channel-audit.md` and the registered consequences that fire with
them. No number below is new.

## The adjudicated rows

| criterion | measured | frozen bar | row | registered consequence that fires |
| --- | --- | --- | --- | --- |
| P1 (band artifact) | band_contrast 0.784 [0.726, 0.841]; frac_alt 0.000 | ≥ 1.30 / ≥ 0.010 | **6 — FAIL** | the §3.1.4 null is **upheld across the frequency band**: no demodulated/Nyquist-band per-direction signal on the frozen tier. The band-artifact account of the frozen-probe null is closed. |
| P2 (frame gain) | frame_gain 1.106 [1.024, 1.188]; bulk_gain 0.701 | ≥ 3.0; 1.3/2.0 | **D — no gain** | A4's tracked-top DC excess does not replicate as a frame gain at ≤ 2× record lr; the weight-anchored-frame family is **not** adjudicated beyond that scope (registered lr-ladder limit, prereg §3). |
| P3 (τ) | τ_cal 0.41 / 0.47 / 0.59 / 0.89, all CI-uppers < 1.0 | < 1.0 at all four K | **DECISIVE, K-stable** | **no positive decorrelation time**; β₁* undefined; the cross-repo "white across steps" premise is wrong in the anti-correlated direction. Survives to K = 64 (the L = 4 form was previously on disk; this is the registered extension). |
| Rider-1 (√B) | ratio 3.662 (excess 0.106 → 0.387) | [2.8, 5.6] / flat < 1.5 | **PASS** | per-direction DC excess scales ≈√B: the per-direction signal is **batch-starved, not absent**. Open question #4 (per-direction critical batch size) gets its first affirmative measured slope. |
| Rider-2 (origin) | ESS/n max/min 1.857, monotone **increasing** in B; φ̂ −0.157 → −0.488 | < 1.3 / within 1.15 of 1.0 | **MIXED** | neither registered branch; the movement is in the *deepening* direction, i.e. away from the sampling account. Reported as structure, not adjudicated as mechanism. |

Estimator health: K1 controls clean (white: contrasts 1.020/0.994, τ_cal
0.989–1.012 at all K; AR(1) φ = −0.34: contrasts 1.024/1.014, τ ladder matching
the committed anchors). No kill clause fired (K2 0/0, K3 spread 0.171 < 0.35,
K4 1.422 < 2.0, K6 0.047 < 0.15). Burn-in sensitivity {5, 15, 25} crosses no
decision line.

## What closes, what opens

**Closed on airbench (this workload, this optimizer, the registered scope):**

- The temporal family, in full: DC estimation (program #4), the demodulated
  band (P1), momentum-window tuning (P3 — β₁* undefined), and the frame
  workaround at ≤ 2× lr (P2). "Averaging washes the signal away" is now
  measured to be unfixable **on the time axis**: there is nothing positively
  correlated to average, at any tested frequency, in either tested frame.

**Opened:**

- The **batch axis** is the surviving variance-reduction axis, with a measured
  ≈√B slope (Rider-1). A batch-axis successor (denser rungs; per-sequence
  concentration / importance-sampling ceiling) is admissible but
  **unregistered** — nothing about it is pre-committed by this program.
- The φ(B) deepening (Phase A discovery, replicated on the frozen tier by
  Rider-2's φ̂ column) is an unexplained trajectory-dynamics regularity.

**Successor decision (same session, human):** the next program is the
**hesstrack lag-ladder run** (sibling repo: 51M LM, AdamW, block-group
granularity) — testing whether τ < 1 anti-correlation and the φ(B) deepening
transfer across optimizer and workload; its pre-registration is
`notes/lagladder-prereg.md` in that repo. The airbench batch-axis successor
remains open behind it.
