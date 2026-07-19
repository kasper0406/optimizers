# Literature Review — Evolved Directions (2026-07-19)

Five parallel literature sweeps over the post-Phase-2 direction proposals
(reports/gate1-decision.md context; prompted by the airbench Phase-2 null).
Each direction has a full report in this directory. Verdicts:

| # | Direction | Verdict | Open core | File |
|---|---|---|---|---|
| A | ρ-feedback global LR control | **Partially reported** | Occupancy-fraction order parameter; nonzero setpoint control; anything in Muon/normalized family (norm/distance signals degenerate there); our empirical characterization | [a-autocorr-lr-control.md](a-autocorr-lr-control.md) |
| B | Per-matrix temporal-stat LR gains | **Unreported as combination (~80%)** | "Temporal third column" of the trust-ratio family (LARS/OrScale = spatial norms; LANTON/NAMO/MoLS = noise magnitude; ours = serial structure). Slot actively being populated in 2026 — move fast | [b-layer-temporal-trust-ratio.md](b-layer-temporal-trust-ratio.md) |
| C | Per-direction routing at longer horizons | **Null unpublished but literature discourages retest** | Our airbench null is neither scooped nor contradicted; short-horizon-bias theory says short horizons *over*-reward damping, so our null mildly discourages long-horizon hopes; Song et al. (ICLR'25) + self-stabilization give principled loss-irrelevance reasons | [c-perdirection-horizons.md](c-perdirection-horizons.md) |
| D | Normalized-update stability laws | **Partially — qualitative claim just published (ICML'26 oral), our quantitative regime open and *named as their open problem*** | Momentum+minibatch Muon threshold constant (Adam analog 38/η exists, Muon analog absent); per-direction HVP ηλ time series (our ~65); momentum-independence; oscillation amplitude law | [d-normalized-stability-theory.md](d-normalized-stability-theory.md) |
| E | Large-batch per-direction SNR | **Partially; per-direction critical batch size open** | First temporal-statistical per-direction classification at LLM scale; empirical per-direction CBS; mechanistic bridge to Muon's post-CBS edge. Crowded field; "random spectra" result warns spectral targeting may be inert | [e-largebatch-snr.md](e-largebatch-snr.md) |

## Re-ranked pursuit plan (agent, under delegation)

1. **A+B merged — "temporal trust ratio" prototype.** One method, granularity as
   an axis: per-matrix (and global-fallback) LR gain driven by negative-ρ
   occupancy with a nonzero setpoint. B is the strongest novelty; A supplies
   the controller framing and the Pflug/GALA lineage positioning. Mandatory
   baselines: OrScale, NAMO/LANTON-style noise scaling, GALA, Prodigy,
   hand-tuned schedule. Cheap on existing infra (airbench first, dev seeds).
2. **D — fold into the measurement paper.** Reframe the Phase-1 writeup around
   the ICML-oral open problem: simultaneous per-direction Euclidean ηλ (HVP)
   and spectral-norm directional smoothness along practical Muon trajectories,
   across β and LR. Our ηλ≈65 + momentum-independence + LR-monotone occupancy
   become the headline empirical laws. Highest scientific value per dollar.
3. **E — shelf** pending explicit human go (nanogpt+ scale, crowded).
4. **C — dropped** (literature cuts against; the null stands as scoped).

## Field-velocity note

Four directly adjacent papers appeared Jan–Jun 2026 (OrScale, NAMO, SpecMuon,
Aurora). The B slot and the D measurement are both time-sensitive.
