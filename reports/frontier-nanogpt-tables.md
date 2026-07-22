# nanogpt frontier transfer (#7) — tables (descriptive)

Pre-registration: `reports/frontier-nanogpt-preregistration.md` · no verdict here

## Loss vs Muon lr per token batch

### chunks=2 (98,304 tokens/step) — valley 3.4304 @ 0.05, floor 3.4404, lr*_cross 0.1277, shoulder 0.1, low-lr penalty 0.02074

| lr | 0.025 | 0.035355 | 0.05 | 0.070711 | 0.1 | 0.141421 |
|---|---|---|---|---|---|---|
| mean loss | 3.4511 | 3.4469 | 3.4304 | 3.4338 | 3.4338 | 3.4431 |
| n | 2 | 2 | 2 | 2 | 2 | 2 |

### chunks=4 (196,608 tokens/step) — valley 3.4131 @ 0.0354, floor 3.4231, lr*_cross 0.06928, shoulder 0.05, low-lr penalty 0.00065

| lr | 0.025 | 0.035355 | 0.05 | 0.070711 | 0.1 | 0.141421 |
|---|---|---|---|---|---|---|
| mean loss | 3.4137 | 3.4131 | 3.415 | 3.4236 | 3.4268 | 3.4476 |
| n | 2 | 2 | 2 | 2 | 2 | 2 |

### chunks=8 (393,216 tokens/step) — valley 3.4404 @ 0.0354, floor 3.4504, lr*_cross 0.07336, shoulder 0.0707, low-lr penalty 0.00121

| lr | 0.025 | 0.035355 | 0.05 | 0.070711 | 0.1 | 0.141421 |
|---|---|---|---|---|---|---|
| mean loss | 3.4416 | 3.4404 | 3.4433 | 3.4493 | 3.46 | 3.4802 |
| n | 2 | 2 | 2 | 2 | 2 | 2 |

### chunks=16 (786,432 tokens/step) — valley 3.5103 @ 0.0354, floor 3.5203, lr*_cross 0.06458, shoulder 0.05, low-lr penalty 0.00235

| lr | 0.025 | 0.035355 | 0.05 | 0.070711 | 0.1 | 0.141421 |
|---|---|---|---|---|---|---|
| mean loss | 3.5127 | 3.5103 | 3.5122 | 3.5232 | 3.543 | 3.5741 |
| n | 2 | 2 | 2 | 2 | 2 | 2 |

## alpha (lr* vs tokens/step)

alpha = -0.2869 over 4 batch sizes · bootstrap median -0.2729, CI95 [-0.3495, 0.2989] (n=2/cell: indicative)

Pre-registered regions: noise-governed ≥ 0.25 · deterministic < 0.1 · airbench transfer point 0.35 [0.30, 0.42].

P2d valley monotone non-decreasing: False · P3d no cliff (all mean losses ≤ 4.0): True

Interpretation lives in `reports/frontier-nanogpt.md`.

