# Directional-smoothness plateau (descriptive)

Sidecars: 9 · plateau window: last 50% of measured steps · minibatch estimate, sum-reduced loss, per-matrix perturbation

| sidecar | lr0 | pooled lr·D_spectral | spread | pooled lr·D_euclid | spread |
|---|---|---|---|---|---|
| airbench_instrumented_seed1300_20260720T081438.instrumentation.json | 0.24 | 208.4 | 519.7 | 2.732 | 2.485 |
| airbench_instrumented_seed1301_20260720T081632.instrumentation.json | 0.24 | 222.6 | 506.6 | 3.109 | 2.581 |
| airbench_instrumented_seed1302_20260720T081816.instrumentation.json | 0.24 | 227.5 | 504.7 | 2.922 | 2.485 |
| airbench_instrumented_seed1310_20260720T082006.instrumentation.json | 0.12 | 124.3 | 298.2 | 1.677 | 1.587 |
| airbench_instrumented_seed1310_20260720T082345.instrumentation.json | 0.24 | 237.5 | 528.3 | 2.808 | 2.44 |
| airbench_instrumented_seed1310_20260720T082712.instrumentation.json | 0.48 | 323.6 | 472.7 | 3.872 | 3.159 |
| airbench_instrumented_seed1311_20260720T082201.instrumentation.json | 0.12 | 122.9 | 305.1 | 1.681 | 1.673 |
| airbench_instrumented_seed1311_20260720T082529.instrumentation.json | 0.24 | 218.7 | 490.2 | 2.782 | 2.367 |
| airbench_instrumented_seed1311_20260720T082856.instrumentation.json | 0.48 | 311.6 | 457.9 | 3.71 | 3.671 |

## LR-ladder invariance (the c/lr claim)

| quantity | 0.12 | 0.24 | 0.48 | max/min |
|---|---|---|---|---|
| spectral | 123.6 | 222.6 | 317.6 | 2.57 |
| euclidean | 1.679 | 2.808 | 3.791 | 2.26 |

A plateau constant that is lr-invariant (ratio ≈ 1) for the spectral
quantity while the Euclidean one scales with lr is the pre-registered
signature. Read is descriptive; no verdict is asserted here.

