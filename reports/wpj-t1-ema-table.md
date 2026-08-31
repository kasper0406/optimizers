# T1 EMA-as-anytime-anneal — analysis

Seed-paired arms: n=20 dev seeds, GPU NVIDIA L40.

Linear-arm (stock anneal) final val acc mean: 0.9318 (TTA 0.9400).

| gamma | const EMA final − linear final | const EMA TTA − linear TTA | harvest epoch (median, reached/total) | linear-arm self-crossing |
|---|---|---|---|---|
| 0.9 | -0.0432 ± 0.0015 | -0.0461 ± 0.0013 | None (0/20) | 8.0 (12/20) |
| 0.96 | -0.0502 ± 0.0014 | -0.0551 ± 0.0017 | None (0/20) | 8 (9/20) |
| 0.98 | -0.0904 ± 0.0033 | -0.0963 ± 0.0034 | None (0/20) | None (0/20) |
| 0.99 | -0.3022 ± 0.0142 | -0.3125 ± 0.0146 | None (0/20) | None (0/20) |

raw constant-arm final − linear final (the anneal's contribution when nothing averages): -0.2476 ± 0.0332

