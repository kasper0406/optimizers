# Teleport-Muon — ON vs OFF (`airbench_teleport`)

Descriptive output of `scripts/analyze_teleport.py`. No pass/fail judgment is made here.

Seed-paired arms: n=20 dev seeds, GPU NVIDIA L40; teleports per run (mean): 7.0.

OFF final: val 0.9317 (TTA 0.9402). ON final: val 0.9274 (TTA 0.9363).

**ON − OFF (paired): val -0.0043 ± 0.0013, TTA -0.0039 ± 0.0008.** Wall-clock overhead: +4.7386 ± 0.0955 s.

| epoch | ON − OFF val |
| --- | --- |
| 1 | -0.0057 ± 0.0162 |
| 2 | -0.0010 ± 0.0262 |
| 3 | +0.0154 ± 0.0173 |
| 4 | +0.0096 ± 0.0127 |
| 5 | -0.0032 ± 0.0044 |
| 6 | -0.0089 ± 0.0035 |
| 7 | -0.0103 ± 0.0019 |
| 8 | -0.0043 ± 0.0013 |

Achieved nuclear-norm ratio at teleports (mean over seeds): first 1.2236, mid 1.2028, last 1.3203.

