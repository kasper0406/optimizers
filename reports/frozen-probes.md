# Frozen-probe tier: long-integration signal detection

Descriptive output of `scripts/analyze_frozen_probes.py`. No pass/fail
judgment is made here; gate decisions are human-only (CLAUDE.md).

Pre-registered question (`src.instrument.tracker.FrozenProbeBank`):
does the cumulative |t| of a frozen, never-reset random probe grow
like sqrt(t) (persistent signal: drift ~ T, noise ~ sqrt(T)), or stay
flat/bounded at the white-noise scale, and does any probe cross
|t| >= 4 by end of run? The tracked/bulk tiers' t is structurally
capped (bounded EMA window + innovation resets).

Runs: 9; frozen probes: 864.
sqrt(t) slope band [0.35, 0.65], flat band [-0.15, 0.15].

## Growth law and threshold crossings

| group | est | slope median | slope IQR | frac in sqrt band | frac flat | median final \|t\| | frac \|t\|>=4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | t_naive | 0.057 | [-0.254, 0.324] | 0.169 | 0.251 | 0.614 | 0.001 |
| pooled | t_nw | 0.042 | [-0.268, 0.335] | 0.176 | 0.253 | 0.874 | 0.009 |
| layers.1.conv1.weight | t_naive | 0.117 | [-0.167, 0.319] | 0.167 | 0.285 | 0.642 | 0.000 |
| layers.1.conv1.weight | t_nw | 0.123 | [-0.133, 0.343] | 0.188 | 0.292 | 1.043 | 0.007 |
| layers.1.conv2.weight | t_naive | 0.139 | [-0.264, 0.430] | 0.194 | 0.215 | 0.661 | 0.007 |
| layers.1.conv2.weight | t_nw | 0.168 | [-0.259, 0.422] | 0.215 | 0.181 | 0.972 | 0.021 |
| layers.2.conv1.weight | t_naive | -0.030 | [-0.360, 0.332] | 0.201 | 0.243 | 0.524 | 0.000 |
| layers.2.conv1.weight | t_nw | -0.022 | [-0.325, 0.266] | 0.174 | 0.278 | 0.694 | 0.007 |
| layers.2.conv2.weight | t_naive | -0.014 | [-0.347, 0.235] | 0.118 | 0.292 | 0.513 | 0.000 |
| layers.2.conv2.weight | t_nw | -0.072 | [-0.383, 0.195] | 0.097 | 0.264 | 0.669 | 0.007 |
| layers.3.conv1.weight | t_naive | -0.004 | [-0.312, 0.306] | 0.125 | 0.243 | 0.640 | 0.000 |
| layers.3.conv1.weight | t_nw | -0.011 | [-0.346, 0.348] | 0.146 | 0.257 | 0.809 | 0.000 |
| layers.3.conv2.weight | t_naive | 0.129 | [-0.195, 0.350] | 0.208 | 0.229 | 0.768 | 0.000 |
| layers.3.conv2.weight | t_nw | 0.132 | [-0.205, 0.392] | 0.236 | 0.250 | 1.160 | 0.014 |

Effective sample size (pooled): median 389.9, IQR [319.6, 466.7]; Newey-West floored on 0 probe(s).

## Tier contrast: final |t| distribution

| beta | tier | n | median | q75 | max | frac >=4 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.9 | tracked:all | 1728 | 0.596 | 1.002 | 2.986 | 0.000 |
| 0.9 | tracked:bulk | 864 | 0.635 | 1.062 | 2.986 | 0.000 |
| 0.9 | tracked:top | 864 | 0.539 | 0.933 | 2.947 | 0.000 |
| 0.99 | tracked:all | 1728 | 0.400 | 0.728 | 2.896 | 0.000 |
| 0.99 | tracked:bulk | 864 | 0.533 | 0.909 | 2.896 | 0.000 |
| 0.99 | tracked:top | 864 | 0.325 | 0.549 | 1.982 | 0.000 |
| - | frozen:t_naive | 864 | 0.614 | 1.134 | 4.603 | 0.001 |
| - | frozen:t_nw | 864 | 0.874 | 1.594 | 5.920 | 0.009 |
