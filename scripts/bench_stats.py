#!/usr/bin/env python
"""Micro-benchmark: scalar vs array-mode regime classification cost per step.

Motivated by the WP1.1 overhead measurement (results/bench_overhead_airbench.json,
37.3% median overhead vs the <10% DoD): the scalar RegimeClassifier costs
~76 us per update call; at airbench shapes (6 matrices x 32 directions x
2 betas = 384 scalar calls/step) that is serial-CPU-bound.  Array mode
(BatchRegimeClassifier) updates a whole k-vector in O(1) numpy calls.

Usage:
    uv run python scripts/bench_stats.py [--k 32] [--steps 2000] [--repeats 3]

Prints per-step cost for one (matrix, beta) group of k directions, both
paths, plus the projected per-training-step cost at airbench shape
(6 matrices x 2 betas).  Measurement only -- no pass/fail judgment.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.stats import BatchRegimeClassifier, RegimeClassifier

# Same threshold style as the equivalence tests: synthetic-scenario values
# with both innovation detectors enabled (the most expensive configuration).
CLF_KWARGS = dict(
    tau_sig=4.0,
    tau_noise=2.5,
    rho_osc=0.4,
    n_min=15,
    z_reset=3.0,
    innov_needed=2,
    innov_window=4,
    z_quiet=0.4,
    quiet_window=6,
)

AIRBENCH_MATRICES = 6
AIRBENCH_BETAS = 2


def _bank(k: int, steps: int, seed: int) -> np.ndarray:
    """Mixed streams: signal / noise / oscillation, one row per direction."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(k):
        kind = i % 3
        if kind == 0:
            rows.append(5.0 + rng.standard_normal(steps))
        elif kind == 1:
            rows.append(rng.standard_normal(steps))
        else:
            rows.append(6.0 * np.power(-1.0, np.arange(steps)) + 0.1 * rng.standard_normal(steps))
    return np.stack(rows)


def bench_scalar(bank: np.ndarray, beta: float) -> float:
    k, steps = bank.shape
    clfs = [RegimeClassifier(beta=beta, **CLF_KWARGS) for _ in range(k)]
    cols = [bank[:, t] for t in range(steps)]
    t0 = time.perf_counter()
    for t in range(steps):
        col = cols[t]
        for i in range(k):
            clfs[i].update(col[i])
    return (time.perf_counter() - t0) / steps


def bench_batch(bank: np.ndarray, beta: float) -> float:
    k, steps = bank.shape
    clf = BatchRegimeClassifier(beta=beta, k=k, **CLF_KWARGS)
    cols = [bank[:, t] for t in range(steps)]
    t0 = time.perf_counter()
    for t in range(steps):
        clf.update(cols[t])
    return (time.perf_counter() - t0) / steps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000, help="dev seed (>= 1000)")
    args = parser.parse_args(argv)
    if args.seed < 1000:
        raise SystemExit("seed must be a dev seed (>= 1000)")

    bank = _bank(args.k, args.steps, args.seed)
    for beta in (0.9, 0.99):
        scalar_s = min(bench_scalar(bank, beta) for _ in range(args.repeats))
        batch_s = min(bench_batch(bank, beta) for _ in range(args.repeats))
        speedup = scalar_s / batch_s if batch_s > 0 else float("inf")
        per_train_step_scalar = scalar_s * AIRBENCH_MATRICES * AIRBENCH_BETAS
        per_train_step_batch = batch_s * AIRBENCH_MATRICES * AIRBENCH_BETAS
        print(
            f"beta={beta}: k={args.k} directions/group | "
            f"scalar {scalar_s * 1e6:8.1f} us/step ({scalar_s / args.k * 1e6:6.2f} us/call) | "
            f"array {batch_s * 1e6:8.1f} us/step | speedup {speedup:6.1f}x"
        )
        print(
            f"          projected airbench stats cost/train-step "
            f"({AIRBENCH_MATRICES} matrices x {AIRBENCH_BETAS} betas): "
            f"scalar {per_train_step_scalar * 1e3:7.2f} ms -> array {per_train_step_batch * 1e3:6.3f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
