#!/usr/bin/env python
"""Program #8 step-time cost: seed-paired TempoMuon-vs-Muon compute overhead.

The eval table (`reports/tempo-eval.md`, `reports/tempo-gate-memo.md` §1) and
the gate's `free_at_record_lr` field are about *accuracy*. On a wall-clock
speedrun testbed the compute side of "switch it on for free" is a separate
question, and the same 800 result files answer it: the arms are seed-paired at
every LR rung, so the per-seed difference in `metrics.time_seconds` (train
time) and `wall_time_s` (whole run) is a paired measurement of what the
controller costs.

What the number is and is not: it is the cost of the *measured configuration* —
one `.item()` sync per matrix per step plus a prev-gradient buffer
(`src/optim/tempomuon.py`), with `history_every` telemetry enabled in the
treatment arm and absent from the control — so it bounds what that
configuration costs, not what the serial-correlation signal must cost.

Arms and LR rungs are read with the same helpers as `scripts/analyze_tempo.py`
(no second copy of the arm-naming rule). Deterministic: sorted keys, no
timestamps, no randomness.

Usage:
    uv run python scripts/analyze_tempo_overhead.py results/*.json
    uv run python scripts/analyze_tempo_overhead.py results/*.json --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scipy import stats as scipy_stats

from scripts.analyze_tempo import load_runs, mean_std, run_lr, run_opt

# metrics.time_seconds = harness-reported train time; wall_time_s = whole run
# (results_io schema, top level) and so carries process setup noise too.
METRICS = ("time_seconds", "wall_time_s")


def metric_value(run: Dict[str, Any], name: str) -> Optional[float]:
    """Read a timing metric from either the metrics dict or the run root."""
    metrics = run.get("metrics") or {}
    if metrics.get(name) is not None:
        return float(metrics[name])
    if run.get(name) is not None:
        return float(run[name])
    return None


def paired_delta(base: Dict[int, float], arm: Dict[int, float]) -> Optional[Dict[str, float]]:
    """Paired stats on arm − base over the seeds present in both, or None."""
    seeds = sorted(set(base) & set(arm))
    if len(seeds) < 2:
        return None
    diffs = [arm[s] - base[s] for s in seeds]
    mean, sd = mean_std(diffs)
    se = sd / math.sqrt(len(diffs))
    tcrit = float(scipy_stats.t.ppf(0.975, len(diffs) - 1))
    base_mean, _ = mean_std([base[s] for s in seeds])
    arm_mean, _ = mean_std([arm[s] for s in seeds])
    return {
        "arm_mean": arm_mean,
        "base_mean": base_mean,
        "ci95_hi": mean + tcrit * se,
        "ci95_lo": mean - tcrit * se,
        "mean": mean,
        "n": len(diffs),
        "pct": 100.0 * mean / base_mean if base_mean else float("nan"),
        "sd": sd,
        "se": se,
        "t": mean / se if se else float("nan"),
    }


def analyze_overhead(
    runs: Sequence[Dict[str, Any]],
    baseline: str = "muon",
    metrics: Sequence[str] = METRICS,
) -> Dict[str, Any]:
    """Seed-paired timing deltas of every non-baseline arm, per LR rung."""
    by_key: Dict[Tuple[str, float, str], Dict[int, float]] = defaultdict(dict)
    for r in runs:
        arm, lr = run_opt(r), run_lr(r)
        for name in metrics:
            v = metric_value(r, name)
            if v is not None:
                by_key[(arm, lr, name)][int(r["seed"])] = v

    arms = sorted({a for a, _, _ in by_key})
    lrs = sorted({l for _, l, _ in by_key})
    if baseline not in arms:
        raise SystemExit(f"baseline arm {baseline!r} not found in {arms}")

    out: Dict[str, Any] = {"arms": arms, "baseline": baseline, "per_metric": {}}
    lines = [
        "# Program #8: seed-paired step-time overhead vs stock Muon",
        "",
        f"- baseline arm: `{baseline}`; paired within seed at each LR rung",
        "- 95% CI is a paired t interval on the per-seed difference",
        "",
    ]
    for name in metrics:
        lines += [
            f"## {name}",
            "",
            "| lr | arm | n | baseline mean (s) | arm mean (s) | "
            "paired Δ ± SE (s) | 95% CI | t | % |",
            "|" + "---|" * 9,
        ]
        rows: Dict[str, Dict[str, Any]] = {}
        for lr in lrs:
            base = by_key.get((baseline, lr, name), {})
            for arm in arms:
                if arm == baseline:
                    continue
                st = paired_delta(base, by_key.get((arm, lr, name), {}))
                if st is None:
                    continue
                rows.setdefault(str(lr), {})[arm] = st
                lines.append(
                    f"| {lr} | {arm} | {st['n']} | {st['base_mean']:.4f} | "
                    f"{st['arm_mean']:.4f} | {st['mean']:+.4f} ± {st['se']:.4f} | "
                    f"[{st['ci95_lo']:+.4f}, {st['ci95_hi']:+.4f}] | "
                    f"{st['t']:.1f} | {st['pct']:+.2f}% |"
                )
        lines.append("")
        out["per_metric"][name] = rows
    out["report"] = "\n".join(lines)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--baseline", default="muon", help="arm name used as control")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)
    out = analyze_overhead(load_runs(args.results), baseline=args.baseline)
    print(out["report"])
    if args.json_out:
        report = out.pop("report")
        Path(args.json_out).write_text(json.dumps(out, indent=2, sort_keys=True))
        out["report"] = report
    return 0


if __name__ == "__main__":
    sys.exit(main())
