"""T1 EMA-as-anytime-anneal analysis (docs/litreview/j-theory-theorem-sweep.md §5).

Reads airbench_ema results JSONs (both arms), pairs them by seed, and answers
the three pre-stated questions from configs/wpj_t1_ema_*.yaml:

  (i)   schedule-free match: final EMA acc (constant arm) vs final annealed
        acc (linear arm), per-seed paired difference with a 95% CI;
  (ii)  harvest epoch: earliest epoch at which the constant arm's EMA val acc
        reaches the linear arm's final (fully annealed) val acc — per gamma,
        per seed, plus the paired summary. Epochs after the crossing are the
        compute the anneal spends that averaging gets for free;
  (iii) anneal contribution: constant-arm RAW final acc vs linear-arm final
        (how much the schedule was doing that the average now carries).

Also reports, on the linear arm alone, whether mid-run EMA approaches the
run's own final annealed accuracy (the within-run form of the prediction).

Usage:
    uv run python scripts/analyze_ema.py [results_dir] [--json OUT] [--md OUT]

Pure stdlib + repo results_io; deterministic; no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _t_crit_975(df: int) -> float:
    """Two-sided 95% t critical value; small table + normal tail (repo has no
    scipy dependency)."""
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000,
    }
    if df in table:
        return table[df]
    for bound in sorted(table):
        if df < bound:
            return table[bound]
    return 1.96


def paired_summary(deltas):
    """Mean, 95% CI half-width, n for a list of per-seed paired differences."""
    n = len(deltas)
    mean = statistics.fmean(deltas)
    if n < 2:
        return {"n": n, "mean": mean, "ci95": None}
    sd = statistics.stdev(deltas)
    return {"n": n, "mean": mean, "ci95": _t_crit_975(n - 1) * sd / math.sqrt(n)}


def load_arms(results_dir: Path):
    """Collect airbench_ema runs, keyed by (lr_schedule, seed).

    Duplicate (schedule, seed) pairs keep the earliest started_at (the
    runbook's re-run rule for preempted spot sweeps)."""
    arms = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("experiment") != "airbench_ema":
            continue
        metrics = data.get("metrics", {})
        schedule = metrics.get("lr_schedule")
        seed = data.get("seed")
        if schedule is None or seed is None:
            continue
        key = (schedule, seed)
        if key in arms and arms[key]["started_at"] <= data.get("started_at", ""):
            continue
        arms[key] = {
            "started_at": data.get("started_at", ""),
            "gpu_type": data.get("gpu_type"),
            "val_accs": metrics["val_accs"],
            "val_acc": metrics["val_acc"],
            "tta_val_acc": metrics.get("tta_val_acc"),
            "ema_val_accs": metrics["ema_val_accs"],
            "ema_tta_val_accs": metrics.get("ema_tta_val_accs") or {},
        }
    return arms


def crossing_epoch(series, target):
    """1-based index of the first entry >= target, or None."""
    for i, value in enumerate(series):
        if value >= target:
            return i + 1
    return None


def analyze(results_dir: Path):
    arms = load_arms(results_dir)
    linear = {seed: run for (s, seed), run in arms.items() if s == "linear"}
    constant = {seed: run for (s, seed), run in arms.items() if s == "constant"}
    paired_seeds = sorted(set(linear) & set(constant))
    if not paired_seeds:
        raise SystemExit(
            f"no seed-paired airbench_ema arms found in {results_dir} "
            f"(linear: {len(linear)}, constant: {len(constant)})"
        )
    gpu_types = {run["gpu_type"] for run in arms.values()}
    if len(gpu_types) > 1:
        raise SystemExit(f"mixed GPU types in EMA arms: {sorted(gpu_types)}")

    gammas = sorted(next(iter(linear.values()))["ema_val_accs"], key=float)

    out = {
        "n_paired_seeds": len(paired_seeds),
        "seeds": paired_seeds,
        "gpu_type": next(iter(gpu_types)),
        "linear_final_val_mean": statistics.fmean(
            linear[s]["val_acc"] for s in paired_seeds
        ),
        "linear_final_tta_mean": statistics.fmean(
            linear[s]["tta_val_acc"] for s in paired_seeds
        ),
        "constant_raw_final_minus_linear_final": paired_summary(
            [constant[s]["val_acc"] - linear[s]["val_acc"] for s in paired_seeds]
        ),
        "per_gamma": {},
    }

    for gamma in gammas:
        # (i) schedule-free match, final TTA readouts
        tta_delta = paired_summary(
            [
                constant[s]["ema_tta_val_accs"][gamma] - linear[s]["tta_val_acc"]
                for s in paired_seeds
                if constant[s]["ema_tta_val_accs"].get(gamma) is not None
                and linear[s]["tta_val_acc"] is not None
            ]
        )
        final_delta = paired_summary(
            [
                constant[s]["ema_val_accs"][gamma][-1] - linear[s]["val_acc"]
                for s in paired_seeds
            ]
        )
        # (ii) harvest epoch vs the PAIRED seed's own annealed final
        crossings = [
            crossing_epoch(constant[s]["ema_val_accs"][gamma], linear[s]["val_acc"])
            for s in paired_seeds
        ]
        reached = [c for c in crossings if c is not None]
        # within-run form on the linear arm: EMA vs that run's own final
        linear_crossings = [
            crossing_epoch(linear[s]["ema_val_accs"][gamma], linear[s]["val_acc"])
            for s in paired_seeds
        ]
        linear_reached = [c for c in linear_crossings if c is not None]
        out["per_gamma"][gamma] = {
            "constant_ema_final_minus_linear_final": final_delta,
            "constant_ema_tta_minus_linear_tta": tta_delta,
            "harvest_epoch": {
                "n_reached": len(reached),
                "n_total": len(crossings),
                "median": statistics.median(reached) if reached else None,
                "per_seed": dict(zip(paired_seeds, crossings)),
            },
            "linear_arm_selfcrossing": {
                "n_reached": len(linear_reached),
                "median": statistics.median(linear_reached)
                if linear_reached
                else None,
            },
        }

    # epoch-resolved mean curves (for the report plot/table)
    curves = defaultdict(dict)
    n_epochs = len(next(iter(linear.values()))["val_accs"])
    for arm_name, arm in (("linear", linear), ("constant", constant)):
        curves[arm_name]["raw"] = [
            statistics.fmean(arm[s]["val_accs"][e] for s in paired_seeds)
            for e in range(n_epochs)
        ]
        for gamma in gammas:
            curves[arm_name][f"ema_{gamma}"] = [
                statistics.fmean(arm[s]["ema_val_accs"][gamma][e] for s in paired_seeds)
                for e in range(n_epochs)
            ]
    out["mean_curves"] = dict(curves)
    return out


def to_markdown(result) -> str:
    lines = [
        "# T1 EMA-as-anytime-anneal — analysis",
        "",
        f"Seed-paired arms: n={result['n_paired_seeds']} dev seeds, "
        f"GPU {result['gpu_type']}.",
        "",
        f"Linear-arm (stock anneal) final val acc mean: "
        f"{result['linear_final_val_mean']:.4f} "
        f"(TTA {result['linear_final_tta_mean']:.4f}).",
        "",
        "| gamma | const EMA final − linear final | const EMA TTA − linear TTA "
        "| harvest epoch (median, reached/total) | linear-arm self-crossing |",
        "|---|---|---|---|---|",
    ]
    for gamma, g in result["per_gamma"].items():
        fd, td, hv, sc = (
            g["constant_ema_final_minus_linear_final"],
            g["constant_ema_tta_minus_linear_tta"],
            g["harvest_epoch"],
            g["linear_arm_selfcrossing"],
        )

        def fmt(d):
            if d["ci95"] is None:
                return f"{d['mean']:+.4f} (n={d['n']})"
            return f"{d['mean']:+.4f} ± {d['ci95']:.4f}"

        lines.append(
            f"| {gamma} | {fmt(fd)} | {fmt(td)} | "
            f"{hv['median']} ({hv['n_reached']}/{hv['n_total']}) | "
            f"{sc['median']} ({sc['n_reached']}/{result['n_paired_seeds']}) |"
        )
    lines += [
        "",
        "raw constant-arm final − linear final (the anneal's contribution "
        "when nothing averages): "
        + (
            f"{result['constant_raw_final_minus_linear_final']['mean']:+.4f} "
            f"± {result['constant_raw_final_minus_linear_final']['ci95']:.4f}"
            if result["constant_raw_final_minus_linear_final"]["ci95"] is not None
            else str(result["constant_raw_final_minus_linear_final"])
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir", nargs="?", default=REPO_ROOT / "results", type=Path
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--md", type=Path, default=None)
    args = parser.parse_args(argv)

    result = analyze(args.results_dir)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    md = to_markdown(result)
    if args.md:
        args.md.write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
