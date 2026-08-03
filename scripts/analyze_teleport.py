"""Teleport-Muon round-2 analysis (litreview j §6 item 4; gate GO in
reports/wpj-mech-round1.md §1).

Pairs `airbench_teleport` ON/OFF arms by seed and reports: final val/TTA
paired deltas with 95% CIs, per-epoch mean curves, and teleport telemetry
(achieved nuclear-norm ratios over training). DESCRIPTIVE OUTPUT ONLY: no
pass/fail judgment, no gate (CLAUDE.md: gate decisions are human-only).

Usage:
    uv run python scripts/analyze_teleport.py [results_dir] [--json OUT] [--md OUT]

Pure stdlib; deterministic; no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _t_crit_975(df: int) -> float:
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
    n = len(deltas)
    if n == 0:
        return None
    mean = statistics.fmean(deltas)
    if n < 2:
        return {"n": n, "mean": mean, "ci95": None}
    sd = statistics.stdev(deltas)
    return {"n": n, "mean": mean, "ci95": _t_crit_975(n - 1) * sd / math.sqrt(n)}


def load_arms(results_dir: Path):
    """(enabled, seed) -> run dict; duplicate seeds keep earliest started_at."""
    arms = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("experiment") != "airbench_teleport":
            continue
        metrics = data.get("metrics", {})
        seed = data.get("seed")
        tp = metrics.get("teleport")
        if seed is None or not isinstance(tp, dict):
            continue
        key = (bool(tp.get("enabled")), seed)
        if key in arms and arms[key]["started_at"] <= data.get("started_at", ""):
            continue
        arms[key] = {
            "started_at": data.get("started_at", ""),
            "gpu_type": data.get("gpu_type"),
            "val_accs": metrics["val_accs"],
            "val_acc": metrics["val_acc"],
            "tta_val_acc": metrics.get("tta_val_acc"),
            "n_teleports": metrics.get("n_teleports", 0),
            "teleport_timeseries": metrics.get("teleport_timeseries", []),
            "time_seconds": metrics.get("time_seconds"),
        }
    return arms


def analyze(results_dir: Path):
    arms = load_arms(results_dir)
    on = {seed: run for (en, seed), run in arms.items() if en}
    off = {seed: run for (en, seed), run in arms.items() if not en}
    paired_seeds = sorted(set(on) & set(off))
    if not paired_seeds:
        raise SystemExit(
            f"no seed-paired airbench_teleport arms found in {results_dir} "
            f"(on: {len(on)}, off: {len(off)})"
        )
    gpu_types = {run["gpu_type"] for run in arms.values()}
    if len(gpu_types) > 1:
        raise SystemExit(f"mixed GPU types: {sorted(map(str, gpu_types))}")

    n_epochs = len(next(iter(on.values()))["val_accs"])
    out = {
        "n_paired_seeds": len(paired_seeds),
        "seeds": paired_seeds,
        "gpu_type": next(iter(gpu_types)),
        "off_final_val_mean": statistics.fmean(
            off[s]["val_acc"] for s in paired_seeds
        ),
        "off_final_tta_mean": statistics.fmean(
            off[s]["tta_val_acc"] for s in paired_seeds
        ),
        "on_final_val_mean": statistics.fmean(on[s]["val_acc"] for s in paired_seeds),
        "on_final_tta_mean": statistics.fmean(
            on[s]["tta_val_acc"] for s in paired_seeds
        ),
        "on_minus_off_val": paired_summary(
            [on[s]["val_acc"] - off[s]["val_acc"] for s in paired_seeds]
        ),
        "on_minus_off_tta": paired_summary(
            [on[s]["tta_val_acc"] - off[s]["tta_val_acc"] for s in paired_seeds]
        ),
        "per_epoch_on_minus_off": [
            paired_summary(
                [
                    on[s]["val_accs"][e] - off[s]["val_accs"][e]
                    for s in paired_seeds
                ]
            )
            for e in range(n_epochs)
        ],
        "mean_curves": {
            "on": [
                statistics.fmean(on[s]["val_accs"][e] for s in paired_seeds)
                for e in range(n_epochs)
            ],
            "off": [
                statistics.fmean(off[s]["val_accs"][e] for s in paired_seeds)
                for e in range(n_epochs)
            ],
        },
        "n_teleports_mean": statistics.fmean(
            on[s]["n_teleports"] for s in paired_seeds
        ),
        "overhead_seconds": paired_summary(
            [
                on[s]["time_seconds"] - off[s]["time_seconds"]
                for s in paired_seeds
                if on[s]["time_seconds"] is not None
                and off[s]["time_seconds"] is not None
            ]
        ),
    }

    # telemetry: mean achieved ratio at early/mid/late teleports (over seeds)
    ratios = {"first": [], "mid": [], "last": []}
    for s in paired_seeds:
        ts = on[s]["teleport_timeseries"]
        if not ts:
            continue
        ratios["first"].append(ts[0]["mean_ratio"])
        ratios["mid"].append(ts[len(ts) // 2]["mean_ratio"])
        ratios["last"].append(ts[-1]["mean_ratio"])
    out["achieved_ratio"] = {
        k: (statistics.fmean(v) if v else None) for k, v in ratios.items()
    }
    return out


def _fmt(summary) -> str:
    if summary is None:
        return "n/a"
    if summary["ci95"] is None:
        return f"{summary['mean']:+.4f} (n={summary['n']})"
    return f"{summary['mean']:+.4f} ± {summary['ci95']:.4f}"


def to_markdown(result) -> str:
    lines = [
        "# Teleport-Muon — ON vs OFF (`airbench_teleport`)",
        "",
        "Descriptive output of `scripts/analyze_teleport.py`. No pass/fail "
        "judgment is made here.",
        "",
        f"Seed-paired arms: n={result['n_paired_seeds']} dev seeds, GPU "
        f"{result['gpu_type']}; teleports per run (mean): "
        f"{result['n_teleports_mean']:.1f}.",
        "",
        f"OFF final: val {result['off_final_val_mean']:.4f} "
        f"(TTA {result['off_final_tta_mean']:.4f}). "
        f"ON final: val {result['on_final_val_mean']:.4f} "
        f"(TTA {result['on_final_tta_mean']:.4f}).",
        "",
        f"**ON − OFF (paired): val {_fmt(result['on_minus_off_val'])}, "
        f"TTA {_fmt(result['on_minus_off_tta'])}.** "
        f"Wall-clock overhead: {_fmt(result['overhead_seconds'])} s.",
        "",
        "| epoch | ON − OFF val |",
        "| --- | --- |",
    ]
    for e, s in enumerate(result["per_epoch_on_minus_off"]):
        lines.append(f"| {e + 1} | {_fmt(s)} |")
    ar = result["achieved_ratio"]
    lines += [
        "",
        "Achieved nuclear-norm ratio at teleports (mean over seeds): "
        f"first {ar['first']:.4f}, mid {ar['mid']:.4f}, last {ar['last']:.4f}."
        if all(v is not None for v in ar.values())
        else "Achieved-ratio telemetry incomplete.",
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
