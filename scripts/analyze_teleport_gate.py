"""Teleportation go/no-go gate analysis — "how much does the gradient vary
along the loss-invariant symmetry orbit?"
(docs/litreview/j-theory-theorem-sweep.md §6, flow-first program item 4 / T6).

Reads `airbench_teleport_gate` results JSONs. Each run snapshots the stock
recipe at a few steps and, at every snapshot, draws random per-channel
conv→BatchNorm rescalings (train-mode loss invariant up to BN eps) and then
runs a constrained random search that maximizes the per-matrix NUCLEAR-norm
sum subject to staying on the level set (|rel_dloss| < invariance_tol). The
nuclear norm is the Muon-relevant gradient size: a Muon step's first-order
loss decrease is <G, polar(G)> = ||G||_*.

This script summarizes, per snapshot step and across seeds:

  (i)   the invariance quality actually achieved — how many of the random
        draws landed on the level set (n_feasible) and the worst realized
        |rel_dloss| over all draws and seeds (the fp16 rounding floor);
  (ii)  what a plain random draw buys — the per-seed median and 90th
        percentile of the drawn nuclear-sum ratios, averaged over seeds;
  (iii) what search buys — the best feasible random draw and the best refined
        point, as mean ± 95% CI over seeds (nuclear-sum ratio), plus the
        refined point's Euclidean (total gradient) ratio for contrast.

The headline quantity for the human's kill decision is
``overall.min_over_snapshots_best_refined_nuclear_mean``: the smallest (i.e.
most conservative over training states) mean best-achievable nuclear-norm
ratio at fixed loss. Teleportation provably accelerates only if the gradient
norm varies along the level set (Zhao et al. arXiv:2205.10637 /
arXiv:2305.13404; Mishkin et al. arXiv:2403.03362).

DESCRIPTIVE OUTPUT ONLY: this script states quantities; it makes no pass/fail
judgment and evaluates no gate (CLAUDE.md: gate decisions are human-only). In
particular it carries no "O(1%) kills the direction" threshold.

Usage:
    uv run python scripts/analyze_teleport_gate.py [results_dir] \
        [--json OUT] [--md OUT]

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

EXPERIMENT = "airbench_teleport_gate"

# Percentiles of the random-draw nuclear-sum ratio reported per snapshot.
DRAW_PERCENTILES = (50.0, 90.0)


def _t_crit_975(df: int) -> float:
    """Two-sided 95% t critical value; small table + normal tail (repo has no
    scipy dependency). Same table as scripts/analyze_ema.py."""
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


def paired_summary(values):
    """Mean, 95% CI half-width, n over seeds.

    ``None`` for an empty list (a cell no seed reported), never 0. Named as in
    scripts/analyze_anneal_branch.py; here the entries are per-seed readouts
    rather than per-seed differences, and the CI is the across-seed CI."""
    n = len(values)
    if n == 0:
        return None
    mean = statistics.fmean(values)
    if n < 2:
        return {"n": n, "mean": mean, "ci95": None}
    sd = statistics.stdev(values)
    return {"n": n, "mean": mean, "ci95": _t_crit_975(n - 1) * sd / math.sqrt(n)}


def _mean_or_none(values):
    """Mean of a list, or None when it is empty (never a silent 0.0)."""
    return statistics.fmean(values) if values else None


def percentile(values, q: float):
    """The q-th percentile (q in [0, 100]) by linear interpolation between the
    two closest ranks — numpy's default method, implemented in pure stdlib
    (the repo's analyzers take no numpy dependency).

    ``None`` for an empty list. Non-finite entries are the caller's problem;
    ``analyze`` filters them out before calling."""
    if not values:
        return None
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (q / 100.0) * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    frac = position - low
    return float(ordered[low] + frac * (ordered[high] - ordered[low]))


def load_gate_runs(results_dir: Path):
    """Collect airbench_teleport_gate runs, keyed by seed.

    Duplicate seeds keep the earliest started_at (the runbook's re-run rule
    for preempted spot sweeps)."""
    runs = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("experiment") != EXPERIMENT:
            continue
        metrics = data.get("metrics", {})
        seed = data.get("seed")
        if seed is None or not metrics.get("snapshots"):
            continue
        if seed in runs and runs[seed]["started_at"] <= data.get("started_at", ""):
            continue
        runs[seed] = {
            "started_at": data.get("started_at", ""),
            "gpu_type": data.get("gpu_type"),
            "invariance_tol": metrics.get("invariance_tol"),
            "pair_names": metrics.get("pair_names") or [],
            "snapshots": metrics["snapshots"],
            "final_val_acc": metrics.get("final_val_acc"),
            "probe_seconds": metrics.get("probe_seconds"),
            "time_seconds": metrics.get("time_seconds"),
        }
    return runs


def _finite(values):
    """Drop non-finite entries (``_ratio`` yields NaN on a zero base norm)."""
    return [v for v in values if v is not None and math.isfinite(v)]


def _record_field(snapshot, record_key: str, field: str):
    """A field of ``snapshot[record_key]`` (best_random / best_refined), or
    None when the record is absent (no feasible point was found)."""
    record = snapshot.get(record_key)
    if not isinstance(record, dict):
        return None
    value = record.get(field)
    if value is None or not math.isfinite(value):
        return None
    return value


def analyze(results_dir: Path):
    runs = load_gate_runs(results_dir)
    if not runs:
        raise SystemExit(f"no {EXPERIMENT} runs found in {results_dir}")
    gpu_types = {run["gpu_type"] for run in runs.values()}
    if len(gpu_types) > 1:
        raise SystemExit(
            f"mixed GPU types across loaded runs: {sorted(map(str, gpu_types))}"
        )
    tols = {run["invariance_tol"] for run in runs.values()}
    tols.discard(None)
    if len(tols) > 1:
        raise SystemExit(f"loaded runs disagree on invariance_tol: {sorted(tols)}")

    seeds = sorted(runs)
    steps = sorted(
        {int(key) for run in runs.values() for key in run["snapshots"]}
    )

    out = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "gpu_type": next(iter(gpu_types)),
        "invariance_tol": next(iter(tols)) if tols else None,
        "snapshot_steps": steps,
        "per_snapshot": {},
        "overall": {},
    }

    for step in steps:
        key = str(step)
        present = [runs[s]["snapshots"][key] for s in seeds if key in runs[s]["snapshots"]]

        draw_p = {}
        for q in DRAW_PERCENTILES:
            per_seed = [
                percentile(
                    _finite([r.get("nuclear_sum_ratio") for r in snap.get("samples", [])]),
                    q,
                )
                for snap in present
            ]
            draw_p[q] = _mean_or_none([v for v in per_seed if v is not None])

        out["per_snapshot"][key] = {
            "n_snapshot_seeds": len(present),
            "n_feasible_mean": _mean_or_none(
                [
                    snap["n_feasible"]
                    for snap in present
                    if snap.get("n_feasible") is not None
                ]
            ),
            "max_abs_rel_dloss_max": max(
                (
                    snap["max_abs_rel_dloss"]
                    for snap in present
                    if snap.get("max_abs_rel_dloss") is not None
                ),
                default=None,
            ),
            "random_draw_nuclear_p50": draw_p[50.0],
            "random_draw_nuclear_p90": draw_p[90.0],
            "best_random_nuclear": paired_summary(
                [
                    v
                    for snap in present
                    if (v := _record_field(snap, "best_random", "nuclear_sum_ratio"))
                    is not None
                ]
            ),
            "best_refined_nuclear": paired_summary(
                [
                    v
                    for snap in present
                    if (v := _record_field(snap, "best_refined", "nuclear_sum_ratio"))
                    is not None
                ]
            ),
            "best_refined_total_grad": paired_summary(
                [
                    v
                    for snap in present
                    if (v := _record_field(snap, "best_refined", "total_grad_ratio"))
                    is not None
                ]
            ),
        }

    refined_means = [
        block["best_refined_nuclear"]["mean"]
        for block in out["per_snapshot"].values()
        if block["best_refined_nuclear"] is not None
    ]
    out["overall"] = {
        "min_over_snapshots_best_refined_nuclear_mean": (
            min(refined_means) if refined_means else None
        )
    }
    return out


def _fmt_summary(summary) -> str:
    if summary is None:
        return "n/a"
    if summary["ci95"] is None:
        return f"{summary['mean']:.4f} (n={summary['n']})"
    return f"{summary['mean']:.4f} ± {summary['ci95']:.4f}"


def _fmt_num(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def to_markdown(result) -> str:
    tol = result["invariance_tol"]
    lines = [
        "# Teleportation gate — gradient-size variation along the "
        "loss-invariant orbit (`airbench_teleport_gate`)",
        "",
        "Descriptive output of `scripts/analyze_teleport_gate.py` "
        "(docs/litreview/j-theory-theorem-sweep.md §6, program item 4 / T6). "
        "No pass/fail judgment is made here.",
        "",
        f"Runs: n={result['n_seeds']} dev seeds "
        f"({', '.join(str(s) for s in result['seeds'])}), "
        f"GPU {result['gpu_type']}; level-set tolerance |rel_dloss| < "
        + ("n/a" if tol is None else f"{tol:g}")
        + ".",
        "",
        "Ratios are relative to the un-teleported base point at the same "
        "training state (1.0000 = no change). "
        "`random p50/p90` are the per-seed median / 90th percentile of the "
        "drawn nuclear-sum ratios, averaged over seeds; `best random` and "
        "`best refined` are mean ± 95% CI over seeds.",
        "",
        "| snapshot step | n_feasible (mean) | worst \\|rel_dloss\\| | "
        "random p50 nuc | random p90 nuc | best random nuc | "
        "best refined nuc | best refined grad |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for step in result["snapshot_steps"]:
        block = result["per_snapshot"][str(step)]
        lines.append(
            f"| {step} | {_fmt_num(block['n_feasible_mean'], 1)} | "
            f"{_fmt_num(block['max_abs_rel_dloss_max'], 6)} | "
            f"{_fmt_num(block['random_draw_nuclear_p50'])} | "
            f"{_fmt_num(block['random_draw_nuclear_p90'])} | "
            f"{_fmt_summary(block['best_random_nuclear'])} | "
            f"{_fmt_summary(block['best_refined_nuclear'])} | "
            f"{_fmt_summary(block['best_refined_total_grad'])} |"
        )
    worst = result["overall"]["min_over_snapshots_best_refined_nuclear_mean"]
    lines += [
        "",
        "Smallest best-refined nuclear-sum ratio over snapshot steps "
        "(the most conservative training state): "
        + (
            "n/a — no snapshot found a feasible refined point."
            if worst is None
            else f"{worst:.4f} "
            f"({(worst - 1.0) * 100:+.2f}% vs the base point)."
        ),
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
