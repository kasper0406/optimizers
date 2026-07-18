#!/usr/bin/env python
"""Results aggregation: results/*.json -> per-config mean/std/95% CI tables.

Usage:
    uv run python scripts/aggregate.py [paths ...] [--metric accuracy]
        [--gpu-type NAME] [--skip-invalid] [--out-md FILE] [--out-csv FILE]

Paths may be results JSON files or directories (default: results/). Every file
is validated through src.results_io.load_result; on top of the schema check,
the reproducibility fields are enforced here:

- git_sha must be a 40-hex commit SHA ("unknown" is rejected — a result
  without provenance cannot enter a comparison table),
- seed, gpu_type, wall_time_s, cost_usd must be present with correct types
  (schema), and a missing cost_usd (null) on a non-local gpu_type is warned
  about loudly (the human fills cloud costs per CLAUDE.md rule 5).

Invalid files abort the aggregation (exit 2) unless --skip-invalid is given,
in which case they are excluded with a warning.

GPU-type rule (plan section 0.1): all runs within any single comparison table
must share one gpu_type. Mixed gpu_types abort with a listing; use --gpu-type
to select the subset you want a table for.

Grouping: one row per (experiment, config path, config sha256). Columns: n,
seed range, <metric> mean/std/95% CI, wall_time_s mean/std/95% CI, total cost.
95% CI uses the Student-t critical value (scipy.stats.t); undefined for n < 2.

Output: markdown table to stdout, plus reports-ready markdown (--out-md) and
CSV (--out-csv). This tool never writes into results/ (append-only).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy import stats as scipy_stats

from src import results_io

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
LOCAL_GPU_TYPES = {"cpu", "mps"}


class AggregateError(ValueError):
    """Raised when aggregation must refuse (invalid files, mixed gpu_type...)."""


# ------------------------------------------------------------------ loading


def check_reproducibility_fields(result: Dict[str, Any], source: str) -> List[str]:
    """Extra checks beyond the results_io schema. Returns problem strings."""
    problems: List[str] = []
    sha = result.get("git_sha")
    if not isinstance(sha, str) or not GIT_SHA_RE.fullmatch(sha):
        problems.append(
            f"git_sha {sha!r} is not a 40-hex commit SHA (results without "
            "provenance cannot be aggregated)"
        )
    return problems


def load_results(
    paths: Iterable[Path], skip_invalid: bool = False
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load + validate results files. Returns (valid results, warnings).

    Each returned result gains a '_source' key with its file path.
    Raises AggregateError on any invalid file unless skip_invalid.
    """
    files: List[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.extend(sorted(p for p in path.glob("*.json") if p.is_file()))
        else:
            files.append(path)
    if not files:
        raise AggregateError("no results JSON files found in the given paths")

    valid: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []
    for path in files:
        try:
            result = results_io.load_result(path)
        except (results_io.ResultsValidationError, ValueError, OSError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        problems = check_reproducibility_fields(result, str(path))
        if problems:
            errors.append(f"{path}: " + "; ".join(problems))
            continue
        if result["cost_usd"] is None and result["gpu_type"] not in LOCAL_GPU_TYPES:
            warnings.append(
                f"{path}: cost_usd is null for gpu_type {result['gpu_type']!r} — "
                "cloud costs must be human-filled before reporting"
            )
        result["_source"] = str(path)
        valid.append(result)

    if errors:
        message = "invalid results file(s):\n  - " + "\n  - ".join(errors)
        if not skip_invalid:
            raise AggregateError(
                message + "\n(use --skip-invalid to aggregate without them)"
            )
        warnings.append("SKIPPED " + message)
    if not valid:
        raise AggregateError("no valid results files to aggregate")
    return valid, warnings


def enforce_single_gpu_type(results: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for result in results:
        counts[result["gpu_type"]] = counts.get(result["gpu_type"], 0) + 1
    if len(counts) > 1:
        listing = ", ".join(f"{k!r}: {v} run(s)" for k, v in sorted(counts.items()))
        raise AggregateError(
            "refusing to aggregate across mixed gpu_type within one comparison "
            f"table (plan section 0.1): {listing}. Re-run with --gpu-type to "
            "select a single type."
        )
    return next(iter(counts))


# ------------------------------------------------------------------- stats


def mean_std_ci95(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n >= 2:
        std = float(arr.std(ddof=1))
        ci95 = float(scipy_stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))
    else:
        std = float("nan")
        ci95 = float("nan")
    return {"n": n, "mean": mean, "std": std, "ci95": ci95}


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def group_results(
    results: List[Dict[str, Any]]
) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for result in results:
        key = (
            result["experiment"],
            result["config"]["path"],
            result["config"]["sha256"],
        )
        groups.setdefault(key, []).append(result)
    return groups


def summarize(results: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    """One summary row per (experiment, config path, config sha256)."""
    rows: List[Dict[str, Any]] = []
    for (experiment, cfg_path, cfg_sha), group in sorted(group_results(results).items()):
        seeds = sorted(r["seed"] for r in group)
        dupes = {s for s in seeds if seeds.count(s) > 1}
        if dupes:
            raise AggregateError(
                f"duplicate seed(s) {sorted(dupes)} for config {cfg_path} "
                f"(sha {cfg_sha[:8]}) — one result per (config, seed) in a table"
            )
        metric_values = []
        missing = []
        for r in group:
            value = r["metrics"].get(metric)
            if _is_number(value):
                metric_values.append(float(value))
            else:
                missing.append(r["_source"])
        if metric_values and missing:
            raise AggregateError(
                f"metric {metric!r} present in some but not all runs of "
                f"{cfg_path} (sha {cfg_sha[:8]}); missing/non-numeric in:\n  - "
                + "\n  - ".join(missing)
            )
        wall = mean_std_ci95([float(r["wall_time_s"]) for r in group])
        costs = [r["cost_usd"] for r in group if _is_number(r["cost_usd"])]
        rows.append(
            {
                "experiment": experiment,
                "config": cfg_path,
                "config_sha256": cfg_sha,
                "gpu_type": group[0]["gpu_type"],
                "n": len(group),
                "seed_min": seeds[0],
                "seed_max": seeds[-1],
                "metric_name": metric,
                "metric": mean_std_ci95(metric_values) if metric_values else None,
                "wall_time_s": wall,
                "total_cost_usd": float(sum(costs)) if costs else None,
            }
        )
    return rows


# ---------------------------------------------------------------- rendering


def _fmt(x: Optional[float], digits: int = 5) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{digits}g}"


def render_markdown(rows: List[Dict[str, Any]], gpu_type: str, metric: str) -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Aggregated results",
        "",
        f"- generated: {stamp}",
        f"- gpu_type: `{gpu_type}` (single-type table per plan section 0.1)",
        f"- metric: `{metric}`; CI = 95% Student-t",
        "",
        f"| config | experiment | n | seeds | {metric} mean | std | 95% CI ± "
        "| wall mean (s) | wall 95% CI ± | total cost (USD) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        m = row["metric"]
        lines.append(
            "| {config} | {experiment} | {n} | {smin}-{smax} | {mm} | {ms} | {mc} "
            "| {wm} | {wc} | {cost} |".format(
                config=row["config"],
                experiment=row["experiment"],
                n=row["n"],
                smin=row["seed_min"],
                smax=row["seed_max"],
                mm=_fmt(m["mean"]) if m else "-",
                ms=_fmt(m["std"]) if m else "-",
                mc=_fmt(m["ci95"]) if m else "-",
                wm=_fmt(row["wall_time_s"]["mean"]),
                wc=_fmt(row["wall_time_s"]["ci95"]),
                cost=_fmt(row["total_cost_usd"], digits=4),
            )
        )
    lines.append("")
    return "\n".join(lines)


CSV_FIELDS = [
    "config", "experiment", "config_sha256", "gpu_type", "n",
    "seed_min", "seed_max", "metric_name",
    "metric_mean", "metric_std", "metric_ci95",
    "wall_time_mean_s", "wall_time_std_s", "wall_time_ci95_s",
    "total_cost_usd",
]


def render_csv(rows: List[Dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        m = row["metric"]
        writer.writerow(
            {
                "config": row["config"],
                "experiment": row["experiment"],
                "config_sha256": row["config_sha256"],
                "gpu_type": row["gpu_type"],
                "n": row["n"],
                "seed_min": row["seed_min"],
                "seed_max": row["seed_max"],
                "metric_name": row["metric_name"],
                "metric_mean": m["mean"] if m else "",
                "metric_std": m["std"] if m else "",
                "metric_ci95": m["ci95"] if m else "",
                "wall_time_mean_s": row["wall_time_s"]["mean"],
                "wall_time_std_s": row["wall_time_s"]["std"],
                "wall_time_ci95_s": row["wall_time_s"]["ci95"],
                "total_cost_usd": "" if row["total_cost_usd"] is None else row["total_cost_usd"],
            }
        )
    return buffer.getvalue()


# --------------------------------------------------------------------- main


def _refuse_results_dir_write(path: Path) -> None:
    resolved = Path(path).resolve()
    results_dir = results_io.RESULTS_DIR.resolve()
    if resolved == results_dir or results_dir in resolved.parents:
        raise AggregateError(
            f"refusing to write {path} under results/ (append-only, results only)"
        )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate results JSONs into per-config mean/std/95% CI tables."
    )
    parser.add_argument(
        "paths", nargs="*", type=Path, default=None,
        help="Results files and/or directories (default: results/)",
    )
    parser.add_argument("--metric", default="accuracy", help="Metric key in metrics{} (default: accuracy)")
    parser.add_argument("--gpu-type", default=None, help="Only aggregate results with this gpu_type")
    parser.add_argument("--skip-invalid", action="store_true", help="Warn on invalid files instead of aborting")
    parser.add_argument("--out-md", type=Path, default=None, help="Write markdown table here (reports-ready)")
    parser.add_argument("--out-csv", type=Path, default=None, help="Write CSV table here")
    args = parser.parse_args(list(argv) if argv is not None else None)

    paths = args.paths if args.paths else [results_io.RESULTS_DIR]
    try:
        for out in (args.out_md, args.out_csv):
            if out is not None:
                _refuse_results_dir_write(out)
        results, warnings = load_results(paths, skip_invalid=args.skip_invalid)
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        if args.gpu_type is not None:
            results = [r for r in results if r["gpu_type"] == args.gpu_type]
            if not results:
                raise AggregateError(f"no results with gpu_type {args.gpu_type!r}")
        gpu_type = enforce_single_gpu_type(results)
        rows = summarize(results, args.metric)
    except AggregateError as exc:
        print(f"aggregate.py: {exc}", file=sys.stderr)
        return 2

    markdown = render_markdown(rows, gpu_type, args.metric)
    print(markdown)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown)
        print(f"wrote {args.out_md}", file=sys.stderr)
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        args.out_csv.write_text(render_csv(rows))
        print(f"wrote {args.out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
