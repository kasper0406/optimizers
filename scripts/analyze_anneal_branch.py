"""Anneal-dissection analysis — "how short is the last mile?"
(docs/litreview/j-theory-theorem-sweep.md §6, flow-first program item 1).

Reads `airbench_anneal_branch` results JSONs (constant-LR base trajectory;
at each branch step the training state is snapshotted and a linear anneal of
length k is branched off over a shared batch stream) and pairs every seed with
its own stock-schedule final from the T1 results (experiment `airbench_ema`
with metrics.lr_schedule == "linear", same dev seeds). It answers:

  (i)   accuracy vs k: for each branch point T_b and anneal length k, the
        per-seed PAIRED difference between the branch accuracy and the SAME
        seed's fully-annealed stock final (raw and TTA), with a 95% CI;
  (ii)  k*: the smallest anneal length whose paired mean val delta lands
        within K_STAR_TOL (0.2pp) of the stock final — the saturation point
        that measures how much of the anneal is fast dynamical relaxation
        rather than slow walking;
  (iii) steps_saved = stock budget − (T_b + k*): the steps that "constant LR
        + short anneal" buys over the tuned schedule at matched accuracy.

Also reports the pre-anneal (base) accuracy at each branch point and the
constant-arm final at the full budget, so the anneal's contribution can be
read off directly.

DESCRIPTIVE OUTPUT ONLY: this script states quantities; it makes no pass/fail
judgment and evaluates no gate (CLAUDE.md: gate decisions are human-only).

Usage:
    uv run python scripts/analyze_anneal_branch.py [results_dir] \
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

# k* rule: the smallest anneal length whose paired mean val delta vs the stock
# final is at least this (i.e. no worse than 0.2pp below the tuned schedule).
K_STAR_TOL = -0.002


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


def paired_summary(deltas):
    """Mean, 95% CI half-width, n for a list of per-seed paired differences.

    ``None`` for an empty list (a cell no paired seed reported), never 0."""
    n = len(deltas)
    if n == 0:
        return None
    mean = statistics.fmean(deltas)
    if n < 2:
        return {"n": n, "mean": mean, "ci95": None}
    sd = statistics.stdev(deltas)
    return {"n": n, "mean": mean, "ci95": _t_crit_975(n - 1) * sd / math.sqrt(n)}


def _mean_or_none(values):
    """Mean of a list, or None when it is empty (never a silent 0.0)."""
    return statistics.fmean(values) if values else None


def load_branch_runs(results_dir: Path):
    """Collect airbench_anneal_branch runs, keyed by seed.

    Duplicate seeds keep the earliest started_at (the runbook's re-run rule
    for preempted spot sweeps)."""
    runs = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("experiment") != "airbench_anneal_branch":
            continue
        metrics = data.get("metrics", {})
        seed = data.get("seed")
        if seed is None or "branches" not in metrics:
            continue
        if seed in runs and runs[seed]["started_at"] <= data.get("started_at", ""):
            continue
        runs[seed] = {
            "started_at": data.get("started_at", ""),
            "gpu_type": data.get("gpu_type"),
            "branch_steps": [int(b) for b in metrics.get("branch_steps", [])],
            "anneal_lengths": [int(k) for k in metrics.get("anneal_lengths", [])],
            "base_val_accs": metrics.get("base_val_accs") or {},
            "branches": metrics["branches"],
            "final_val_acc": metrics.get("final_val_acc"),
            "final_tta_val_acc": metrics.get("final_tta_val_acc"),
        }
    return runs


def load_stock_finals(results_dir: Path):
    """Collect the stock-schedule finals: airbench_ema runs on the LINEAR arm,
    keyed by seed (earliest started_at wins, as above).

    metrics.val_acc / metrics.tta_val_acc of that arm are the fully-annealed
    finals the branches are compared against (see scripts/analyze_ema.py)."""
    runs = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("experiment") != "airbench_ema":
            continue
        metrics = data.get("metrics", {})
        seed = data.get("seed")
        if seed is None or metrics.get("lr_schedule") != "linear":
            continue
        if seed in runs and runs[seed]["started_at"] <= data.get("started_at", ""):
            continue
        runs[seed] = {
            "started_at": data.get("started_at", ""),
            "gpu_type": data.get("gpu_type"),
            "val_acc": metrics.get("val_acc"),
            "tta_val_acc": metrics.get("tta_val_acc"),
            "steps": metrics.get("steps"),
        }
    return runs


def k_star(per_k, tol: float = K_STAR_TOL):
    """Smallest k whose paired mean val delta is >= tol; None if none is."""
    for key in sorted(per_k, key=int):
        summary = per_k[key]["val_delta"]
        if summary is not None and summary["mean"] >= tol:
            return int(key)
    return None


def analyze(results_dir: Path):
    branch = load_branch_runs(results_dir)
    stock = load_stock_finals(results_dir)
    paired_seeds = sorted(set(branch) & set(stock))
    if not paired_seeds:
        raise SystemExit(
            f"no seed-paired anneal-branch runs found in {results_dir} "
            f"(airbench_anneal_branch: {len(branch)}, "
            f"airbench_ema linear: {len(stock)})"
        )
    gpu_types = {run["gpu_type"] for run in branch.values()} | {
        run["gpu_type"] for run in stock.values()
    }
    if len(gpu_types) > 1:
        raise SystemExit(
            f"mixed GPU types across loaded runs: {sorted(map(str, gpu_types))}"
        )

    # stock budget: the step count the tuned schedule spends (steps_saved is
    # measured against it). Paired stock runs must agree on it.
    budgets = {
        stock[s]["steps"] for s in paired_seeds if stock[s]["steps"] is not None
    }
    if len(budgets) > 1:
        raise SystemExit(
            f"paired stock (linear) runs disagree on the step budget: "
            f"{sorted(budgets)}"
        )
    total_steps = int(next(iter(budgets))) if budgets else None

    reference = branch[paired_seeds[0]]
    branch_steps = reference["branch_steps"]
    anneal_lengths = reference["anneal_lengths"]

    out = {
        "n_paired_seeds": len(paired_seeds),
        "seeds": paired_seeds,
        "gpu_type": next(iter(gpu_types)),
        "total_steps": total_steps,
        "k_star_tol": K_STAR_TOL,
        "branch_steps": branch_steps,
        "anneal_lengths": anneal_lengths,
        "stock_final_val_mean": statistics.fmean(
            stock[s]["val_acc"] for s in paired_seeds
        ),
        "stock_final_tta_mean": _mean_or_none(
            [
                stock[s]["tta_val_acc"]
                for s in paired_seeds
                if stock[s]["tta_val_acc"] is not None
            ]
        ),
        "constant_final_val_mean": _mean_or_none(
            [
                branch[s]["final_val_acc"]
                for s in paired_seeds
                if branch[s]["final_val_acc"] is not None
            ]
        ),
        "constant_final_tta_mean": _mean_or_none(
            [
                branch[s]["final_tta_val_acc"]
                for s in paired_seeds
                if branch[s]["final_tta_val_acc"] is not None
            ]
        ),
        "per_branch": {},
    }

    for t_b in branch_steps:
        key_b = str(t_b)
        per_k = {}
        for k in anneal_lengths:
            key_k = str(k)
            cells = [
                (branch[s]["branches"].get(key_b, {}).get(key_k), stock[s])
                for s in paired_seeds
            ]
            per_k[key_k] = {
                "val_delta": paired_summary(
                    [
                        cell["val_acc"] - ref["val_acc"]
                        for cell, ref in cells
                        if cell is not None
                        and cell.get("val_acc") is not None
                        and ref["val_acc"] is not None
                    ]
                ),
                "tta_delta": paired_summary(
                    [
                        cell["tta_val_acc"] - ref["tta_val_acc"]
                        for cell, ref in cells
                        if cell is not None
                        and cell.get("tta_val_acc") is not None
                        and ref["tta_val_acc"] is not None
                    ]
                ),
            }
        best = k_star(per_k)
        out["per_branch"][key_b] = {
            "base_val_mean": _mean_or_none(
                [
                    branch[s]["base_val_accs"][key_b]
                    for s in paired_seeds
                    if branch[s]["base_val_accs"].get(key_b) is not None
                ]
            ),
            "per_k": per_k,
            "k_star": best,
            "steps_saved": (
                total_steps - (t_b + best)
                if best is not None and total_steps is not None
                else None
            ),
        }
    return out


def _fmt_delta(summary) -> str:
    if summary is None:
        return "n/a"
    if summary["ci95"] is None:
        return f"{summary['mean']:+.4f} (n={summary['n']})"
    return f"{summary['mean']:+.4f} ± {summary['ci95']:.4f}"


def _fmt_acc(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def to_markdown(result) -> str:
    lines = [
        "# Anneal dissection — accuracy vs anneal length "
        "(`airbench_anneal_branch`)",
        "",
        "Descriptive output of `scripts/analyze_anneal_branch.py` "
        "(docs/litreview/j-theory-theorem-sweep.md §6, program item 1). "
        "No pass/fail judgment is made here.",
        "",
        f"Seed-paired runs: n={result['n_paired_seeds']} dev seeds, "
        f"GPU {result['gpu_type']}; stock budget "
        f"{result['total_steps']} steps.",
        "",
        "Baseline = each seed's OWN stock linear-schedule final "
        "(`airbench_ema`, lr_schedule=linear): val "
        f"{_fmt_acc(result['stock_final_val_mean'])} "
        f"(TTA {_fmt_acc(result['stock_final_tta_mean'])}).",
        "",
        "Constant-arm final at the full budget (no anneal at all): val "
        f"{_fmt_acc(result['constant_final_val_mean'])} "
        f"(TTA {_fmt_acc(result['constant_final_tta_mean'])}).",
        "",
        f"k* rule: smallest anneal length k whose paired mean val delta is "
        f">= {result['k_star_tol']:+.4f}.",
        "",
    ]
    for t_b in result["branch_steps"]:
        block = result["per_branch"][str(t_b)]
        lines += [
            f"## Branch step {t_b} "
            f"(pre-anneal base val {_fmt_acc(block['base_val_mean'])})",
            "",
            "| k | val delta vs stock final | TTA delta vs stock final |",
            "| --- | --- | --- |",
        ]
        for k in result["anneal_lengths"]:
            cell = block["per_k"][str(k)]
            lines.append(
                f"| {k} | {_fmt_delta(cell['val_delta'])} | "
                f"{_fmt_delta(cell['tta_delta'])} |"
            )
        if block["k_star"] is None:
            lines += [
                "",
                f"k* = none — no tested anneal length reaches within "
                f"{abs(result['k_star_tol']):.4f} of the stock final at "
                f"branch step {t_b}.",
                "",
            ]
        else:
            saved = block["steps_saved"]
            lines += [
                "",
                f"k* = {block['k_star']} → steps saved "
                + (
                    f"{saved} ({t_b} + {block['k_star']} vs "
                    f"{result['total_steps']} stock steps)."
                    if saved is not None
                    else "n/a (no stock step budget in the paired runs)."
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
