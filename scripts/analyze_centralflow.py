"""Central-flow Muon v0 analysis — "does the explicit curvature-penalty term
reproduce at cold LR what the high-LR oscillation does?"
(docs/litreview/j-theory-theorem-sweep.md §6, flow-first program item 2).

Reads `airbench_centralflow` results JSONs. The experiment is a 2x2 over
(Muon LR scale, explicit central-flow term on/off), all arms uncompiled and on
dev seeds:

  A = (lr_scale 1.00, term off)  — stock reference
  B = (lr_scale 0.25, term off)  — cold LR, the accuracy the low-LR run gets
                                   without any stand-in for the oscillation
  C = (lr_scale 0.25, term on)   — the mechanism arm: cold LR + the explicit
                                   time-averaged curvature-penalty term
  D = (lr_scale 1.00, term on)   — the term on top of stock LR

Every seed present in ALL FOUR arms is paired, and the script reports the four
paired differences that carry the reading:

  C − B  the mechanism effect: what the explicit term buys at cold LR;
  C − A  the recovery gap: how far the mechanism arm still sits from stock;
  B − A  the cost of the cold LR the term has to make up;
  D − A  the term on top of stock LR (is it additive, or only a stand-in?).

``recovery_fraction`` = mean(C − B) / mean(A − B) summarizes C − B as a
fraction of the gap the cold LR opened: 1.0 would mean the explicit term fully
substitutes for the high-LR oscillation, 0.0 that it does nothing. It is
``None`` when the stock-vs-cold gap is not a positive, resolvable quantity
(mean(A − B) < 1e-4), where the ratio would be meaningless.

Also reports the term's own telemetry on the arms that run it (C and D): the
penalty-gradient norm at the first and last logged step (is the term growing
or dying?) and the mean tracked curvature at the middle of the run.

DESCRIPTIVE OUTPUT ONLY: this script states quantities; it makes no pass/fail
judgment and evaluates no gate (CLAUDE.md: gate decisions are human-only).

Usage:
    uv run python scripts/analyze_centralflow.py [results_dir] \
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

EXPERIMENT = "airbench_centralflow"

# Arm identity = (cf.lr_scale, cf.enabled), the only two config knobs the 2x2
# varies (configs/wpj_cf_{stock,cold,cold_on,stock_on}.yaml).
ARMS = {
    "A": (1.00, False),
    "B": (0.25, False),
    "C": (0.25, True),
    "D": (1.00, True),
}
ARM_LABELS = {
    "A": "stock LR, term off (reference)",
    "B": "cold LR (0.25x), term off",
    "C": "cold LR (0.25x) + CF term (mechanism arm)",
    "D": "stock LR + CF term",
}
# The four paired differences, as (name, arm, baseline arm).
DELTAS = (
    ("C_minus_B", "C", "B"),
    ("C_minus_A", "C", "A"),
    ("B_minus_A", "B", "A"),
    ("D_minus_A", "D", "A"),
)
DELTA_LABELS = {
    "C_minus_B": "mechanism effect (term at cold LR)",
    "C_minus_A": "recovery gap vs stock",
    "B_minus_A": "cost of the cold LR",
    "D_minus_A": "term on top of stock LR",
}
# Below this the stock-vs-cold gap is not resolvable and recovery_fraction is
# reported as None rather than a ratio of noise.
RECOVERY_MIN_GAP = 1e-4
# cf.lr_scale is a float read back from JSON; match arms with a tolerance.
LR_SCALE_TOL = 1e-9


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


def arm_of(cf) -> str | None:
    """The arm letter for a run's ``metrics.cf`` block, or None when the
    (lr_scale, enabled) pair is not one of the four 2x2 cells."""
    if not isinstance(cf, dict) or cf.get("lr_scale") is None:
        return None
    try:
        scale = float(cf["lr_scale"])
    except (TypeError, ValueError):
        return None
    enabled = bool(cf.get("enabled"))
    for arm, (arm_scale, arm_enabled) in ARMS.items():
        if enabled == arm_enabled and abs(scale - arm_scale) <= LR_SCALE_TOL:
            return arm
    return None


def load_arms(results_dir: Path):
    """Collect airbench_centralflow runs, keyed by (arm, seed).

    Duplicate (arm, seed) pairs keep the earliest started_at (the runbook's
    re-run rule for preempted spot sweeps). Runs whose cf block is outside the
    2x2 are ignored."""
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
        arm = arm_of(metrics.get("cf"))
        if seed is None or arm is None or metrics.get("val_acc") is None:
            continue
        key = (arm, seed)
        if key in runs and runs[key]["started_at"] <= data.get("started_at", ""):
            continue
        runs[key] = {
            "started_at": data.get("started_at", ""),
            "gpu_type": data.get("gpu_type"),
            "cf": metrics.get("cf") or {},
            "val_accs": metrics.get("val_accs") or [],
            "val_acc": metrics.get("val_acc"),
            "tta_val_acc": metrics.get("tta_val_acc"),
            "cf_timeseries": metrics.get("cf_timeseries") or [],
            "time_seconds": metrics.get("time_seconds"),
            "steps": metrics.get("steps"),
        }
    return runs


def _curve_mean(runs, seeds):
    """Element-wise mean of the per-epoch val_accs over seeds, truncated to the
    shortest curve present (None when no seed reported one)."""
    curves = [runs[s]["val_accs"] for s in seeds if runs[s]["val_accs"]]
    if not curves:
        return None
    width = min(len(c) for c in curves)
    return [statistics.fmean(c[i] for c in curves) for i in range(width)]


def _telemetry(runs, seeds):
    """Central-flow term telemetry over the seeds of one term-on arm."""
    series = [runs[s]["cf_timeseries"] for s in seeds if runs[s]["cf_timeseries"]]
    if not series:
        return {
            "n_seeds_with_timeseries": 0,
            "penalty_norm_first": None,
            "penalty_norm_last": None,
            "curvature_mean_mid": None,
        }
    return {
        "n_seeds_with_timeseries": len(series),
        "penalty_norm_first": _mean_or_none(
            [ts[0]["penalty_grad_norm"] for ts in series]
        ),
        "penalty_norm_last": _mean_or_none(
            [ts[-1]["penalty_grad_norm"] for ts in series]
        ),
        "curvature_mean_mid": _mean_or_none(
            [ts[len(ts) // 2]["curvature_mean"] for ts in series]
        ),
    }


def recovery_fraction(deltas, min_gap: float = RECOVERY_MIN_GAP):
    """mean(C − B) / mean(A − B), or None when the stock-vs-cold gap is not a
    positive, resolvable quantity (A must exceed B by at least min_gap)."""
    mechanism = deltas.get("C_minus_B", {}).get("val_delta")
    cold_cost = deltas.get("B_minus_A", {}).get("val_delta")
    if mechanism is None or cold_cost is None:
        return None
    gap = -cold_cost["mean"]  # A − B
    if gap < min_gap:
        return None
    return mechanism["mean"] / gap


def analyze(results_dir: Path):
    runs = load_arms(results_dir)
    if not runs:
        raise SystemExit(f"no {EXPERIMENT} runs found in {results_dir}")
    per_arm_seeds = {
        arm: {seed for (a, seed) in runs if a == arm} for arm in ARMS
    }
    paired_seeds = sorted(set.intersection(*per_arm_seeds.values()))
    if not paired_seeds:
        counts = ", ".join(f"{arm}: {len(per_arm_seeds[arm])}" for arm in sorted(ARMS))
        raise SystemExit(
            f"no seed-paired {EXPERIMENT} arms found in {results_dir} ({counts})"
        )
    gpu_types = {run["gpu_type"] for run in runs.values()}
    if len(gpu_types) > 1:
        raise SystemExit(
            f"mixed GPU types across loaded runs: {sorted(map(str, gpu_types))}"
        )

    by_arm = {
        arm: {seed: runs[(arm, seed)] for seed in paired_seeds} for arm in ARMS
    }

    out = {
        "n_paired_seeds": len(paired_seeds),
        "seeds": paired_seeds,
        "gpu_type": next(iter(gpu_types)),
        "arms": {arm: {"lr_scale": ARMS[arm][0], "enabled": ARMS[arm][1]} for arm in ARMS},
        "per_arm": {},
        "paired_deltas": {},
        "cf_telemetry": {},
    }

    for arm in sorted(ARMS):
        arm_runs = by_arm[arm]
        out["per_arm"][arm] = {
            "label": ARM_LABELS[arm],
            "final_val_mean": _mean_or_none(
                [arm_runs[s]["val_acc"] for s in paired_seeds]
            ),
            "final_tta_mean": _mean_or_none(
                [
                    arm_runs[s]["tta_val_acc"]
                    for s in paired_seeds
                    if arm_runs[s]["tta_val_acc"] is not None
                ]
            ),
            "val_accs_mean": _curve_mean(arm_runs, paired_seeds),
        }

    for name, arm, base in DELTAS:
        out["paired_deltas"][name] = {
            "label": DELTA_LABELS[name],
            "val_delta": paired_summary(
                [
                    by_arm[arm][s]["val_acc"] - by_arm[base][s]["val_acc"]
                    for s in paired_seeds
                    if by_arm[arm][s]["val_acc"] is not None
                    and by_arm[base][s]["val_acc"] is not None
                ]
            ),
            "tta_delta": paired_summary(
                [
                    by_arm[arm][s]["tta_val_acc"] - by_arm[base][s]["tta_val_acc"]
                    for s in paired_seeds
                    if by_arm[arm][s]["tta_val_acc"] is not None
                    and by_arm[base][s]["tta_val_acc"] is not None
                ]
            ),
        }

    for arm in ("C", "D"):
        out["cf_telemetry"][arm] = _telemetry(by_arm[arm], paired_seeds)

    out["recovery_fraction"] = recovery_fraction(out["paired_deltas"])
    return out


def _fmt_delta(summary) -> str:
    if summary is None:
        return "n/a"
    if summary["ci95"] is None:
        return f"{summary['mean']:+.4f} (n={summary['n']})"
    return f"{summary['mean']:+.4f} ± {summary['ci95']:.4f}"


def _fmt_num(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def to_markdown(result) -> str:
    lines = [
        "# Central-flow Muon v0 — does the explicit term stand in for the "
        "high-LR oscillation? (`airbench_centralflow`)",
        "",
        "Descriptive output of `scripts/analyze_centralflow.py` "
        "(docs/litreview/j-theory-theorem-sweep.md §6, program item 2). "
        "No pass/fail judgment is made here.",
        "",
        f"Seed-paired 2x2: n={result['n_paired_seeds']} dev seeds "
        f"({', '.join(str(s) for s in result['seeds'])}), "
        f"GPU {result['gpu_type']}. Every seed is present in all four arms.",
        "",
        "| arm | LR scale | CF term | final val | final TTA | |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in sorted(ARMS):
        block = result["per_arm"][arm]
        cfg = result["arms"][arm]
        lines.append(
            f"| {arm} | {cfg['lr_scale']:.2f} | "
            f"{'on' if cfg['enabled'] else 'off'} | "
            f"{_fmt_num(block['final_val_mean'])} | "
            f"{_fmt_num(block['final_tta_mean'])} | {block['label']} |"
        )
    lines += [
        "",
        "## Paired differences (per-seed, mean ± 95% CI)",
        "",
        "| difference | val | TTA | reading |",
        "| --- | --- | --- | --- |",
    ]
    for name, _arm, _base in DELTAS:
        block = result["paired_deltas"][name]
        lines.append(
            f"| {name.replace('_minus_', ' − ')} | "
            f"{_fmt_delta(block['val_delta'])} | "
            f"{_fmt_delta(block['tta_delta'])} | {block['label']} |"
        )
    fraction = result["recovery_fraction"]
    lines += [
        "",
        "recovery_fraction = mean(C − B) / mean(A − B) = "
        + (
            "n/a — the stock-vs-cold gap is not positive and resolvable "
            f"(< {RECOVERY_MIN_GAP:g}), so the ratio is undefined."
            if fraction is None
            else f"{fraction:.3f} "
            f"({fraction * 100:.1f}% of the accuracy the cold LR gave up is "
            "recovered by the explicit term)."
        ),
        "",
        "## Central-flow term telemetry (term-on arms)",
        "",
    ]
    for arm in ("C", "D"):
        tel = result["cf_telemetry"][arm]
        lines.append(
            f"- Arm {arm} ({ARM_LABELS[arm]}): penalty grad norm "
            f"{_fmt_num(tel['penalty_norm_first'], 6)} (first logged step) → "
            f"{_fmt_num(tel['penalty_norm_last'], 6)} (last); mid-run mean "
            f"curvature {_fmt_num(tel['curvature_mean_mid'], 6)}; "
            f"{tel['n_seeds_with_timeseries']}/{result['n_paired_seeds']} "
            "seeds logged a timeseries."
        )
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
