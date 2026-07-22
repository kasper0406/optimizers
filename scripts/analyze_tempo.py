#!/usr/bin/env python
"""Program #8 analysis: TempoMuon temporal trust ratio (deterministic).

Subcommands:
    passive  -- Phase A: per-matrix rho_hat vs LR under passive (kappa=0)
                TempoMuon; monotonicity check mirroring the per-direction
                WP1.2 result at matrix granularity.
    compare  -- Phase B: accuracy vs LR per arm (stock muon / tempomuon
                per-matrix / tempomuon global), seed-paired deltas.

Both read append-only results JSONs (scripts/run.py schema) and print a
markdown report to stdout; --json writes the aggregate numbers next to it.

Usage:
    uv run python scripts/analyze_tempo.py passive results/*.json
    uv run python scripts/analyze_tempo.py compare results/*.json --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Mid-training window for rho summaries: past EMA burn-in, before the last
# quarter of the linear anneal (occupancy collapses there, WP1.2 / program #2).
RHO_WINDOW = (50, 150)


def load_runs(paths: Sequence[str]) -> List[Dict[str, Any]]:
    runs = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        d["_path"] = p
        runs.append(d)
    return runs


def _opt_cfg(run: Dict[str, Any]) -> Dict[str, Any]:
    # results_io schema: config = {contents: <resolved dict>, path, sha256}
    return run["config"]["contents"]["optimizer"]


def run_lr(run: Dict[str, Any]) -> float:
    return float(_opt_cfg(run)["lr"])


def run_opt(run: Dict[str, Any]) -> str:
    name = _opt_cfg(run)["name"]
    if name == "tempomuon":
        scope = _opt_cfg(run).get("scope", "per_matrix")
        kappa = float(_opt_cfg(run).get("kappa", 0.0))
        if kappa == 0.0:
            return "tempomuon-passive"
        return f"tempomuon-{scope}"
    return name


def acc(run: Dict[str, Any]) -> float:
    return float(run["metrics"]["tta_val_acc"])


def window_rho(run: Dict[str, Any], lo: int, hi: int) -> Dict[str, float]:
    """Mean rho_hat per matrix over history steps in [lo, hi]."""
    hist = run["metrics"]["tempo_stats"]["history"]
    acc_: Dict[str, List[float]] = defaultdict(list)
    for snap in hist:
        if lo <= snap["step"] <= hi:
            for label, rho in snap["rho"].items():
                if rho is not None:
                    acc_[label].append(rho)
    return {k: sum(v) / len(v) for k, v in sorted(acc_.items()) if v}


def window_gain(run: Dict[str, Any], lo: int, hi: int) -> Dict[str, float]:
    hist = run["metrics"]["tempo_stats"]["history"]
    out: Dict[str, List[float]] = defaultdict(list)
    for snap in hist:
        if lo <= snap["step"] <= hi:
            gains = snap.get("gain")
            if gains is None and "global_gain" in snap:
                gains = {"GLOBAL": snap["global_gain"]}
            for label, g in (gains or {}).items():
                out[label].append(g)
    return {k: sum(v) / len(v) for k, v in sorted(out.items()) if v}


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def ranks(v: Sequence[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0  # average rank across the tie block
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return num / den if den else float("nan")


def mean_std(v: Sequence[float]) -> Tuple[float, float]:
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return m, math.sqrt(var)


# ------------------------------------------------------------------ passive


def analyze_passive(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    runs = [r for r in runs if "tempo_stats" in r["metrics"]]
    if not runs:
        raise SystemExit("no runs with tempo_stats found")
    by_lr: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_lr[run_lr(r)].append(r)

    lo, hi = RHO_WINDOW
    lines = [
        "# Program #8 Phase A: passive matrix-level rho vs LR",
        "",
        f"- runs: {len(runs)}; window: steps {lo}-{hi}; "
        "rho = mean bias-corrected EMA[cos(G_t,G_t-1)] per matrix",
        "",
    ]
    matrix_labels: List[str] = sorted(
        {k for r in runs for k in window_rho(r, lo, hi)}
    )
    header = "| lr | n | tta_acc mean±std | agg rho mean±std | " + " | ".join(
        matrix_labels
    ) + " |"
    lines += [header, "|" + "---|" * (4 + len(matrix_labels))]

    agg_by_lr: Dict[float, List[float]] = {}
    out: Dict[str, Any] = {"window": [lo, hi], "per_lr": {}}
    for lr in sorted(by_lr):
        rs = by_lr[lr]
        accs = [acc(r) for r in rs]
        per_matrix: Dict[str, List[float]] = defaultdict(list)
        aggs = []
        for r in rs:
            wr = window_rho(r, lo, hi)
            for k, v in wr.items():
                per_matrix[k].append(v)
            aggs.append(sum(wr.values()) / len(wr))
        agg_by_lr[lr] = aggs
        am, asd = mean_std(accs)
        gm, gsd = mean_std(aggs)
        cells = []
        for label in matrix_labels:
            mm, _ = mean_std(per_matrix[label]) if per_matrix[label] else (float("nan"), 0)
            cells.append(f"{mm:+.3f}")
        lines.append(
            f"| {lr} | {len(rs)} | {am:.4f}±{asd:.4f} | {gm:+.3f}±{gsd:.3f} | "
            + " | ".join(cells)
            + " |"
        )
        out["per_lr"][str(lr)] = {
            "n": len(rs),
            "acc_mean": am,
            "acc_std": asd,
            "agg_rho_mean": gm,
            "agg_rho_std": gsd,
            "per_matrix_rho_mean": {
                k: mean_std(v)[0] for k, v in sorted(per_matrix.items())
            },
        }

    xs, ys = [], []
    for lr, aggs in agg_by_lr.items():
        for a in aggs:
            xs.append(lr)
            ys.append(a)
    rho_s = spearman(xs, ys)
    out["spearman_lr_vs_agg_rho"] = rho_s
    lines += [
        "",
        f"Spearman(lr, per-run aggregate rho) = **{rho_s:+.3f}** "
        "(LR-monotonicity check; per-direction WP1.2 analogue predicts < 0)",
    ]
    out["report"] = "\n".join(lines)
    return out


# ------------------------------------------------------------------ compare


def analyze_compare(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm_lr: Dict[Tuple[str, float], Dict[int, float]] = defaultdict(dict)
    for r in runs:
        by_arm_lr[(run_opt(r), run_lr(r))][int(r["seed"])] = acc(r)

    arms = sorted({a for a, _ in by_arm_lr})
    lrs = sorted({l for _, l in by_arm_lr})
    baseline = "muon"
    lines = [
        "# Program #8 Phase B: accuracy vs LR per arm",
        "",
        "| lr | " + " | ".join(arms) + " | " + " | ".join(
            f"Δ({a}−{baseline}) paired" for a in arms if a != baseline
        ) + " |",
    ]
    lines.append("|" + "---|" * (1 + len(arms) + (len(arms) - 1)))
    out: Dict[str, Any] = {"arms": arms, "per_lr": {}}
    for lr in lrs:
        cells, deltas = [], []
        row: Dict[str, Any] = {}
        base = by_arm_lr.get((baseline, lr), {})
        for a in arms:
            d = by_arm_lr.get((a, lr), {})
            if d:
                m, s = mean_std(list(d.values()))
                cells.append(f"{m:.4f}±{s:.4f} (n={len(d)})")
                row[a] = {"mean": m, "std": s, "n": len(d)}
            else:
                cells.append("—")
        for a in arms:
            if a == baseline:
                continue
            d = by_arm_lr.get((a, lr), {})
            common = sorted(set(d) & set(base))
            if common:
                diffs = [d[s] - base[s] for s in common]
                dm, ds = mean_std(diffs)
                se = ds / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
                deltas.append(f"{dm*100:+.3f}pp ± {se*100:.3f} (n={len(diffs)})")
                row[f"delta_{a}"] = {
                    "mean_pp": dm * 100,
                    "se_pp": se * 100,
                    "n": len(diffs),
                }
            else:
                deltas.append("—")
        lines.append(f"| {lr} | " + " | ".join(cells + deltas) + " |")
        out["per_lr"][str(lr)] = row
    out["report"] = "\n".join(lines)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["passive", "compare"])
    ap.add_argument("results", nargs="+")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)
    runs = load_runs(args.results)
    out = analyze_passive(runs) if args.mode == "passive" else analyze_compare(runs)
    print(out["report"])
    if args.json_out:
        report = out.pop("report")
        Path(args.json_out).write_text(json.dumps(out, indent=2, sort_keys=True))
        out["report"] = report
    return 0


if __name__ == "__main__":
    sys.exit(main())
