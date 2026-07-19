#!/usr/bin/env python
"""Deterministic, DESCRIPTIVE analysis for the Phase-1 disambiguation runs.

Two independent sections (each optional, at least one required):

1. Sampling comparison (``--baseline`` vs ``--with-replacement``): phase-wise
   fractions of direction-snapshots with lag-1 autocorrelation rho < -0.2,
   per beta and per direction kind (all / top / bulk), with bootstrap CIs
   over runs (each run = one sidecar; the per-run statistic is the pooled
   snapshot fraction) and the bootstrap CI of the between-set difference.
   Reported for the same two variants as scripts/analyze_phase1.py: raw
   snapshots and burn-in (n_since_reset >= 10).

2. HVP calibration agreement (``--hvp``): for HVP-enabled sidecars, the
   implied eta*lambda (amplitude ratio) vs HVP-measured eta*lambda
   (lr * lambda_hvp) pairs of oscillating-classified directions -- the exact
   matching rule of the eta-lambda calibration plot
   (src.instrument.plots.collect_calibration_points) -- summarized per beta
   as n, Pearson r, Spearman rho, and median relative error
   |implied - hvp| / |hvp|.

DESCRIPTIVE OUTPUT ONLY: this script states quantities and intervals; it
makes no pass/fail judgment (Gate decisions are human-only, CLAUDE.md).
Deterministic: fixed bootstrap seed, sorted keys, no timestamps -- identical
inputs produce byte-identical outputs.

Usage:
    uv run python scripts/analyze_disambiguation.py \
        --baseline results/wp12_sidecars/ \
        --with-replacement results/withreplacement_sidecars/ \
        --hvp results/hvp_sidecars/ [--lr 0.24] \
        --out-md reports/wp12-disambiguation.md \
        --out-json reports/wp12-disambiguation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.instrument.plots import (  # noqa: E402
    _lr_from_sidecar_siblings,
    collect_calibration_points,
    load_sidecar_directory,
)

RHO_THRESHOLD = -0.2
BURN_IN_VARIANTS = [(0, "raw"), (10, "burn_in_10")]
KINDS = ("all", "top", "bulk")
CI_LO, CI_HI = 2.5, 97.5


# ------------------------------------------------------------------- loading


def load_sidecars(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """(run_name, log) per ``*.instrumentation.json`` sidecar in a directory
    (or a single sidecar file), sorted by name for determinism."""
    p = Path(path)
    files = sorted(p.glob("*.instrumentation.json")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no *.instrumentation.json sidecars in {p}")
    return [(f.name, json.loads(f.read_text())) for f in files]


# --------------------------------------------- section 1: sampling comparison


def per_run_fraction(
    log: Dict[str, Any],
    beta: str,
    phase_lo: int,
    phase_hi: int,
    burn_in: int,
    kind: str,
) -> Tuple[Optional[float], int]:
    """(fraction of snapshots with rho < RHO_THRESHOLD, n snapshots) for one
    run, pooled over matrices/directions; fraction None when n == 0."""
    n = 0
    hits = 0
    for mat in log["matrices"].values():
        for d in mat["directions"]:
            if kind != "all" and d["kind"] != kind:
                continue
            pb = d["per_beta"][beta]
            for i, step in enumerate(pb["step"]):
                if not (phase_lo < step <= phase_hi):
                    continue
                if pb["n_since_reset"][i] < burn_in:
                    continue
                # Same snapshot filter as scripts/analyze_phase1.py
                # (var > 0), so fractions are comparable with the WP1.2
                # report tables.
                var = pb["var"][i]
                if var is None or var <= 0:
                    continue
                rho = pb["rho"][i]
                if rho is None:
                    continue
                n += 1
                hits += rho < RHO_THRESHOLD
    return (hits / n if n else None, n)


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int
) -> Tuple[float, float, float]:
    """(mean, ci_lo, ci_hi) of the mean under a bootstrap over runs."""
    means = np.array(
        [
            values[rng.integers(0, len(values), size=len(values))].mean()
            for _ in range(n_boot)
        ]
    )
    lo, hi = np.percentile(means, [CI_LO, CI_HI])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_diff_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int
) -> Tuple[float, float, float]:
    """(mean difference a-b, ci_lo, ci_hi) under independent bootstraps."""
    diffs = np.array(
        [
            a[rng.integers(0, len(a), size=len(a))].mean()
            - b[rng.integers(0, len(b), size=len(b))].mean()
            for _ in range(n_boot)
        ]
    )
    lo, hi = np.percentile(diffs, [CI_LO, CI_HI])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def compare_sampling(
    baseline: List[Tuple[str, Dict[str, Any]]],
    withrepl: List[Tuple[str, Dict[str, Any]]],
    phases: List[Tuple[int, int]],
    n_boot: int,
) -> Dict[str, Any]:
    betas = baseline[0][1]["betas"]
    for name, log in baseline + withrepl:
        if log["betas"] != betas:
            raise ValueError(f"{name}: betas {log['betas']} != {betas}")
    # One RNG, fixed seed, consumed in deterministic (sorted) cell order.
    rng = np.random.default_rng(0)
    out: Dict[str, Any] = {
        "rho_threshold": RHO_THRESHOLD,
        "n_boot": n_boot,
        "ci_percentiles": [CI_LO, CI_HI],
        "n_runs": {"baseline": len(baseline), "with_replacement": len(withrepl)},
        "runs": {
            "baseline": [name for name, _ in baseline],
            "with_replacement": [name for name, _ in withrepl],
        },
        "phases": phases,
        "betas": betas,
        "variants": {},
    }
    for burn_in, vname in BURN_IN_VARIANTS:
        cells = []
        for beta in betas:
            for pi, (lo, hi) in enumerate(phases):
                for kind in KINDS:
                    sets = {}
                    fracs_by_set = {}
                    for set_name, runs in (
                        ("baseline", baseline),
                        ("with_replacement", withrepl),
                    ):
                        fracs, counts, used = [], [], []
                        for run_name, log in runs:
                            f, n = per_run_fraction(log, beta, lo, hi, burn_in, kind)
                            if f is not None:
                                fracs.append(f)
                                counts.append(n)
                                used.append(run_name)
                        arr = np.array(fracs, dtype=float)
                        if len(arr):
                            mean, ci_lo, ci_hi = bootstrap_mean_ci(arr, rng, n_boot)
                        else:
                            mean = ci_lo = ci_hi = None
                        fracs_by_set[set_name] = arr
                        sets[set_name] = {
                            "n_runs_used": len(arr),
                            "n_snapshots_total": int(sum(counts)),
                            "per_run_frac": [round(f, 6) for f in fracs],
                            "mean_frac": _r(mean),
                            "ci": [_r(ci_lo), _r(ci_hi)],
                        }
                    a, b = fracs_by_set["baseline"], fracs_by_set["with_replacement"]
                    if len(a) and len(b):
                        d, d_lo, d_hi = bootstrap_diff_ci(a, b, rng, n_boot)
                        diff = {"mean": _r(d), "ci": [_r(d_lo), _r(d_hi)]}
                    else:
                        diff = {"mean": None, "ci": [None, None]}
                    cells.append(
                        {
                            "beta": beta,
                            "phase": pi + 1,
                            "kind": kind,
                            **{k: sets[k] for k in sorted(sets)},
                            "difference_baseline_minus_withrepl": diff,
                        }
                    )
        out["variants"][vname] = {"burn_in": burn_in, "cells": cells}
    return out


def _r(x: Optional[float], nd: int = 6) -> Optional[float]:
    return None if x is None else round(float(x), nd)


# ------------------------------------------- section 2: HVP calibration stats


def hvp_agreement(hvp_dir: Path, lr: Optional[float]) -> Dict[str, Any]:
    from scipy import stats as sstats

    merged = load_sidecar_directory(Path(hvp_dir))
    if lr is None:
        lr = _lr_from_sidecar_siblings(Path(hvp_dir))
    if lr is None:
        raise SystemExit(
            "--lr is required for the HVP calibration section unless the "
            "sidecars' sibling results JSONs record the optimizer lr"
        )
    if not merged["hvp_enabled"]:
        raise SystemExit(
            f"sidecars in {hvp_dir} carry no HVP records (hvp_enabled false); "
            "point --hvp at HVP-enabled instrumented runs"
        )
    points = collect_calibration_points(merged, lr=lr)
    n_runs = len({name.split(":")[0] for name in merged["matrices"]})
    out: Dict[str, Any] = {"lr": lr, "n_runs": n_runs, "per_beta": {}}
    for beta in merged["betas"]:
        xs, ys = points[beta]
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        entry: Dict[str, Any] = {
            "n_pairs": int(len(x)),
            "n_nonfinite_dropped": int(len(xs) - len(x)),
        }
        if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
            entry["pearson_r"] = _r(float(np.corrcoef(x, y)[0, 1]))
            entry["spearman_rho"] = _r(float(sstats.spearmanr(x, y).statistic))
        else:
            entry["pearson_r"] = None
            entry["spearman_rho"] = None
        if len(x):
            rel = np.abs(y - x) / np.maximum(np.abs(x), 1e-12)
            entry["median_rel_err"] = _r(float(np.median(rel)))
            entry["median_implied"] = _r(float(np.median(y)))
            entry["median_hvp_eta_lambda"] = _r(float(np.median(x)))
        else:
            entry["median_rel_err"] = None
            entry["median_implied"] = None
            entry["median_hvp_eta_lambda"] = None
        out["per_beta"][beta] = entry
    return out


# ------------------------------------------------------------------ markdown


def to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Phase-1 disambiguation analysis (descriptive; no pass/fail)",
        "",
    ]
    comp = report.get("sampling_comparison")
    if comp:
        lines += [
            "## Sampling ablation: fraction of snapshots with "
            f"rho < {comp['rho_threshold']}",
            "",
            f"Baseline runs: {comp['n_runs']['baseline']} · with-replacement "
            f"runs: {comp['n_runs']['with_replacement']} · bootstrap over runs, "
            f"B={comp['n_boot']}, CI {comp['ci_percentiles']}% · phases "
            f"{comp['phases']}",
            "",
        ]
        for vname, variant in comp["variants"].items():
            lines += [
                f"### Variant: {vname} (burn_in={variant['burn_in']})",
                "",
                "| beta | phase | kind | baseline mean [CI] | with-repl mean [CI] "
                "| diff (base − w.r.) [CI] |",
                "|---|---|---|---|---|---|",
            ]
            for c in variant["cells"]:
                b = c["baseline"]
                w = c["with_replacement"]
                d = c["difference_baseline_minus_withrepl"]
                lines.append(
                    f"| {c['beta']} | {c['phase']} | {c['kind']} | "
                    f"{_fmt(b['mean_frac'])} [{_fmt(b['ci'][0])}, {_fmt(b['ci'][1])}] | "
                    f"{_fmt(w['mean_frac'])} [{_fmt(w['ci'][0])}, {_fmt(w['ci'][1])}] | "
                    f"{_fmt(d['mean'])} [{_fmt(d['ci'][0])}, {_fmt(d['ci'][1])}] |"
                )
            lines.append("")
    cal = report.get("hvp_calibration")
    if cal:
        lines += [
            "## Implied vs HVP eta*lambda (oscillating-classified snapshots)",
            "",
            f"HVP runs: {cal['n_runs']} · lr = {cal['lr']} · pairs matched by "
            "the calibration-plot rule (latest snapshot at/before each HVP "
            "record, regime == oscillating)",
            "",
            "| beta | n pairs | Pearson r | Spearman rho | median rel err "
            "| median implied | median lr*lambda |",
            "|---|---|---|---|---|---|---|",
        ]
        for beta, e in cal["per_beta"].items():
            lines.append(
                f"| {beta} | {e['n_pairs']} | {_fmt(e['pearson_r'])} | "
                f"{_fmt(e['spearman_rho'])} | {_fmt(e['median_rel_err'])} | "
                f"{_fmt(e['median_implied'])} | {_fmt(e['median_hvp_eta_lambda'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _fmt(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}"


# ---------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, help="dir of WP1.2 baseline sidecars")
    ap.add_argument(
        "--with-replacement", type=Path, dest="with_replacement",
        help="dir of with-replacement ablation sidecars",
    )
    ap.add_argument("--hvp", type=Path, help="dir of HVP-enabled sidecars")
    ap.add_argument("--lr", type=float, default=None, help="optimizer lr (HVP section)")
    ap.add_argument("--phase-len", type=int, default=50)
    ap.add_argument("--total-steps", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args(argv)

    have_comparison = args.baseline is not None and args.with_replacement is not None
    if (args.baseline is None) != (args.with_replacement is None):
        ap.error("--baseline and --with-replacement must be given together")
    if not have_comparison and args.hvp is None:
        ap.error("nothing to do: give --baseline/--with-replacement and/or --hvp")

    report: Dict[str, Any] = {}
    if have_comparison:
        phases = [
            (lo, min(lo + args.phase_len, args.total_steps))
            for lo in range(0, args.total_steps, args.phase_len)
        ]
        report["sampling_comparison"] = compare_sampling(
            load_sidecars(args.baseline),
            load_sidecars(args.with_replacement),
            phases,
            args.n_boot,
        )
    if args.hvp is not None:
        report["hvp_calibration"] = hvp_agreement(args.hvp, args.lr)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(to_markdown(report))

    # Console summary (descriptive).
    comp = report.get("sampling_comparison")
    if comp:
        for vname, variant in comp["variants"].items():
            for c in variant["cells"]:
                if c["kind"] != "all":
                    continue
                d = c["difference_baseline_minus_withrepl"]
                print(
                    f"{vname} beta={c['beta']} phase={c['phase']}: "
                    f"baseline={_fmt(c['baseline']['mean_frac'])} "
                    f"withrepl={_fmt(c['with_replacement']['mean_frac'])} "
                    f"diff={_fmt(d['mean'])} CI=[{_fmt(d['ci'][0])}, {_fmt(d['ci'][1])}]"
                )
    cal = report.get("hvp_calibration")
    if cal:
        for beta, e in cal["per_beta"].items():
            print(
                f"hvp beta={beta}: n={e['n_pairs']} pearson={_fmt(e['pearson_r'])} "
                f"median_rel_err={_fmt(e['median_rel_err'])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
