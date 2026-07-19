#!/usr/bin/env python
"""Deterministic, DESCRIPTIVE analysis for the Gate-1 A4 mechanism probes.

Compares the phase-wise fraction of direction-snapshots with lag-1
autocorrelation rho < -0.2 between the WP1.2 instrumented BASELINE
(stock recipe: lr=0.24, momentum=0.6, nesterov) and the mechanism-probe
runs:

    --mom0       momentum = 0, nesterov off   (configs/dev/instrumented_airbench_mom0.yaml)
    --lrhalf     lr = 0.12 (record x 0.5)     (configs/dev/instrumented_airbench_lrhalf.yaml)
    --lrquarter  lr = 0.06 (record x 0.25)    (configs/dev/instrumented_airbench_lrquarter.yaml)

For every (beta, phase, direction kind in {all, top, bulk}) cell it reports
the per-config mean fraction with a bootstrap CI over runs and the
bootstrapped difference (probe - baseline) -- the bulk-vs-top split is the
Gate-1 anomaly check (bulk out-oscillated top in phase 1). Context, stated
here and nowhere judged: the momentum-overshoot prediction is a LARGE DROP
of the negative-rho fraction at momentum = 0 and a MONOTONE DECREASE down
the LR ladder; this script only states the numbers and intervals.

Statistics, thresholds, snapshot filters, burn-in variants, and bootstrap
procedure are exactly those of scripts/analyze_disambiguation.py (imported,
not reimplemented). DESCRIPTIVE OUTPUT ONLY: no pass/fail judgment (gate
decisions are human checkpoints, CLAUDE.md). Deterministic: fixed bootstrap
seed, fixed cell order -- identical inputs produce byte-identical outputs.

Usage:
    uv run python scripts/analyze_mechanism.py \
        --baseline results/wp12_sidecars/ \
        --mom0 results/mom0_sidecars/ \
        --lrhalf results/lrhalf_sidecars/ \
        --lrquarter results/lrquarter_sidecars/ \
        --out-md reports/wp22-mechanism-probes.md \
        --out-json reports/wp22-mechanism-probes.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_disambiguation():
    """Import scripts/analyze_disambiguation.py (shared statistics core)."""
    spec = importlib.util.spec_from_file_location(
        "analyze_disambiguation_core", REPO_ROOT / "scripts" / "analyze_disambiguation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ad = _load_disambiguation()

# Fixed probe order (baseline first, then decreasing-momentum / LR ladder).
PROBE_ORDER = ("baseline", "mom0", "lrhalf", "lrquarter")
PROBE_LABEL = {
    "baseline": "baseline (lr 0.24, mom 0.6)",
    "mom0": "momentum 0",
    "lrhalf": "lr x0.5 (0.12)",
    "lrquarter": "lr x0.25 (0.06)",
}


def compare_probes(
    runs_by_set: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    phases: List[Tuple[int, int]],
    n_boot: int,
) -> Dict[str, Any]:
    """Phase-wise rho < -0.2 fractions per config with bootstrap CIs and the
    per-cell (probe - baseline) bootstrap difference, per direction kind."""
    sets_present = [s for s in PROBE_ORDER if s in runs_by_set]
    betas = runs_by_set["baseline"][0][1]["betas"]
    for set_name in sets_present:
        for name, log in runs_by_set[set_name]:
            if log["betas"] != betas:
                raise ValueError(f"{name}: betas {log['betas']} != {betas}")
    rng = np.random.default_rng(0)  # one RNG, fixed seed, fixed cell order
    out: Dict[str, Any] = {
        "rho_threshold": ad.RHO_THRESHOLD,
        "n_boot": n_boot,
        "ci_percentiles": [ad.CI_LO, ad.CI_HI],
        "sets": sets_present,
        "n_runs": {s: len(runs_by_set[s]) for s in sets_present},
        "runs": {s: [name for name, _ in runs_by_set[s]] for s in sets_present},
        "phases": phases,
        "betas": betas,
        "prediction_context": (
            "momentum-overshoot predicts a large drop at momentum=0 and a "
            "monotone decrease with lr; stated for context, not judged here"
        ),
        "variants": {},
    }
    for burn_in, vname in ad.BURN_IN_VARIANTS:
        cells = []
        for beta in betas:
            for pi, (lo, hi) in enumerate(phases):
                for kind in ad.KINDS:
                    per_set: Dict[str, Any] = {}
                    fracs_by_set: Dict[str, np.ndarray] = {}
                    for set_name in sets_present:
                        fracs, counts = [], []
                        for _run_name, log in runs_by_set[set_name]:
                            f, n = ad.per_run_fraction(
                                log, beta, lo, hi, burn_in, kind
                            )
                            if f is not None:
                                fracs.append(f)
                                counts.append(n)
                        arr = np.array(fracs, dtype=float)
                        if len(arr):
                            mean, ci_lo, ci_hi = ad.bootstrap_mean_ci(
                                arr, rng, n_boot
                            )
                        else:
                            mean = ci_lo = ci_hi = None
                        fracs_by_set[set_name] = arr
                        per_set[set_name] = {
                            "n_runs_used": len(arr),
                            "n_snapshots_total": int(sum(counts)),
                            "per_run_frac": [round(f, 6) for f in fracs],
                            "mean_frac": ad._r(mean),
                            "ci": [ad._r(ci_lo), ad._r(ci_hi)],
                        }
                    diffs: Dict[str, Any] = {}
                    base = fracs_by_set.get("baseline", np.array([]))
                    for set_name in sets_present:
                        if set_name == "baseline":
                            continue
                        probe = fracs_by_set[set_name]
                        if len(base) and len(probe):
                            d, d_lo, d_hi = ad.bootstrap_diff_ci(
                                probe, base, rng, n_boot
                            )
                            diffs[set_name] = {
                                "mean": ad._r(d),
                                "ci": [ad._r(d_lo), ad._r(d_hi)],
                            }
                        else:
                            diffs[set_name] = {"mean": None, "ci": [None, None]}
                    cells.append(
                        {
                            "beta": beta,
                            "phase": pi + 1,
                            "kind": kind,
                            "sets": per_set,
                            "difference_probe_minus_baseline": diffs,
                        }
                    )
        out["variants"][vname] = {"burn_in": burn_in, "cells": cells}
    return out


# ------------------------------------------------------------------ markdown


def to_markdown(report: Dict[str, Any]) -> str:
    comp = report["mechanism_comparison"]
    sets_present = comp["sets"]
    lines = [
        "# Gate-1 A4 mechanism probes: phase-wise rho < "
        f"{comp['rho_threshold']} fractions (descriptive; no pass/fail)",
        "",
        " · ".join(
            f"{PROBE_LABEL[s]}: {comp['n_runs'][s]} run(s)" for s in sets_present
        ),
        "",
        f"Bootstrap over runs, B={comp['n_boot']}, CI {comp['ci_percentiles']}% "
        f"· phases {comp['phases']}",
        "",
        f"Context (not judged here): {comp['prediction_context']}.",
        "",
    ]
    probes = [s for s in sets_present if s != "baseline"]
    for vname, variant in comp["variants"].items():
        lines += [f"## Variant: {vname} (burn_in={variant['burn_in']})", ""]
        for kind in ad.KINDS:
            header = "| beta | phase | baseline [CI] |"
            rule = "|---|---|---|"
            for s in probes:
                header += f" {PROBE_LABEL[s]} [CI] | Δ vs base [CI] |"
                rule += "---|---|"
            lines += [f"### kind = {kind} (bulk-vs-top split)", "", header, rule]
            for c in variant["cells"]:
                if c["kind"] != kind:
                    continue
                b = c["sets"].get("baseline", {})
                row = (
                    f"| {c['beta']} | {c['phase']} | "
                    f"{_fmt(b.get('mean_frac'))} "
                    f"[{_fmt(b.get('ci', [None, None])[0])}, "
                    f"{_fmt(b.get('ci', [None, None])[1])}] |"
                )
                for s in probes:
                    e = c["sets"][s]
                    d = c["difference_probe_minus_baseline"][s]
                    row += (
                        f" {_fmt(e['mean_frac'])} "
                        f"[{_fmt(e['ci'][0])}, {_fmt(e['ci'][1])}] |"
                        f" {_fmt(d['mean'])} "
                        f"[{_fmt(d['ci'][0])}, {_fmt(d['ci'][1])}] |"
                    )
                lines.append(row)
            lines.append("")
    return "\n".join(lines) + "\n"


def _fmt(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}"


# ---------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline", type=Path, required=True,
        help="dir of WP1.2 baseline instrumented sidecars",
    )
    ap.add_argument("--mom0", type=Path, help="dir of momentum=0 probe sidecars")
    ap.add_argument("--lrhalf", type=Path, help="dir of lr x0.5 probe sidecars")
    ap.add_argument("--lrquarter", type=Path, help="dir of lr x0.25 probe sidecars")
    ap.add_argument("--phase-len", type=int, default=50)
    ap.add_argument("--total-steps", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args(argv)

    probe_dirs = {
        "mom0": args.mom0,
        "lrhalf": args.lrhalf,
        "lrquarter": args.lrquarter,
    }
    if all(v is None for v in probe_dirs.values()):
        ap.error("nothing to compare: give at least one of --mom0/--lrhalf/--lrquarter")

    runs_by_set = {"baseline": ad.load_sidecars(args.baseline)}
    for name, path in probe_dirs.items():
        if path is not None:
            runs_by_set[name] = ad.load_sidecars(path)

    phases = [
        (lo, min(lo + args.phase_len, args.total_steps))
        for lo in range(0, args.total_steps, args.phase_len)
    ]
    report = {
        "mechanism_comparison": compare_probes(runs_by_set, phases, args.n_boot)
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(to_markdown(report))

    # Console summary (descriptive): the all-directions rows.
    comp = report["mechanism_comparison"]
    for vname, variant in comp["variants"].items():
        for c in variant["cells"]:
            if c["kind"] != "all":
                continue
            parts = [
                f"{s}={_fmt(c['sets'][s]['mean_frac'])}" for s in comp["sets"]
            ]
            print(f"{vname} beta={c['beta']} phase={c['phase']}: " + " ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
