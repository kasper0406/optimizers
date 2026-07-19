#!/usr/bin/env python
"""Deterministic Phase-1 descriptive statistics from instrumentation sidecars.

Computes, for the two pre-registered quantities in
criteria/phase1_preregistration.md (DESCRIPTIVE OUTPUT ONLY — this script
makes no pass/fail judgment; Gate 1 is human-only):

1. Per (matrix, phase) cell: BIC-preferred GMM component count fitted on
   per-direction-snapshot points in (log10 SNR, rho) space, pooled across all
   runs. Reported per beta, k in {1, 2, 3}, fixed seeds — byte-stable output.
2. The rho < -0.2 population per phase (fraction of direction-snapshots and
   count of distinct directions), per beta. Phase bins are contiguous
   T_refresh-aligned windows (default 4 x 50 steps); the airbench LR schedule
   decays linearly, so phase 1 is the high-LR phase.

Sensitivity variants reported alongside (per the pre-registration scope note:
unspecified choices are reported across reasonable variants, never selected):
raw snapshots vs burn-in (n_since_reset >= 10).

Usage: uv run python scripts/analyze_phase1.py results/ --out-md reports/... --out-json reports/...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sklearn.mixture import GaussianMixture

SNR_FLOOR = 1e-8


def load_sidecars(paths):
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.instrumentation.json")) if p.is_dir() else [p]
    return [(f.name, json.loads(f.read_text())) for f in files]


def collect_points(sidecars, beta, phase_lo, phase_hi, burn_in):
    """Per (matrix) dict of (log10 SNR, rho) arrays plus direction identities."""
    out = {}
    for run_name, sc in sidecars:
        for mat, m in sc["matrices"].items():
            pts = out.setdefault(mat, {"x": [], "rho": [], "dir_ids": []})
            for d in m["directions"]:
                pb = d["per_beta"][beta]
                for i, step in enumerate(pb["step"]):
                    if not (phase_lo < step <= phase_hi):
                        continue
                    if pb["n_since_reset"][i] < burn_in:
                        continue
                    var = pb["var"][i]
                    if var is None or var <= 0:
                        continue
                    snr = abs(pb["mu"][i]) / max(np.sqrt(var), SNR_FLOOR)
                    pts["x"].append(np.log10(max(snr, SNR_FLOOR)))
                    pts["rho"].append(pb["rho"][i])
                    pts["dir_ids"].append((run_name, d["kind"], d["index"]))
    return out


def bic_preferred_k(x, rho, max_k=3):
    pts = np.column_stack([x, rho])
    if len(pts) < 10 * max_k:
        return None, []
    bics = []
    for k in range(1, max_k + 1):
        gm = GaussianMixture(n_components=k, random_state=0, n_init=1)
        gm.fit(pts)
        bics.append(float(gm.bic(pts)))
    return int(np.argmin(bics)) + 1, bics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--phase-len", type=int, default=50)
    ap.add_argument("--total-steps", type=int, default=200)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args(argv)

    sidecars = load_sidecars(args.paths)
    betas = sidecars[0][1]["betas"]
    phases = [
        (lo, min(lo + args.phase_len, args.total_steps))
        for lo in range(0, args.total_steps, args.phase_len)
    ]

    report = {"n_runs": len(sidecars), "betas": betas, "phases": phases, "variants": {}}
    for burn_in, vname in [(0, "raw"), (10, "burn_in_10")]:
        variant = {"cells": [], "rho_lt_-0.2": []}
        for beta in betas:
            cells_ge2 = 0
            cells_total = 0
            for pi, (lo, hi) in enumerate(phases):
                pts_by_mat = collect_points(sidecars, beta, lo, hi, burn_in)
                for mat, pts in sorted(pts_by_mat.items()):
                    k, bics = bic_preferred_k(pts["x"], pts["rho"])
                    if k is not None:
                        cells_total += 1
                        cells_ge2 += k >= 2
                    variant["cells"].append(
                        dict(beta=beta, phase=pi + 1, matrix=mat, n=len(pts["x"]),
                             preferred_k=k, bics=[round(b, 1) for b in bics])
                    )
                # rho < -0.2 population, pooled over matrices for this phase
                rho_all = np.concatenate(
                    [np.asarray(p["rho"]) for p in pts_by_mat.values()]
                ) if pts_by_mat else np.array([])
                dirs_all = [
                    did for p in pts_by_mat.values()
                    for did, r in zip(p["dir_ids"], p["rho"]) if r < -0.2
                ]
                variant["rho_lt_-0.2"].append(
                    dict(beta=beta, phase=pi + 1, n_snapshots=int(len(rho_all)),
                         frac_snapshots=float(np.mean(rho_all < -0.2)) if len(rho_all) else None,
                         distinct_directions=len(set(dirs_all)))
                )
            variant[f"frac_cells_ge2_beta_{beta}"] = (
                cells_ge2 / cells_total if cells_total else None
            )
            variant[f"n_cells_beta_{beta}"] = cells_total
        report["variants"][vname] = variant

    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.out_md:
        lines = [
            "# Phase-1 pre-registered quantities (descriptive; no pass/fail)",
            f"\nRuns: {report['n_runs']} · phases: {phases} · GMM k∈{{1,2,3}}, "
            "random_state=0, fit on (log10 SNR, ρ) pooled across runs\n",
        ]
        for vname, v in report["variants"].items():
            lines.append(f"## Variant: {vname}\n")
            for beta in betas:
                frac = v[f"frac_cells_ge2_beta_{beta}"]
                lines.append(
                    f"- β={beta}: BIC prefers ≥2 components in "
                    f"**{frac:.0%}** of {v[f'n_cells_beta_{beta}']} (matrix, phase) cells"
                )
            lines.append("\n| β | phase | snapshots | frac ρ<−0.2 | distinct dirs ρ<−0.2 |")
            lines.append("|---|---|---|---|---|")
            for r in v["rho_lt_-0.2"]:
                lines.append(
                    f"| {r['beta']} | {r['phase']} | {r['n_snapshots']} | "
                    f"{r['frac_snapshots']:.3f} | {r['distinct_directions']} |"
                )
            lines.append("")
        args.out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "variants"}))
    for vname, v in report["variants"].items():
        for beta in betas:
            print(f"{vname} beta={beta}: frac_cells_ge2="
                  f"{v[f'frac_cells_ge2_beta_{beta}']:.3f} of {v[f'n_cells_beta_{beta}']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
