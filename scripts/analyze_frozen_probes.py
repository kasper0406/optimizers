#!/usr/bin/env python
"""Deterministic, DESCRIPTIVE analysis of the frozen-probe tier.

Answers, with numbers only, the pre-registered question stated in
``src.instrument.tracker.FrozenProbeBank``:

    Along a frozen (never-refreshed, never-reset) random probe direction, does
    the cumulative t-statistic of the gradient projection grow like sqrt(t) --
    the signature of a persistent, non-zero-mean component, since drift
    accumulates like T while zero-mean noise accumulates like sqrt(T) -- or
    does it stay flat/bounded at the white-noise scale?  Does ANY frozen probe
    cross the conventional |t| >= 4 by the end of the run?  How does that
    compare with the tracked/bulk tiers, whose t is structurally capped by a
    bounded EMA window plus innovation resets?

Sections produced:

1. **Growth law.** Per matrix and pooled: the OLS slope of log|t| on log(step)
   over the snapshot series of every frozen probe, reported as the median
   slope with an inter-quartile range and the fraction of probes whose slope
   lands in [0.35, 0.65] (the sqrt(t) band) vs near 0 (the bounded band).
   Reported for both the naive and the Newey-West-adjusted t.
2. **Threshold crossings.** Fraction of frozen probes with final |t| >= 4
   (and >= 2, >= 3), per matrix and pooled, for both estimators, plus the
   effective sample size distribution.
3. **Tier comparison.** The final-snapshot |t| distribution of the frozen tier
   against that of the tracked (top-k1) and bulk (k2) tiers, per beta -- the
   structural-cap contrast.

DESCRIPTIVE OUTPUT ONLY: this script states quantities; it makes no pass/fail
judgment and evaluates no gate (CLAUDE.md: gate decisions are human-only).
Deterministic: sorted keys, no timestamps, no randomness -- identical inputs
produce byte-identical outputs.

Usage:
    uv run python scripts/analyze_frozen_probes.py \
        --sidecars results/smoothness_sidecars/ \
        --out-md reports/frozen-probes.md \
        --out-json reports/frozen-probes.json \
        [--out-plot reports/figures/frozen-probes.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.instrument.schema import (  # noqa: E402
    SIDECAR_SUFFIX,
    load_instrumentation,
)

T_THRESHOLDS = (2.0, 3.0, 4.0)
SQRT_BAND = (0.35, 0.65)  # log-log slope band consistent with |t| ~ sqrt(t)
FLAT_BAND = (-0.15, 0.15)  # ... and with a bounded, non-growing |t|
ESTIMATORS = ("t_naive", "t_nw")


# ------------------------------------------------------------------ loading


def load_sidecars(dir_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Every ``*.instrumentation.json`` under ``dir_path``, sorted by name."""
    paths = sorted(Path(dir_path).glob(f"*{SIDECAR_SUFFIX}"))
    if not paths:
        raise SystemExit(f"no {SIDECAR_SUFFIX} files under {dir_path}")
    return [(p.name[: -len(SIDECAR_SUFFIX)], load_instrumentation(p)) for p in paths]


def iter_frozen(
    sidecars: Sequence[Tuple[str, Dict[str, Any]]]
) -> List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]:
    """(run, matrix, frozen_block, probe) for every frozen probe found."""
    out = []
    for run, log in sidecars:
        for name in sorted(log.get("matrices", {})):
            block = log["matrices"][name].get("frozen_probes")
            if not block:
                continue
            for probe in block.get("probes", []):
                out.append((run, name, block, probe))
    return out


# -------------------------------------------------------------- growth law


def growth_slope(steps: Sequence[float], t_series: Sequence[float]) -> Optional[float]:
    """OLS slope of log|t| on log(step), over snapshots with |t| > 0.

    ``None`` when fewer than three usable points exist (a probe whose series is
    too short or degenerate is reported as missing, never as slope 0).
    """
    s = np.asarray(steps, dtype=np.float64)
    t = np.abs(np.asarray(t_series, dtype=np.float64))
    ok = (s > 0) & (t > 0) & np.isfinite(t)
    if int(ok.sum()) < 3:
        return None
    return float(np.polyfit(np.log(s[ok]), np.log(t[ok]), 1)[0])


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def _fraction(values: Sequence[float], lo: float, hi: float) -> Optional[float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return float(np.mean((v >= lo) & (v <= hi)))


def analyze_group(entries: Sequence[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    """Growth-law + threshold statistics for one group of frozen probes."""
    out: Dict[str, Any] = {"n_probes": len(entries)}
    for est in ESTIMATORS:
        slopes = [
            growth_slope(block["snapshot_steps"], probe[est])
            for _, _, block, probe in entries
        ]
        finals = [abs(float(probe["final"][est])) for _, _, _, probe in entries]
        out[est] = {
            "slope": _summary(slopes),
            "n_slope_missing": sum(1 for s in slopes if s is None),
            "frac_slope_in_sqrt_band": _fraction(slopes, *SQRT_BAND),
            "frac_slope_in_flat_band": _fraction(slopes, *FLAT_BAND),
            "final_abs_t": _summary(finals),
            "frac_crossing": {
                f"{thr:g}": (
                    float(np.mean(np.asarray(finals) >= thr)) if finals else None
                )
                for thr in T_THRESHOLDS
            },
        }
    out["ess"] = _summary([float(p["final"]["ess"]) for _, _, _, p in entries])
    out["n_observations"] = _summary(
        [float(p["final"]["n"]) for _, _, _, p in entries]
    )
    out["n_nw_floored"] = sum(
        1 for _, _, _, p in entries if p["final"].get("nw_floored")
    )
    return out


# ----------------------------------------------------------- tier contrast


def tracked_tier_final_t(
    sidecars: Sequence[Tuple[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Final-snapshot |t_stat| of the tracked/bulk tiers, per beta and kind."""
    pools: Dict[str, Dict[str, List[float]]] = {}
    for _, log in sidecars:
        for name in sorted(log.get("matrices", {})):
            for d in log["matrices"][name].get("directions", []):
                kind = d.get("kind", "unknown")
                for beta, series in (d.get("per_beta") or {}).items():
                    ts = series.get("t_stat") or []
                    if not ts:
                        continue
                    value = abs(float(ts[-1]))
                    pools.setdefault(beta, {}).setdefault(kind, []).append(value)
                    pools[beta].setdefault("all", []).append(value)
    return {
        beta: {
            kind: {
                **_summary(vals),
                "frac_crossing": {
                    f"{thr:g}": float(np.mean(np.asarray(vals) >= thr))
                    for thr in T_THRESHOLDS
                },
            }
            for kind, vals in sorted(by_kind.items())
        }
        for beta, by_kind in sorted(pools.items())
    }


# ------------------------------------------------------------------ report


def build_report(sidecars: Sequence[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    entries = iter_frozen(sidecars)
    if not entries:
        raise SystemExit(
            "no frozen_probes blocks in these sidecars -- the runs were made "
            "with instrumentation.frozen_probes off"
        )
    by_matrix: Dict[str, List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]] = {}
    for entry in entries:
        by_matrix.setdefault(entry[1], []).append(entry)
    return {
        "n_runs": len(sidecars),
        "runs": sorted(run for run, _ in sidecars),
        "n_frozen_probes": len(entries),
        "pooled": analyze_group(entries),
        "per_matrix": {name: analyze_group(g) for name, g in sorted(by_matrix.items())},
        "tracked_tier_final_abs_t": tracked_tier_final_t(sidecars),
        "bands": {
            "sqrt_band": list(SQRT_BAND),
            "flat_band": list(FLAT_BAND),
            "thresholds": list(T_THRESHOLDS),
        },
    }


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Frozen-probe tier: long-integration signal detection",
        "",
        "Descriptive output of `scripts/analyze_frozen_probes.py`. No pass/fail",
        "judgment is made here; gate decisions are human-only (CLAUDE.md).",
        "",
        "Pre-registered question (`src.instrument.tracker.FrozenProbeBank`):",
        "does the cumulative |t| of a frozen, never-reset random probe grow",
        "like sqrt(t) (persistent signal: drift ~ T, noise ~ sqrt(T)), or stay",
        "flat/bounded at the white-noise scale, and does any probe cross",
        "|t| >= 4 by end of run? The tracked/bulk tiers' t is structurally",
        "capped (bounded EMA window + innovation resets).",
        "",
        f"Runs: {report['n_runs']}; frozen probes: {report['n_frozen_probes']}.",
        f"sqrt(t) slope band {report['bands']['sqrt_band']}, "
        f"flat band {report['bands']['flat_band']}.",
        "",
        "## Growth law and threshold crossings",
        "",
        "| group | est | slope median | slope IQR | frac in sqrt band | frac flat "
        "| median final \\|t\\| | frac \\|t\\|>=4 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    groups = [("pooled", report["pooled"])] + sorted(report["per_matrix"].items())
    for gname, g in groups:
        for est in ESTIMATORS:
            e = g[est]
            s = e["slope"]
            lines.append(
                f"| {gname} | {est} | {_fmt(s['median'])} | "
                f"[{_fmt(s['q25'])}, {_fmt(s['q75'])}] | "
                f"{_fmt(e['frac_slope_in_sqrt_band'])} | "
                f"{_fmt(e['frac_slope_in_flat_band'])} | "
                f"{_fmt(e['final_abs_t']['median'])} | "
                f"{_fmt(e['frac_crossing']['4'])} |"
            )
    ess = report["pooled"]["ess"]
    lines += [
        "",
        f"Effective sample size (pooled): median {_fmt(ess['median'], 1)}, "
        f"IQR [{_fmt(ess['q25'], 1)}, {_fmt(ess['q75'], 1)}]; "
        f"Newey-West floored on {report['pooled']['n_nw_floored']} probe(s).",
        "",
        "## Tier contrast: final |t| distribution",
        "",
        "| beta | tier | n | median | q75 | max | frac >=4 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for beta, kinds in report["tracked_tier_final_abs_t"].items():
        for kind, st in kinds.items():
            lines.append(
                f"| {beta} | tracked:{kind} | {st['n']} | {_fmt(st['median'])} | "
                f"{_fmt(st['q75'])} | {_fmt(st['max'])} | "
                f"{_fmt(st['frac_crossing']['4'])} |"
            )
    for est in ESTIMATORS:
        e = report["pooled"][est]["final_abs_t"]
        lines.append(
            f"| - | frozen:{est} | {e['n']} | {_fmt(e['median'])} | "
            f"{_fmt(e['q75'])} | {_fmt(e['max'])} | "
            f"{_fmt(report['pooled'][est]['frac_crossing']['4'])} |"
        )
    lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------------- plot


def make_plot(
    sidecars: Sequence[Tuple[str, Dict[str, Any]]], out_path: Path
) -> Optional[Path]:
    """|t| vs sqrt(step) per matrix (+ pooled), one thin line per probe.

    On the sqrt(t) axis a persistent-signal probe is a straight line through
    the origin and a pure-noise probe is a flat band; the sqrt-x axis is the
    whole point of the plot. Returns None if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a repo dependency
        return None

    entries = iter_frozen(sidecars)
    by_matrix: Dict[str, List[Any]] = {}
    for entry in entries:
        by_matrix.setdefault(entry[1], []).append(entry)
    names = sorted(by_matrix) + ["pooled"]
    ncol = min(3, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        group = entries if name == "pooled" else by_matrix[name]
        for _, _, block, probe in group:
            steps = np.asarray(block["snapshot_steps"], dtype=float)
            t = np.abs(np.asarray(probe["t_nw"], dtype=float))
            if steps.size:
                ax.plot(np.sqrt(steps), t, lw=0.6, alpha=0.5, color="#4477aa")
        ax.axhline(4.0, color="#cc3311", lw=1.0, ls="--", label="|t| = 4")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("sqrt(step)")
        ax.set_ylabel("|t| (Newey-West)")
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(
        "Frozen probes: cumulative |t| vs sqrt(step)\n"
        "(straight through origin = persistent signal; flat = noise)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sidecars", type=Path, required=True,
        help="directory of *.instrumentation.json sidecars with frozen_probes",
    )
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-plot", type=Path)
    args = ap.parse_args(argv)

    sidecars = load_sidecars(args.sidecars)
    report = build_report(sidecars)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(to_markdown(report))
    if args.out_plot:
        made = make_plot(sidecars, args.out_plot)
        if made is None:
            print("matplotlib unavailable; skipped the plot", file=sys.stderr)

    pooled = report["pooled"]
    for est in ESTIMATORS:
        e = pooled[est]
        print(
            f"frozen {est}: n={pooled['n_probes']} "
            f"slope_median={_fmt(e['slope']['median'])} "
            f"frac_sqrt_band={_fmt(e['frac_slope_in_sqrt_band'])} "
            f"median_final_|t|={_fmt(e['final_abs_t']['median'])} "
            f"frac_|t|>=4={_fmt(e['frac_crossing']['4'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
