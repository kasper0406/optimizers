#!/usr/bin/env python
"""Occupancy-LR law: is negative-rho occupancy a state function of the LR?

DESCRIPTIVE, zero-new-compute analysis over the existing airbench
instrumented sidecars (results/*.instrumentation.json). For every run,
beta and snapshot step t it computes

    occupancy(t) = fraction of direction-snapshots with lag-1 rho < -0.2
                   (same snapshot filters as scripts/analyze_phase1.py:
                    var > 0, rho not None, plus the raw / burn_in >= 10
                    variants)

and pairs it with the instantaneous filter learning rate of the airbench
schedule. Schedule provenance (src/optim/airbench_zoo.py): the filter
param groups are set to  initial_lr * (1 - step/total_train_steps)  with
the 0-based loop step BEFORE optimizer.step(); the instrumentation hub
increments its 1-based step counter inside after_step(), so a sidecar
snapshot at step s was produced by an update taken at

    lr(s) = lr0 * (1 - (s - 1) / total_steps)        (total_steps = 200).

(The whiten-bias group of optimizer1 anneals on its own shorter schedule;
tracked matrices are all filter matrices, so only the filter LR is used.)

The collapse test pools (lr, occupancy) points across all runs and configs
(the lr0 ladder 0.24 / 0.12 / 0.06 plus momentum=0, with-replacement and
HVP/compile-off probes) and quantifies how much of the occupancy variance
a single monotone curve occupancy = f(lr) explains:

* R^2 of an isotonic (monotone non-decreasing in lr) fit on the pooled
  points, and on baseline points only;
* a variance decomposition over lr-decile bins (between-bin share = the
  "explained by lr" fraction; within-bin variance further split into a
  between-config and a residual part);
* the matched-lr comparison: for every non-baseline point, the baseline
  occupancy at the SAME instantaneous lr (per-run linear interpolation of
  the 20 baseline trajectories in lr, then averaged) and the offset
  probe - baseline. This is the statistic that breaks the within-run
  lr-vs-time confound;
* an early-vs-late check: points with lr in [0.75*lr0_probe, lr0_probe]
  occur EARLY in the probe runs but LATE in the baseline runs -- their
  occupancies are compared directly;
* the baseline isotonic curve sampled on a fixed lr grid and at the
  record schedule's phase midpoints (the would-be controller calibration
  curve).

DESCRIPTIVE OUTPUT ONLY: no pass/fail judgment (gate decisions are human
checkpoints, CLAUDE.md). Deterministic: no RNG anywhere, sorted keys,
fixed rounding, no timestamps -- identical inputs give byte-identical
JSON and markdown.

Usage:
    uv run python scripts/analyze_occupancy_lr.py results/ \
        --out-md reports/occupancy-lr-law.md \
        --out-json reports/occupancy-lr-law.json \
        --fig-dir reports/figures/occupancy_lr
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

RHO_THRESHOLD = -0.2
BURN_IN_VARIANTS = [(0, "raw"), (10, "burn_in_10")]
MIN_SNAPSHOTS_PER_POINT = 20  # occupancy from fewer direction-snapshots is dropped
TOTAL_STEPS_DEFAULT = 200

# Fixed config order (baseline first; then the LR ladder; then the probes).
CONFIG_ORDER = ("baseline", "lrhalf", "lrquarter", "mom0", "withrep", "hvp_compileoff")
CONFIG_LABEL = {
    "baseline": "baseline (lr0 0.24)",
    "lrhalf": "lr0 0.12 (x0.5)",
    "lrquarter": "lr0 0.06 (x0.25)",
    "mom0": "momentum 0 (lr0 0.24)",
    "withrep": "with-replacement (lr0 0.24)",
    "hvp_compileoff": "HVP on / compile off (lr0 0.24)",
}
# Only these configs share the record lr0 and differ from baseline in one knob.
SAME_LR0_PROBES = ("mom0", "withrep", "hvp_compileoff")
LR_LADDER_PROBES = ("lrhalf", "lrquarter")

# Fixed lr grid for the reported baseline curve (controller calibration curve).
CURVE_LR_GRID = [0.24, 0.21, 0.18, 0.15, 0.12, 0.09, 0.06, 0.03, 0.012, 0.0012]
# Record-schedule phase midpoints (T_refresh-aligned 4 x 50 phases, t = mid).
PHASE_MIDPOINT_STEPS = [25, 75, 125, 175]


def _r(x: Optional[float], nd: int = 6) -> Optional[float]:
    return None if x is None else round(float(x), nd)


# ------------------------------------------------------------------- loading


def classify_config(results: Dict[str, Any]) -> Tuple[str, float]:
    """(config_key, lr0) from a run's results JSON (metrics + config contents).

    The seed->config mapping is NOT trusted from any listing: it is derived
    from each sidecar's sibling results JSON (metrics.optimizer_lr,
    metrics.probe_overrides, metrics.sampling, recipe.compile +
    instrumentation.hvp).
    """
    metrics = results.get("metrics", {})
    contents = results.get("config", {}).get("contents", {})
    lr0 = float(metrics["optimizer_lr"])
    overrides = metrics.get("probe_overrides") or {}
    if overrides.get("momentum") == 0.0:
        return "mom0", lr0
    if lr0 == 0.12:
        return "lrhalf", lr0
    if lr0 == 0.06:
        return "lrquarter", lr0
    if metrics.get("sampling") == "with_replacement":
        return "withrep", lr0
    recipe = contents.get("recipe", {})
    instr = contents.get("instrumentation", {})
    if recipe.get("compile") is False and instr.get("hvp") is True:
        return "hvp_compileoff", lr0
    return "baseline", lr0


def load_runs(paths: List[Path]) -> List[Dict[str, Any]]:
    """One record per sidecar: name, config key, lr0, total_steps, log."""
    files: List[Path] = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.instrumentation.json")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no *.instrumentation.json sidecars in {paths}")
    runs = []
    for f in files:
        sibling = f.with_name(f.name[: -len(".instrumentation.json")] + ".json")
        if not sibling.exists():
            raise FileNotFoundError(f"{f}: sibling results JSON {sibling} missing")
        results = json.loads(sibling.read_text())
        config, lr0 = classify_config(results)
        runs.append(
            {
                "name": f.name,
                "config": config,
                "lr0": lr0,
                "total_steps": int(results["metrics"]["steps"]),
                "seed": results.get("seed"),
                "log": json.loads(f.read_text()),
            }
        )
    return runs


# -------------------------------------------------------------- occupancy(t)


def lr_at(step: int, lr0: float, total_steps: int) -> float:
    """Instantaneous filter LR of the update that produced snapshot `step`
    (1-based sidecar step; see module docstring for the off-by-one)."""
    return lr0 * (1.0 - (step - 1) / total_steps)


def occupancy_series(
    log: Dict[str, Any], beta: str, burn_in: int, min_n: int = MIN_SNAPSHOTS_PER_POINT
) -> List[Tuple[int, float, int]]:
    """[(step, occupancy, n_snapshots)] pooled over all matrices/directions.

    Snapshot filters exactly as scripts/analyze_phase1.py /
    analyze_disambiguation.per_run_fraction: var > 0, rho not None,
    n_since_reset >= burn_in. Steps with fewer than min_n surviving
    direction-snapshots are dropped (unstable fractions).
    """
    counts: Dict[int, List[int]] = {}
    for mat in log["matrices"].values():
        for d in mat["directions"]:
            pb = d["per_beta"][beta]
            for i, step in enumerate(pb["step"]):
                if pb["n_since_reset"][i] < burn_in:
                    continue
                var = pb["var"][i]
                if var is None or var <= 0:
                    continue
                rho = pb["rho"][i]
                if rho is None:
                    continue
                c = counts.setdefault(int(step), [0, 0])
                c[1] += 1
                c[0] += rho < RHO_THRESHOLD
    return [
        (step, hits / n, n)
        for step, (hits, n) in sorted(counts.items())
        if n >= min_n
    ]


def collect_points(
    runs: List[Dict[str, Any]], beta: str, burn_in: int
) -> Dict[str, Any]:
    """Pooled point arrays: config, run name, t, lr, occupancy, n."""
    config: List[str] = []
    run_name: List[str] = []
    t: List[int] = []
    lr: List[float] = []
    occ: List[float] = []
    n: List[int] = []
    for run in runs:
        for step, frac, cnt in occupancy_series(run["log"], beta, burn_in):
            config.append(run["config"])
            run_name.append(run["name"])
            t.append(step)
            lr.append(lr_at(step, run["lr0"], run["total_steps"]))
            occ.append(frac)
            n.append(cnt)
    return {
        "config": np.array(config),
        "run": np.array(run_name),
        "t": np.array(t, dtype=int),
        "lr": np.array(lr, dtype=float),
        "occ": np.array(occ, dtype=float),
        "n": np.array(n, dtype=int),
    }


# ------------------------------------------------------------------ statistics


def isotonic_fit(lr: np.ndarray, occ: np.ndarray):
    """Monotone non-decreasing occupancy = f(lr); returns (predict_fn, r2)."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(lr, occ)
    pred = iso.predict(lr)
    ss_res = float(np.sum((occ - pred) ** 2))
    ss_tot = float(np.sum((occ - occ.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return iso.predict, r2


def decile_decomposition(
    x: np.ndarray, occ: np.ndarray, config: np.ndarray
) -> Dict[str, Any]:
    """Variance decomposition of occupancy over decile bins of x.

    x is the conditioning variable (instantaneous lr for the state-function
    test; training step t for the path-dependence counterpart). between-bin
    share = fraction of total occupancy variance explained by the x-decile
    alone; the within-bin remainder is split into a between-config part
    (configs differing at matched x) and a residual (runs and steps within
    a config at matched x).
    """
    edges = np.quantile(x, np.linspace(0, 1, 11))
    bin_idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 9)
    grand = occ.mean()
    ss_tot = float(np.sum((occ - grand) ** 2))
    ss_between_bins = 0.0
    ss_config_within = 0.0
    ss_resid = 0.0
    bins = []
    for b in range(10):
        mask = bin_idx == b
        if not mask.any():
            continue
        o = occ[mask]
        c = config[mask]
        bin_mean = o.mean()
        ss_between_bins += len(o) * (bin_mean - grand) ** 2
        cfg_spread = {}
        for key in sorted(set(c)):
            oc = o[c == key]
            ss_config_within += len(oc) * (oc.mean() - bin_mean) ** 2
            ss_resid += float(np.sum((oc - oc.mean()) ** 2))
            cfg_spread[key] = {"n": int(len(oc)), "mean_occ": _r(oc.mean())}
        bins.append(
            {
                "bin": b + 1,
                "lo": _r(edges[b]),
                "hi": _r(edges[b + 1]),
                "n": int(len(o)),
                "mean_occ": _r(bin_mean),
                "sd_occ": _r(o.std(ddof=0)),
                "configs": cfg_spread,
            }
        )
    ss_within = ss_config_within + ss_resid
    return {
        "decile_edges": [_r(e) for e in edges],
        "r2_between_bins": _r(ss_between_bins / ss_tot) if ss_tot > 0 else None,
        "within_bin_config_share": (
            _r(ss_config_within / ss_within) if ss_within > 0 else None
        ),
        "ss_total": _r(ss_tot),
        "ss_between_bins": _r(ss_between_bins),
        "ss_within_bins": _r(ss_within),
        "ss_between_configs_within_bins": _r(ss_config_within),
        "bins": bins,
    }


def baseline_interp_curves(
    points: Dict[str, Any],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float, float]:
    """Per-baseline-run (lr ascending, occ) arrays + common lr support."""
    curves = []
    lo, hi = -np.inf, np.inf
    for name in sorted(set(points["run"][points["config"] == "baseline"])):
        mask = points["run"] == name
        order = np.argsort(points["lr"][mask])
        x = points["lr"][mask][order]
        y = points["occ"][mask][order]
        curves.append((x, y))
        lo = max(lo, x.min())
        hi = min(hi, x.max())
    return curves, float(lo), float(hi)


def matched_lr_offsets(points: Dict[str, Any]) -> Dict[str, Any]:
    """Per non-baseline config: occupancy offset vs the baseline trajectories
    interpolated at the SAME instantaneous lr (the confound-breaking stat)."""
    curves, lo, hi = baseline_interp_curves(points)
    out: Dict[str, Any] = {"baseline_lr_support": [_r(lo), _r(hi)]}
    for cfg in CONFIG_ORDER:
        if cfg == "baseline" or cfg not in set(points["config"]):
            continue
        mask = (points["config"] == cfg) & (points["lr"] >= lo) & (points["lr"] <= hi)
        if not mask.any():
            out[cfg] = None
            continue
        lr = points["lr"][mask]
        occ = points["occ"][mask]
        runs = points["run"][mask]
        base = np.mean([np.interp(lr, x, y) for x, y in curves], axis=0)
        delta = occ - base
        per_run = [
            {
                "run": name,
                "n": int(np.sum(runs == name)),
                "mean_delta": _r(delta[runs == name].mean()),
            }
            for name in sorted(set(runs))
        ]
        run_means = np.array([r["mean_delta"] for r in per_run], dtype=float)
        out[cfg] = {
            "n_points": int(mask.sum()),
            "mean_delta": _r(delta.mean()),
            "mean_abs_delta": _r(np.abs(delta).mean()),
            "sd_over_run_means": _r(run_means.std(ddof=0)),
            "per_run": per_run,
        }
    return out


def early_vs_late(points: Dict[str, Any]) -> Dict[str, Any]:
    """For each LR-ladder probe: occupancy at lr in [0.75*lr0, lr0] --
    reached EARLY (t small) by the probe, LATE (t large) by baseline."""
    out: Dict[str, Any] = {}
    for cfg in LR_LADDER_PROBES:
        pmask = points["config"] == cfg
        if not pmask.any():
            continue
        lr0 = float(points["lr"][pmask].max())
        w_lo, w_hi = 0.75 * lr0, lr0
        pm = pmask & (points["lr"] >= w_lo) & (points["lr"] <= w_hi)
        bm = (
            (points["config"] == "baseline")
            & (points["lr"] >= w_lo)
            & (points["lr"] <= w_hi)
        )
        if not pm.any() or not bm.any():
            out[cfg] = None
            continue
        out[cfg] = {
            "lr_window": [_r(w_lo), _r(w_hi)],
            "probe": {
                "n_points": int(pm.sum()),
                "mean_t": _r(points["t"][pm].mean(), 1),
                "mean_occ": _r(points["occ"][pm].mean()),
            },
            "baseline": {
                "n_points": int(bm.sum()),
                "mean_t": _r(points["t"][bm].mean(), 1),
                "mean_occ": _r(points["occ"][bm].mean()),
            },
            "probe_minus_baseline": _r(
                points["occ"][pm].mean() - points["occ"][bm].mean()
            ),
        }
    return out


def analyze_beta(points: Dict[str, Any]) -> Dict[str, Any]:
    """All statistics for one (variant, beta) point cloud."""
    lr, occ, config = points["lr"], points["occ"], points["config"]
    _, r2_pooled = isotonic_fit(lr, occ)
    bmask = config == "baseline"
    base_predict, r2_base = isotonic_fit(lr[bmask], occ[bmask])

    # Residuals of every config against the baseline-only isotonic curve.
    residuals: Dict[str, Any] = {}
    for cfg in CONFIG_ORDER:
        m = config == cfg
        if not m.any():
            continue
        res = occ[m] - base_predict(lr[m])
        run_means = [
            _r(res[points["run"][m] == name].mean())
            for name in sorted(set(points["run"][m]))
        ]
        residuals[cfg] = {
            "n_points": int(m.sum()),
            "mean_residual": _r(res.mean()),
            "sd_over_run_means": _r(np.array(run_means, dtype=float).std(ddof=0)),
            "per_run_mean": run_means,
        }

    # Controller calibration curve: baseline isotonic on fixed lr grid +
    # the record schedule's phase midpoints.
    grid = np.array(CURVE_LR_GRID, dtype=float)
    curve = base_predict(grid)
    phase_lrs = np.array(
        [lr_at(t, 0.24, TOTAL_STEPS_DEFAULT) for t in PHASE_MIDPOINT_STEPS]
    )
    phase_curve = base_predict(phase_lrs)

    return {
        "n_points": int(len(occ)),
        "n_points_per_config": {
            cfg: int(np.sum(config == cfg))
            for cfg in CONFIG_ORDER
            if np.any(config == cfg)
        },
        "collapse": {
            "r2_isotonic_pooled": _r(r2_pooled),
            "r2_isotonic_baseline_only": _r(r2_base),
            "deciles": decile_decomposition(lr, occ, config),
            # Path-dependence counterpart: the same decomposition binned on
            # training step t instead of lr. Within a run t and lr/lr0 are
            # the same variable; across the lr0 ladder they differ, so
            # comparing the two R^2 values says which conditioning variable
            # the pooled cloud collapses on better.
            "deciles_t": decile_decomposition(
                points["t"].astype(float), occ, config
            ),
        },
        "matched_lr": matched_lr_offsets(points),
        "early_vs_late": early_vs_late(points),
        "residuals_vs_baseline_curve": residuals,
        "baseline_curve": {
            "lr": [_r(v) for v in grid],
            "occupancy": [_r(v) for v in curve],
            "phase_midpoints": [
                {
                    "t": t,
                    "lr": _r(l),
                    "occupancy": _r(o),
                }
                for t, l, o in zip(PHASE_MIDPOINT_STEPS, phase_lrs, phase_curve)
            ],
        },
    }


# -------------------------------------------------------------------- figures

FIG_COLOR = {
    "baseline": "#7a7975",  # recessive gray: the 20-run reference mass
    "lrhalf": "#2a78d6",  # blue
    "lrquarter": "#4a3aa7",  # violet
    "mom0": "#eb6834",  # orange
    "withrep": "#008300",  # green
    "hvp_compileoff": "#eda100",  # yellow
}
FIG_MARKER = {
    "baseline": "",
    "lrhalf": "o",
    "lrquarter": "s",
    "mom0": "^",
    "withrep": "D",
    "hvp_compileoff": "v",
}


def make_figures(
    per_beta_points: Dict[str, Dict[str, Any]], vname: str, fig_dir: Path
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    betas = sorted(per_beta_points)

    # ---- collapse plot: occupancy vs lr, one panel per beta ----
    fig, axes = plt.subplots(1, len(betas), figsize=(11, 4.4), sharey=True)
    for ax, beta in zip(np.atleast_1d(axes), betas):
        pts = per_beta_points[beta]
        for cfg in CONFIG_ORDER:
            m = pts["config"] == cfg
            if not m.any():
                continue
            first = True
            for name in sorted(set(pts["run"][m])):
                rm = m & (pts["run"] == name)
                order = np.argsort(pts["lr"][rm])
                ax.plot(
                    pts["lr"][rm][order],
                    pts["occ"][rm][order],
                    color=FIG_COLOR[cfg],
                    marker=FIG_MARKER[cfg] or None,
                    markersize=3.5,
                    linewidth=1.0 if cfg == "baseline" else 1.4,
                    alpha=0.35 if cfg == "baseline" else 0.9,
                    label=CONFIG_LABEL[cfg] if first else None,
                )
                first = False
        # baseline isotonic reference curve
        bmask = pts["config"] == "baseline"
        predict, _ = isotonic_fit(pts["lr"][bmask], pts["occ"][bmask])
        gx = np.linspace(pts["lr"].min(), pts["lr"].max(), 200)
        ax.plot(gx, predict(gx), color="#0b0b0b", linestyle="--", linewidth=1.6,
                label="baseline isotonic fit")
        ax.set_title(f"beta = {beta}", fontsize=11)
        ax.set_xlabel("instantaneous filter LR")
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("occupancy: frac(rho < -0.2)")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.suptitle(
        f"Occupancy vs instantaneous LR, all runs/configs ({vname})", fontsize=12
    )
    fig.tight_layout()
    out = fig_dir / f"collapse_{vname}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    written.append(str(out))

    # ---- residuals vs t at matched lr ----
    fig, axes = plt.subplots(1, len(betas), figsize=(11, 4.4), sharey=True)
    for ax, beta in zip(np.atleast_1d(axes), betas):
        pts = per_beta_points[beta]
        bmask = pts["config"] == "baseline"
        predict, _ = isotonic_fit(pts["lr"][bmask], pts["occ"][bmask])
        res = pts["occ"] - predict(pts["lr"])
        for cfg in CONFIG_ORDER:
            m = pts["config"] == cfg
            if not m.any():
                continue
            ax.scatter(
                pts["t"][m],
                res[m],
                s=8 if cfg == "baseline" else 16,
                color=FIG_COLOR[cfg],
                marker=FIG_MARKER[cfg] or "o",
                alpha=0.3 if cfg == "baseline" else 0.85,
                label=CONFIG_LABEL[cfg],
                linewidths=0,
            )
        ax.axhline(0.0, color="#0b0b0b", linewidth=1.0, linestyle="--")
        ax.set_title(f"beta = {beta}", fontsize=11)
        ax.set_xlabel("training step t")
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("occupancy - baseline isotonic f(lr)")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="lower left", frameon=False)
    fig.suptitle(
        f"Residuals vs step at matched LR ({vname}) -- path dependence check",
        fontsize=12,
    )
    fig.tight_layout()
    out = fig_dir / f"residuals_vs_t_{vname}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    written.append(str(out))
    return written


# ------------------------------------------------------------------- markdown


def _fmt(x: Optional[float], nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Occupancy-LR law (descriptive; no pass/fail)",
        "",
        "Is the negative-autocorrelation occupancy — the fraction of tracked",
        "direction-snapshots with lag-1 rho < -0.2 — a *state function* of the",
        "instantaneous learning rate? Computed from the existing airbench",
        "instrumented sidecars only (zero new compute) by",
        "`scripts/analyze_occupancy_lr.py`; numbers below are generated from",
        "the same computation that writes `occupancy-lr-law.json`.",
        "",
        "## What was computed",
        "",
        "- Per run, per beta, per snapshot step t (t in {1, 5, ..., 200}):",
        "  occupancy(t) = fraction of direction-snapshots with rho < -0.2,",
        "  pooled over all 6 tracked matrices x 32 directions, with the same",
        "  snapshot filters as `scripts/analyze_phase1.py` (var > 0, rho not",
        "  None), in two variants: raw and burn-in (n_since_reset >= 10).",
        f"  Steps with < {report['min_snapshots_per_point']} surviving",
        "  direction-snapshots are dropped.",
        "- Instantaneous LR: the airbench filter groups anneal linearly",
        "  (`src/optim/airbench_zoo.py`: `lr = initial_lr * (1 - step/200)`,",
        "  set with the 0-based loop step before `optimizer.step()`; the hub's",
        "  sidecar step counter is 1-based), so a snapshot at sidecar step s",
        "  was produced by an update at lr(s) = lr0 * (1 - (s-1)/200).",
        "  The whiten-bias group of optimizer1 has its own shorter schedule;",
        "  all tracked matrices are filter matrices, so only the filter LR is",
        "  used. lr0 is read per run from metrics.optimizer_lr.",
        "- Config identity is derived per run from the sidecar's sibling",
        "  results JSON (probe_overrides / optimizer_lr / sampling / recipe),",
        "  never from a seed listing.",
        "",
        "## Data",
        "",
    ]
    for cfg in CONFIG_ORDER:
        info = report["configs"].get(cfg)
        if info:
            lines.append(
                f"- **{CONFIG_LABEL[cfg]}**: {info['n_runs']} run(s), "
                f"seeds {info['seeds']}"
            )
    lines += [
        "",
        "## Collapse statistics",
        "",
        "R^2 of a single monotone (isotonic, non-decreasing in lr) curve",
        "fitted to the pooled (lr, occupancy) points of ALL runs and configs;",
        "decile decomposition: between-bin share = variance explained by the",
        "lr decile alone; the within-bin remainder splits into a between-config",
        "part (configs differing at matched lr) and a residual. The last two",
        "columns repeat the decomposition binned on training step t instead",
        "of lr — the path-dependence counterpart (within a run t and lr are",
        "the same variable; across the lr0 ladder they differ).",
        "",
        "| variant | beta | n points | R^2 isotonic (pooled) | R^2 isotonic (baseline only) | R^2 between lr-deciles | config share of within-lr-bin var | R^2 between t-deciles | config share of within-t-bin var |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for vname, variant in report["variants"].items():
        for beta, b in variant["betas"].items():
            c = b["collapse"]
            lines.append(
                f"| {vname} | {beta} | {b['n_points']} | "
                f"{_fmt(c['r2_isotonic_pooled'])} | "
                f"{_fmt(c['r2_isotonic_baseline_only'])} | "
                f"{_fmt(c['deciles']['r2_between_bins'])} | "
                f"{_fmt(c['deciles']['within_bin_config_share'])} | "
                f"{_fmt(c['deciles_t']['r2_between_bins'])} | "
                f"{_fmt(c['deciles_t']['within_bin_config_share'])} |"
            )
    lines += [
        "",
        "Note the isotonic R^2 is depressed relative to the (unconstrained)",
        "decile R^2 wherever the pooled relationship is non-monotone: the",
        "highest lr values only ever occur in the first ~30 steps of a run,",
        "where occupancy is still climbing out of the start-of-training",
        "transient, so the top of the lr range carries a dip that a monotone",
        "fit cannot represent (see the decile table below and the collapse",
        "figure).",
        "",
        "### Occupancy by lr decile and config (burn_in_10)",
        "",
        "Per-decile mean occupancy per config — the collapse test in table",
        "form: a state function of lr means matching columns within a row.",
        "",
    ]
    for vname, variant in report["variants"].items():
        if vname != "burn_in_10":
            continue
        for beta, b in variant["betas"].items():
            bins = b["collapse"]["deciles"]["bins"]
            present = [
                cfg
                for cfg in CONFIG_ORDER
                if any(cfg in bn["configs"] for bn in bins)
            ]
            lines.append(f"**beta {beta}** (columns: mean occupancy; n in JSON):")
            lines.append("")
            lines.append(
                "| lr bin | " + " | ".join(CONFIG_LABEL[c] for c in present) + " |"
            )
            lines.append("|---|" + "---|" * len(present))
            for bn in bins:
                cells = [
                    _fmt(bn["configs"][c]["mean_occ"]) if c in bn["configs"] else "—"
                    for c in present
                ]
                lines.append(
                    f"| {_fmt(bn['lo'], 4)}–{_fmt(bn['hi'], 4)} | "
                    + " | ".join(cells)
                    + " |"
                )
            lines.append("")
    lines += [
        "",
        "## Matched-lr cross-config comparison (the confound-breaking statistic)",
        "",
        "Within a run, lr(t) and training progress t are perfectly confounded;",
        "only the cross-config comparison at matched instantaneous lr breaks",
        "this. For every non-baseline point, Delta = occupancy minus the mean",
        "of the 20 baseline trajectories linearly interpolated at the SAME lr.",
        "",
        "| variant | beta | config | n points | mean Delta | mean abs Delta | sd over run means |",
        "|---|---|---|---|---|---|---|",
    ]
    for vname, variant in report["variants"].items():
        for beta, b in variant["betas"].items():
            for cfg in CONFIG_ORDER:
                m = b["matched_lr"].get(cfg)
                if not isinstance(m, dict):
                    continue
                lines.append(
                    f"| {vname} | {beta} | {CONFIG_LABEL[cfg]} | {m['n_points']} | "
                    f"{_fmt(m['mean_delta'])} | {_fmt(m['mean_abs_delta'])} | "
                    f"{_fmt(m['sd_over_run_means'])} |"
                )
    lines += [
        "",
        "## Early-vs-late at the same lr (path-dependence check)",
        "",
        "Points with lr in [0.75 lr0_probe, lr0_probe] occur EARLY in the",
        "probe runs (first ~50 steps) but LATE in the baseline runs (which",
        "only anneal down to that lr after 100+/150+ steps). If occupancy were",
        "a pure state function of lr, the two groups would match.",
        "",
        "| variant | beta | probe | lr window | probe occ (mean t) | baseline occ (mean t) | probe - baseline |",
        "|---|---|---|---|---|---|---|",
    ]
    for vname, variant in report["variants"].items():
        for beta, b in variant["betas"].items():
            for cfg in LR_LADDER_PROBES:
                e = b["early_vs_late"].get(cfg)
                if not e:
                    continue
                lines.append(
                    f"| {vname} | {beta} | {CONFIG_LABEL[cfg]} | "
                    f"[{_fmt(e['lr_window'][0])}, {_fmt(e['lr_window'][1])}] | "
                    f"{_fmt(e['probe']['mean_occ'])} (t≈{e['probe']['mean_t']:.0f}) | "
                    f"{_fmt(e['baseline']['mean_occ'])} (t≈{e['baseline']['mean_t']:.0f}) | "
                    f"{_fmt(e['probe_minus_baseline'])} |"
                )
    lines += [
        "",
        "## Config offsets vs the baseline curve",
        "",
        "Mean residual of each config's points against the baseline-only",
        "isotonic curve f(lr) (positive = above the baseline curve).",
        "",
        "| variant | beta | config | n points | mean residual | sd over run means |",
        "|---|---|---|---|---|---|",
    ]
    for vname, variant in report["variants"].items():
        for beta, b in variant["betas"].items():
            for cfg in CONFIG_ORDER:
                r = b["residuals_vs_baseline_curve"].get(cfg)
                if not r:
                    continue
                lines.append(
                    f"| {vname} | {beta} | {CONFIG_LABEL[cfg]} | {r['n_points']} | "
                    f"{_fmt(r['mean_residual'])} | {_fmt(r['sd_over_run_means'])} |"
                )
    lines += [
        "",
        "## Baseline occupancy-vs-lr curve (would-be controller calibration)",
        "",
        "The baseline isotonic curve sampled on a fixed lr grid, and at the",
        "record schedule's phase midpoints (t = 25/75/125/175). If occupancy",
        "is treated as a controlled variable, this curve is the calibration",
        "between an occupancy setpoint and the LR that produces it — subject",
        "to every caveat below. (In the raw variant the value at lr = 0.24",
        "reflects only the degenerate t = 1 snapshot — lr0 occurs exactly",
        "once per run, at the first step, where the EMA statistics are",
        "essentially unformed; use the burn_in_10 rows.)",
        "",
    ]
    for vname, variant in report["variants"].items():
        for beta, b in variant["betas"].items():
            bc = b["baseline_curve"]
            lines.append(f"**{vname}, beta {beta}** — f(lr) on the grid:")
            lines.append("")
            lines.append("| lr | " + " | ".join(_fmt(v) for v in bc["lr"]) + " |")
            lines.append("|---|" + "---|" * len(bc["lr"]))
            lines.append(
                "| occupancy | "
                + " | ".join(_fmt(v) for v in bc["occupancy"])
                + " |"
            )
            lines.append("")
            mids = ", ".join(
                f"t={p['t']} (lr {_fmt(p['lr'])}): {_fmt(p['occupancy'])}"
                for p in bc["phase_midpoints"]
            )
            lines.append(f"Phase midpoints: {mids}")
            lines.append("")
    lines += [
        "## Figures",
        "",
        "`reports/figures/occupancy_lr/`: `collapse_<variant>.png` (occupancy",
        "vs lr, all runs, colored by config, baseline isotonic fit dashed) and",
        "`residuals_vs_t_<variant>.png` (residuals against the baseline curve",
        "vs training step, i.e. the path-dependence view).",
        "",
        "## Reading (descriptive; refers to the 32-run 2026-07-19 sidecar set)",
        "",
        "- Configs sharing the record LR trajectory (with-replacement,",
        "  HVP/compile-off) sit on the baseline curve in every decile — the",
        "  collapse is exact when the lr *path* is identical, so the",
        "  measurement itself is stable.",
        "- momentum=0 runs a small, roughly constant amount above the curve",
        "  at all lr (the level shift already reported in",
        "  `wp22-mechanism-probes.md`), i.e. approximately a parallel curve,",
        "  not a reshaped one.",
        "- The lr0-ladder runs are NOT on the baseline curve at matched lr:",
        "  each ladder run reproduces the baseline's *shape in scheduled",
        "  time* (early dip near its own lr0, mid-run hump, anneal-tail",
        "  collapse) shifted to its own lr scale. They sit above the",
        "  baseline at lrs just below their lr0 (their own hump vs the",
        "  baseline's anneal) and below it near their lr0 (their own early",
        "  transient vs the baseline's hump). At the very bottom of the",
        "  anneal (lr < ~0.02) the configs reconverge at beta 0.9, while at",
        "  beta 0.99 the ladder runs remain ~0.10-0.15 below the baseline",
        "  (the slow EMA carries their lower recent history into the tail).",
        "- Binning on training step t explains the pooled cloud about as",
        "  well as binning on lr, and neither dominates (collapse table).",
        "  This is expected: 26 of the 32 runs share lr0 = 0.24, so t and",
        "  lr are collinear in most of the pooled data and pooled R^2",
        "  cannot separate them. The separation lives entirely in the",
        "  ladder runs' matched-lr deviations (above) together with the",
        "  matched-phase lr0 dependence already shown in",
        "  `wp22-mechanism-probes.md`: each is nonzero, so the data are",
        "  consistent with occupancy depending on BOTH the instantaneous lr",
        "  level and the position in the schedule (equivalently lr/lr0),",
        "  not on instantaneous lr alone. With n=2 ladder runs per rung",
        "  this is a description, not an estimate of the two contributions.",
        "- For a setpoint controller this means the baseline curve below is",
        "  a calibration of occupancy against the *record schedule*, valid",
        "  along that trajectory; it is not a schedule-free lr-occupancy",
        "  law, and transferring it to a different schedule shape is not",
        "  supported by this data.",
        "",
        "## Caveats (read before using any number above)",
        "",
        "- **Within-run confound.** Inside any single run, lr(t) and training",
        "  progress t are perfectly confounded; the within-run collapse can",
        "  never distinguish 'occupancy tracks lr' from 'occupancy tracks",
        "  time'. Only the lr0-ladder configs (0.12, 0.06) break this, and",
        "  they have n = 2 runs each — the matched-lr and early-vs-late",
        "  tables are the load-bearing statistics, at probe-side n of 2.",
        "- **What CAN be concluded** is limited to: whether the 2+2 ladder",
        "  runs' (lr, occupancy) points lie on / off the 20-run baseline",
        "  curve at overlapping lr, plus the analogous check for momentum=0,",
        "  with-replacement and HVP/compile-off at lr0 = 0.24. What CANNOT:",
        "  any claim about schedules not in the data (e.g. LR increases,",
        "  constant-LR long runs), any per-direction claim (occupancy is a",
        "  population fraction), or causality of lr vs co-annealed dynamics.",
        "- **Estimator memory.** rho is an EMA-lag-1 statistic; at beta 0.99",
        "  the effective window (~100 steps) spans a large stretch of the",
        "  anneal, so 'instantaneous' lr attribution is smeared — beta 0.9",
        "  (~10-step window) is the cleaner state-function probe; treat the",
        "  beta 0.99 columns as slow-averaged.",
        "- **Snapshot cadence and resets.** Occupancy is sampled every 5",
        "  steps; subspace refreshes at t = 50/100/150 reset the EMAs and",
        "  transiently depress negative-rho fractions (the burn_in_10 variant",
        "  drops n_since_reset < 10 snapshots and is the cleaner one; raw is",
        "  reported for sensitivity). Early-t points also carry re-tracked",
        "  directions whose identity changed at refresh.",
        "- **Single task / arch / scale.** airbench94 (CIFAR-10, 200 steps,",
        "  batch 2000) only; nothing here says nanogpt or any other scale",
        "  behaves the same.",
        "- **Occupancy definition.** frac(rho < -0.2) over the tracked",
        "  subspace (top-16 + 16 bulk probes per matrix); the threshold is",
        "  the pre-registered -0.2, not tuned here; top and bulk directions",
        "  are pooled (they differ in level, see wp22-mechanism-probes.md).",
        "- **n = 2-3 per probe config**; run-to-run sd columns are computed",
        "  over 2-3 run means and are indicative only. No inferential claims",
        "  are made anywhere in this file.",
        "",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="results dir(s) or sidecar file(s)")
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--fig-dir", type=Path)
    args = ap.parse_args(argv)

    runs = load_runs([Path(p) for p in args.paths])
    betas = runs[0]["log"]["betas"]
    for run in runs:
        if run["log"]["betas"] != betas:
            raise ValueError(f"{run['name']}: betas {run['log']['betas']} != {betas}")

    configs: Dict[str, Any] = {}
    for run in runs:
        c = configs.setdefault(
            run["config"], {"lr0": run["lr0"], "n_runs": 0, "seeds": [], "runs": []}
        )
        c["n_runs"] += 1
        c["seeds"].append(run["seed"])
        c["runs"].append(run["name"])

    report: Dict[str, Any] = {
        "rho_threshold": RHO_THRESHOLD,
        "min_snapshots_per_point": MIN_SNAPSHOTS_PER_POINT,
        "lr_schedule": "lr(s) = lr0 * (1 - (s-1)/total_steps), filter groups",
        "betas": betas,
        "n_runs": len(runs),
        "configs": configs,
        "variants": {},
    }
    figures: List[str] = []
    for burn_in, vname in BURN_IN_VARIANTS:
        per_beta_points = {beta: collect_points(runs, beta, burn_in) for beta in betas}
        report["variants"][vname] = {
            "burn_in": burn_in,
            "betas": {beta: analyze_beta(per_beta_points[beta]) for beta in betas},
        }
        if args.fig_dir:
            figures += make_figures(per_beta_points, vname, args.fig_dir)
    if figures:
        report["figures"] = [str(Path(f)) for f in figures]

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(to_markdown(report))

    # Console summary: headline collapse + matched-lr numbers.
    for vname, variant in report["variants"].items():
        for beta, b in variant["betas"].items():
            c = b["collapse"]
            print(
                f"{vname} beta={beta}: n={b['n_points']} "
                f"r2_iso_pooled={_fmt(c['r2_isotonic_pooled'])} "
                f"r2_lr_deciles={_fmt(c['deciles']['r2_between_bins'])} "
                f"config_share_within_lr={_fmt(c['deciles']['within_bin_config_share'])} "
                f"r2_t_deciles={_fmt(c['deciles_t']['r2_between_bins'])} "
                f"config_share_within_t={_fmt(c['deciles_t']['within_bin_config_share'])}"
            )
            for cfg in CONFIG_ORDER:
                m = b["matched_lr"].get(cfg)
                if isinstance(m, dict):
                    print(
                        f"  matched-lr {cfg}: mean_delta={_fmt(m['mean_delta'])} "
                        f"(n={m['n_points']})"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
