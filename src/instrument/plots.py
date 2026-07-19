"""Deterministic Phase-1 plots from instrumentation JSONs (plan section 1.2).

Generates, from a results/sidecar JSON alone (no live training):

1. ``regime_scatter.png``  -- each tracked direction as a point in
   (|mu|/sqrt(var), rho) space, colored by the HVP-measured curvature lambda
   (gray when no HVP was recorded), faceted over training phases x betas.
   Hypothesis: three clusters (signal: high SNR, rho >= 0; noise: low SNR,
   rho ~ 0; oscillation: rho strongly negative).  Null: a single smear.
2. ``regime_occupancy.png`` -- fraction of tracked directions per regime over
   training, one panel per beta (stacked area).
3. ``eta_lambda_calibration.png`` -- for directions classified oscillating,
   implied eta*lambda from the amplitude ratio vs HVP-measured eta*lambda
   (= lr * lambda_hvp), with the y = x reference line.

Everything is deterministic: sorted iteration order, fixed figure geometry,
fixed dpi, no timestamps.  Usage::

    uv run python -m src.instrument.plots <log.json | sidecar_dir> <out_dir> [--lr LR]

The input may be a single instrumentation JSON (sidecar or full results
file) or a **directory** of ``*.instrumentation.json`` sidecars (e.g. a
synced ``results/`` after the WP1.2 seed sweep): all sidecars are merged
into one log, with matrix records namespaced by run stem, so the three plots
pool tracked directions across runs/seeds.

``--lr`` is required for plot 3 unless it can be recovered from the JSON --
``config.contents.optimizer.lr`` or ``metrics.optimizer_lr`` of a full
results file, or (directory mode) of the sidecars' sibling results JSONs,
which must then agree on a single value.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from src.instrument.schema import (
    SIDECAR_SUFFIX,
    iter_directions,
    load_instrumentation,
)

__all__ = [
    "plot_regime_scatter",
    "plot_regime_occupancy",
    "plot_eta_lambda_calibration",
    "collect_calibration_points",
    "load_sidecar_directory",
    "make_all_plots",
]

# --------------------------------------------------------------------- style
# Palette per the dataviz reference instance (validated: all-pairs CVD and
# normal-vision floors pass for the 3 regime hues on the light surface; the
# yellow's <3:1 surface contrast is relieved by the always-on legend labels).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NO_HVP_GRAY = "#b0afa9"
REGIME_COLORS = {
    "signal": "#2a78d6",  # blue
    "noise": "#eda100",  # yellow
    "oscillating": "#e34948",  # red
}
REGIME_ORDER = ["signal", "noise", "oscillating"]
# Sequential blue ramp (light -> dark) for the lambda magnitude encoding.
_SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LAMBDA_CMAP = LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE)

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.edgecolor": GRID,
    "grid.color": GRID,
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "svg.hashsalt": "wp11",
}
_DPI = 150
_SNR_FLOOR = 1e-6


# ------------------------------------------------------------------- helpers


def _betas(log: Dict[str, Any]) -> List[str]:
    return list(log["betas"])


def _max_step(log: Dict[str, Any]) -> int:
    mx = 0
    for _, mat, _ in iter_directions(log):
        if mat["steps"]:
            mx = max(mx, mat["steps"][-1])
    return mx


def _facet_edges(max_step: int, n_facets: int) -> List[int]:
    """Right edges of n_facets contiguous training phases."""
    return [max(1, round(max_step * (i + 1) / n_facets)) for i in range(n_facets)]


def _snapshot_index_at(series_steps: Sequence[int], step: int) -> Optional[int]:
    """Index of the last snapshot at or before ``step`` (None if none)."""
    idx = None
    for i, s in enumerate(series_steps):
        if s <= step:
            idx = i
        else:
            break
    return idx


def _lambda_at(direction: Dict[str, Any], step: int) -> Optional[float]:
    """Latest HVP lambda recorded at or before ``step``."""
    steps = direction["lambda_hvp"]["step"]
    values = direction["lambda_hvp"]["value"]
    idx = _snapshot_index_at(steps, step)
    return None if idx is None else float(values[idx])


def _lr_from_json(path: Path) -> Optional[float]:
    """Try to read optimizer lr from a full results JSON.

    Checks ``config.contents.optimizer.lr`` first, then
    ``metrics.optimizer_lr`` (experiments that pin the optimizer themselves,
    e.g. airbench_instrumented, record the pinned lr in metrics).
    """
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except Exception:
        return None
    try:
        return float(obj["config"]["contents"]["optimizer"]["lr"])
    except Exception:
        pass
    try:
        return float(obj["metrics"]["optimizer_lr"])
    except Exception:
        return None


def _sidecar_paths(dir_path: Path) -> List[Path]:
    paths = sorted(Path(dir_path).glob(f"*{SIDECAR_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(
            f"no *{SIDECAR_SUFFIX} sidecar files found in {dir_path}"
        )
    return paths


def load_sidecar_directory(dir_path: Path) -> Dict[str, Any]:
    """Merge every ``*.instrumentation.json`` sidecar in a directory into one
    validated log; matrix records are namespaced ``<run_stem>:<matrix>`` so
    the plots pool tracked directions across runs/seeds."""
    merged: Optional[Dict[str, Any]] = None
    for path in _sidecar_paths(dir_path):
        log = load_instrumentation(path)
        stem = path.name[: -len(SIDECAR_SUFFIX)]
        if merged is None:
            merged = {
                "instrumentation_schema_version": log[
                    "instrumentation_schema_version"
                ],
                "betas": list(log["betas"]),
                "hvp_enabled": bool(log["hvp_enabled"]),
                "matrices": {},
            }
        elif list(log["betas"]) != merged["betas"]:
            raise ValueError(
                f"{path}: betas {log['betas']} != {merged['betas']} of earlier "
                "sidecars; cannot merge a mixed-beta directory"
            )
        else:
            merged["hvp_enabled"] = merged["hvp_enabled"] or bool(log["hvp_enabled"])
        for name, mat in log["matrices"].items():
            merged["matrices"][f"{stem}:{name}"] = mat
    return merged


def _lr_from_sidecar_siblings(dir_path: Path) -> Optional[float]:
    """Directory mode: recover lr from the sidecars' sibling results JSONs
    (<stem>.instrumentation.json -> <stem>.json); all found values must agree."""
    lrs = set()
    for path in _sidecar_paths(dir_path):
        sibling = path.with_name(path.name[: -len(SIDECAR_SUFFIX)] + ".json")
        lr = _lr_from_json(sibling)
        if lr is not None:
            lrs.add(lr)
    if len(lrs) > 1:
        raise ValueError(
            f"sibling results JSONs in {dir_path} disagree on optimizer lr "
            f"({sorted(lrs)}); pass --lr explicitly"
        )
    return lrs.pop() if lrs else None


# --------------------------------------------------------------- plot 1 of 3


def plot_regime_scatter(
    log: Dict[str, Any], out_path: Path, *, n_facets: int = 4
) -> Path:
    """(|mu|/sqrt(var), rho) scatter, colored by lambda, facets x betas."""
    betas = _betas(log)
    max_step = _max_step(log)
    edges = _facet_edges(max_step, n_facets)

    lambdas = [
        abs(v)
        for _, _, d in iter_directions(log)
        for v in d["lambda_hvp"]["value"]
    ]
    norm = Normalize(vmin=0.0, vmax=max(lambdas) if lambdas else 1.0)

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            len(betas),
            n_facets,
            figsize=(2.9 * n_facets, 2.7 * len(betas)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for r, beta in enumerate(betas):
            for c, edge in enumerate(edges):
                ax = axes[r][c]
                xs, ys, cs = [], [], []
                xg, yg = [], []
                for _, _, d in iter_directions(log):
                    series = d["per_beta"][beta]
                    idx = _snapshot_index_at(series["step"], edge)
                    if idx is None:
                        continue
                    var = max(series["var"][idx], 0.0)
                    snr = abs(series["mu"][idx]) / math.sqrt(max(var, _SNR_FLOOR))
                    rho = series["rho"][idx]
                    lam = _lambda_at(d, edge)
                    if lam is None:
                        xg.append(max(snr, _SNR_FLOOR))
                        yg.append(rho)
                    else:
                        xs.append(max(snr, _SNR_FLOOR))
                        ys.append(rho)
                        cs.append(abs(lam))
                ax.axhline(0.0, color=GRID, linewidth=0.8, zorder=1)
                if xg:
                    ax.scatter(
                        xg, yg, s=14, color=NO_HVP_GRAY, linewidths=0,
                        alpha=0.8, zorder=2, label="no HVP",
                    )
                if xs:
                    ax.scatter(
                        xs, ys, s=16, c=cs, cmap=LAMBDA_CMAP, norm=norm,
                        linewidths=0.5, edgecolors=SURFACE, zorder=3,
                    )
                ax.set_xscale("log")
                ax.set_ylim(-1.05, 1.05)
                if r == 0:
                    ax.set_title(f"steps ≤ {edge}", fontsize=9, color=TEXT_PRIMARY)
                if c == 0:
                    ax.set_ylabel(f"β = {beta}\nlag-1 autocorr ρ")
                if r == len(betas) - 1:
                    ax.set_xlabel("|μ| / √var  (SNR)")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=LAMBDA_CMAP)
        cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("|λ| (HVP)", color=TEXT_SECONDARY)
        fig.suptitle(
            "Regime scatter: tracked directions in (SNR, ρ) space over training",
            fontsize=11,
            color=TEXT_PRIMARY,
        )
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
    return Path(out_path)


# --------------------------------------------------------------- plot 2 of 3


def plot_regime_occupancy(log: Dict[str, Any], out_path: Path) -> Path:
    """Fraction of tracked directions per regime vs step, per beta."""
    betas = _betas(log)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            len(betas), 1, figsize=(7.5, 2.4 * len(betas)), squeeze=False, sharex=True
        )
        for r, beta in enumerate(betas):
            ax = axes[r][0]
            counts: Dict[int, Dict[str, int]] = {}
            for _, _, d in iter_directions(log):
                series = d["per_beta"][beta]
                for step, regime in zip(series["step"], series["regime"]):
                    counts.setdefault(step, {k: 0 for k in REGIME_ORDER})
                    counts[step][regime] += 1
            steps = sorted(counts)
            if steps:
                totals = np.array(
                    [sum(counts[s].values()) for s in steps], dtype=float
                )
                fractions = [
                    np.array([counts[s][reg] for s in steps]) / totals
                    for reg in REGIME_ORDER
                ]
                ax.stackplot(
                    steps,
                    *fractions,
                    labels=REGIME_ORDER,
                    colors=[REGIME_COLORS[k] for k in REGIME_ORDER],
                    edgecolor=SURFACE,
                    linewidth=0.8,
                    step="post",  # regimes are step-sampled, not continuous
                )
            ax.set_ylim(0, 1)
            ax.set_ylabel(f"β = {beta}\nfraction of directions")
            if r == 0:
                ax.legend(
                    loc="upper right", frameon=True, framealpha=0.9,
                    facecolor=SURFACE, edgecolor=GRID,
                )
        axes[-1][0].set_xlabel("training step")
        fig.suptitle(
            "Regime occupancy over training", fontsize=11, color=TEXT_PRIMARY
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
    return Path(out_path)


# --------------------------------------------------------------- plot 3 of 3


def collect_calibration_points(
    log: Dict[str, Any], *, lr: float
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Per beta, the (x, y) pairs of the eta*lambda calibration plot:
    x = HVP-measured eta*lambda (lr * lambda_hvp), y = implied eta*lambda
    from the amplitude ratio, one pair per HVP record whose latest snapshot
    at or before the record step is classified oscillating.

    Shared by :func:`plot_eta_lambda_calibration` and the disambiguation
    analysis (scripts/analyze_disambiguation.py) so both consume the exact
    same matching rule.
    """
    out: Dict[str, Tuple[List[float], List[float]]] = {}
    for beta in _betas(log):
        xs: List[float] = []
        ys: List[float] = []
        for _, _, d in iter_directions(log):
            series = d["per_beta"][beta]
            for hstep, lam in zip(
                d["lambda_hvp"]["step"], d["lambda_hvp"]["value"]
            ):
                idx = _snapshot_index_at(series["step"], hstep)
                if idx is None or series["regime"][idx] != "oscillating":
                    continue
                xs.append(lr * float(lam))
                ys.append(float(series["implied_eta_lambda"][idx]))
        out[beta] = (xs, ys)
    return out


def plot_eta_lambda_calibration(
    log: Dict[str, Any], out_path: Path, *, lr: float
) -> Path:
    """Implied eta*lambda (amplitude ratio) vs HVP eta*lambda = lr * lambda,
    for snapshots where the direction is classified oscillating."""
    betas = _betas(log)
    points = collect_calibration_points(log, lr=lr)
    markers = ["o", "^", "s", "D"]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        any_points = False
        all_vals: List[float] = []
        for bi, beta in enumerate(betas):
            xs, ys = points[beta]
            if xs:
                any_points = True
                all_vals += xs + ys
                ax.scatter(
                    xs, ys, s=22,
                    color=REGIME_COLORS["oscillating"],
                    marker=markers[bi % len(markers)],
                    alpha=0.85, linewidths=0.5, edgecolors=SURFACE,
                    label=f"β = {beta}",
                )
        lo, hi = (min(all_vals), max(all_vals)) if all_vals else (0.0, 2.0)
        pad = 0.1 * (hi - lo or 1.0)
        lo, hi = lo - pad, hi + pad
        ax.plot(
            [lo, hi], [lo, hi], linestyle="--", linewidth=1.2,
            color=TEXT_SECONDARY, zorder=1, label="y = x",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("HVP-measured η·λ  (lr × λ_HVP)")
        ax.set_ylabel("implied η·λ  (amplitude ratio)")
        ax.set_title(
            "Oscillating directions: implied vs measured η·λ",
            fontsize=11, color=TEXT_PRIMARY,
        )
        if not any_points:
            ax.text(
                0.5, 0.5, "no oscillating directions with HVP records",
                transform=ax.transAxes, ha="center", va="center",
                color=TEXT_SECONDARY,
            )
        ax.legend(
            loc="upper left", frameon=True, framealpha=0.9,
            facecolor=SURFACE, edgecolor=GRID,
        )
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
    return Path(out_path)


# ----------------------------------------------------------------------- all


def make_all_plots(
    json_path: Path,
    out_dir: Path,
    *,
    lr: Optional[float] = None,
    n_facets: int = 4,
) -> List[Path]:
    """Generate the three Phase-1 plots from an instrumentation JSON or a
    directory of ``*.instrumentation.json`` sidecars (merged across runs)."""
    json_path = Path(json_path)
    if json_path.is_dir():
        log = load_sidecar_directory(json_path)
        if lr is None:
            lr = _lr_from_sidecar_siblings(json_path)
    else:
        log = load_instrumentation(json_path)
        if lr is None:
            lr = _lr_from_json(json_path)
    if lr is None:
        raise ValueError(
            "lr is required for the eta*lambda calibration plot; pass --lr "
            "or point at results JSONs that record the optimizer lr"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_regime_scatter(log, out_dir / "regime_scatter.png", n_facets=n_facets),
        plot_regime_occupancy(log, out_dir / "regime_occupancy.png"),
        plot_eta_lambda_calibration(
            log, out_dir / "eta_lambda_calibration.png", lr=lr
        ),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_path",
        type=Path,
        help="instrumentation sidecar / results JSON, or a directory of "
        "*.instrumentation.json sidecars (merged across runs)",
    )
    parser.add_argument("out_dir", type=Path, help="directory for the three PNGs")
    parser.add_argument("--lr", type=float, default=None, help="optimizer lr (for plot 3)")
    parser.add_argument("--n-facets", type=int, default=4)
    args = parser.parse_args(argv)
    paths = make_all_plots(args.json_path, args.out_dir, lr=args.lr, n_facets=args.n_facets)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
