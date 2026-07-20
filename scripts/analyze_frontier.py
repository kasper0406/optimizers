#!/usr/bin/env python
"""Stability-frontier analysis (program #6, lr x batch).

Answers the PRE-REGISTERED questions of
``reports/stability-frontier-preregistration.md`` from frontier results JSONs
(+ instrumentation sidecars). DESCRIPTIVE OUTPUT ONLY -- the report states
where the measured quantities land relative to the pre-registered regions; no
pass/fail is asserted here. Deterministic: fixed bootstrap RNG seed, sorted
keys, byte-stable output.

Definitions implemented exactly as pre-registered:

- ref_acc(B)      = max over rungs {0.5x, 1x} of mean TTA accuracy at B
- floor(B)        = ref_acc(B) - 1.0pp
- lr_shoulder(B)  = largest rung with mean acc >= floor(B) below the FIRST
                    floor crossing; later recoveries flagged, not used
- alpha           = OLS slope of log lr_shoulder vs log B; quantization
                    uncertainty = +/- half a rung ratio at every B endpoint;
                    seed bootstrap (B=2000 draws, rng seed 0) secondary
- P2 candidates (B in {500,1000,2000,4000} only, pre-declared):
    occupancy    whole-run pooled frac(rho_1 < -0.2), beta 0.9, burn-in 10
                 (filters identical to scripts/analyze_occupancy_lr.py)
    spectral     pooled plateau median of lr * D_smooth_spectral
    euclidean    pooled plateau median of lr * D_smooth_frobenius
                 (plateau window: last 50% of measured steps, the
                 scripts/analyze_smoothness.py convention)
    hvp_q90      q90 of eta*lambda = lr_at(step) * lambda_hvp pooled over
                 directions/refreshes (linear-decay schedule lr_at as in
                 scripts/analyze_occupancy_lr.py)
  evaluated at each B's shoulder rung AND at the fixed 1x rung (control).
  Pre-registered frontier-tracking signature: across-B max/min ratio < 1.5
  at the shoulder AND >= 1.5 at the 1x control.

Usage:
    uv run python scripts/analyze_frontier.py results/ \
        [--out-md reports/stability-frontier-tables.md] \
        [--out-json reports/stability-frontier.json] \
        [--out-plot-dir reports/figures]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CONFIG_TAG = "frontier_lrxbatch"  # identifies frontier runs via config.path
STOCK_LR = 0.24
RUNGS = [0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.96, 1.44]
BATCHES = [500, 1000, 2000, 4000, 8000]
P2_BATCHES = [500, 1000, 2000, 4000]  # B=8000 excluded in the pre-registration
FLOOR_DROP_PP = 1.0
REF_RUNGS = [0.12, 0.24]  # the two lowest rungs define ref_acc(B)
TRACKING_RATIO = 1.5
BOOT_N = 2000
BOOT_SEED = 0


def _load_helper(name: str):
    spec = importlib.util.spec_from_file_location(
        f"rm_frontier_{name}", REPO_ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_occ = _load_helper("analyze_occupancy_lr")   # occupancy_series, lr_at
_smooth = _load_helper("analyze_smoothness")  # plateau_stats, SPEC/EUC keys


def _r(x: Optional[float], nd: int = 6) -> Optional[float]:
    return None if x is None else round(float(x), nd)


# ------------------------------------------------------------------ loading


def load_frontier_runs(results_dir: Path) -> List[Dict[str, Any]]:
    """One record per frontier results JSON (config.path tags the sweep)."""
    runs = []
    for f in sorted(results_dir.glob("airbench_instrumented_seed*.json")):
        if f.name.endswith(".instrumentation.json"):
            continue
        data = json.loads(f.read_text())
        cfg = data.get("config") or {}
        if CONFIG_TAG not in str(cfg.get("path", "")):
            continue
        contents = cfg.get("contents") or {}
        bsz = int(contents["train"]["batch_size"])
        lr = float(contents["probe_overrides"]["lr"])
        acc = data["metrics"].get("tta_val_acc")  # stored as a fraction
        sidecar = data["metrics"].get("instrumentation_sidecar")
        runs.append(
            {
                "file": f.name,
                "batch_size": bsz,
                "lr": lr,
                "seed": int(data["seed"]),
                "acc": 100.0 * float(acc) if acc is not None else None,  # percent
                "steps": int(data["metrics"]["steps"]),
                "sidecar": str(results_dir / sidecar) if sidecar else None,
            }
        )
    # Dedupe by (batch, lr, seed), keeping the LATEST file: the first launch
    # of the sweep was aborted after a few runs (provenance flag fix) and
    # results/ is append-only, so its files remain; the relaunch re-ran the
    # same cells. Sorted glob order = timestamp order within a seed.
    latest: Dict[Tuple[int, float, int], Dict[str, Any]] = {}
    for r in runs:
        latest[(r["batch_size"], r["lr"], r["seed"])] = r
    return sorted(latest.values(), key=lambda r: r["file"])


def cell_table(runs) -> Dict[int, Dict[float, List[Dict[str, Any]]]]:
    table: Dict[int, Dict[float, List[Dict[str, Any]]]] = {}
    for r in runs:
        table.setdefault(r["batch_size"], {}).setdefault(r["lr"], []).append(r)
    return table


# ------------------------------------------------------------- P1: shoulder


def shoulder_from_means(mean_acc: Dict[float, float]) -> Dict[str, Any]:
    """First-crossing shoulder per the pre-registered definition.

    mean_acc: rung lr -> mean accuracy (percent). Returns shoulder lr (None
    if the lowest rung is already below floor), floor, ref_acc, and flags for
    post-crossing recoveries.
    """
    rungs = [lr for lr in RUNGS if lr in mean_acc]
    refs = [mean_acc[lr] for lr in REF_RUNGS if lr in mean_acc]
    if not refs:
        return {"shoulder": None, "ref_acc": None, "floor": None, "recoveries": []}
    ref = max(refs)
    floor = ref - FLOOR_DROP_PP
    shoulder = None
    crossed = False
    recoveries = []
    for lr in rungs:
        if not crossed:
            if mean_acc[lr] >= floor:
                shoulder = lr
            else:
                crossed = True
        elif mean_acc[lr] >= floor:
            recoveries.append(lr)
    return {
        "shoulder": shoulder,
        "ref_acc": _r(ref, 4),
        "floor": _r(floor, 4),
        "recoveries": recoveries,
    }


def fit_alpha(shoulders: Dict[int, float]) -> Optional[Dict[str, Any]]:
    pts = sorted((b, lr) for b, lr in shoulders.items() if lr is not None)
    if len(pts) < 2:
        return None
    x = np.log([b for b, _ in pts])
    y = np.log([lr for _, lr in pts])
    slope, intercept = np.polyfit(x, y, 1)
    # rung-quantization envelope: the true shoulder lies within +/- half a
    # rung ratio of the measured rung; the extreme slopes tilt the endpoints
    # in opposite directions.
    half = 0.5 * math.log(RUNGS[1] / RUNGS[0])  # rungs are ~log-uniform
    span = x[-1] - x[0]
    return {
        "alpha": _r(slope),
        "intercept": _r(intercept),
        "n_points": len(pts),
        "alpha_quantization_halfwidth": _r(2 * half / span if span else None),
    }


def bootstrap_alpha(table, rng_seed=BOOT_SEED, n_boot=BOOT_N) -> Optional[Dict[str, Any]]:
    """Seed bootstrap: resample per-cell run accuracies with replacement,
    recompute every shoulder and the fit. Secondary to the quantization
    envelope (n=2 per cell)."""
    rng = np.random.default_rng(rng_seed)
    alphas = []
    for _ in range(n_boot):
        shoulders = {}
        for bsz, cells in table.items():
            mean_acc = {}
            for lr, cell in cells.items():
                accs = [r["acc"] for r in cell if r["acc"] is not None]
                if accs:
                    draw = rng.choice(accs, size=len(accs), replace=True)
                    mean_acc[lr] = float(np.mean(draw))
            sh = shoulder_from_means(mean_acc)["shoulder"]
            if sh is not None:
                shoulders[bsz] = sh
        fit = fit_alpha(shoulders)
        if fit is not None:
            alphas.append(fit["alpha"])
    if not alphas:
        return None
    arr = np.sort(np.asarray(alphas))
    return {
        "n_boot": len(alphas),
        "alpha_median": _r(float(np.median(arr))),
        "alpha_ci95": [
            _r(float(arr[int(0.025 * len(arr))])),
            _r(float(arr[min(len(arr) - 1, int(0.975 * len(arr)))])),
        ],
    }


# ----------------------------------------------------------- P2: candidates


def run_occupancy(log: Dict[str, Any], beta: str = "0.9", burn_in: int = 10) -> Optional[float]:
    """Whole-run pooled frac(rho < -0.2): total hits / total surviving
    snapshots, via the tested per-step series (min_n filter retained)."""
    series = _occ.occupancy_series(log, beta, burn_in)
    n = sum(cnt for _, _, cnt in series)
    if n == 0:
        return None
    hits = sum(frac * cnt for _, frac, cnt in series)
    return hits / n


def run_smoothness(log: Dict[str, Any], key: str, frac: float = 0.5) -> Optional[float]:
    sm = log.get("smoothness")
    if not isinstance(sm, dict):
        return None
    meds = []
    for m in sm["matrices"].values():
        stats = _smooth.plateau_stats(m.get(key, []), m.get("step", []), frac)
        if stats:
            meds.append(stats["median"])
    return st.median(meds) if meds else None


def run_hvp_q90(log: Dict[str, Any], lr0: float, total_steps: int) -> Optional[float]:
    vals = []
    for mat in log["matrices"].values():
        for d in mat["directions"]:
            lam = d.get("lambda_hvp") or {}
            for step, value in zip(lam.get("step", []), lam.get("value", [])):
                if value is not None and math.isfinite(value):
                    vals.append(_occ.lr_at(int(step), lr0, total_steps) * float(value))
    if not vals:
        return None
    return float(np.quantile(np.asarray(vals), 0.90))


P2_KEYS = ("occupancy", "spectral", "euclidean", "hvp_q90")


def p2_cell(cell_runs: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Pool candidate quantities over a cell's runs (mean of per-run values)."""
    per_run: Dict[str, List[float]] = {k: [] for k in P2_KEYS}
    for r in cell_runs:
        if not r["sidecar"] or not Path(r["sidecar"]).exists():
            continue
        log = json.loads(Path(r["sidecar"]).read_text())
        vals = {
            "occupancy": run_occupancy(log),
            "spectral": run_smoothness(log, _smooth.SPEC),
            "euclidean": run_smoothness(log, _smooth.EUC),
            "hvp_q90": run_hvp_q90(log, r["lr"], r["steps"]),
        }
        for k, v in vals.items():
            if v is not None and math.isfinite(v):
                per_run[k].append(v)
    return {k: (_r(st.mean(v)) if v else None) for k, v in per_run.items()}


def p2_tables(table, shoulders) -> Dict[str, Any]:
    out: Dict[str, Any] = {"at_shoulder": {}, "at_1x": {}, "ratios": {}}
    for bsz in P2_BATCHES:
        cells = table.get(bsz, {})
        sh = shoulders.get(bsz)
        if sh is not None and sh in cells:
            out["at_shoulder"][str(bsz)] = p2_cell(cells[sh])
        if STOCK_LR in cells:
            out["at_1x"][str(bsz)] = p2_cell(cells[STOCK_LR])
    for key in P2_KEYS:
        entry = {}
        for which in ("at_shoulder", "at_1x"):
            vals = [v[key] for v in out[which].values() if v.get(key) is not None]
            entry[which] = (
                _r(max(vals) / min(vals), 4) if len(vals) > 1 and min(vals) else None
            )
        r_sh, r_1x = entry["at_shoulder"], entry["at_1x"]
        entry["tracking_signature"] = (
            None
            if r_sh is None or r_1x is None
            else bool(r_sh < TRACKING_RATIO <= r_1x)
        )
        out["ratios"][key] = entry
    return out


# ------------------------------------------------------------------- report


def build_report(results_dir: Path) -> Dict[str, Any]:
    runs = load_frontier_runs(results_dir)
    if not runs:
        raise SystemExit(f"no frontier runs (config.path ~ {CONFIG_TAG}) in {results_dir}")
    table = cell_table(runs)

    frontier = {}
    shoulders = {}
    for bsz in sorted(table):
        mean_acc = {
            lr: st.mean([r["acc"] for r in cell if r["acc"] is not None])
            for lr, cell in table[bsz].items()
            if any(r["acc"] is not None for r in cell)
        }
        sh = shoulder_from_means(mean_acc)
        n_seeds = {lr: len(cell) for lr, cell in sorted(table[bsz].items())}
        frontier[str(bsz)] = {
            "mean_acc": {f"{lr:g}": _r(v, 4) for lr, v in sorted(mean_acc.items())},
            "n_runs": n_seeds,
            **sh,
        }
        shoulders[bsz] = sh["shoulder"]
        # P3: graceful-degradation check -- min mean accuracy across rungs
        frontier[str(bsz)]["min_mean_acc"] = _r(min(mean_acc.values()), 4)

    report = {
        "n_runs": len(runs),
        "frontier": frontier,
        "alpha_fit": fit_alpha(shoulders),
        "alpha_bootstrap": bootstrap_alpha(table),
        "p2": p2_tables(table, shoulders),
        "definitions": {
            "floor_drop_pp": FLOOR_DROP_PP,
            "ref_rungs": REF_RUNGS,
            "tracking_ratio": TRACKING_RATIO,
            "p2_batches": P2_BATCHES,
            "occupancy": "beta 0.9, burn_in 10, whole-run pooled",
            "plateau": "last 50% of measured steps, analyze_smoothness convention",
        },
    }
    return report


def to_markdown(report: Dict[str, Any]) -> str:
    def fmt(x, nd=4):
        return f"{x:.{nd}g}" if isinstance(x, (int, float)) else "—"

    L = ["# Stability frontier (lr × batch) — tables (descriptive)", ""]
    L.append(f"Runs: {report['n_runs']} · pre-registration: "
             "`reports/stability-frontier-preregistration.md` · no verdict here")
    L += ["", "## Accuracy vs lr rung per batch size", ""]
    header = "| B | " + " | ".join(f"{lr:g}" for lr in RUNGS) + " | ref | floor | shoulder | recoveries |"
    L.append(header)
    L.append("|---" * (len(RUNGS) + 5) + "|")
    for bsz, e in sorted(report["frontier"].items(), key=lambda kv: int(kv[0])):
        row = [bsz]
        for lr in RUNGS:
            row.append(fmt(e["mean_acc"].get(f"{lr:g}")))
        row += [fmt(e["ref_acc"]), fmt(e["floor"]),
                fmt(e["shoulder"], 3),
                ",".join(f"{r:g}" for r in e["recoveries"]) or "—"]
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## P1 — shoulder scaling", ""]
    fit, boot = report["alpha_fit"], report["alpha_bootstrap"]
    if fit:
        L.append(f"alpha = {fmt(fit['alpha'])} over {fit['n_points']} batch sizes "
                 f"(quantization half-width ±{fmt(fit['alpha_quantization_halfwidth'])}"
                 + (f"; seed bootstrap median {fmt(boot['alpha_median'])}, "
                    f"CI95 [{fmt(boot['alpha_ci95'][0])}, {fmt(boot['alpha_ci95'][1])}]"
                    if boot else "") + ")")
    else:
        L.append("alpha not fittable (fewer than 2 shoulders located)")
    L.append("")
    L.append("Pre-registered regions: noise-governed alpha ≥ 0.25 · "
             "deterministic |alpha| < 0.1 · between = ambiguous.")
    L += ["", "## P2 — invariant candidates (at-shoulder vs at-1× control)", ""]
    for which, title in (("at_shoulder", "At the per-B shoulder rung"),
                         ("at_1x", "At the fixed 1× rung (control)")):
        L += [f"### {title}", ""]
        L.append("| B | " + " | ".join(P2_KEYS) + " |")
        L.append("|---" * (len(P2_KEYS) + 1) + "|")
        for bsz, vals in sorted(report["p2"][which].items(), key=lambda kv: int(kv[0])):
            L.append(f"| {bsz} | " + " | ".join(fmt(vals.get(k)) for k in P2_KEYS) + " |")
        L.append("")
    L += ["### Across-B max/min ratios", "",
          "| candidate | at shoulder | at 1× | tracking signature (<1.5 and ≥1.5) |",
          "|---|---|---|---|"]
    for key in P2_KEYS:
        e = report["p2"]["ratios"][key]
        sig = e["tracking_signature"]
        L.append(f"| {key} | {fmt(e['at_shoulder'])} | {fmt(e['at_1x'])} | "
                 f"{'—' if sig is None else sig} |")
    L += ["", "## P3 — graceful degradation", "",
          "| B | min mean acc across rungs |", "|---|---|"]
    for bsz, e in sorted(report["frontier"].items(), key=lambda kv: int(kv[0])):
        L.append(f"| {bsz} | {fmt(e['min_mean_acc'])} |")
    L += ["", "Reads are descriptive; region calls and any interpretation live in",
          "`reports/stability-frontier.md`.", ""]
    return "\n".join(L) + "\n"


def make_plots(report: Dict[str, Any], out_dir: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for bsz, e in sorted(report["frontier"].items(), key=lambda kv: int(kv[0])):
        lrs = sorted(float(k) for k in e["mean_acc"])
        accs = [e["mean_acc"][f"{lr:g}"] for lr in lrs]
        (line,) = ax.plot([lr / STOCK_LR for lr in lrs], accs, "o-", label=f"B={bsz}")
        if e["floor"] is not None:
            ax.axhline(e["floor"], color=line.get_color(), lw=0.6, ls=":", alpha=0.5)
        if e["shoulder"] is not None:
            ax.axvline(e["shoulder"] / STOCK_LR, color=line.get_color(), lw=0.6,
                       ls="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("lr multiplier (× record 0.24)")
    ax.set_ylabel("mean TTA accuracy (%)")
    ax.set_title("Accuracy vs Muon lr per batch size (floors dotted, shoulders dashed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "frontier_acc_vs_lr.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    pts = [(int(b), e["shoulder"]) for b, e in report["frontier"].items()
           if e["shoulder"] is not None]
    if len(pts) >= 2:
        pts.sort()
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        bs = [b for b, _ in pts]
        sh = [s for _, s in pts]
        ax.plot(bs, sh, "o", color="#1f6f8b")
        fit = report["alpha_fit"]
        if fit:
            xs = np.array([min(bs), max(bs)], dtype=float)
            ys = np.exp(fit["intercept"]) * xs ** fit["alpha"]
            ax.plot(xs, ys, "-", color="#1f6f8b", alpha=0.7,
                    label=f"fit: lr* ∝ B^{fit['alpha']:.2f}")
        for alpha_ref, ls in ((0.5, "--"), (0.0, ":")):
            ys = sh[len(sh) // 2] * (np.array(bs, dtype=float) / bs[len(bs) // 2]) ** alpha_ref
            ax.plot(bs, ys, ls, color="gray", alpha=0.6, label=f"α={alpha_ref:g} ref")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_ylabel("shoulder lr")
        ax.set_title("Useful-lr shoulder vs batch size")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / "frontier_shoulder_vs_batch.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)

    fig, axes = plt.subplots(1, len(P2_KEYS), figsize=(3.2 * len(P2_KEYS), 3.4))
    for ax, key in zip(np.atleast_1d(axes), P2_KEYS):
        for which, marker, label in (("at_shoulder", "o-", "at shoulder"),
                                     ("at_1x", "s--", "at 1×")):
            items = sorted(report["p2"][which].items(), key=lambda kv: int(kv[0]))
            bs = [int(b) for b, v in items if v.get(key) is not None]
            ys = [v[key] for _, v in items if v.get(key) is not None]
            if bs:
                ax.plot(bs, ys, marker, label=label, ms=4)
        ax.set_xscale("log")
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("batch size")
        ax.legend(fontsize=7)
    fig.suptitle("P2 invariant candidates across batch size", fontsize=10)
    fig.tight_layout()
    p = out_dir / "frontier_p2_candidates.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-plot-dir", type=Path)
    args = ap.parse_args(argv)

    report = build_report(args.results_dir)
    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.out_md:
        args.out_md.write_text(to_markdown(report))
    if args.out_plot_dir:
        make_plots(report, args.out_plot_dir)
    print(json.dumps(
        {
            "n_runs": report["n_runs"],
            "shoulders": {b: e["shoulder"] for b, e in report["frontier"].items()},
            "alpha_fit": report["alpha_fit"],
            "p2_ratios": report["p2"]["ratios"],
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
