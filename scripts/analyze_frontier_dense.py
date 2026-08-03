#!/usr/bin/env python
"""Frontier sharpening analysis (program #6b).

Implements the PRE-REGISTERED estimators of
``reports/frontier-sharpening-preregistration.md``:

Part 1 (dense ladder, B in {1000, 2000, 4000}, n=5 seeds):
- ref(B)   = max mean TTA accuracy over that B's rung list
- floor(B) = ref(B) - 1.0pp
- lr*_cross(B): first adjacent rung pair with mean acc_i >= floor >
  acc_{i+1}, linearly interpolated in (log lr, acc); undefined if the
  lowest rung is below floor or no crossing occurs in the band
- alpha    = OLS slope of log lr*_cross vs log B over the three B
- seed bootstrap: resample per-cell accuracies with replacement, 2000
  draws, rng seed 0; 95% CI on alpha

Part 2 (step-matched B=8000, epochs 32, n=2):
- peak-referenced shoulder (largest rung within 1.0pp of the ladder max,
  scanning down from the peak to the first crossing) + the Part-1
  interpolated crossing on the same ladder

DESCRIPTIVE OUTPUT ONLY. Deterministic: fixed bootstrap RNG, sorted keys.

Usage:
    uv run python scripts/analyze_frontier_dense.py results/ \
        [--out-md ...] [--out-json ...] [--out-plot-dir ...]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DENSE_TAGS = {
    "frontier_dense_b1000": 1000,
    "frontier_dense_b2000": 2000,
    "frontier_dense_b4000": 4000,
}
STEPMATCH_TAG = "frontier_b8000_stepmatched"
FLOOR_DROP_PP = 1.0
BOOT_N = 2000
BOOT_SEED = 0


def _r(x: Optional[float], nd: int = 6) -> Optional[float]:
    return None if x is None else round(float(x), nd)


# ------------------------------------------------------------------ loading


def load_runs(results_dir: Path) -> Tuple[Dict[int, Dict[float, List[float]]], Dict[float, List[float]]]:
    """Return (dense: B -> lr -> [acc%], stepmatched: lr -> [acc%]),
    deduped by (tag, lr, seed) keeping the latest file."""
    latest: Dict[Tuple[str, float, int], Tuple[str, float]] = {}
    for f in sorted(results_dir.glob("airbench_instrumented_seed*.json")):
        if f.name.endswith(".instrumentation.json"):
            continue
        data = json.loads(f.read_text())
        cfg = data.get("config") or {}
        path = str(cfg.get("path", ""))
        tag = next((t for t in list(DENSE_TAGS) + [STEPMATCH_TAG] if t in path), None)
        if tag is None:
            continue
        # frontier_dense_b1000 would also substring-match a hypothetical
        # longer tag; exact directory-style match guards against that.
        contents = cfg.get("contents") or {}
        lr = float(contents["probe_overrides"]["lr"])
        acc = data["metrics"].get("tta_val_acc")
        if acc is None:
            continue
        latest[(tag, lr, int(data["seed"]))] = (f.name, 100.0 * float(acc))
    dense: Dict[int, Dict[float, List[float]]] = {}
    stepm: Dict[float, List[float]] = {}
    for (tag, lr, _seed), (_fname, acc) in sorted(latest.items()):
        if tag == STEPMATCH_TAG:
            stepm.setdefault(lr, []).append(acc)
        else:
            dense.setdefault(DENSE_TAGS[tag], {}).setdefault(lr, []).append(acc)
    return dense, stepm


# ------------------------------------------------------ pre-reg estimators


def interpolated_crossing(mean_acc: Dict[float, float], floor: float) -> Optional[float]:
    """First adjacent pair bracketing the floor, interpolated in (log lr, acc)."""
    rungs = sorted(mean_acc)
    if not rungs or mean_acc[rungs[0]] < floor:
        return None
    for lo, hi in zip(rungs, rungs[1:]):
        a_lo, a_hi = mean_acc[lo], mean_acc[hi]
        if a_lo >= floor > a_hi:
            frac = (a_lo - floor) / (a_lo - a_hi)
            return math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
    return None


def crossing_for_cells(cells: Dict[float, List[float]]) -> Dict[str, Any]:
    mean_acc = {lr: st.mean(accs) for lr, accs in cells.items() if accs}
    ref = max(mean_acc.values())
    floor = ref - FLOOR_DROP_PP
    return {
        "mean_acc": {f"{lr:g}": _r(v, 4) for lr, v in sorted(mean_acc.items())},
        "n_per_cell": {f"{lr:g}": len(cells[lr]) for lr in sorted(cells)},
        "ref_acc": _r(ref, 4),
        "floor": _r(floor, 4),
        "lr_cross": _r(interpolated_crossing(mean_acc, floor)),
    }


def peak_referenced_shoulder(mean_acc: Dict[float, float]) -> Optional[float]:
    """Largest rung within 1.0pp of the ladder max, scanning down from the
    peak rung to the first crossing (pre-declared for Part 2)."""
    if not mean_acc:
        return None
    rungs = sorted(mean_acc)
    floor = max(mean_acc.values()) - FLOOR_DROP_PP
    peak_rung = max(mean_acc, key=lambda lr: mean_acc[lr])
    shoulder = peak_rung
    for lr in rungs[rungs.index(peak_rung) + 1:]:
        if mean_acc[lr] >= floor:
            shoulder = lr
        else:
            break
    return shoulder


def alpha_from_crossings(crossings: Dict[int, Optional[float]]) -> Optional[Dict[str, Any]]:
    pts = sorted((b, c) for b, c in crossings.items() if c is not None)
    if len(pts) < 2:
        return None
    x = np.log([b for b, _ in pts])
    y = np.log([c for _, c in pts])
    slope, intercept = np.polyfit(x, y, 1)
    return {"alpha": _r(slope), "intercept": _r(intercept), "n_points": len(pts)}


def bootstrap_alpha(dense: Dict[int, Dict[float, List[float]]]) -> Optional[Dict[str, Any]]:
    rng = np.random.default_rng(BOOT_SEED)
    alphas = []
    n_undefined = 0
    for _ in range(BOOT_N):
        crossings = {}
        for bsz, cells in dense.items():
            mean_acc = {}
            for lr, accs in cells.items():
                if accs:
                    draw = rng.choice(accs, size=len(accs), replace=True)
                    mean_acc[lr] = float(np.mean(draw))
            floor = max(mean_acc.values()) - FLOOR_DROP_PP
            crossings[bsz] = interpolated_crossing(mean_acc, floor)
        fit = alpha_from_crossings(crossings)
        if fit is not None and fit["n_points"] == len(dense):
            alphas.append(fit["alpha"])
        else:
            n_undefined += 1
    if not alphas:
        return None
    arr = np.sort(np.asarray(alphas))
    return {
        "n_boot": len(alphas),
        "n_draws_with_undefined_crossing": n_undefined,
        "alpha_median": _r(float(np.median(arr))),
        "alpha_ci95": [
            _r(float(arr[int(0.025 * len(arr))])),
            _r(float(arr[min(len(arr) - 1, int(0.975 * len(arr)))])),
        ],
    }


# ------------------------------------------------------------------- report


def build_report(results_dir: Path) -> Dict[str, Any]:
    dense, stepm = load_runs(results_dir)
    if not dense:
        raise SystemExit(f"no dense-ladder runs (tags {sorted(DENSE_TAGS)}) in {results_dir}")

    part1 = {str(b): crossing_for_cells(cells) for b, cells in sorted(dense.items())}
    crossings = {b: part1[str(b)]["lr_cross"] for b in dense}
    report: Dict[str, Any] = {
        "part1": part1,
        "alpha_fit": alpha_from_crossings(crossings),
        "alpha_bootstrap": bootstrap_alpha(dense),
        "definitions": {
            "floor_drop_pp": FLOOR_DROP_PP,
            "estimator": "log-linear interpolated first floor crossing",
            "bootstrap": {"n": BOOT_N, "rng_seed": BOOT_SEED},
        },
    }
    if stepm:
        entry = crossing_for_cells(stepm)
        mean_acc = {float(k): v for k, v in entry["mean_acc"].items()}
        entry["peak_ref_shoulder"] = peak_referenced_shoulder(mean_acc)
        report["part2_stepmatched_b8000"] = entry
    return report


def to_markdown(report: Dict[str, Any]) -> str:
    def fmt(x, nd=4):
        return f"{x:.{nd}g}" if isinstance(x, (int, float)) else "—"

    L = ["# Frontier sharpening (#6b) — tables (descriptive)", ""]
    L.append("Pre-registration: `reports/frontier-sharpening-preregistration.md` · no verdict here")
    L += ["", "## Part 1 — dense ladders", ""]
    for bsz, e in sorted(report["part1"].items(), key=lambda kv: int(kv[0])):
        L.append(f"### B = {bsz}  (ref {fmt(e['ref_acc'])}, floor {fmt(e['floor'])}, "
                 f"lr*_cross {fmt(e['lr_cross'])})")
        L.append("")
        rungs = sorted(e["mean_acc"], key=float)
        L.append("| lr | " + " | ".join(rungs) + " |")
        L.append("|---" * (len(rungs) + 1) + "|")
        L.append("| mean acc | " + " | ".join(fmt(e["mean_acc"][r]) for r in rungs) + " |")
        L.append("| n | " + " | ".join(str(e["n_per_cell"][r]) for r in rungs) + " |")
        L.append("")
    fit, boot = report["alpha_fit"], report["alpha_bootstrap"]
    L += ["## alpha", ""]
    if fit:
        line = f"alpha = {fmt(fit['alpha'])} over {fit['n_points']} batch sizes"
        if boot:
            line += (f" · seed bootstrap median {fmt(boot['alpha_median'])}, "
                     f"CI95 [{fmt(boot['alpha_ci95'][0])}, {fmt(boot['alpha_ci95'][1])}] "
                     f"({boot['n_boot']}/{BOOT_N} draws valid)")
        L.append(line)
    else:
        L.append("alpha not fittable")
    L.append("")
    L.append("Pre-registered regions: noise-governed alpha ≥ 0.25 · deterministic |alpha| < 0.1.")
    p2 = report.get("part2_stepmatched_b8000")
    if p2:
        L += ["", "## Part 2 — step-matched B=8000 (epochs 32, ~200 steps)", ""]
        rungs = sorted(p2["mean_acc"], key=float)
        L.append("| lr | " + " | ".join(rungs) + " |")
        L.append("|---" * (len(rungs) + 1) + "|")
        L.append("| mean acc | " + " | ".join(fmt(p2["mean_acc"][r]) for r in rungs) + " |")
        L += ["",
              f"peak-referenced shoulder: {fmt(p2['peak_ref_shoulder'], 3)} · "
              f"interpolated lr*_cross: {fmt(p2['lr_cross'])} · "
              f"ref {fmt(p2['ref_acc'])}",
              "",
              "Pre-registered read: shoulder ≥ 0.72 → undertraining explained the",
              "program-#6 fallback; ≤ 0.48 → the rightward shift saturates by B=8000."]
    L += ["", "Reads are descriptive; interpretation lives in `reports/frontier-sharpening.md`.", ""]
    return "\n".join(L) + "\n"


def make_plots(report: Dict[str, Any], out_dir: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for bsz, e in sorted(report["part1"].items(), key=lambda kv: int(kv[0])):
        lrs = sorted(float(k) for k in e["mean_acc"])
        accs = [e["mean_acc"][f"{lr:g}"] for lr in lrs]
        (line,) = ax.plot(lrs, accs, "o-", ms=4, label=f"B={bsz}")
        ax.axhline(e["floor"], color=line.get_color(), lw=0.6, ls=":", alpha=0.5)
        if e["lr_cross"]:
            ax.axvline(e["lr_cross"], color=line.get_color(), lw=0.8, ls="--", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Muon lr")
    ax.set_ylabel("mean TTA accuracy (%)")
    ax.set_title("Dense ladders: floors dotted, interpolated crossings dashed (n=5)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "sharpening_dense_ladders.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    fit = report["alpha_fit"]
    pts = [(int(b), e["lr_cross"]) for b, e in report["part1"].items() if e["lr_cross"]]
    if fit and len(pts) >= 2:
        pts.sort()
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        bs = [b for b, _ in pts]
        cr = [c for _, c in pts]
        ax.plot(bs, cr, "o", color="#1f6f8b", label="interpolated crossings")
        xs = np.array([min(bs), max(bs)], dtype=float)
        ax.plot(xs, np.exp(fit["intercept"]) * xs ** fit["alpha"], "-",
                color="#1f6f8b", alpha=0.7, label=f"fit: lr* ∝ B^{fit['alpha']:.3f}")
        for aref, ls in ((0.5, "--"), (0.0, ":")):
            ax.plot(bs, cr[1] * (np.array(bs) / bs[1]) ** aref, ls, color="gray",
                    alpha=0.6, label=f"α={aref:g} ref")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_ylabel("lr*_cross")
        boot = report["alpha_bootstrap"]
        sub = (f"CI95 [{boot['alpha_ci95'][0]:.3f}, {boot['alpha_ci95'][1]:.3f}]"
               if boot else "")
        ax.set_title(f"Interpolated frontier crossing vs batch {sub}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / "sharpening_alpha_fit.png"
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
            "crossings": {b: e["lr_cross"] for b, e in report["part1"].items()},
            "alpha_fit": report["alpha_fit"],
            "alpha_bootstrap": report["alpha_bootstrap"],
            "stepmatched": {
                k: report.get("part2_stepmatched_b8000", {}).get(k)
                for k in ("peak_ref_shoulder", "lr_cross", "ref_acc")
            },
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
