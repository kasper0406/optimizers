#!/usr/bin/env python
"""Program #7 analysis: nanogpt frontier transfer (lr x token batch).

Implements the PRE-REGISTERED estimators of
``reports/frontier-nanogpt-preregistration.md`` (loss convention — lower is
better):

- valley(B)      = min over rungs of mean final val loss
- floor(B)       = valley(B) + 0.010
- lr*_cross(B)   = first upward adjacent pair with mean_i <= floor <
                   mean_{i+1}, interpolated linearly in (log lr, loss);
                   scanning starts AT the valley rung
- shoulder(B)    = largest rung >= valley rung with mean <= floor, up to
                   the first crossing (secondary)
- alpha          = OLS slope of log lr*_cross vs log tokens-per-step;
                   seed bootstrap (2000 draws, rng seed 0) indicative at n=2
- low-lr penalty = mean(0.5x rung) - valley, per B (the "low lr becomes the
                   losing side" signature)

Deterministic; descriptive only.

Usage:
    uv run python scripts/analyze_frontier_nanogpt.py results/ \
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

TAG_PREFIX = "frontier_nanogpt_c"
CHUNK_TOKENS = 49_152
FLOOR_DELTA = 0.010
RUNGS = [0.025, 0.035355, 0.05, 0.070711, 0.1, 0.141421]
BOOT_N = 2000
BOOT_SEED = 0


def _r(x: Optional[float], nd: int = 6) -> Optional[float]:
    return None if x is None else round(float(x), nd)


def load_runs(results_dir: Path) -> Dict[int, Dict[float, List[float]]]:
    """chunks -> muon_lr -> [final val loss], deduped (chunks, lr, seed)->latest.

    Files listed in results/INVALID_RUNS.json (append-only tombstones for
    corrupt artifacts, e.g. the checkpoint-collision incident) are excluded.
    """
    tomb = results_dir / "INVALID_RUNS.json"
    invalid = (
        {e["file"] for e in json.loads(tomb.read_text()).get("invalid", [])}
        if tomb.exists()
        else set()
    )
    latest: Dict[Tuple[int, float, int], float] = {}
    for f in sorted(results_dir.glob("nanogpt_seed*.json")):
        if f.name in invalid:
            continue
        d = json.loads(f.read_text())
        path = str((d.get("config") or {}).get("path", ""))
        if TAG_PREFIX not in path:
            continue
        ng = d["config"]["contents"]["nanogpt"]
        if ng.get("max_steps") is not None:
            continue  # smoke runs never enter the analysis
        chunks = int(ng["chunks_per_step"])
        lr = float(ng["muon_lr"])
        final = d["metrics"].get("final_val_loss")
        if final is None:
            continue
        latest[(chunks, lr, int(d["seed"]))] = float(final)
    table: Dict[int, Dict[float, List[float]]] = {}
    for (chunks, lr, _seed), v in sorted(latest.items()):
        table.setdefault(chunks, {}).setdefault(lr, []).append(v)
    return table


def crossing_from_means(mean_loss: Dict[float, float]) -> Dict[str, Any]:
    rungs = sorted(mean_loss)
    valley_lr = min(mean_loss, key=lambda lr: mean_loss[lr])
    valley = mean_loss[valley_lr]
    floor = valley + FLOOR_DELTA
    i0 = rungs.index(valley_lr)
    cross = None
    shoulder = valley_lr
    for lo, hi in zip(rungs[i0:], rungs[i0 + 1:]):
        m_lo, m_hi = mean_loss[lo], mean_loss[hi]
        if m_lo <= floor < m_hi:
            frac = (floor - m_lo) / (m_hi - m_lo)
            cross = math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
            break
        if m_hi <= floor:
            shoulder = hi
    else:
        shoulder = shoulder if cross is None else shoulder
    if cross is not None:
        # shoulder = last rung at-or-below floor before the crossing pair's hi
        shoulder = max(
            (lr for lr in rungs[i0:] if lr <= cross and mean_loss[lr] <= floor),
            default=valley_lr,
        )
    low_rung = rungs[0]
    return {
        "valley_lr": valley_lr,
        "valley": _r(valley, 5),
        "floor": _r(floor, 5),
        "lr_cross": _r(cross),
        "shoulder": shoulder,
        "low_lr_penalty": _r(mean_loss[low_rung] - valley, 5),
        "mean_loss": {f"{lr:g}": _r(v, 5) for lr, v in sorted(mean_loss.items())},
    }


def alpha_fit(crossings: Dict[int, Optional[float]]) -> Optional[Dict[str, Any]]:
    pts = sorted((c * CHUNK_TOKENS, x) for c, x in crossings.items() if x is not None)
    if len(pts) < 2:
        return None
    x = np.log([b for b, _ in pts])
    y = np.log([v for _, v in pts])
    slope, intercept = np.polyfit(x, y, 1)
    return {"alpha": _r(slope), "intercept": _r(intercept), "n_points": len(pts)}


def bootstrap_alpha(table) -> Optional[Dict[str, Any]]:
    rng = np.random.default_rng(BOOT_SEED)
    alphas = []
    for _ in range(BOOT_N):
        crossings = {}
        for chunks, cells in table.items():
            mean_loss = {}
            for lr, vals in cells.items():
                if vals:
                    draw = rng.choice(vals, size=len(vals), replace=True)
                    mean_loss[lr] = float(np.mean(draw))
            crossings[chunks] = crossing_from_means(mean_loss)["lr_cross"]
        fit = alpha_fit(crossings)
        if fit is not None and fit["n_points"] == len(table):
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


def build_report(results_dir: Path) -> Dict[str, Any]:
    table = load_runs(results_dir)
    if not table:
        raise SystemExit(f"no '{TAG_PREFIX}*' runs in {results_dir}")
    arms = {}
    crossings = {}
    for chunks, cells in sorted(table.items()):
        mean_loss = {lr: st.mean(v) for lr, v in cells.items() if v}
        entry = crossing_from_means(mean_loss)
        entry["n_per_cell"] = {f"{lr:g}": len(cells[lr]) for lr in sorted(cells)}
        entry["tokens_per_step"] = chunks * CHUNK_TOKENS
        entry["max_mean_loss"] = _r(max(mean_loss.values()), 5)
        arms[str(chunks)] = entry
        crossings[chunks] = entry["lr_cross"]
    valleys = [arms[str(c)]["valley_lr"] for c in sorted(table)]
    report = {
        "arms": arms,
        "alpha_fit": alpha_fit(crossings),
        "alpha_bootstrap": bootstrap_alpha(table),
        "valley_monotone_nondecreasing": all(
            b >= a for a, b in zip(valleys, valleys[1:])
        ),
        "p3_no_cliff": all(
            arms[str(c)]["max_mean_loss"] <= 4.0 for c in sorted(table)
        ),
        "definitions": {
            "floor_delta": FLOOR_DELTA,
            "estimator": "upward log-linear interpolated floor crossing from the valley",
            "bootstrap": {"n": BOOT_N, "rng_seed": BOOT_SEED},
        },
    }
    return report


def to_markdown(rep: Dict[str, Any]) -> str:
    def fmt(x, nd=4):
        return f"{x:.{nd}g}" if isinstance(x, (int, float)) else "—"

    L = ["# nanogpt frontier transfer (#7) — tables (descriptive)", ""]
    L.append("Pre-registration: `reports/frontier-nanogpt-preregistration.md` · no verdict here")
    L += ["", "## Loss vs Muon lr per token batch", ""]
    for chunks, e in sorted(rep["arms"].items(), key=lambda kv: int(kv[0])):
        L.append(f"### chunks={chunks} ({e['tokens_per_step']:,} tokens/step) — "
                 f"valley {fmt(e['valley'], 5)} @ {fmt(e['valley_lr'], 3)}, "
                 f"floor {fmt(e['floor'], 5)}, lr*_cross {fmt(e['lr_cross'], 4)}, "
                 f"shoulder {fmt(e['shoulder'], 3)}, low-lr penalty {fmt(e['low_lr_penalty'], 4)}")
        L.append("")
        rungs = sorted(e["mean_loss"], key=float)
        L.append("| lr | " + " | ".join(rungs) + " |")
        L.append("|---" * (len(rungs) + 1) + "|")
        L.append("| mean loss | " + " | ".join(fmt(e["mean_loss"][r], 5) for r in rungs) + " |")
        L.append("| n | " + " | ".join(str(e["n_per_cell"][r]) for r in rungs) + " |")
        L.append("")
    fit, boot = rep["alpha_fit"], rep["alpha_bootstrap"]
    L += ["## alpha (lr* vs tokens/step)", ""]
    if fit:
        line = f"alpha = {fmt(fit['alpha'])} over {fit['n_points']} batch sizes"
        if boot:
            line += (f" · bootstrap median {fmt(boot['alpha_median'])}, "
                     f"CI95 [{fmt(boot['alpha_ci95'][0])}, {fmt(boot['alpha_ci95'][1])}] "
                     "(n=2/cell: indicative)")
        L.append(line)
    else:
        L.append("alpha not fittable")
    L += ["", "Pre-registered regions: noise-governed ≥ 0.25 · deterministic < 0.1 · "
          "airbench transfer point 0.35 [0.30, 0.42].",
          "", f"P2d valley monotone non-decreasing: {rep['valley_monotone_nondecreasing']} · "
          f"P3d no cliff (all mean losses ≤ 4.0): {rep['p3_no_cliff']}",
          "", "Interpretation lives in `reports/frontier-nanogpt.md`.", ""]
    return "\n".join(L) + "\n"


def make_plot(rep: Dict[str, Any], out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for chunks, e in sorted(rep["arms"].items(), key=lambda kv: int(kv[0])):
        lrs = sorted(float(k) for k in e["mean_loss"])
        (line,) = axes[0].plot(
            lrs, [e["mean_loss"][f"{lr:g}"] for lr in lrs], "o-", ms=4,
            label=f"{e['tokens_per_step']//1024}K tok/step",
        )
        axes[0].axhline(e["floor"], color=line.get_color(), lw=0.6, ls=":", alpha=0.5)
        if e["lr_cross"]:
            axes[0].axvline(e["lr_cross"], color=line.get_color(), lw=0.8, ls="--", alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Muon lr")
    axes[0].set_ylabel("final val loss")
    axes[0].set_title("nanogpt: loss vs lr per token batch (floors dotted, crossings dashed)")
    axes[0].legend(fontsize=8)
    fit = rep["alpha_fit"]
    pts = [(e["tokens_per_step"], e["lr_cross"])
           for e in rep["arms"].values() if e["lr_cross"]]
    if fit and len(pts) >= 2:
        pts.sort()
        bs = [b for b, _ in pts]
        cr = [c for _, c in pts]
        axes[1].plot(bs, cr, "o", color="#1f6f8b", label="lr*_cross")
        xs = np.array([min(bs), max(bs)], dtype=float)
        axes[1].plot(xs, np.exp(fit["intercept"]) * xs ** fit["alpha"], "-",
                     color="#1f6f8b", alpha=0.7, label=f"fit: B^{fit['alpha']:.3f}")
        for aref, ls in ((0.35, "--"), (0.0, ":")):
            axes[1].plot(bs, cr[1] * (np.array(bs) / bs[1]) ** aref, ls,
                         color="gray", alpha=0.6, label=f"α={aref:g} ref")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("tokens per step")
        axes[1].set_ylabel("lr*_cross")
        axes[1].set_title("Frontier crossing vs token batch")
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "frontier_nanogpt_transfer.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-plot-dir", type=Path)
    args = ap.parse_args(argv)

    rep = build_report(args.results_dir)
    if args.out_json:
        args.out_json.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    if args.out_md:
        args.out_md.write_text(to_markdown(rep))
    if args.out_plot_dir:
        make_plot(rep, args.out_plot_dir)
    print(json.dumps(
        {
            "crossings": {c: e["lr_cross"] for c, e in rep["arms"].items()},
            "alpha_fit": rep["alpha_fit"],
            "alpha_bootstrap": rep["alpha_bootstrap"],
            "valley_monotone": rep["valley_monotone_nondecreasing"],
            "no_cliff": rep["p3_no_cliff"],
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
