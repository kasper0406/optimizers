#!/usr/bin/env python
"""Directional-smoothness plateau analysis (brainstorm program #3).

Reads instrumentation sidecars carrying the v2 ``smoothness`` block and
answers the PRE-REGISTERED question stated in ``src/instrument/smoothness.py``:

    Does the generalized (spectral-norm) directional smoothness along the
    actual trajectory equilibrate at c/lr for some constant c -- the Muon
    analogue of GD's 2 and Adam's ~38 -- while the Euclidean quantity does
    not?  Signature: a plateau of the dimensionless product
    ``lr * D_smooth_spectral``; and because "equilibrates at c/lr" is an
    lr-scaling claim, the plateau constant must agree across the lr ladder.

DESCRIPTIVE OUTPUT ONLY -- no pass/fail language; the plateau read is a
human/gate judgment.  Deterministic (no RNG, sorted keys, byte-stable).

Caveats carried into the report (see the module docstring of
src/instrument/smoothness.py): minibatch estimate of a full-batch quantity;
sum-reduced loss (constants inherit that scale, reported alongside
batch_size); per-matrix perturbation, not the joint step; compile disabled.

Usage:
    uv run python scripts/analyze_smoothness.py <sidecar-dir-or-files...> \
        [--plateau-frac 0.5] [--out-md ...] [--out-json ...] [--out-plot ...]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SPEC = "lr_times_d_smooth_spectral"
EUC = "lr_times_d_smooth_frobenius"


def load_sidecars(paths):
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.instrumentation.json")) if p.is_dir() else [p]
    out = []
    for f in files:
        data = json.loads(f.read_text())
        if isinstance(data.get("smoothness"), dict):
            out.append((f.name, data))
    return out


def run_lr(name, log):
    """Nominal lr0 for a run: max recorded lr across matrices (step-0 value)."""
    lrs = [
        m["lr"][0]
        for m in log["smoothness"]["matrices"].values()
        if m.get("lr")
    ]
    return max(lrs) if lrs else float("nan")


def plateau_stats(series, steps, frac):
    """Median/IQR over the last `frac` of measured steps (the plateau window)."""
    if not series:
        return None
    cut = steps[int(len(steps) * (1.0 - frac))] if steps else 0
    vals = [v for v, s in zip(series, steps) if s >= cut and v is not None and math.isfinite(v)]
    if len(vals) < 3:
        return None
    vals_sorted = sorted(vals)
    q1 = vals_sorted[len(vals_sorted) // 4]
    q3 = vals_sorted[(3 * len(vals_sorted)) // 4]
    return {
        "n": len(vals),
        "median": st.median(vals),
        "iqr": q3 - q1,
        "rel_iqr": (q3 - q1) / abs(st.median(vals)) if st.median(vals) else None,
        "min": min(vals),
        "max": max(vals),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--plateau-frac", type=float, default=0.5,
                    help="tail fraction of measured steps treated as the plateau window")
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-plot", type=Path)
    args = ap.parse_args(argv)

    sidecars = load_sidecars(args.paths)
    if not sidecars:
        raise SystemExit("no sidecars with a 'smoothness' block found")

    report = {"plateau_frac": args.plateau_frac, "runs": [], "by_lr": {}}
    for name, log in sorted(sidecars):
        sm = log["smoothness"]
        lr0 = run_lr(name, log)
        entry = {
            "sidecar": name, "lr0": lr0, "t_meas": sm.get("t_meas"),
            "batch_size": sm.get("batch_size"),
            "loss_reduction": sm.get("loss_reduction"),
            "n_measured_steps": sm.get("n_measured_steps"),
            "matrices": {},
        }
        for mname, m in sorted(sm["matrices"].items()):
            steps = m.get("step", [])
            entry["matrices"][mname] = {
                "spectral": plateau_stats(m.get(SPEC, []), steps, args.plateau_frac),
                "euclidean": plateau_stats(m.get(EUC, []), steps, args.plateau_frac),
            }
        # pooled across matrices: median of per-matrix plateau medians
        for key in ("spectral", "euclidean"):
            meds = [v[key]["median"] for v in entry["matrices"].values() if v[key]]
            entry[f"pooled_{key}_median"] = st.median(meds) if meds else None
            entry[f"pooled_{key}_spread"] = (max(meds) - min(meds)) if len(meds) > 1 else None
        report["runs"].append(entry)
        report["by_lr"].setdefault(f"{lr0:.4g}", []).append(entry)

    # ladder comparison: is the spectral constant lr-invariant, and the Euclidean not?
    ladder = {}
    for lr_key, entries in sorted(report["by_lr"].items()):
        for key in ("spectral", "euclidean"):
            vals = [e[f"pooled_{key}_median"] for e in entries if e[f"pooled_{key}_median"]]
            ladder.setdefault(key, {})[lr_key] = {
                "n_runs": len(vals),
                "median": st.median(vals) if vals else None,
                "spread": (max(vals) - min(vals)) if len(vals) > 1 else None,
            }
    for key, per_lr in ladder.items():
        meds = [v["median"] for v in per_lr.values() if v["median"]]
        ladder[key]["across_lr_ratio_max_over_min"] = (
            max(meds) / min(meds) if len(meds) > 1 and min(meds) else None
        )
    report["ladder"] = ladder

    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    if args.out_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        for name, log in sorted(sidecars):
            lr0 = run_lr(name, log)
            for ax, key, title in (
                (axes[0], SPEC, "lr · D_smooth  (spectral)"),
                (axes[1], EUC, "lr · D_smooth  (Euclidean)"),
            ):
                for mname, m in sorted(log["smoothness"]["matrices"].items()):
                    ax.plot(m.get("step", []), m.get(key, []), alpha=0.35, lw=0.9,
                            label=f"lr0={lr0:.3g}" if mname.endswith("conv1.weight") else None)
                ax.set_title(title)
                ax.set_xlabel("step")
                ax.set_yscale("log")
        handles, labels = axes[0].get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        if uniq:
            axes[0].legend(uniq.values(), uniq.keys(), fontsize=8)
        fig.suptitle("Directional smoothness along the trajectory (plateau = constant)")
        fig.tight_layout()
        args.out_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_plot, dpi=130)
        plt.close(fig)

    if args.out_md:
        L = ["# Directional-smoothness plateau (descriptive)", ""]
        L.append(f"Sidecars: {len(sidecars)} · plateau window: last "
                 f"{args.plateau_frac:.0%} of measured steps · "
                 "minibatch estimate, sum-reduced loss, per-matrix perturbation")
        L.append("")
        L.append("| sidecar | lr0 | pooled lr·D_spectral | spread | pooled lr·D_euclid | spread |")
        L.append("|---|---|---|---|---|---|")
        for e in report["runs"]:
            def fmt(x):
                return f"{x:.4g}" if isinstance(x, (int, float)) else "—"
            L.append(f"| {e['sidecar']} | {e['lr0']:.4g} | {fmt(e['pooled_spectral_median'])} | "
                     f"{fmt(e['pooled_spectral_spread'])} | {fmt(e['pooled_euclidean_median'])} | "
                     f"{fmt(e['pooled_euclidean_spread'])} |")
        L += ["", "## LR-ladder invariance (the c/lr claim)", ""]
        L.append("| quantity | " + " | ".join(sorted(report["by_lr"])) + " | max/min |")
        L.append("|---" * (len(report["by_lr"]) + 2) + "|")
        for key in ("spectral", "euclidean"):
            row = [key]
            for lr_key in sorted(report["by_lr"]):
                v = ladder[key][lr_key]["median"]
                row.append(f"{v:.4g}" if v else "—")
            r = ladder[key].get("across_lr_ratio_max_over_min")
            row.append(f"{r:.3g}" if r else "—")
            L.append("| " + " | ".join(row) + " |")
        L += ["", "A plateau constant that is lr-invariant (ratio ≈ 1) for the spectral",
              "quantity while the Euclidean one scales with lr is the pre-registered",
              "signature. Read is descriptive; no verdict is asserted here.", ""]
        args.out_md.write_text("\n".join(L) + "\n")

    print(json.dumps({"n_sidecars": len(sidecars), "ladder": ladder}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
