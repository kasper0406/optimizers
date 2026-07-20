#!/usr/bin/env python
"""WP0.2 analysis: our nanogpt port vs the pinned record's published trace.

DESCRIPTIVE ONLY. This script reports overlays, deviations, and distributions;
it never emits a pass/fail. The reproduction-quality judgment is the human's
at the WP0.2 checkpoint, against `criteria/nanogpt_tolerance.yaml`.

Usage::

    uv run python scripts/analyze_nanogpt.py --results results/nanogpt_seed*.json
    uv run python scripts/analyze_nanogpt.py --results ... \
        --out-md reports/wp02-nanogpt-repro.md --out-png reports/figures/wp02-overlay.png

What it produces:

1. **Overlay** of our val-loss-vs-tokens curve(s) against the record's
   published trace (parsed out of the record log file itself), plus the max
   absolute deviation at the shared token checkpoints (every 125 steps =
   49.15M tokens).
2. **Steps-to-target** (val loss <= 3.28), linearly interpolated between the
   125-step validation points, for each of our seeds, with mean/std.
3. **Our seed variance vs the record's n=20 distribution** — our final val
   loss mean/std next to the record's 3.2791 / 0.0013, with a Welch t-test
   reported as a descriptive statistic (not a gate).
4. **Power note**: the n our variance implies for a 1% and a 3%
   steps-to-target effect at 80% power — the number WP3.x needs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nanogpt.config import RECORD_TOKENS_PER_STEP  # noqa: E402
from src.nanogpt.record_log import (  # noqa: E402
    RecordTrace,
    deviation_at_checkpoints,
    record_validation_traces,
    steps_to_target,
)


def load_runs(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    runs = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        if data.get("experiment") != "nanogpt":
            continue
        m = data["metrics"]
        runs.append(
            {
                "path": Path(path),
                "seed": data["seed"],
                "gpu_type": data["gpu_type"],
                "git_sha": data["git_sha"][:8],
                "cost_usd": data.get("cost_usd"),
                "wall_time_s": data["wall_time_s"],
                "train_time_s": m.get("train_time_s"),
                "steps": [int(p["step"]) for p in m["val_curve"]],
                "tokens": [int(p["tokens"]) for p in m["val_curve"]],
                "losses": [float(p["val_loss"]) for p in m["val_curve"]],
                "final_val_loss": m.get("final_val_loss"),
                "steps_to_target": m.get("steps_to_target_loss"),
                "target": m.get("target_val_loss", 3.28),
                "record_faithful": m.get("record_faithful"),
                "deviations": m.get("deviations", {}),
                "device_count": m.get("device_count"),
                "accumulation_factor": m.get("accumulation_factor"),
                "tokens_per_step": m.get("tokens_per_step"),
                "precision_mode": m.get("precision_mode"),
            }
        )
    return runs


def welch_t(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    if va + vb == 0:
        return None
    return (st.mean(a) - st.mean(b)) / math.sqrt(va + vb)


def n_for_effect(sd: float, effect: float, power_z: float = 2.802) -> Optional[int]:
    """Per-arm n for a two-sided 0.05 test at 80% power (z_{a/2}+z_b = 2.802)."""
    if effect <= 0 or sd <= 0:
        return None
    return math.ceil(2 * (power_z * sd / effect) ** 2)


def build_report(runs: List[Dict[str, Any]], record: List[RecordTrace]) -> str:
    ref: RecordTrace = record[0]
    rec_finals = [t.final_val_loss for t in record]
    rec_steps_to = [s for s in (t.steps_to_target(3.28) for t in record) if s is not None]

    out: List[str] = []
    w = out.append
    w("# WP0.2 — nanogpt port vs pinned record (descriptive)\n")
    w(f"Record: **2025-07-12_BosAlign**, n={len(record)} same-script validation runs "
      f"in `{ref.path.parent.relative_to(REPO_ROOT)}` (script md5 `{ref.script_md5[:8]}`).")
    w(f"Record final val loss: mean {st.mean(rec_finals):.4f}, "
      f"std {st.stdev(rec_finals):.4f}, n={len(rec_finals)}.")
    if rec_steps_to:
        w(f"Record steps-to-3.28 (interpolated from its own 125-step trace): "
          f"mean {st.mean(rec_steps_to):.1f}, std {st.stdev(rec_steps_to):.1f} "
          f"of {ref.total_steps} total.\n")

    if not runs:
        w("\n**No local runs supplied** — record-side summary only. "
          "Pass `--results results/nanogpt_*.json` once cloud runs have synced.\n")
        return "\n".join(out)

    # ---- provenance / deviation audit ------------------------------------
    w("\n## Runs\n")
    w("| seed | GPU | git | device_count x accum | tokens/step | precision | final val | steps→3.28 | train s | $ |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        s2t = "—" if r["steps_to_target"] is None else f"{r['steps_to_target']:.1f}"
        w(f"| {r['seed']} | {r['gpu_type']} | {r['git_sha']} | "
          f"{r['device_count']} x {r['accumulation_factor']} | {r['tokens_per_step']:,} | "
          f"{r['precision_mode']} | {r['final_val_loss']:.4f} | {s2t} | "
          f"{r['train_time_s']:.0f} | {r['cost_usd'] if r['cost_usd'] is not None else '—'} |")

    unfaithful = [r for r in runs if not r["record_faithful"]]
    if unfaithful:
        w("\n**Deviation flags active** (these runs are NOT record-faithful):\n")
        for r in unfaithful:
            for k, v in r["deviations"].items():
                w(f"- seed {r['seed']} — `{k}`: {v}")
    tps = {r["tokens_per_step"] for r in runs}
    if tps != {RECORD_TOKENS_PER_STEP}:
        w(f"\n**WARNING: tokens/step {sorted(tps)} != record {RECORD_TOKENS_PER_STEP}** — "
          "the runs are not on the record's token batch; the overlay is not comparable.")

    # ---- overlay ---------------------------------------------------------
    w("\n## Overlay vs the record trace\n")
    w("Deviation = our val loss − record val loss at the same step "
      f"(token checkpoints of {ref.tokens_per_step:,} tokens/step).\n")
    w("| seed | max |dev| | at step | dev @25% | dev @50% | dev @75% | dev @final |")
    w("|---|---|---|---|---|---|---|")
    for r in runs:
        rows = deviation_at_checkpoints((r["steps"], r["losses"]), ref)
        if not rows:
            w(f"| {r['seed']} | (no shared checkpoints) | | | | | |")
            continue
        worst = max(rows, key=lambda row: abs(row[3]))
        def at(frac: float) -> str:
            target_step = int(frac * ref.total_steps)
            near = min(rows, key=lambda row: abs(row[0] - target_step))
            return f"{near[3]:+.4f}"
        w(f"| {r['seed']} | {abs(worst[3]):.4f} | {worst[0]} | "
          f"{at(0.25)} | {at(0.5)} | {at(0.75)} | {rows[-1][3]:+.4f} |")

    # ---- distributions ---------------------------------------------------
    finals = [r["final_val_loss"] for r in runs if r["final_val_loss"] is not None]
    ours_s2t = [r["steps_to_target"] for r in runs if r["steps_to_target"] is not None]
    w("\n## Distributions\n")
    w(f"Ours: final val loss mean {st.mean(finals):.4f}"
      + (f", std {st.stdev(finals):.4f}" if len(finals) > 1 else " (n=1, no std)")
      + f", n={len(finals)}.")
    w(f"Record: mean {st.mean(rec_finals):.4f}, std {st.stdev(rec_finals):.4f}, n={len(rec_finals)}.")
    t = welch_t(finals, rec_finals)
    if t is not None:
        w(f"Welch t (ours vs record, descriptive only): t = {t:+.2f}.")
    if ours_s2t:
        w(f"\nOurs: steps-to-{runs[0]['target']} mean {st.mean(ours_s2t):.1f}"
          + (f", std {st.stdev(ours_s2t):.1f}" if len(ours_s2t) > 1 else " (n=1)") + ".")
        if len(ours_s2t) > 1 and st.mean(ours_s2t) > 0:
            sd = st.stdev(ours_s2t)
            mean = st.mean(ours_s2t)
            w("\n### Power (per-arm n, two-sided 0.05, 80% power)\n")
            w("| effect on steps-to-target | absolute steps | required n/arm |")
            w("|---|---|---|")
            for pct in (0.01, 0.03, 0.05):
                eff = pct * mean
                w(f"| {pct*100:.0f}% | {eff:.1f} | {n_for_effect(sd, eff)} |")
            w("\nBasis: our own seed std above (small n — treat as an order-of-magnitude "
              "input to WP3.x, not a precise number).")
    else:
        w(f"\nNo run reached the target loss {runs[0]['target']} — steps-to-target "
          "is undefined and is NOT extrapolated.")

    w("\n---\nDescriptive only; no pass/fail. Reproduction quality is judged by the "
      "human against `criteria/nanogpt_tolerance.yaml`.")
    return "\n".join(out)


def plot(runs, record: List[RecordTrace], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for t in record:
        ax.plot([x / 1e6 for x in t.tokens], t.val_losses, color="0.7", lw=0.8,
                label="record (n=%d)" % len(record) if t is record[0] else None)
    for r in runs:
        ax.plot([x / 1e6 for x in r["tokens"]], r["losses"], lw=1.5, label=f"ours seed {r['seed']}")
    ax.axhline(3.28, color="crimson", ls="--", lw=0.8, label="target 3.28")
    ax.set_xlabel("tokens (M)")
    ax.set_ylabel("val loss")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title("loss vs tokens")

    ax = axes[1]
    ref = record[0]
    for r in runs:
        rows = deviation_at_checkpoints((r["steps"], r["losses"]), ref)
        if rows:
            ax.plot([row[0] for row in rows], [row[3] for row in rows], marker="o", ms=3,
                    label=f"seed {r['seed']}")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("ours − record (val loss)")
    ax.set_title("deviation at shared checkpoints")
    ax.legend(fontsize=7)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, nargs="*", default=[], help="nanogpt results JSONs")
    ap.add_argument("--record-dir", type=Path, default=None, help="record log directory")
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-png", type=Path, default=None)
    args = ap.parse_args(argv)

    record = record_validation_traces(args.record_dir) if args.record_dir else record_validation_traces()
    runs = load_runs(args.results)
    report = build_report(runs, record)
    print(report)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(report + "\n")
        print(f"\nwrote {args.out_md}")
    if args.out_png:
        plot(runs, record, args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
