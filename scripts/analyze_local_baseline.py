#!/usr/bin/env python
"""Local nanogpt baseline aggregation: own-harness sigma + power table.

Reads the results JSONs of a baseline config (default: the runs whose
embedded config path matches ``nanogpt_local_baseline``) and produces the
testbed-validation numbers project-state next-step #4 asks for:

- final val loss mean, sd, and the chi-square 95% CI on sd (df = n-1);
- per-checkpoint across-seed sd profile (the record's shrinks ~5x over
  training; ours is the yardstick future A/B comparisons will divide by);
- steps-to-target censoring count (the WP0.2 critique: on this harness the
  record's 3.28 target may be unreachable, making final-val-at-fixed-steps
  the only uncensored endpoint);
- end-of-run loss-per-step slope (mean over runs of the last two val
  checkpoints), to convert loss effects into steps-equivalents;
- a power table: seeds per arm for 80% power at alpha .05 (two-sided,
  unpaired; ``scripts/analyze_nanogpt.py::n_for_effect`` convention) for a
  range of loss effects, plus their steps-equivalents.

Deterministic; descriptive only.

Usage:
    uv run python scripts/analyze_local_baseline.py results/ \
        [--config-tag nanogpt_local_baseline] [--out-md ...] [--out-json ...]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TAG = "nanogpt_local_baseline"
POWER_Z = 2.802  # z_{0.975} + z_{0.80}; matches scripts/analyze_nanogpt.py
EFFECTS = [0.001, 0.00125, 0.0025, 0.005, 0.010]
# chi-square df=9 quantiles for the sd CI (scipy-free)
CHI2_9 = {"q025": 2.700, "q975": 19.023}


def n_per_arm(sd: float, effect: float) -> int:
    """Seeds per arm, unpaired two-sample, 80% power at alpha .05."""
    return max(2, math.ceil(2.0 * (sd / effect) ** 2 * POWER_Z**2))


def load(results_dir: Path, tag: str) -> List[Dict[str, Any]]:
    runs = []
    for f in sorted(results_dir.glob("nanogpt_seed*.json")):
        d = json.loads(f.read_text())
        if tag not in str((d.get("config") or {}).get("path", "")):
            continue
        m = d["metrics"]
        runs.append(
            {
                "seed": d["seed"],
                "file": f.name,
                "final": float(m["final_val_loss"]),
                "curve": {int(p["step"]): float(p["val_loss"]) for p in m["val_curve"]},
                "steps_to_target": m.get("steps_to_target"),
                "git_sha": d.get("git_sha"),
                "git_dirty": d.get("git_dirty"),
            }
        )
    return runs


def build_report(results_dir: Path, tag: str) -> Dict[str, Any]:
    runs = load(results_dir, tag)
    if len(runs) < 2:
        raise SystemExit(f"need >= 2 completed '{tag}' runs in {results_dir}, found {len(runs)}")
    finals = [r["final"] for r in runs]
    n = len(finals)
    mean = st.mean(finals)
    sd = st.stdev(finals)
    sd_ci = (
        [sd * math.sqrt(9 / CHI2_9["q975"]), sd * math.sqrt(9 / CHI2_9["q025"])]
        if n - 1 == 9
        else None  # CI constants are df=9; other n reported without one
    )

    steps = sorted(set.intersection(*(set(r["curve"]) for r in runs)))
    profile = {
        str(s): {
            "mean": round(st.mean([r["curve"][s] for r in runs]), 5),
            "sd": round(st.stdev([r["curve"][s] for r in runs]), 6),
        }
        for s in steps
    }

    # end-of-run slope: loss removed per step between the last two checkpoints
    s_prev, s_last = steps[-2], steps[-1]
    slopes = [(r["curve"][s_prev] - r["curve"][s_last]) / (s_last - s_prev) for r in runs]
    slope = st.mean(slopes)

    censored = sum(1 for r in runs if r["steps_to_target"] is None)
    power = [
        {
            "effect_loss": e,
            "effect_steps_equiv": round(e / slope, 1) if slope > 0 else None,
            "n_per_arm": n_per_arm(sd, e),
        }
        for e in EFFECTS
    ]

    return {
        "config_tag": tag,
        "n_runs": n,
        "seeds": sorted(r["seed"] for r in runs),
        "final_val_loss": {
            "per_seed": {str(r["seed"]): round(r["final"], 5) for r in runs},
            "mean": round(mean, 5),
            "sd": round(sd, 6),
            "sd_ci95": [round(x, 6) for x in sd_ci] if sd_ci else None,
            "se_mean": round(sd / math.sqrt(n), 6),
        },
        "checkpoint_sd_profile": profile,
        "end_slope_loss_per_step": round(slope, 8),
        "steps_to_target_censored": f"{censored}/{n}",
        "git_shas": sorted({r["git_sha"] for r in runs}),
        "any_git_dirty": any(r["git_dirty"] for r in runs),
        "power_table": power,
    }


def to_markdown(rep: Dict[str, Any]) -> str:
    f = rep["final_val_loss"]
    L = [
        "# nanogpt local baseline — own-harness sigma (descriptive)",
        "",
        f"Config tag: `{rep['config_tag']}` · n = {rep['n_runs']} seeds "
        f"{rep['seeds'][0]}–{rep['seeds'][-1]} · SHA(s): {', '.join(rep['git_shas'])}",
        "",
        f"**final val loss = {f['mean']} ± {f['sd']} (sd)**, SE(mean) {f['se_mean']}"
        + (f", sd CI95 [{f['sd_ci95'][0]}, {f['sd_ci95'][1]}]" if f["sd_ci95"] else ""),
        "",
        f"steps-to-target(3.28) censored: {rep['steps_to_target_censored']} — "
        "final-val-at-1750-steps is the uncensored endpoint on this harness.",
        "",
        "## Per-seed finals",
        "",
        "| seed | final val |",
        "|---|---|",
    ]
    for seed, v in sorted(f["per_seed"].items(), key=lambda kv: int(kv[0])):
        L.append(f"| {seed} | {v} |")
    L += ["", "## Across-seed sd by checkpoint", "", "| step | mean | sd |", "|---|---|---|"]
    for s, e in sorted(rep["checkpoint_sd_profile"].items(), key=lambda kv: int(kv[0])):
        L.append(f"| {s} | {e['mean']} | {e['sd']} |")
    L += [
        "",
        f"End-of-run slope: {rep['end_slope_loss_per_step']} loss/step "
        "(mean of last-two-checkpoint differences).",
        "",
        "## Power (80% power, alpha .05 two-sided, unpaired)",
        "",
        "| loss effect | ~steps equiv | seeds/arm |",
        "|---|---|---|",
    ]
    for row in rep["power_table"]:
        L.append(
            f"| {row['effect_loss']} | {row['effect_steps_equiv']} | {row['n_per_arm']} |"
        )
    L += ["", "Descriptive only; testbed interpretation lives in "
          "`reports/nanogpt-local-baseline.md`.", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--config-tag", default=DEFAULT_TAG)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args(argv)

    rep = build_report(args.results_dir, args.config_tag)
    if args.out_json:
        args.out_json.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    if args.out_md:
        args.out_md.write_text(to_markdown(rep))
    print(json.dumps(
        {k: rep[k] for k in ("n_runs", "final_val_loss", "steps_to_target_censored")},
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
