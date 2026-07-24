#!/usr/bin/env python
"""Program #22 BBP Phase A-empirical analysis (prereg reports/bbp-prereg.md,
including AMENDMENTS A1 §3b and A2 §3c).

Computes only the registered quantities:
  (V)  vacuity guard — >=50% of matrices with a_hat dynamic range >= 0.2
  (S') saturation    — median_m a_hat(128)/a_hat(256) >= 0.9 at the MID
                       checkpoint, evaluated on the HELD-OUT arm (A1); the
                       shard-3 arm is reported alongside as the seen-data
                       comparison.
The momentum-corrected training point is b_eff = 8*(1+beta)/(1-beta) = 312
chunks (beta=0.95), just above the measured grid top of 256 — which is why
(S') tests flatness approaching it rather than the originally-registered
e(8)/e(64), whose evaluation points both landed off-grid (amendment A2).

Reports; the gate call is human.

Usage: uv run python scripts/analyze_bbp.py --out reports/bbp-phase-a.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BETA = 0.95
B_EFF_FACTOR = (1 + BETA) / (1 - BETA)  # 39
RECORD_B = 8  # chunks per record optimizer step
# Criterion (S') per prereg AMENDMENT A2: the momentum-corrected training
# point b_eff(8)=312 sits just above the grid top (256), so saturation is
# tested by flatness approaching it. The originally-registered e(8)/e(64) was
# vacuous — both points clamped off-grid, making the ratio identically 1.
SAT_NUM, SAT_DEN = 128, 256  # (S') ratio a_hat(128)/a_hat(256)


def load_probes() -> Dict[str, Dict]:
    """tag -> probe metrics, newest per config tag."""
    out: Dict[str, Dict] = {}
    for f in sorted(glob.glob(str(REPO_ROOT / "results" / "bbp_probe_seed*.json"))):
        d = json.load(open(f))
        tag = Path(str((d.get("config") or {}).get("path", ""))).stem
        m = d.get("metrics") or {}
        if "curves" in m:
            out[tag] = m
    return out


def interp_a(curve: Dict, b_target: float, allow_clamp: bool = True) -> Optional[float]:
    """a_hat at arbitrary b by linear interpolation in log b.

    ``allow_clamp=False`` returns None off-grid instead of clamping. Criteria
    MUST use that mode: prereg amendment A2 exists because clamped off-grid
    readings silently made criterion (S) identically 1.0.
    """
    bs, a = curve["b_chunks"], curve["a_hat"]
    if not allow_clamp and not (bs[0] <= b_target <= bs[-1]):
        return None
    lb, lt = [math.log(x) for x in bs], math.log(b_target)
    if lt <= lb[0]:
        return a[0]
    if lt >= lb[-1]:
        return a[-1]
    for i in range(1, len(lb)):
        if lt <= lb[i]:
            w = (lt - lb[i - 1]) / (lb[i] - lb[i - 1])
            return a[i - 1] + w * (a[i] - a[i - 1])
    return a[-1]


def analyze_probe(m: Dict) -> Dict:
    curves = m["curves"]
    per_matrix = {}
    for name, c in curves.items():
        rng = max(c["a_hat"]) - min(c["a_hat"])
        # (S'): on-grid flatness approaching the momentum-corrected training
        # point; None (not clamped) if either point is off-grid.
        s_num = interp_a(c, SAT_NUM, allow_clamp=False)
        s_den = interp_a(c, SAT_DEN, allow_clamp=False)
        raw_8 = interp_a(c, 8, allow_clamp=False)
        raw_64 = interp_a(c, 64, allow_clamp=False)
        per_matrix[name] = {
            "dynamic_range": rng,
            "a_at_record_b": interp_a(c, RECORD_B),
            "a_at_128": s_num,
            "a_at_256": s_den,
            "sat_ratio": (s_num / s_den) if (s_num and s_den) else None,
            "raw_ratio_8_over_64": (raw_8 / raw_64) if (raw_8 and raw_64) else None,
            "b_eff_training_point": RECORD_B * B_EFF_FACTOR,
            "b_max": max(c["b_chunks"]),
        }
    ranges = [v["dynamic_range"] for v in per_matrix.values()]
    sats = [v["sat_ratio"] for v in per_matrix.values() if v["sat_ratio"] is not None]
    a_rec = [v["a_at_record_b"] for v in per_matrix.values()]
    return {
        "source": m.get("bbp_source"),
        "data_file_index": m.get("data_file_index"),
        "n_matrices": len(per_matrix),
        "median_a_at_record_b": st.median(a_rec),
        "median_sat_ratio": st.median(sats) if sats else None,
        "frac_matrices_range_ge_02": sum(r >= 0.2 for r in ranges) / len(ranges),
        "median_dynamic_range": st.median(ranges),
        "per_matrix": per_matrix,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "bbp-phase-a.json"))
    args = ap.parse_args(argv)

    probes = load_probes()
    out: Dict = {"beta": BETA, "b_eff_factor": B_EFF_FACTOR, "probes": {}}
    for tag, m in sorted(probes.items()):
        res = analyze_probe(m)
        out["probes"][tag] = res
        print(f"{tag:<28} shard {res['data_file_index']}  "
              f"a(record B) {res['median_a_at_record_b']:.4f}  "
              f"sat e(8)/e(64) {res['median_sat_ratio']:.4f}  "
              f"range>=0.2 {res['frac_matrices_range_ge_02']:.2f}", flush=True)

    held = {k: v for k, v in out["probes"].items() if "midheld" in k}
    seen = {k: v for k, v in out["probes"].items() if k.startswith("bbp_probe_mid_")}
    def agg(group, key):
        vals = [v[key] for v in group.values() if v[key] is not None]
        return st.median(vals) if vals else None

    out["registered"] = {
        "V_vacuity_pass": all(v["frac_matrices_range_ge_02"] >= 0.5
                              for v in out["probes"].values()),
        "S_criterion": "(S') median a_hat(128)/a_hat(256) >= 0.9 (amendment A2)",
        "S_arm": "held_out_shard8 (amendment A1)",
        "S_median_sat_ratio_heldout": agg(held, "median_sat_ratio"),
        "S_pass": (agg(held, "median_sat_ratio") or 0) >= 0.9 if held else None,
        "seen_arm_median_sat_ratio": agg(seen, "median_sat_ratio"),
        "seen_minus_heldout_sat": (
            (agg(seen, "median_sat_ratio") - agg(held, "median_sat_ratio"))
            if held and seen else None),
        "note": "gate call is human (prereg §3, §3b)",
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\n" + json.dumps(out["registered"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
