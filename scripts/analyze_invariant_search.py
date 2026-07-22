#!/usr/bin/env python
"""Program #11: composite frontier-invariant search (offline).

Pre-registration: reports/invariant-search-prereg.md (commit 0ecd919,
before any fit). Fit set = program-#6 coarse grid shoulders; validation
= program-#6b dense ladders + step-matched B=8000, held out. Criterion
and permutation control per the prereg.

Usage:
    uv run python scripts/analyze_invariant_search.py [--features-cache F.json]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BURNIN_SPIKE = 5
MAD = 1.4826

FIT_SHOULDERS = {}       # B -> shoulder lr (program #6, stability-frontier.json)
VAL_SHOULDERS = {}       # B -> nearest dense rung to interpolated crossing


def load_shoulders() -> None:
    sf = json.load(open("reports/stability-frontier.json"))
    for b, cell in sf["frontier"].items():
        if cell.get("shoulder") is not None:  # fixed-budget B=8000 has none
            FIT_SHOULDERS[int(b)] = float(cell["shoulder"])
    sh = json.load(open("reports/frontier-sharpening.json"))
    for b, cell in sh["part1"].items():
        rungs = [float(r) for r in cell["mean_acc"]]
        cross = float(cell["lr_cross"])
        VAL_SHOULDERS[int(b)] = min(rungs, key=lambda r: abs(math.log(r / cross)))
    VAL_SHOULDERS[8000] = float(sh["part2_stepmatched_b8000"]["peak_ref_shoulder"])


# ------------------------------------------------------------------ features


def spike_rate(s: np.ndarray, refresh: List[int]) -> Optional[float]:
    n = len(s)
    bounds = [r - 1 for r in refresh if 0 <= r - 1 < n] + [n]
    hits = tot = 0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        w = s[lo + BURNIN_SPIKE:hi]
        w = w[np.isfinite(w)]
        if len(w) < 30:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad <= 0:
            continue
        z = (w - med) / (MAD * mad)
        hits += int(np.sum(np.abs(z) > 3.0))
        tot += len(w)
    return hits / tot if tot else None


def run_features(side: Dict[str, Any], lr: float) -> Dict[str, Optional[float]]:
    occ_n = occ_d = 0
    rhos: List[float] = []
    sigmas: List[float] = []
    lambdas: List[float] = []
    spikes_h = spikes_t = 0
    for mat in side.get("matrices", {}).values():
        refresh = mat.get("refresh_steps") or []
        for d in mat.get("directions") or []:
            pb = (d.get("per_beta") or {}).get("0.9") or {}
            rr = pb.get("rho") or []
            nn = pb.get("n_since_reset") or []
            for r, n in zip(rr, nn):
                if n is not None and n >= 10 and r is not None:
                    occ_d += 1
                    rhos.append(r)
                    if r < -0.2:
                        occ_n += 1
            lam = d.get("lambda_hvp")
            if isinstance(lam, list):
                lambdas += [v for v in lam if v is not None]
            elif isinstance(lam, dict):
                lambdas += [v for v in (lam.get("value") or []) if v is not None]
            elif isinstance(lam, (int, float)):
                lambdas.append(lam)
            sg = d.get("sigma")
            if isinstance(sg, dict):
                vals = sg.get("value") or []
                half = vals[len(vals) // 2:]
                sigmas += [v for v in half if v is not None]
            s = np.asarray(d.get("s") or [], float)
            if len(s) >= 100:
                sr = spike_rate(s, refresh)
                if sr is not None:
                    n_eff = len(s)
                    spikes_h += sr * n_eff
                    spikes_t += n_eff
    smf: List[float] = []
    sms: List[float] = []
    gn: List[float] = []
    for mat in (side.get("smoothness") or {}).get("matrices", {}).values():
        f = mat.get("lr_times_d_smooth_frobenius") or []
        sp = mat.get("lr_times_d_smooth_spectral") or []
        smf += [v for v in f[len(f) // 2:] if v is not None]
        sms += [v for v in sp[len(sp) // 2:] if v is not None]
    for mat in side.get("matrices", {}).values():
        g = mat.get("grad_fro_norm") or []
        gn += [v for v in g[len(g) // 2:] if v is not None]

    def mean(v):
        return float(np.mean(v)) if v else None

    return {
        "occupancy": occ_n / occ_d if occ_d else None,
        "lr_dsmooth_fro": mean(smf),
        "lr_dsmooth_spec": mean(sms),
        "hvp_q90": float(np.quantile([lr * l for l in lambdas], 0.9)) if lambdas else None,
        "grad_norm": mean(gn),
        "sigma": mean(sigmas),
        "spike_rate": spikes_h / spikes_t if spikes_t else None,
        "rho_mean": mean(rhos),
    }


def classify(path: str) -> Optional[Tuple[str, int, float]]:
    """(set, B, lr) from the companion results JSON config path."""
    base = path.replace(".instrumentation", "")
    try:
        cfg = json.load(open(base))["config"]["contents"]
    except Exception:
        return None
    tag = os.path.basename(json.load(open(base))["config"].get("path", ""))
    lr = float((cfg.get("probe_overrides") or {}).get("lr")
               or cfg["optimizer"]["lr"])
    b = int(cfg.get("train", {}).get("batch_size"))
    if tag.startswith("frontier_lrxbatch__") :
        return ("fit", b, lr)
    if tag.startswith("frontier_dense_"):
        return ("val", b, lr)
    if tag.startswith("frontier_b8000_stepmatched"):
        return ("val", 8000, lr)
    return None  # fixed-budget b8000 (trend-break) and others: excluded


def extract_all(cache: Optional[str]) -> Dict[str, Any]:
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    cells: Dict[str, List[Dict]] = defaultdict(list)
    for f in sorted(glob.glob("results/*.instrumentation.json")):
        cls = classify(f)
        if cls is None:
            continue
        which, b, lr = cls
        side = json.load(open(f))
        cells[f"{which}|{b}|{lr}"].append(run_features(side, lr))
    out: Dict[str, Any] = {}
    for key, runs in cells.items():
        agg = {}
        for feat in runs[0]:
            vals = [r[feat] for r in runs if r[feat] is not None]
            agg[feat] = float(np.mean(vals)) if vals else None
        out[key] = agg
    if cache:
        Path(cache).write_text(json.dumps(out, indent=1, sort_keys=True))
    return out


# ---------------------------------------------------------------- the search

FEATURES = ["occupancy", "lr_dsmooth_fro", "lr_dsmooth_spec", "hvp_q90",
            "grad_norm", "sigma", "spike_rate", "rho_mean"]


def cell_lookup(cells, which, b, lr, feat):
    key = f"{which}|{b}|{lr}"
    v = cells.get(key, {}).get(feat)
    if v is None or v <= 0:
        return None
    return v


def composite_ratio(vals: List[float]) -> float:
    return max(vals) / min(vals)


def eval_pair(cells, fi, fj, fit_shoulders, val_shoulders):
    """Fit exponent a on fit shoulders; return (a, val_ratio, min lr-var)."""
    fit_pts = []
    for b, lr in fit_shoulders.items():
        x = cell_lookup(cells, "fit", b, lr, fi)
        y = cell_lookup(cells, "fit", b, lr, fj)
        if x and y:
            fit_pts.append((math.log(x), math.log(y)))
    if len(fit_pts) < 3:
        return None
    # min over a of range(log x + a log y): 1-D; solve by minimizing variance
    X = np.array([p[0] for p in fit_pts])
    Y = np.array([p[1] for p in fit_pts])
    if np.var(Y) < 1e-12:
        return None
    a = -np.cov(X, Y, bias=True)[0, 1] / np.var(Y)
    # validation
    val_logs = []
    for b, lr in val_shoulders.items():
        x = cell_lookup(cells, "val", b, lr, fi)
        y = cell_lookup(cells, "val", b, lr, fj)
        if x and y:
            val_logs.append(math.log(x) + a * math.log(y))
    if len(val_logs) < 3:
        return None
    val_ratio = math.exp(max(val_logs) - min(val_logs))
    # within-B lr variation on validation ladders
    lr_vars = []
    val_bs = sorted({int(k.split("|")[1]) for k in cells if k.startswith("val|")})
    for b in val_bs:
        logs = []
        for key, feats in cells.items():
            wh, bb, lr = key.split("|")
            if wh != "val" or int(bb) != b:
                continue
            x = feats.get(fi); y = feats.get(fj)
            if x and y and x > 0 and y > 0:
                logs.append(math.log(x) + a * math.log(y))
        if len(logs) >= 3:
            lr_vars.append(math.exp(max(logs) - min(logs)))
    if not lr_vars:
        return None
    return a, val_ratio, min(lr_vars)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-cache", default="reports/invariant-search-features.json")
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args(argv)
    load_shoulders()
    cells = extract_all(args.features_cache)
    print(f"cells: {len(cells)} (fit shoulders {FIT_SHOULDERS}, "
          f"val shoulders {VAL_SHOULDERS})\n")

    results = []
    for fi, fj in itertools.permutations(FEATURES, 2):
        r = eval_pair(cells, fi, fj, FIT_SHOULDERS, VAL_SHOULDERS)
        if r:
            results.append((fi, fj, *r))
    # singles baseline
    singles = []
    for f in FEATURES:
        logs = []
        for b, lr in VAL_SHOULDERS.items():
            v = cell_lookup(cells, "val", b, lr, f)
            if v:
                logs.append(math.log(v))
        if len(logs) >= 3:
            singles.append((f, math.exp(max(logs) - min(logs))))

    print("singles (validation shoulder ratio; pre-registered bar <= 1.5):")
    for f, r in sorted(singles, key=lambda t: t[1]):
        print(f"  {f:16s} {r:6.2f}")
    print("\npairs, sorted by validation shoulder ratio "
          "(PASS = ratio <= 1.5 AND min within-B lr-variation >= 3.0):")
    results.sort(key=lambda t: t[3])
    passes = 0
    for fi, fj, a, vr, lv in results[:12]:
        ok = vr <= 1.5 and lv >= 3.0
        passes += ok
        print(f"  {fi:16s} + {a:+6.2f}*{fj:16s} | shoulder ratio {vr:6.2f} | "
              f"min lr-var {lv:6.2f} | {'PASS' if ok else 'fail'}")
    total_pass = sum(1 for *_, vr, lv in results if vr <= 1.5 and lv >= 3.0)
    print(f"\nreal passes: {total_pass} / {len(results)} evaluated pairs")

    # permutation control
    rng = np.random.default_rng(20260722)
    fit_rungs = defaultdict(list)
    val_rungs = defaultdict(list)
    for key in cells:
        wh, b, lr = key.split("|")
        (fit_rungs if wh == "fit" else val_rungs)[int(b)].append(float(lr))
    null_counts = []
    for _ in range(args.n_perm):
        fs = {b: rng.choice(v) for b, v in fit_rungs.items() if b in FIT_SHOULDERS}
        vs = {b: rng.choice(v) for b, v in val_rungs.items() if b in VAL_SHOULDERS}
        c = 0
        for fi, fj in itertools.permutations(FEATURES, 2):
            r = eval_pair(cells, fi, fj, fs, vs)
            if r and r[1] <= 1.5 and r[2] >= 3.0:
                c += 1
        null_counts.append(c)
    print(f"permutation null (n={args.n_perm}): mean passing pairs "
          f"{np.mean(null_counts):.2f}, q95 {np.quantile(null_counts, 0.95):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
