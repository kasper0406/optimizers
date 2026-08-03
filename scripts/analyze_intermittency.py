#!/usr/bin/env python
"""Intermittency scan over per-direction projection series (offline, no GPU).

Hypothesis (2026-07-22, user): sporadic-but-real signal is heavy-tailed in
the per-direction projections s_i(t) even where mean-based statistics are
null (the t-ceiling result and program #4 are both mean-based; a stationary
Gaussian AR(1) — the measured negative-rho structure — is marginally
Gaussian, so serial correlation does NOT confound a kurtosis/spike null).

Method, per direction:
- split s(t) into refresh windows (direction is re-anchored at each
  refresh; quasi-stationary within);
- robust-standardize each window: z = (s - median) / (1.4826 * MAD)
  (MAD, not sd: a spike inflates the sample sd and masks itself);
- pool z across windows; compute excess kurtosis g2 = E[z^4]/E[z^2]^2 - 3
  and spike rates p3 = frac(|z| > 3), p4 = frac(|z| > 4);
- split-half check: spike counts in odd vs even windows (real
  per-direction structure correlates; sampling noise does not).

Null: simulated AR(1)-Gaussian directions with the same window geometry
and the same standardization pipeline (MAD-on-50-samples biases g2, so the
null is empirical, not the textbook sqrt(24/n)).

Usage:
    uv run python scripts/analyze_intermittency.py results/*.instrumentation.json \
        [--json out.json] [--null-dirs 4000]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

MIN_WINDOW = 30
MAD_SCALE = 1.4826


def robust_pooled_z(s: np.ndarray, refresh_steps: Sequence[int],
                    burnin: int = 0) -> Optional[np.ndarray]:
    """Per-refresh-window MAD standardization, pooled. None if unusable.

    ``burnin`` drops the first N samples of every window: the top-anchored
    directions spike deterministically right after subspace re-anchoring
    (93% of naive spikes in the first 5 steps — the anchoring artifact),
    so the intermittency question is only meaningful past that.
    """
    n = len(s)
    bounds = [r - 1 for r in refresh_steps if 0 <= r - 1 < n] + [n]
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        w = s[lo + burnin:hi]
        w = w[np.isfinite(w)]
        if len(w) < MIN_WINDOW:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad <= 0:
            continue
        out.append((w - med) / (MAD_SCALE * mad))
    if not out:
        return None
    return np.concatenate(out)


def direction_stats(z: np.ndarray) -> Dict[str, float]:
    m2 = float(np.mean(z**2))
    m4 = float(np.mean(z**4))
    return {
        "n": len(z),
        "g2": m4 / (m2 * m2) - 3.0,
        "p3": float(np.mean(np.abs(z) > 3.0)),
        "p4": float(np.mean(np.abs(z) > 4.0)),
    }


def split_half_counts(s: np.ndarray, refresh_steps: Sequence[int],
                      burnin: int = 0) -> Tuple[int, int]:
    """Spike counts (|z|>3) in odd vs even refresh windows."""
    n = len(s)
    bounds = [r - 1 for r in refresh_steps if 0 <= r - 1 < n] + [n]
    counts = [0, 0]
    for j, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        w = s[lo + burnin:hi]
        w = w[np.isfinite(w)]
        if len(w) < MIN_WINDOW:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad <= 0:
            continue
        z = (w - med) / (MAD_SCALE * mad)
        counts[j % 2] += int(np.sum(np.abs(z) > 3.0))
    return counts[0], counts[1]


def modal_regime(direction: Dict[str, Any]) -> str:
    pb = direction.get("per_beta") or {}
    key = "0.99" if "0.99" in pb else (next(iter(pb)) if pb else None)
    if key is None:
        return "?"
    regimes = pb[key].get("regime") or []
    if not regimes:
        return "?"
    return Counter(regimes).most_common(1)[0][0]


def simulate_null(n_dirs: int, n_windows: int = 16, window: int = 50,
                  rho_range: Tuple[float, float] = (-0.6, 0.1),
                  seed: int = 20260722, burnin: int = 0) -> Dict[str, np.ndarray]:
    """AR(1)-Gaussian directions through the identical pipeline."""
    rng = np.random.default_rng(seed)
    g2s, p3s, p4s = [], [], []
    refresh = [1 + j * window for j in range(n_windows)]
    for _ in range(n_dirs):
        rho = rng.uniform(*rho_range)
        eps = rng.standard_normal(n_windows * window)
        s = np.empty_like(eps)
        s[0] = eps[0]
        c = math.sqrt(1 - rho * rho)
        for t in range(1, len(eps)):
            s[t] = rho * s[t - 1] + c * eps[t]
        z = robust_pooled_z(s, refresh, burnin=burnin)
        st = direction_stats(z)
        g2s.append(st["g2"]); p3s.append(st["p3"]); p4s.append(st["p4"])
    return {"g2": np.array(g2s), "p3": np.array(p3s), "p4": np.array(p4s)}


def load_sidecar_group_info(path: str) -> Dict[str, Any]:
    """lr / batch metadata from the companion results JSON."""
    base = path.replace(".instrumentation", "")
    try:
        cfg = json.load(open(base))["config"]["contents"]
        lr = (cfg.get("probe_overrides") or {}).get("lr") or cfg["optimizer"]["lr"]
        batch = cfg.get("train", {}).get("batch_size")
        return {"lr": float(lr), "batch": int(batch)}
    except Exception:
        return {"lr": None, "batch": None}


def analyze(paths: Sequence[str], null_dirs: int = 4000,
            burnin: int = 0) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        info = load_sidecar_group_info(path)
        try:
            side = json.load(open(path))
        except Exception:
            continue
        for mname, mat in (side.get("matrices") or {}).items():
            refresh = mat.get("refresh_steps") or []
            for direction in mat.get("directions") or []:
                s = np.asarray(direction.get("s") or [], dtype=np.float64)
                if len(s) < 100:
                    continue
                z = robust_pooled_z(s, refresh, burnin=burnin)
                if z is None or len(z) < 200:
                    continue
                st = direction_stats(z)
                odd, even = split_half_counts(s, refresh, burnin=burnin)
                rows.append({
                    "file": os.path.basename(path),
                    "lr": info["lr"], "batch": info["batch"],
                    "matrix": mname, "kind": direction.get("kind", "?"),
                    "regime": modal_regime(direction),
                    "spikes_odd": odd, "spikes_even": even,
                    **st,
                })
    if not rows:
        raise SystemExit("no usable directions found")

    null = simulate_null(null_dirs, burnin=burnin)
    thr = {k: float(np.quantile(null[k], 0.99)) for k in ("g2", "p3", "p4")}
    null_summary = {
        k: {"mean": float(null[k].mean()), "sd": float(null[k].std()),
            "q99": thr[k]} for k in ("g2", "p3", "p4")
    }

    def agg(sub: List[Dict[str, Any]]) -> Dict[str, float]:
        g2 = np.array([r["g2"] for r in sub])
        p4 = np.array([r["p4"] for r in sub])
        return {
            "n_dirs": len(sub),
            "g2_median": float(np.median(g2)),
            "g2_q90": float(np.quantile(g2, 0.90)),
            "p4_mean": float(np.mean(p4)),
            "frac_g2_gt_null99": float(np.mean(g2 > thr["g2"])),
            "frac_p4_gt_null99": float(np.mean(p4 > thr["p4"])),
        }

    out: Dict[str, Any] = {"null": null_summary, "overall": agg(rows)}

    by_batch: Dict[int, List] = defaultdict(list)
    by_kind: Dict[str, List] = defaultdict(list)
    by_regime: Dict[str, List] = defaultdict(list)
    by_lrband: Dict[str, List] = defaultdict(list)
    for r in rows:
        if r["batch"] is not None:
            by_batch[r["batch"]].append(r)
        by_kind[r["kind"]].append(r)
        by_regime[r["regime"]].append(r)
        if r["lr"] is not None and r["batch"] == 1000:
            band = "lr<=0.32" if r["lr"] <= 0.32 else ("lr<=0.55" if r["lr"] <= 0.55 else "lr>0.55")
            by_lrband[band].append(r)
    out["by_batch"] = {str(k): agg(v) for k, v in sorted(by_batch.items())}
    out["by_kind"] = {k: agg(v) for k, v in sorted(by_kind.items())}
    out["by_regime"] = {k: agg(v) for k, v in sorted(by_regime.items())}
    out["by_lr_band_b1000"] = {k: agg(v) for k, v in sorted(by_lrband.items())}

    odd = np.array([r["spikes_odd"] for r in rows], dtype=float)
    even = np.array([r["spikes_even"] for r in rows], dtype=float)
    if odd.std() > 0 and even.std() > 0:
        out["split_half_pearson"] = float(np.corrcoef(odd, even)[0, 1])

    # ---- report ----
    L = ["# Intermittency scan: per-direction kurtosis / spike-rate vs AR(1)-Gaussian null", ""]
    L.append(f"- window burn-in: {burnin} steps dropped after each subspace refresh")
    L.append(f"- directions analyzed: {len(rows)} (from {len(set(r['file'] for r in rows))} runs)")
    L.append(f"- null (n={null_dirs} simulated AR(1) dirs, same pipeline): "
             f"g2 mean {null_summary['g2']['mean']:+.3f} sd {null_summary['g2']['sd']:.3f} "
             f"q99 {null_summary['g2']['q99']:+.3f}; p4 q99 {null_summary['p4']['q99']:.5f}")
    o = out["overall"]
    L.append(f"- observed: g2 median {o['g2_median']:+.3f}, q90 {o['g2_q90']:+.3f}; "
             f"p4 mean {o['p4_mean']:.5f}")
    L.append(f"- **fraction of directions above null q99: g2 {o['frac_g2_gt_null99']:.1%}, "
             f"p4 {o['frac_p4_gt_null99']:.1%}** (null expectation 1%)")
    if "split_half_pearson" in out:
        L.append(f"- split-half (odd vs even windows) spike-count Pearson: "
                 f"**{out['split_half_pearson']:+.3f}** (sampling noise -> ~0)")
    for title, key in (("by batch size", "by_batch"), ("by kind", "by_kind"),
                       ("by modal regime", "by_regime"), ("by lr band (B=1000)", "by_lr_band_b1000")):
        L += ["", f"## {title}", "", "| group | n dirs | g2 median | g2 q90 | p4 mean | frac g2>null99 | frac p4>null99 |",
              "|---|---|---|---|---|---|---|"]
        for k, v in out[key].items():
            L.append(f"| {k} | {v['n_dirs']} | {v['g2_median']:+.3f} | {v['g2_q90']:+.3f} | "
                     f"{v['p4_mean']:.5f} | {v['frac_g2_gt_null99']:.1%} | {v['frac_p4_gt_null99']:.1%} |")
    out["report"] = "\n".join(L)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sidecars", nargs="+")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--null-dirs", type=int, default=4000)
    ap.add_argument("--burnin", type=int, default=0)
    args = ap.parse_args(argv)
    out = analyze(args.sidecars, null_dirs=args.null_dirs, burnin=args.burnin)
    print(out["report"])
    if args.json_out:
        report = out.pop("report")
        Path(args.json_out).write_text(json.dumps(out, indent=2, sort_keys=True))
        out["report"] = report
    return 0


if __name__ == "__main__":
    sys.exit(main())
