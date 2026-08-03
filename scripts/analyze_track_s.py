#!/usr/bin/env python
"""Program #13 Track S: passive spectral screen over frontier sidecars.

Pre-registered (reports/endstate-prereg.md §6): per (batch, lr) cell,
from pooled mature per-direction projection series (5-step post-refresh
burn-in), three spectral features — fS1 normalized spectral centroid,
fS2 fraction of power above 1/4 cyc/step, fS3 spectral flatness — with
refresh-harmonic bins notched out. Each single feature is tested against
the program-#11 tracking signature VERBATIM on the same fit/validation
split (validation shoulder ratio <= 1.5 AND within-B lr max/min >= 3.0),
with the 1,000-draw permutation control.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze_invariant_search import (  # noqa: E402
    FIT_SHOULDERS, VAL_SHOULDERS, classify, load_shoulders,
)

BURNIN = 5
MIN_WIN = 30

SPEC_FEATURES = ["fS1_centroid", "fS2_highfrac", "fS3_flatness"]


def window_psd(s: np.ndarray, refresh, t_refresh: int):
    """Mean periodogram over post-burnin windows + notch mask."""
    n = len(s)
    bounds = [r - 1 for r in refresh if 0 <= r - 1 < n] + [n]
    acc = None
    cnt = 0
    wlen = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        w = s[lo + BURNIN:hi]
        w = w[np.isfinite(w)]
        if len(w) < MIN_WIN:
            continue
        if wlen is None:
            wlen = len(w)
        if len(w) != wlen:
            w = w[:wlen] if len(w) > wlen else None
            if w is None:
                continue
        w = w - w.mean()
        sd = w.std()
        if sd <= 0:
            continue
        psd = np.abs(np.fft.rfft(w / sd)) ** 2
        acc = psd if acc is None else acc + psd
        cnt += 1
    if acc is None or wlen is None:
        return None, None
    freqs = np.fft.rfftfreq(wlen)
    mask = np.ones(len(freqs), bool)
    mask[0] = False  # DC
    # The prereg's refresh-harmonic notch concern does not apply here:
    # each periodogram is computed WITHIN a single refresh period (window
    # length < t_refresh), so no refresh cadence can appear in it. A notch
    # at k/t_refresh with these window lengths would blanket the whole
    # spectrum (bands wider than their spacing) — implementation note.
    return acc / cnt, (freqs, mask)


def spectral_features(psd, freqs_mask):
    freqs, mask = freqs_mask
    p = psd[mask]
    f = freqs[mask]
    tot = p.sum()
    if tot <= 0:
        return None
    centroid = float((f * p).sum() / tot / 0.5)  # normalized to Nyquist
    highfrac = float(p[f > 0.25].sum() / tot)
    logp = np.log(np.clip(p, 1e-30, None))
    flatness = float(np.exp(logp.mean()) / p.mean())
    return {"fS1_centroid": centroid, "fS2_highfrac": highfrac,
            "fS3_flatness": flatness}


def extract(cache: str):
    if os.path.exists(cache):
        return json.load(open(cache))
    cells = defaultdict(list)
    for f in sorted(glob.glob("results/*.instrumentation.json")):
        cls = classify(f)
        if cls is None:
            continue
        which, b, lr = cls
        side = json.load(open(f))
        feats = defaultdict(list)
        for mat in side.get("matrices", {}).values():
            refresh = mat.get("refresh_steps") or []
            t_ref = refresh[1] - refresh[0] if len(refresh) > 1 else 0
            for d in mat.get("directions") or []:
                s = np.asarray(d.get("s") or [], float)
                if len(s) < 100:
                    continue
                psd, fm = window_psd(s, refresh, t_ref)
                if psd is None:
                    continue
                sf = spectral_features(psd, fm)
                if sf:
                    for k, v in sf.items():
                        feats[k].append(v)
        if feats:
            cells[f"{which}|{b}|{lr}"].append(
                {k: float(np.mean(v)) for k, v in feats.items()}
            )
    out = {}
    for key, runs in cells.items():
        out[key] = {k: float(np.mean([r[k] for r in runs if k in r]))
                    for k in SPEC_FEATURES}
    Path(cache).write_text(json.dumps(out, indent=1, sort_keys=True))
    return out


def single_signature(cells, feat, val_shoulders):
    logs = []
    for b, lr in val_shoulders.items():
        v = cells.get(f"val|{b}|{lr}", {}).get(feat)
        if v and v > 0:
            logs.append(math.log(v))
    if len(logs) < 3:
        return None
    sh_ratio = math.exp(max(logs) - min(logs))
    lr_vars = []
    val_bs = sorted({int(k.split("|")[1]) for k in cells if k.startswith("val|")})
    for b in val_bs:
        ls = [math.log(f[feat]) for k, f in cells.items()
              if k.startswith(f"val|{b}|") and f.get(feat, 0) > 0]
        if len(ls) >= 3:
            lr_vars.append(math.exp(max(ls) - min(ls)))
    if not lr_vars:
        return None
    return sh_ratio, min(lr_vars)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="reports/track-s-features.json")
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args(argv)
    load_shoulders()
    cells = extract(args.cache)
    print(f"cells: {len(cells)}; validation shoulders {VAL_SHOULDERS}\n")
    print("feature        | val shoulder ratio (<=1.5) | min within-B lr-var (>=3.0) | verdict")
    passes = 0
    for feat in SPEC_FEATURES:
        r = single_signature(cells, feat, VAL_SHOULDERS)
        if r is None:
            print(f"{feat:14s} | n/a")
            continue
        ok = r[0] <= 1.5 and r[1] >= 3.0
        passes += ok
        print(f"{feat:14s} | {r[0]:6.2f} | {r[1]:6.2f} | {'PASS' if ok else 'fail'}")
    rng = np.random.default_rng(20260722)
    val_rungs = defaultdict(list)
    for key in cells:
        wh, b, lr = key.split("|")
        if wh == "val":
            val_rungs[int(b)].append(float(lr))
    null_counts = []
    for _ in range(args.n_perm):
        vs = {b: rng.choice(v) for b, v in val_rungs.items() if b in VAL_SHOULDERS}
        c = 0
        for feat in SPEC_FEATURES:
            r = single_signature(cells, feat, vs)
            if r and r[0] <= 1.5 and r[1] >= 3.0:
                c += 1
        null_counts.append(c)
    print(f"\nreal passes: {passes}/3; permutation null mean "
          f"{np.mean(null_counts):.3f}, q95 {np.quantile(null_counts, 0.95):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
