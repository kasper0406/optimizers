#!/usr/bin/env python
"""Program #13 Stage-1 endpoint analysis (reports/endstate-prereg.md §4-§6).

Computes O1-O7 per run from checkpoints + logits bundles, aggregates per
cell, evaluates P1a/P1b/P2/P3/P5, writes reports/endstate-features.json
(durable; checkpoints themselves are scratch and deleted afterwards).

Usage:
    uv run python scripts/analyze_endstate.py [--iters 20]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.instrument.endstate import endpoint_lambda1  # noqa: E402
from src.optim.airbench_zoo import load_vendor_airbench  # noqa: E402

PEAK_LR = 0.24
PROBE_SEED = 20260722


def load_runs():
    """All program-#13 runs with endpoint artifacts, keyed by arm."""
    runs = []
    for f in sorted(glob.glob("results/airbench_smoke_seed14*.json")):
        d = json.load(open(f))
        m = d["metrics"]
        if "endpoint_path" not in m or "logits_path" not in m:
            continue
        path = d["config"].get("path", "")
        cfg = d["config"]["contents"]
        lr = float(cfg["optimizer"]["lr"])
        epochs = int(cfg["train"]["epochs"])
        b = int(cfg["train"]["batch_size"])
        if "endstate_b1000" in path:
            arm = "ladder"
        elif "endstate_placebo" in path:
            arm = "placebo"
        elif "endstate_smoke_hooks" in path:
            arm = "replicate"
        else:
            continue
        if not (Path(m["endpoint_path"]).exists() and Path(m["logits_path"]).exists()):
            continue
        runs.append({
            "file": f, "arm": arm, "lr": lr, "epochs": epochs, "B": b,
            "seed": d["seed"], "acc": float(m["tta_val_acc"]),
            "ckpt": m["endpoint_path"], "logits": m["logits_path"],
        })
    return runs


def margins_q10(logits: torch.Tensor, labels: torch.Tensor) -> float:
    lt = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), -1e9)
    return float(torch.quantile(lt - masked.max(1).values, 0.10).item())


def mean_ce(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(torch.nn.functional.cross_entropy(
        logits.float(), labels, label_smoothing=0.2).item())


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0); Y = Y - Y.mean(0)
    xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    return float(xy / (xx * yy))


def ridge_probe(train_X, train_y, test_X, test_y, lam=1.0) -> float:
    X = np.concatenate([train_X, np.ones((len(train_X), 1))], 1)
    Y = np.eye(10)[train_y]
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    Xt = np.concatenate([test_X, np.ones((len(test_X), 1))], 1)
    return float(((Xt @ W).argmax(1) == test_y).mean())


class FeatureExtractor:
    def __init__(self):
        self.ab = load_vendor_airbench()
        self.test_loader = self.ab.CifarLoader("data/cifar10", train=False, batch_size=2000)
        train_loader = self.ab.CifarLoader(
            "data/cifar10", train=True, batch_size=1000,
            aug=dict(flip=True, translate=2))
        self.slab_images = train_loader.images[:2000]
        self.slab_labels = train_loader.labels[:2000]
        self.normalize = train_loader.normalize
        self.model = self.ab.CifarNet().cuda().to(memory_format=torch.channels_last)

    @torch.no_grad()
    def features(self, sd_path):
        self.model.load_state_dict(torch.load(sd_path, map_location="cuda"))
        self.model.eval()
        feats = {}
        h = self.model.head.register_forward_hook(
            lambda mod, inp, out: feats.setdefault("x", []).append(inp[0].float().cpu()))
        for imgs in self.test_loader.normalize(self.test_loader.images).split(2000):
            self.model(imgs)
        test_feats = torch.cat(feats.pop("x"))
        for imgs in self.normalize(self.slab_images).split(2000):
            self.model(imgs)
        slab_feats = torch.cat(feats.pop("x"))
        h.remove()
        return test_feats.numpy(), slab_feats.numpy()

    def lam1(self, sd_path, iters):
        self.model.load_state_dict(torch.load(sd_path, map_location="cuda"))
        self.model.eval()
        return endpoint_lambda1(
            self.model, self.normalize(self.slab_images),
            self.slab_labels, iters=iters, seed=PROBE_SEED)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="reports/endstate-features.json")
    args = ap.parse_args(argv)
    runs = load_runs()
    print(f"{len(runs)} runs with artifacts "
          f"({sum(r['arm']=='ladder' for r in runs)} ladder, "
          f"{sum(r['arm']=='placebo' for r in runs)} placebo, "
          f"{sum(r['arm']=='replicate' for r in runs)} replicate)")
    fx = FeatureExtractor()
    test_labels = fx.test_loader.labels.cpu()
    slab_labels = fx.slab_labels.cpu()

    # per-run observables + cached correctness vectors and features
    correct = {}
    feat_cache = {}
    for r in runs:
        bundle = torch.load(r["logits"], map_location="cpu")
        tta = bundle["test_tta"]
        r["O1_margin_q10"] = margins_q10(tta, test_labels)
        r["O6_gen_gap"] = mean_ce(bundle["train_tta"], slab_labels) - mean_ce(tta, test_labels)
        correct[r["file"]] = (tta.argmax(1) == test_labels).numpy()
        tf, sf = fx.features(r["ckpt"])
        feat_cache[r["file"]] = tf[:2000]  # CKA subset
        r["O3_probe_acc"] = ridge_probe(sf, slab_labels.numpy(), tf, test_labels.numpy())
        r["O4_lambda1"] = fx.lam1(r["ckpt"], args.iters)

    # O2: LOSO difficulty from the 10 ladder peak-rung runs
    peak = [r for r in runs if r["arm"] == "ladder" and r["lr"] == PEAK_LR]
    peak_correct = np.stack([correct[r["file"]] for r in peak])
    for r in runs:
        if r in peak:
            others = np.stack([correct[p["file"]] for p in peak if p is not r])
        else:
            others = peak_correct
        diff = others.mean(0)  # fraction correct = easiness
        hardest = np.argsort(diff)[: len(diff) // 5]
        r["O2_hard_quintile_acc"] = float(correct[r["file"]][hardest].mean())
        r["_hardest_idx"] = hardest

    # O5/O7: same-seed comparisons to the peak rung
    peak_by_seed = {r["seed"]: r for r in peak}
    for r in runs:
        pk = peak_by_seed.get(r["seed"])
        if pk is None or r is pk:
            r["O5_cka_to_peak"] = None
            r["O7_reldist"] = None
            continue
        r["O5_cka_to_peak"] = linear_cka(feat_cache[r["file"]], feat_cache[pk["file"]])
        a = torch.load(r["ckpt"], map_location="cpu")
        b = torch.load(pk["ckpt"], map_location="cpu")
        num = den = 0.0
        for k in a:
            if a[k].is_floating_point():
                num += (a[k].float() - b[k].float()).norm() ** 2
                den += b[k].float().norm() ** 2
        r["O7_reldist"] = float((num / den).sqrt())
    # replicate-pair CKA floor
    reps = [r for r in runs if r["arm"] == "replicate"]
    rep_ckas = [linear_cka(feat_cache[reps[i]["file"]], feat_cache[reps[j]["file"]])
                for i in range(len(reps)) for j in range(i + 1, len(reps))]

    for r in runs:
        r.pop("_hardest_idx", None)
    out = {"runs": [{k: v for k, v in r.items()} for r in runs],
           "replicate_pair_cka": rep_ckas}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"features written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
