#!/usr/bin/env python
"""Program #15 Phase A: dose-path geometry (reports/dosepath-prereg.md).

M1 mode connectivity (linear path peak->shoulder, BN-stat repair),
M2 straightness cosines, M3 sharp-or-sloppy Rayleigh/lambda1,
M4 layer profile. Writes reports/dosepath-features.json.
"""

from __future__ import annotations

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
from src.instrument.hvp import (  # noqa: E402
    default_ce_loss, fp32_functional_loss, fp32_overrides,
)
from src.optim.airbench_zoo import load_vendor_airbench  # noqa: E402

RUNGS = [0.24, 0.37, 0.48, 0.64]
PEAK, SHOULDER = 0.24, 0.48
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_ckpts():
    out = defaultdict(dict)  # seed -> lr -> ckpt path
    for f in sorted(glob.glob("results/airbench_smoke_seed14*.json")):
        d = json.load(open(f))
        m = d["metrics"]
        if "dosepath" not in d["config"].get("path", ""):
            continue
        if "endpoint_path" not in m or not Path(m["endpoint_path"]).exists():
            continue
        lr = float(d["config"]["contents"]["optimizer"]["lr"])
        out[d["seed"]][lr] = m["endpoint_path"]
    return out


def flat_params(sd):
    return torch.cat([v.float().reshape(-1) for k, v in sorted(sd.items())
                      if v.is_floating_point() and "running" not in k
                      and "num_batches" not in k])


def main():
    ab = load_vendor_airbench()
    test_loader = ab.CifarLoader("data/cifar10", train=False, batch_size=2000)
    train_loader = ab.CifarLoader("data/cifar10", train=True, batch_size=1000,
                                  aug=dict(flip=True, translate=2))
    slab = train_loader.normalize(train_loader.images[:2000])
    slab_labels = train_loader.labels[:2000]
    model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
    ck = load_ckpts()
    seeds = sorted(s for s in ck if all(lr in ck[s] for lr in RUNGS))
    print(f"{len(seeds)} complete seeds")

    def bn_repair_and_eval(sd):
        model.load_state_dict(sd)
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.reset_running_stats()
        model.train()
        with torch.no_grad():
            for _ in range(4):
                for chunk in slab.split(1000):
                    model(chunk)
        model.eval()
        return ab.evaluate(model, test_loader, tta_level=2)

    rows = []
    for s in seeds:
        sds = {lr: torch.load(ck[s][lr], map_location="cuda") for lr in RUNGS}
        # M1: linear path peak->shoulder with BN repair (incl. endpoints for
        # a consistent convention)
        path = []
        for a in ALPHAS:
            sd = {k: ((1 - a) * sds[PEAK][k].float() + a * sds[SHOULDER][k].float()).to(sds[PEAK][k].dtype)
                  for k in sds[PEAK]}
            path.append(bn_repair_and_eval(sd))
        barrier = min(path) - min(path[0], path[-1])
        # M2: straightness
        w = {lr: flat_params(sds[lr]) for lr in RUNGS}
        v_mid = w[0.37] - w[PEAK]
        v_sh = w[SHOULDER] - w[PEAK]
        v_far = w[0.64] - w[PEAK]
        cos_a = float(torch.dot(v_mid, v_sh) / (v_mid.norm() * v_sh.norm()))
        cos_b = float(torch.dot(v_sh, v_far) / (v_sh.norm() * v_far.norm()))
        # M3: Rayleigh of normalized dose direction at shoulder vs lambda1
        model.load_state_dict(sds[SHOULDER])
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        overrides, leaves = fp32_overrides(model, grad_param_ids={id(p) for p in params})
        leaf_list = [leaves[id(p)] for p in params]
        loss = fp32_functional_loss(model, overrides, slab, slab_labels, default_ce_loss())
        grads = torch.autograd.grad(loss, leaf_list, create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        # dose direction restricted to trainable params, in leaf order
        name_of = {id(p): n for n, p in model.named_parameters()}
        sd_sh, sd_pk = sds[SHOULDER], sds[PEAK]
        dose = torch.cat([(sd_sh[name_of[id(p)]].float() - sd_pk[name_of[id(p)]].float()).reshape(-1)
                          for p in params])
        dose = (dose / dose.norm()).cuda()
        hv = torch.autograd.grad(torch.dot(flat_grad, dose), leaf_list, retain_graph=False)
        rayleigh = float(torch.dot(dose, torch.cat([h.reshape(-1) for h in hv])))
        lam1 = endpoint_lambda1(model, slab, slab_labels, iters=20)
        # M4: layer profile of the dose vector
        prof = {}
        tot = 0.0
        for p in params:
            n = name_of[id(p)]
            e = float((sd_sh[n].float() - sd_pk[n].float()).norm() ** 2)
            prof[n] = e
            tot += e
        prof = {k: round(v / tot, 4) for k, v in prof.items()}
        rows.append({"seed": s, "path_acc": path, "barrier": barrier,
                     "cos_mid_sh": cos_a, "cos_sh_far": cos_b,
                     "rayleigh": rayleigh, "lambda1": lam1,
                     "ratio": rayleigh / lam1 if lam1 else None,
                     "layer_profile": prof})
        print(f"seed {s}: barrier {barrier*100:+.2f}pp | path "
              + " ".join(f"{a:.3f}" for a in path)
              + f" | cos {cos_a:+.2f}/{cos_b:+.2f} | Rayleigh/l1 {rayleigh/lam1:+.3f}")

    Path("reports/dosepath-features.json").write_text(json.dumps({"rows": rows}, indent=1))
    b = [r["barrier"] for r in rows]
    print(f"\nM1 barrier: mean {np.mean(b)*100:+.2f}pp (connected >= -0.5pp; barriered <= -5pp)")
    print(f"M2 cos(mid,shoulder): {np.mean([r['cos_mid_sh'] for r in rows]):+.2f}; "
          f"cos(shoulder,far): {np.mean([r['cos_sh_far'] for r in rows]):+.2f} (chord >= 0.7)")
    print(f"M3 Rayleigh/lambda1: mean {np.mean([r['ratio'] for r in rows]):+.3f} "
          f"(sloppy <= 0.1; sharp >= 0.5)")


if __name__ == "__main__":
    sys.exit(main())
