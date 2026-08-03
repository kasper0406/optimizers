#!/usr/bin/env python
"""Wave-1 Phase B analysis — program #18 (schedule-free tail graft).

Gate record: reports/wave1-phase-b-gate.md. Computes ONLY the registered
quantities (plus labeled descriptives):

- paired deltas B(s) - A(s), s in eval seeds: mean, t, 95% CI
  (A = existing n=10 baseline runs, same seed, same hot fingerprint)
- paired deltas B(s) - C_polyak(s): C_polyak evaluated by full-val forward
  passes from the arm-C tail artifact (40 chunks, window at step 1750)
- flatness: val(xbar) at steps 1375 and 1500 vs endpoint, per seed
- restated criteria booleans (WIN / ANNEAL-REPLACED / KILL); gate call human.

Usage: uv run python scripts/analyze_wave1_phaseb.py --out reports/wave1-phaseb.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

EVAL_SEEDS = [1710, 1711, 1712, 1713]


def completed_run(seed: int, config_tag: str) -> Dict:
    hits = []
    for f in sorted(glob.glob(str(REPO_ROOT / "results" / f"nanogpt_seed{seed}_*.json"))):
        d = json.load(open(f))
        if Path(str((d.get("config") or {}).get("path", ""))).name != config_tag + ".yaml":
            continue
        if (d.get("metrics") or {}).get("final_val_loss") is None:
            continue
        hits.append(d)
    if not hits:
        raise SystemExit(f"no completed run for {config_tag} seed {seed}")
    return hits[-1]


def paired_stats(deltas: List[float]) -> Dict:
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    se = math.sqrt(var / n)
    # t crit for 95% two-sided, df=3
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}[n - 1]
    return {
        "n": n, "mean": mean, "sd": math.sqrt(var), "se": se,
        "t": mean / se if se else None,
        "ci95": [mean - tcrit * se, mean + tcrit * se],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=EVAL_SEEDS)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "wave1-phaseb.json"))
    args = ap.parse_args(argv)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "wave1_readout", REPO_ROOT / "scripts" / "analyze_wave1_readout.py"
    )
    readout_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readout_mod)
    ShardEvaluator, find_artifact = readout_mod.ShardEvaluator, readout_mod.find_artifact
    from src.nanogpt.config import NanoGPTConfig  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = NanoGPTConfig.from_config(
        yaml.safe_load(open(REPO_ROOT / "configs/dev/wave1_constlr_acc.yaml"))
    )
    ev = ShardEvaluator(cfg, device, compile_model=not args.no_compile)

    out: Dict = {"seeds": args.seeds, "per_seed": {}}
    d_BA, d_BCpol = [], []
    flat_ok = []
    for seed in args.seeds:
        B = completed_run(seed, "wave1_sf_k10_r07")
        A = completed_run(seed, "nanogpt_local_baseline")
        art_c_path, C = find_artifact(
            seed, str(REPO_ROOT / "results" / "nanogpt_seed*.json"), "wave1_constlr_acc"
        )
        art = torch.load(art_c_path, map_location="cpu", weights_only=False)
        ev.load_weights(art["polyak"])
        s1, s2 = ev.shard_losses()
        c_polyak_full = (s1 + s2) / 2  # 20+20 chunks, equal weight == full val

        b_final = B["metrics"]["final_val_loss"]
        a_final = A["metrics"]["final_val_loss"]
        curve = {p["step"]: p["val_loss"] for p in B["metrics"]["val_curve"]}
        flat = (abs(curve[1375] - b_final) <= 0.01) and (abs(curve[1500] - b_final) <= 0.01)
        d_BA.append(b_final - a_final)
        d_BCpol.append(b_final - c_polyak_full)
        flat_ok.append(flat)
        out["per_seed"][str(seed)] = {
            "B_xbar": b_final,
            "B_z": B["metrics"].get("final_val_loss_z"),
            "A": a_final,
            "C_final": C["metrics"]["final_val_loss"],
            "C_polyak_full_val": c_polyak_full,
            "B_minus_A": b_final - a_final,
            "B_minus_Cpolyak": b_final - c_polyak_full,
            "xbar_at_1375": curve[1375], "xbar_at_1500": curve[1500],
            "flatness_ok": flat,
        }
        print(f"seed {seed}: B {b_final:.5f}  A {a_final:.5f}  "
              f"C_pol {c_polyak_full:.5f}  dBA {b_final - a_final:+.5f}", flush=True)

    sBA = paired_stats(d_BA)
    sBC = paired_stats(d_BCpol)
    win = sBA["mean"] <= -0.0025 and sBA["ci95"][1] < 0
    kill = sBA["mean"] >= 0.0025
    annealed = (
        sBA["mean"] < 0.0025
        and all(d < -0.005 for d in d_BCpol)
        and all(flat_ok)
    )
    out["paired_B_minus_A"] = sBA
    out["paired_B_minus_Cpolyak"] = sBC
    out["registered"] = {
        "WIN": bool(win),
        "ANNEAL_REPLACED": bool(annealed and not win),
        "KILL": bool(kill),
        "flatness_all": all(flat_ok),
        "note": "gate verdict is human (reports/wave1-phase-b-gate.md)",
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out["registered"][k] for k in ("WIN", "ANNEAL_REPLACED", "KILL")}))
    print("paired B-A:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sBA.items() if k != "ci95"},
          "ci95:", [round(x, 6) for x in sBA["ci95"]])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
