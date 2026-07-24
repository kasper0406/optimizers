#!/usr/bin/env python
"""Program #20 Gauge Ledger — Phase A analysis (prereg reports/gauge-ledger-prereg.md).

Stages:
  ab  — criteria (a) radial dominance and (b) tangential de-anti-alignment,
        CPU-only, from stored Wave-1 arm-A/arm-C artifacts (seeds 1511-1513);
        plus registered exploratory breakdowns (control group, per-layer,
        Q-vs-K, embed rows).
  c   — criterion (c) perpendicularity + zero-fit norm-growth law, from the
        gauge replay artifacts (constant-LR, seeds 1511-1512).
  d   — criterion (d) ±10% roster radial rescale, 4 full-val forward passes
        (GPU), on seed-1511 arm-C Polyak and arm-A final endpoints; plus the
        exploratory gauge-transport eval.

All numeric definitions follow prereg §3-4 exactly; this script reports, the
gate call is human.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

HEAD_DIM = 128
SEEDS_AB = [1511, 1512, 1513]
CONSTLR_TAG = "wave1_constlr_acc"
WSD_TAG = "wave1_wsd_acc"


def _readout_mod():
    spec = importlib.util.spec_from_file_location(
        "wave1_readout", REPO_ROOT / "scripts" / "analyze_wave1_readout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_arm_artifacts(seed: int):
    mod = _readout_mod()
    glob_pat = str(REPO_ROOT / "results" / f"nanogpt_seed{seed}_*.json")
    c_path, _ = mod.find_artifact(seed, glob_pat, CONSTLR_TAG)
    a_path, _ = mod.find_artifact(seed, glob_pat, WSD_TAG)
    art_c = torch.load(c_path, map_location="cpu", weights_only=False)
    art_a = torch.load(a_path, map_location="cpu", weights_only=False)
    return art_c, art_a


def roster_blocks(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Per-head Q and K blocks of every qkv_w (prereg §2 primary roster)."""
    out = {}
    for name, t in tensors.items():
        if name.endswith("qkv_w") and t.ndim == 3 and t.shape[0] == 3:
            heads = t.shape[1] // HEAD_DIM
            for s, stag in ((0, "Q"), (1, "K")):
                for h in range(heads):
                    out[f"{name}.{stag}{h}"] = t[s, h * HEAD_DIM:(h + 1) * HEAD_DIM, :]
    return out


def control_matrices(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Non-invariant Muon-owned group: V slices, c_proj, MLP (prereg §2)."""
    out = {}
    for name, t in tensors.items():
        if name.endswith("qkv_w") and t.ndim == 3:
            out[f"{name}.V"] = t[2]
        elif ("c_proj" in name or "mlp" in name or "c_fc" in name) and t.ndim == 2:
            out[name] = t
    return out


def decompose(units_w2, units_w1, units_a) -> Dict[str, Dict[str, float]]:
    """Per-unit v/D radial-tangential scalars at base point W2 (prereg §3)."""
    per_unit = {}
    for name in units_w2:
        W2, W1, A = units_w2[name].double(), units_w1[name].double(), units_a[name].double()
        v, D = W2 - W1, A - W2
        nrm = W2.norm()
        rhat = W2 / nrm
        v_rad = (v * rhat).sum()
        d_rad = (D * rhat).sum()
        v_tan = v - v_rad * rhat
        d_tan = D - d_rad * rhat
        per_unit[name] = {
            "v_rad": float(v_rad), "d_rad": float(d_rad),
            "v2": float((v * v).sum()), "d2": float((D * D).sum()),
            "vt2": float((v_tan * v_tan).sum()), "dt2": float((d_tan * d_tan).sum()),
            "vt_dt": float((v_tan * d_tan).sum()),
            "v_d": float((v * D).sum()),
        }
    return per_unit


def aggregate(per_unit: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    import math
    s = lambda k: sum(u[k] for u in per_unit.values())
    rad_frac_d = sum(u["d_rad"] ** 2 for u in per_unit.values()) / s("d2")
    rad_frac_v = sum(u["v_rad"] ** 2 for u in per_unit.values()) / s("v2")
    cos_tan = s("vt_dt") / math.sqrt(s("vt2") * s("dt2"))
    cos_full = s("v_d") / math.sqrt(s("v2") * s("d2"))
    dtan_frac = math.sqrt(s("dt2") / s("d2"))
    return {
        "radial_frac_D": rad_frac_d, "radial_frac_v": rad_frac_v,
        "cos_v_D_full": cos_full, "cos_vtan_Dtan": cos_tan,
        "Dtan_over_D": dtan_frac, "n_units": len(per_unit),
    }


def bootstrap_cos(per_unit, n=10000, seed=0) -> Tuple[float, float]:
    import math
    g = torch.Generator().manual_seed(seed)
    units = list(per_unit.values())
    vals = []
    for _ in range(n):
        idx = torch.randint(0, len(units), (len(units),), generator=g)
        vt2 = dt2 = vtdt = 0.0
        for i in idx.tolist():
            u = units[i]
            vt2 += u["vt2"]; dt2 += u["dt2"]; vtdt += u["vt_dt"]
        vals.append(vtdt / math.sqrt(vt2 * dt2))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def stage_ab(out: Dict) -> None:
    out["ab"] = {}
    for seed in SEEDS_AB:
        art_c, art_a = load_arm_artifacts(seed)
        w2, w1, a_final = art_c["w2"], art_c["w1"], art_a["final"]
        res: Dict = {}
        pu_roster = decompose(roster_blocks(w2), roster_blocks(w1), roster_blocks(a_final))
        res["roster"] = aggregate(pu_roster)
        lo, hi = bootstrap_cos(pu_roster)
        res["roster"]["cos_tan_bootstrap_ci95"] = [lo, hi]
        pu_ctrl = decompose(control_matrices(w2), control_matrices(w1), control_matrices(a_final))
        res["control_noninvariant"] = aggregate(pu_ctrl)
        # exploratory: Q vs K split
        for tag in ("Q", "K"):
            sub = {k: v for k, v in pu_roster.items() if f".{tag}" in k}
            res[f"roster_{tag}_only"] = aggregate(sub)
        # secondary: embed rows (Adam-owned; descriptive)
        try:
            e2, e1, ea = art_c["w2"]["embed.weight"], art_c["w1"]["embed.weight"], art_a["final"]["embed.weight"]
            rows = {"embed": None}
            # row-wise: treat each row as a unit, vectorized
            W2, W1, A = e2.double(), e1.double(), ea.double()
            v, D = W2 - W1, A - W2
            nrm = W2.norm(dim=1, keepdim=True).clamp_min(1e-12)
            rhat = W2 / nrm
            d_rad = (D * rhat).sum(1)
            res["embed_rows_secondary"] = {
                "radial_frac_D": float((d_rad ** 2).sum() / (D * D).sum()),
            }
        except KeyError:
            pass
        out["ab"][str(seed)] = res
        r = res["roster"]
        print(f"seed {seed} ROSTER: radial_frac_D={r['radial_frac_D']:.3f} "
              f"cos_full={r['cos_v_D_full']:.3f} cos_tan={r['cos_vtan_Dtan']:.3f} "
              f"Dtan/D={r['Dtan_over_D']:.3f} ci95=[{r['cos_tan_bootstrap_ci95'][0]:.3f},"
              f"{r['cos_tan_bootstrap_ci95'][1]:.3f}]", flush=True)
        c = res["control_noninvariant"]
        print(f"          CONTROL: radial_frac_D={c['radial_frac_D']:.3f} "
              f"cos_full={c['cos_v_D_full']:.3f} cos_tan={c['cos_vtan_Dtan']:.3f}", flush=True)

    # registered verdicts for (a) and (b)
    rr = [out["ab"][str(s)]["roster"] for s in SEEDS_AB]
    out["criteria"] = out.get("criteria", {})
    out["criteria"]["a_radial_dominance"] = all(r["radial_frac_D"] >= 0.5 for r in rr)
    b_cos = sum(1 for r in rr if abs(r["cos_vtan_Dtan"]) < 0.2) >= 2
    b_dtan = all(r["Dtan_over_D"] >= 0.3 for r in rr)
    b_ci = all(r["cos_tan_bootstrap_ci95"][0] >= -0.4 for r in rr)
    out["criteria"]["b_tangential_dealignment"] = bool(b_cos and b_dtan and b_ci)
    out["criteria"]["b_parts"] = {"cos_2of3": b_cos, "dtan_ge_03_all": b_dtan, "ci_excl_m04_all": b_ci}


def stage_c(out: Dict) -> None:
    import glob as globmod
    res = {}
    per_block_ok, growth_ok = [], []
    for seed in (1511, 1512):
        hits = []
        for f in sorted(globmod.glob(str(REPO_ROOT / "results" / f"nanogpt_seed{seed}_*.json"))):
            d = json.load(open(f))
            if Path(str(d["config"].get("path", ""))).name != "gauge_replay_constlr.yaml":
                continue
            if "gauge_artifact" in (d.get("metrics") or {}):
                hits.append(d)
        if not hits:
            raise SystemExit(f"no gauge replay result for seed {seed}")
        art = torch.load(REPO_ROOT / hits[-1]["metrics"]["gauge_artifact"],
                         map_location="cpu", weights_only=False)
        seed_res = {"blocks": 0, "perp_pass": 0, "growth_pass": 0}
        for name, scal in art["qk_blocks"].items():  # (T, 3, 2, H): [w2, wv, v2]
            lr = art["matrices"][name]["eff_lr"]  # (T,)
            T = scal.shape[0]
            for s in range(2):
                for h in range(scal.shape[3]):
                    w2 = scal[:, 0, s, h].double()
                    wv = scal[:, 1, s, h].double()
                    v2 = scal[:, 2, s, h].double()
                    lrd = lr.double()
                    perp = float((2 * wv.abs() / v2.clamp_min(1e-30)).median())
                    pred = w2[0] + torch.cumsum(lrd**2 * v2, 0)[:-1]
                    rel = float(((pred - w2[1:]) / w2[1:]).pow(2).mean().sqrt())
                    seed_res["blocks"] += 1
                    seed_res["perp_pass"] += int(perp <= 0.2)
                    seed_res["growth_pass"] += int(rel <= 0.05)
                    per_block_ok.append(perp <= 0.2)
                    growth_ok.append(rel <= 0.05)
        res[str(seed)] = seed_res
        print(f"seed {seed}: {seed_res}", flush=True)
    out["c"] = res
    frac_perp = sum(per_block_ok) / len(per_block_ok)
    frac_growth = sum(growth_ok) / len(growth_ok)
    out["criteria"] = out.get("criteria", {})
    out["criteria"]["c_perp_frac"] = frac_perp
    out["criteria"]["c_growth_frac"] = frac_growth
    out["criteria"]["c_norm_growth_law"] = bool(frac_perp >= 0.8 and frac_growth >= 0.8)


def _rescale_roster(tensors: Dict[str, torch.Tensor], factor: float) -> Dict[str, torch.Tensor]:
    out = {}
    for name, t in tensors.items():
        if name.endswith("qkv_w") and t.ndim == 3 and t.shape[0] == 3:
            t = t.clone()
            t[0] *= factor
            t[1] *= factor
        out[name] = t
    return out


def _rescale_roster_to_target(src: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]):
    out = {}
    for name, t in src.items():
        if name.endswith("qkv_w") and t.ndim == 3 and t.shape[0] == 3:
            t = t.clone()
            heads = t.shape[1] // HEAD_DIM
            for s in (0, 1):
                for h in range(heads):
                    sl = (s, slice(h * HEAD_DIM, (h + 1) * HEAD_DIM))
                    t[sl] *= (tgt[name][sl].norm() / t[sl].norm().clamp_min(1e-12))
        out[name] = t
    return out


def stage_d(out: Dict) -> None:
    mod = _readout_mod()
    cfg_mod = importlib.import_module("src.nanogpt.config")
    cfg = cfg_mod.NanoGPTConfig.from_config(
        yaml.safe_load(open(REPO_ROOT / "configs/dev/wave1_constlr_acc.yaml")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = mod.ShardEvaluator(cfg, device, compile_model=torch.cuda.is_available())
    art_c, art_a = load_arm_artifacts(1511)
    res = {}

    def val(tag, weights):
        ev.load_weights(weights)
        s1, s2 = ev.shard_losses()
        res[tag] = (s1 + s2) / 2
        print(f"{tag}: {res[tag]:.5f}", flush=True)

    for base_tag, base in (("c_polyak", art_c["polyak"]), ("a_final", art_a["final"])):
        val(f"{base_tag}", base)
        val(f"{base_tag}_x1.1", _rescale_roster(base, 1.1))
        val(f"{base_tag}_x0.9", _rescale_roster(base, 0.9))
    # exploratory: gauge transport of arm-C polyak roster norms to arm-A's
    val("c_polyak_transported_to_a_norms", _rescale_roster_to_target(art_c["polyak"], art_a["final"]))
    out["d"] = res
    deltas = [abs(res[f"{b}_x{f}"] - res[b]) for b in ("c_polyak", "a_final") for f in ("1.1", "0.9")]
    out["criteria"] = out.get("criteria", {})
    out["criteria"]["d_max_rescale_delta"] = max(deltas)
    out["criteria"]["d_function_null"] = bool(max(deltas) < 0.0025)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stages", nargs="+", choices=["ab", "c", "d"])
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "gauge-ledger-phase-a.json"))
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for st in args.stages:
        {"ab": stage_ab, "c": stage_c, "d": stage_d}[st](out)
    out_path.write_text(json.dumps(out, indent=2))
    print("criteria so far:", json.dumps(out.get("criteria", {}), indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
