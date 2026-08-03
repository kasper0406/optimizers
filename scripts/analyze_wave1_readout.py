#!/usr/bin/env python
"""Program #17 drift-completion readout evaluation (Wave 1).

Pre-registered design: reports/wave1-anneal-decomposition-prereg.md §1.
Inputs: tail artifacts written by the arm-C (constant-LR + accumulators) and
arm-A (WSD + accumulators) runs. Forward passes only — no training.

For each seed it evaluates, on val shard-1 (val chunks 0-19) and shard-2
(chunks 20-39), the readout family W(alpha) = W2 + alpha*(W2 - W1) from the
arm-C artifact, plus the registered references (arm-C raw final iterate,
arm-C Polyak, arm-A final iterate) and the labeled-exploratory WSM-style
convex merges w*W2 + (1-w)*Polyak. Mechanism readouts: cos(v, D) and
||D||/||v|| with v = W2 - W1, D = A_final - W2 (same-seed pairs share the
prefix, so D is the anneal displacement).

Selection protocol (registered): alpha* = argmin on --select-seed shard-1,
frozen; recovery r = (L(W2) - L(W(alpha*))) / (L(W2) - L_WSD) per held-out
seed x shard, L_WSD from the same-seed arm-A final iterate on that shard.
PASS iff r >= 0.40 on all four held-out cells; FAIL iff any r < 0.40 or
alpha* <= 0.5. This script only REPORTS the numbers; the gate call is human.

Usage:
  uv run python scripts/analyze_wave1_readout.py \
      --select-seed 1511 --heldout-seeds 1512 1513 \
      --out reports/wave1-readout.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from src.nanogpt.config import NanoGPTConfig
from src.nanogpt.data import RecordDataGenerator
from src.nanogpt.model import GPT, next_multiple_of_n

ALPHA_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # prereg §1
MERGE_GRID = [0.25, 0.5, 0.75]  # exploratory (labeled)
SHARD_CHUNKS = 20  # val chunks per shard; 40 total (prereg §1)
FINAL_STEP = 1750


def find_artifact(seed: int, results_glob: str, config_tag: str) -> Tuple[Path, Dict]:
    """Locate the newest completed results JSON for (config_tag, seed) and
    return its tail artifact path + the parsed results JSON."""
    hits = []
    for f in sorted(glob.glob(results_glob)):
        d = json.load(open(f))
        cfg = d.get("config") or {}
        if Path(str(cfg.get("path", ""))).name != config_tag + ".yaml":
            continue
        if d.get("seed") != seed:
            continue
        m = d.get("metrics") or {}
        if m.get("final_val_loss") is None or "tail_artifact" not in m:
            continue
        hits.append((f, d))
    if not hits:
        raise SystemExit(f"no completed results with a tail artifact for {config_tag} seed {seed}")
    f, d = hits[-1]
    art = Path(d["metrics"]["tail_artifact"])
    if not art.is_absolute():
        art = REPO_ROOT / art
    if not art.exists():
        raise SystemExit(f"artifact {art} (from {f}) is missing on disk")
    return art, d


def build_model(cfg: NanoGPTConfig, device: torch.device) -> torch.nn.Module:
    model = GPT(
        vocab_size=next_multiple_of_n(cfg.vocab_size, n=128),
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        model_dim=cfg.model_dim,
        max_seq_len=max(cfg.train_seq_len, cfg.val_seq_len),
        world_size=cfg.record_world_size,
        use_fp8=(cfg.precision_mode == "fp8"),
        attention_impl=cfg.attention_impl,
        head_chunk_rows=cfg.head_chunk_rows,
    ).to(device)
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()
    model.eval()
    return model


class ShardEvaluator:
    """Evaluate a weight assignment on the two fixed val half-shards."""

    def __init__(self, cfg: NanoGPTConfig, device: torch.device, compile_model: bool):
        self.cfg = cfg
        self.device = device
        self.raw_model = build_model(cfg, device)
        self.model = torch.compile(self.raw_model, dynamic=False) if compile_model else self.raw_model
        self.param_names = {n for n, _ in self.raw_model.named_parameters()}
        from src.nanogpt.train import window_size_blocks_value

        self.window = torch.tensor(
            window_size_blocks_value(FINAL_STEP, cfg), dtype=torch.int32, device=device
        )

    @torch.no_grad()
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        missing = self.param_names - set(weights)
        if missing:
            raise SystemExit(f"weight dict missing parameters: {sorted(missing)[:5]} ...")
        params = dict(self.raw_model.named_parameters())
        for n, t in weights.items():
            params[n].copy_(t.to(device=self.device, dtype=params[n].dtype))

    @torch.no_grad()
    def shard_losses(self) -> Tuple[float, float]:
        cfg = self.cfg
        loader = RecordDataGenerator(
            cfg.val_files,
            local_batch_size=cfg.val_seq_len,
            record_world_size=cfg.record_world_size,
            device_count=1,
            rank=0,
            align_to_bos=cfg.val_align_to_bos,
            device=self.device,
        )
        val_steps = cfg.val_tokens // (cfg.record_world_size * cfg.val_seq_len)
        sums = [torch.zeros((), device=self.device), torch.zeros((), device=self.device)]
        counts = [0, 0]
        chunk = 0
        for _ in range(val_steps):
            for inputs, targets in loader.next_step():
                shard = 0 if chunk < SHARD_CHUNKS else 1
                sums[shard] = sums[shard] + self.model(inputs, targets, self.window)
                counts[shard] += 1
                chunk += 1
        assert counts == [SHARD_CHUNKS, SHARD_CHUNKS], counts
        return float(sums[0].item()) / counts[0], float(sums[1].item()) / counts[1]


def lincomb(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], wa: float, wb: float):
    return {n: wa * a[n] + wb * b[n] for n in a}


def dot_norm(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Tuple[float, float, float]:
    dot = na = nb = 0.0
    for n in a:
        x, y = a[n].double(), b[n].double()
        dot += float((x * y).sum())
        na += float((x * x).sum())
        nb += float((y * y).sum())
    return dot, math.sqrt(na), math.sqrt(nb)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select-seed", type=int, default=1511)
    ap.add_argument("--heldout-seeds", type=int, nargs="+", default=[1512, 1513])
    ap.add_argument("--constlr-tag", default="wave1_constlr_acc")
    ap.add_argument("--wsd-tag", default="wave1_wsd_acc")
    ap.add_argument("--results-glob", default=str(REPO_ROOT / "results" / "nanogpt_seed*.json"))
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/dev/wave1_constlr_acc.yaml"))
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "wave1-readout.json"))
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = NanoGPTConfig.from_config(yaml.safe_load(open(args.config)))
    ev = ShardEvaluator(cfg, device, compile_model=not args.no_compile)

    seeds = [args.select_seed] + list(args.heldout_seeds)
    out: Dict = {"alpha_grid": ALPHA_GRID, "seeds": seeds, "per_seed": {}}

    for seed in seeds:
        art_c_path, res_c = find_artifact(seed, args.results_glob, args.constlr_tag)
        art_a_path, res_a = find_artifact(seed, args.results_glob, args.wsd_tag)
        art_c = torch.load(art_c_path, map_location="cpu", weights_only=False)
        art_a = torch.load(art_a_path, map_location="cpu", weights_only=False)
        w1, w2, polyak, c_final = art_c["w1"], art_c["w2"], art_c["polyak"], art_c["final"]
        a_final = art_a["final"]

        v = {n: w2[n] - w1[n] for n in w2}
        d = {n: a_final[n] - w2[n] for n in w2}
        dot, nv, nd = dot_norm(v, d)
        rec: Dict = {
            "cos_v_D": dot / (nv * nd) if nv * nd else None,
            "norm_D_over_v": nd / nv if nv else None,
            "counts": art_c["counts"],
            "artifact_c": str(art_c_path),
            "artifact_a": str(art_a_path),
            "final_val_c": res_c["metrics"]["final_val_loss"],
            "final_val_a": res_a["metrics"]["final_val_loss"],
            "readouts": {},
        }

        def evaluate(name: str, weights: Dict[str, torch.Tensor]) -> Tuple[float, float]:
            ev.load_weights(weights)
            s1, s2 = ev.shard_losses()
            rec["readouts"][name] = {"shard1": s1, "shard2": s2}
            print(f"seed {seed} {name}: shard1 {s1:.5f} shard2 {s2:.5f}", flush=True)
            return s1, s2

        for alpha in ALPHA_GRID:
            evaluate(f"alpha_{alpha:g}", lincomb(w2, v, 1.0, alpha) if alpha else w2)
        evaluate("c_final", c_final)
        evaluate("c_polyak", polyak)
        evaluate("a_final", a_final)
        for w in MERGE_GRID:  # exploratory, labeled (prereg §1)
            evaluate(f"merge_w2_{w:g}", lincomb(w2, polyak, w, 1.0 - w))
        out["per_seed"][str(seed)] = rec

    # ---- registered selection + recovery ---------------------------------
    sel = out["per_seed"][str(args.select_seed)]["readouts"]
    alpha_star = min(ALPHA_GRID, key=lambda a: sel[f"alpha_{a:g}"]["shard1"])
    out["alpha_star"] = alpha_star
    out["alpha_star_shard1_curve"] = {f"{a:g}": sel[f"alpha_{a:g}"]["shard1"] for a in ALPHA_GRID}
    out["recovery"] = {}
    for seed in args.heldout_seeds:
        r = out["per_seed"][str(seed)]["readouts"]
        for shard in ("shard1", "shard2"):
            l_w2 = r["alpha_0"][shard]
            l_alpha = r[f"alpha_{alpha_star:g}"][shard]
            l_wsd = r["a_final"][shard]
            denom = l_w2 - l_wsd
            out["recovery"][f"seed{seed}_{shard}"] = {
                "L_W2": l_w2, "L_Walpha": l_alpha, "L_WSD": l_wsd,
                "r": (l_w2 - l_alpha) / denom if denom else None,
            }
    rs = [c["r"] for c in out["recovery"].values() if c["r"] is not None]
    out["registered"] = {
        "alpha_star": alpha_star,
        "alpha_star_gt_half": alpha_star > 0.5,
        "min_heldout_recovery": min(rs) if rs else None,
        "all_cells_ge_040": bool(rs) and all(r >= 0.40 for r in rs),
        "note": "gate call is human (prereg §1); this block only restates the registered quantities",
    }

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    print(json.dumps(out["registered"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
