"""Program #22 BBP Phase A-empirical: frozen-checkpoint msign alignment probe.

Pre-registration: reports/bbp-prereg.md. At a frozen parameter point, stream
M chunk gradients (the record's own 49,152-token BOS-aligned chunks from a
registered fixed data offset) and estimate the split-half alignment curve
a(b) of the Newton-Schulz orthogonalization via a doubling merge tree:
every pairwise merge of two independent b-chunk half-sums contributes one
cos(msign(A), msign(B)) sample at that b. No optimizer steps; training is
never touched.

Registered estimator details (§1): msign = the record's 5-step bf16
Newton-Schulz (what Muon actually computes); â(b) = sqrt(max(mean cos, 0)).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.nanogpt.config import NanoGPTConfig
from src.nanogpt.data import RecordDataGenerator
from src.nanogpt.model import GPT, next_multiple_of_n
from src.nanogpt.optim import zeropower_via_newtonschulz5
from src.nanogpt.train import REPO_ROOT, window_size_blocks_value


def _build_frozen_model(cfg: NanoGPTConfig, device: torch.device):
    model = GPT(
        vocab_size=next_multiple_of_n(cfg.vocab_size, n=128),
        num_layers=cfg.num_layers, num_heads=cfg.num_heads, model_dim=cfg.model_dim,
        max_seq_len=max(cfg.train_seq_len, cfg.val_seq_len),
        world_size=cfg.record_world_size,
        use_fp8=(cfg.precision_mode == "fp8"),
        attention_impl=cfg.attention_impl,
        head_chunk_rows=cfg.head_chunk_rows,
    ).to(device)
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()
    return model


def _load_weights(raw_model, spec: Dict[str, Any], device) -> str:
    ckpt, art = spec.get("checkpoint"), spec.get("artifact")
    assert (ckpt is None) != (art is None), "exactly one of checkpoint/artifact"
    if ckpt is not None:
        path = Path(ckpt) if Path(ckpt).is_absolute() else REPO_ROOT / ckpt
        state = torch.load(path, map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        return f"checkpoint:{path.name}@step{state.get('step')}"
    path = Path(art) if Path(art).is_absolute() else REPO_ROOT / art
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    weights = artifact["final"]
    params = dict(raw_model.named_parameters())
    missing = set(params) - set(weights)
    assert not missing, f"artifact missing params: {sorted(missing)[:4]}"
    with torch.no_grad():
        for n, t in weights.items():
            params[n].copy_(t.to(device=params[n].device, dtype=params[n].dtype))
    return f"artifact:{path.name}"


def run_bbp_probe(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    cfg = NanoGPTConfig.from_config(config)
    spec = dict(config.get("bbp", {}) or {})
    n_chunks = int(spec.get("n_chunks", 512))
    window_step = int(spec["window_step"])
    file_index = int(spec.get("data_file_index", 3))

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)

    raw_model = _build_frozen_model(cfg, device)
    source = _load_weights(raw_model, spec, device)
    model = torch.compile(raw_model, dynamic=False) if cfg.compile else raw_model
    model.train()  # gradient path identical to training (no dropout in this arch)

    hidden = [(n, p) for n, p in raw_model.blocks.named_parameters()
              if p.ndim >= 2 and "embed" not in n]
    names = [n for n, _ in hidden]

    window = torch.tensor(window_size_blocks_value(window_step, cfg),
                          dtype=torch.int32, device=device)
    loader = RecordDataGenerator(
        cfg.train_files, local_batch_size=cfg.train_seq_len,
        record_world_size=cfg.effective_chunks, device_count=1, rank=0,
        align_to_bos=cfg.train_align_to_bos, device=device,
    )
    loader.load_state_dict({"file_index": file_index, "pos": 0})

    def msign(t_cpu: torch.Tensor) -> torch.Tensor:
        return zeropower_via_newtonschulz5(t_cpu.to(device).bfloat16(), 5)

    def cos_records(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, float]:
        out = {}
        for n in names:
            ma, mb = msign(a[n]).float(), msign(b[n]).float()
            out[n] = float((ma * mb).sum() / (ma.norm() * mb.norm()))
        return out

    # streaming doubling tree (binary-counter merge)
    alignments: Dict[int, Dict[str, List[float]]] = {}
    stack: List = []  # (level, {name: cpu fp32 sum})
    produced = 0
    micro_iter = None
    while produced < n_chunks:
        if micro_iter is None:
            micro_iter = loader.next_step()
        try:
            inputs, targets = next(micro_iter)
        except StopIteration:
            micro_iter = None
            continue
        model.zero_grad(set_to_none=True)
        model(inputs, targets, window).backward()
        grads = {n: p.grad.detach().float().cpu().clone() for n, p in hidden}
        produced += 1
        node = (0, grads)
        while stack and stack[-1][0] == node[0]:
            lvl, other = stack.pop()
            rec = alignments.setdefault(lvl, {n: [] for n in names})
            for n, c in cos_records(other, node[1]).items():
                rec[n].append(round(c, 6))
            node = (lvl + 1, {n: other[n] + node[1][n] for n in names})
        stack.append(node)
        if produced % 64 == 0:
            print(f"chunks {produced}/{n_chunks}", flush=True)
    model.zero_grad(set_to_none=True)

    # â(b) curves: level ℓ merges compare half-sums of b = 2^ℓ chunks
    curves: Dict[str, Dict[str, Any]] = {}
    for lvl, rec in sorted(alignments.items()):
        b = 2 ** lvl
        med = []
        for n in names:
            cs = rec[n]
            mean_c = sum(cs) / len(cs)
            a_hat = (max(mean_c, 0.0)) ** 0.5
            curves.setdefault(n, {"b_chunks": [], "a_hat": [], "n_merges": []})
            curves[n]["b_chunks"].append(b)
            curves[n]["a_hat"].append(round(a_hat, 5))
            curves[n]["n_merges"].append(len(cs))
            med.append(a_hat)

    return {
        "bbp_source": source,
        "window_step": window_step,
        "n_chunks": n_chunks,
        "data_file_index": file_index,
        "chunk_tokens": cfg.train_seq_len,
        "curves": curves,
        "raw_cos": {str(2 ** lvl): rec for lvl, rec in sorted(alignments.items())},
        "nanogpt_config": cfg.to_dict(),
    }
