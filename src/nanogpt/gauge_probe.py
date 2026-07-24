"""Program #20 gauge probe (PORT CHANGE P7, measurement-only).

Pre-registration: reports/gauge-ledger-prereg.md §6. Logs, per Muon optimizer
step and per Muon matrix, the scalars ||W||^2_F, <W, V>_F, ||V||^2_F and
eff_lr, where V is the Newton-Schulz output BEFORE the -eff_lr*V application
— everything needed to reconstruct weight-norm trajectories, update
perpendicularity, and the effective LR eta_eff = eff_lr*||V||/||W||, without
storing iterates. For merged QKV parameters it additionally logs the same
scalars per primary-roster block (per-head Q and K slices, the exactly
scale-invariant units).

Attach via ``muon.gauge_probe = GaugeProbe(named_params)`` (mirrors the P6
tempo-probe pattern); ``None`` leaves Muon.step byte-identical in behaviour.
All per-step work is device-side reductions appended to lists (no syncs);
``flush()`` stacks to CPU tensors once at end of training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

QK_SLICES = 2  # slices 0 (Q) and 1 (K) of the merged (3, hdim, dim) qkv_w
HEAD_DIM = 128  # model.py CausalSelfAttention head_dim (fixed by the record)


def _is_qkv(name: str, p: Tensor) -> bool:
    return name.endswith("qkv_w") and p.ndim == 3 and p.shape[0] == 3 and p.shape[1] % HEAD_DIM == 0


class GaugeProbe:
    def __init__(self, named_params: List[Tuple[str, Tensor]]):
        self._names = {id(p): n for n, p in named_params}
        self._qkv = {id(p) for n, p in named_params if _is_qkv(n, p)}
        # per param name -> list of per-step scalar tensors
        self._log: Dict[str, Dict[str, List[Tensor]]] = {}
        self._block_log: Dict[str, List[Tensor]] = {}  # name -> [(3, 2, H) stacked scalars]
        self.steps = 0

    @torch.no_grad()
    def observe(self, p: Tensor, v: Tensor, eff_lr: float) -> None:
        """Called once per matrix per step, W pre-update, V the NS output."""
        name = self._names.get(id(p))
        if name is None:
            return
        w, u = p.detach(), v.detach().to(p.dtype)
        rec = self._log.setdefault(name, {"w2": [], "wv": [], "v2": [], "eff_lr": []})
        rec["w2"].append((w * w).sum())
        rec["wv"].append((w * u).sum())
        rec["v2"].append((u * u).sum())
        rec["eff_lr"].append(torch.as_tensor(float(eff_lr)))
        if id(p) in self._qkv:
            heads = p.shape[1] // HEAD_DIM
            wb = w[:QK_SLICES].view(QK_SLICES, heads, HEAD_DIM, -1)
            ub = u[:QK_SLICES].view(QK_SLICES, heads, HEAD_DIM, -1)
            scal = torch.stack(
                [(wb * wb).sum(dim=(2, 3)), (wb * ub).sum(dim=(2, 3)), (ub * ub).sum(dim=(2, 3))]
            )  # (3, 2, heads): [w2, wv, v2] x [Q, K] x head
            self._block_log.setdefault(name, []).append(scal)

    def begin_step(self) -> None:
        self.steps += 1

    def flush(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"steps": self.steps, "head_dim": HEAD_DIM, "matrices": {}, "qk_blocks": {}}
        for name, rec in self._log.items():
            out["matrices"][name] = {
                k: torch.stack(vs).float().cpu() for k, vs in rec.items()
            }
        for name, scals in self._block_log.items():
            out["qk_blocks"][name] = torch.stack(scals).float().cpu()  # (T, 3, 2, H)
        return out
