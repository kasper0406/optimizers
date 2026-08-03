"""Program #8 passive tempo probe for the nanogpt port (PORT CHANGE P6).

Measures, per observed hidden matrix per step, two serial-alignment
statistics of the (cross-rank-averaged) gradient stream inside Muon.step:

- ``cos_gg``: cos(G_t, G_{t-1}) — the airbench-calibrated program-#8
  signal (requires storing the previous gradient; kept in bf16 to halve
  memory; fp32 arithmetic for the dot/norms per the fp16-overflow lesson,
  reports/tempo-phase-a.md §2.1).
- ``cos_gm``: cos(G_t, momentum_buffer_{t-1}) — a zero-extra-memory
  alignment in the same family (GALA-flavored); logged side-by-side to
  test whether it can replace cos_gg at scale.

Passive only: nothing feeds back into the update. Observations are 0-dim
GPU tensors buffered on-device and synced to CPU every ``flush_every``
steps (avoids a per-matrix ``.item()`` sync per step).

``subset``: observe every ``subset``-th matrix (by construction order) to
cap the prev-grad memory (124M model: all hidden matrices ~170 MB bf16;
subset=2 → ~85 MB against a 31.4 GB peak).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


class TempoProbe:
    def __init__(self, subset: int = 2, flush_every: int = 10) -> None:
        if subset < 1:
            raise ValueError(f"subset must be >= 1, got {subset}")
        if flush_every < 1:
            raise ValueError(f"flush_every must be >= 1, got {flush_every}")
        self.subset = subset
        self.flush_every = flush_every
        self._param_idx: Dict[int, int] = {}  # id(param) -> matrix index
        self._shapes: Dict[int, List[int]] = {}
        self._prev: Dict[int, torch.Tensor] = {}  # matrix idx -> bf16 flat grad
        self._pending: List[tuple] = []  # (step, idx, cos_gg 0-dim, cos_gm 0-dim)
        self._rows: List[Dict[str, Any]] = []  # flushed CPU rows
        self._step = 0

    def begin_step(self) -> None:
        """Call once per optimizer step, before any observe()."""
        self._step += 1
        if self._step % self.flush_every == 0:
            self._flush()

    @torch.no_grad()
    def observe(self, param: torch.Tensor, grad: torch.Tensor,
                momentum_buffer: Optional[torch.Tensor]) -> None:
        """Called from Muon.step with the averaged grad, pre-momentum-update."""
        key = id(param)
        if key not in self._param_idx:
            self._param_idx[key] = len(self._param_idx)
            self._shapes[self._param_idx[key]] = list(param.shape)
        idx = self._param_idx[key]
        if idx % self.subset != 0:
            return
        g32 = grad.detach().reshape(-1).float()
        gnorm = g32.norm()
        cos_gg = None
        prev = self._prev.get(idx)
        if prev is not None:
            p32 = prev.float()
            cos_gg = torch.dot(g32, p32) / (gnorm * p32.norm()).clamp_min(1e-30)
        cos_gm = None
        if momentum_buffer is not None:
            m32 = momentum_buffer.detach().reshape(-1).float()
            cos_gm = torch.dot(g32, m32) / (gnorm * m32.norm()).clamp_min(1e-30)
        if prev is None:
            self._prev[idx] = g32.bfloat16()
        else:
            prev.copy_(g32.bfloat16())
        self._pending.append((self._step, idx, cos_gg, cos_gm))

    def _flush(self) -> None:
        for step, idx, cos_gg, cos_gm in self._pending:
            self._rows.append(
                {
                    "step": step,
                    "matrix": idx,
                    "cos_gg": None if cos_gg is None else float(cos_gg.item()),
                    "cos_gm": None if cos_gm is None else float(cos_gm.item()),
                }
            )
        self._pending = []

    def to_log(self) -> Dict[str, Any]:
        self._flush()
        return {
            "subset": self.subset,
            "flush_every": self.flush_every,
            "matrices": {
                str(i): self._shapes[i]
                for i in sorted(self._shapes)
                if i % self.subset == 0
            },
            "rows": self._rows,
        }
