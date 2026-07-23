"""Wave-1 tail-phase machinery (programs #17/#18/#19).

Pre-registered design: reports/wave1-anneal-decomposition-prereg.md §0.
Config surface: src/nanogpt/config.py TailConfig. Loop integration:
src/nanogpt/train.py (PORT CHANGE T9). Three pieces:

- :class:`TailAccumulators` — streaming, spike-gated fp32 means of the
  post-update iterate (W1/W2 half-windows + Polyak-from-start), plus the
  artifact writer. Measurement only; never touches the update path.
- :class:`ScheduleFreeTail` — anchored-averaging state for
  ``tail.mode: schedule_free``: stash z, evaluate gradients at
  y = (1-rho)*z + rho*xbar, restore z, update the equal-weight Polyak mean
  xbar after the optimizer step. The optimizer itself is stock.
- :func:`ramp_chunk_schedule` — the deterministic per-step chunk counts for
  ``tail.mode: batch_ramp``: B(u) = B0/max(w(u), 1/8), w(u) = 1 - 0.95*u
  (0.95 = 1 - the record's min_lr_frac), whole 49,152-token chunks, consuming
  exactly the record's tail token budget.

Everything here operates on ``dict(name -> Parameter)`` of the raw
(uncompiled) model; in-place writes are visible to the compiled wrapper
because storage is shared.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from src.nanogpt.config import NanoGPTConfig, TailConfig

# The pinned record's LR floor; the ramp's w(u) is defined against the RECORD
# schedule (prereg §0.5), not against the running config's min_lr_frac (ramp
# configs pin their own min_lr_frac to 1.0 so the hot phase is constant-LR).
RECORD_MIN_LR_FRAC = 0.05


def ramp_chunk_schedule(cfg: NanoGPTConfig) -> List[int]:
    """Chunk counts per tail step for batch_ramp; sums to the exact budget."""
    rws = cfg.effective_chunks
    total_chunks = (cfg.num_iterations - cfg.tail.start_step) * rws
    ks: List[int] = []
    consumed = 0
    while consumed < total_chunks:
        u = consumed / total_chunks
        w = 1.0 - (1.0 - RECORD_MIN_LR_FRAC) * u
        k = int(round(rws / max(w, 1.0 / 8.0)))
        k = max(rws, min(8 * rws, k))
        k = min(k, total_chunks - consumed)
        ks.append(k)
        consumed += k
    assert sum(ks) == total_chunks
    return ks


class TailAccumulators:
    """Streaming spike-gated means of the post-update iterate (prereg §0.3).

    ``observe`` is called once per trained step >= tail.start_step with that
    step's train loss (any consistent scale — only the z-score matters) AFTER
    ``opt.step()``. The spike gate excludes steps whose loss z-score against a
    running EMA mean/var exceeds ``spike_z`` (one-sided, upward); the first
    ``spike_warmup`` tail steps are always included while the EMA warms up.
    """

    def __init__(self, tail: TailConfig, named_params: Dict[str, Tensor]):
        self.tail = tail
        self._params = named_params
        # Buffers live on CPU: three fp32 model-sized means on a 32 GB card
        # pushed the GPU smoke to 14 MB free at peak. The per-step cost is one
        # ~0.7 GB device-to-host copy + CPU lerps (~0.1-0.3 s), tail-only.
        self.w1: Dict[str, Tensor] = {}
        self.w2: Dict[str, Tensor] = {}
        self.polyak: Dict[str, Tensor] = {}
        self._stage: Dict[str, Tensor] = {}
        self.n1 = self.n2 = self.n_polyak = 0
        self.steps_seen = 0
        self._ema_mean: Optional[float] = None
        self._ema_sq: Optional[float] = None
        self.gate_log: List[Dict[str, Any]] = []

    def _ensure(self, buf: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if not buf:
            for n, p in self._params.items():
                buf[n] = torch.zeros(p.shape, dtype=torch.float32, device="cpu")
        return buf

    @torch.no_grad()
    def observe(self, step: int, train_loss: float) -> None:
        self.steps_seen += 1
        z = 0.0
        if self._ema_mean is not None:
            var = max(self._ema_sq - self._ema_mean**2, 1e-12)
            z = (train_loss - self._ema_mean) / math.sqrt(var)
        included = z <= self.tail.spike_z or self.steps_seen <= self.tail.spike_warmup
        beta = self.tail.spike_beta
        if self._ema_mean is None:
            self._ema_mean, self._ema_sq = train_loss, train_loss**2
        else:
            self._ema_mean = beta * self._ema_mean + (1 - beta) * train_loss
            self._ema_sq = beta * self._ema_sq + (1 - beta) * train_loss**2
        self.gate_log.append(
            {"step": step, "train_loss": train_loss, "z": round(z, 3), "included": included}
        )
        if not included:
            return
        targets = [(self._ensure(self.polyak), "n_polyak")]
        if self.tail.w1_window[0] <= step < self.tail.w1_window[1]:
            targets.append((self._ensure(self.w1), "n1"))
        if self.tail.w2_window[0] <= step < self.tail.w2_window[1]:
            targets.append((self._ensure(self.w2), "n2"))
        # one device-to-host fp32 staging copy per param per step, shared by
        # all destination buffers
        if not self._stage:
            self._stage = {
                n: torch.empty(p.shape, dtype=torch.float32, device="cpu",
                               pin_memory=p.is_cuda)
                for n, p in self._params.items()
            }
        for name, p in self._params.items():
            self._stage[name].copy_(p.detach())
        for buf, counter in targets:
            n = getattr(self, counter) + 1
            setattr(self, counter, n)
            for name in self._params:
                buf[name].lerp_(self._stage[name], 1.0 / n)

    # ------------------------------------------------------- checkpointing
    def state_dict(self) -> Optional[Dict[str, Any]]:
        if self.steps_seen == 0:
            return None
        return {
            "w1": self.w1, "w2": self.w2, "polyak": self.polyak,
            "n1": self.n1, "n2": self.n2, "n_polyak": self.n_polyak,
            "steps_seen": self.steps_seen,
            "ema_mean": self._ema_mean, "ema_sq": self._ema_sq,
            "gate_log": self.gate_log,
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if state is None:
            return
        self.w1, self.w2, self.polyak = state["w1"], state["w2"], state["polyak"]
        self.n1, self.n2, self.n_polyak = state["n1"], state["n2"], state["n_polyak"]
        self.steps_seen = state["steps_seen"]
        self._ema_mean, self._ema_sq = state["ema_mean"], state["ema_sq"]
        self.gate_log = list(state["gate_log"])

    # ------------------------------------------------------------ artifact
    def artifact(self) -> Dict[str, Any]:
        cpu = lambda buf: {n: t.detach().cpu() for n, t in buf.items()}
        return {
            "w1": cpu(self.w1), "w2": cpu(self.w2), "polyak": cpu(self.polyak),
            "final": {n: p.detach().float().cpu() for n, p in self._params.items()},
            "counts": {"n1": self.n1, "n2": self.n2, "n_polyak": self.n_polyak},
            "windows": {"w1": list(self.tail.w1_window), "w2": list(self.tail.w2_window)},
            "gate_log": self.gate_log,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "n1": self.n1, "n2": self.n2, "n_polyak": self.n_polyak,
            "steps_seen": self.steps_seen,
            "excluded": sum(1 for e in self.gate_log if not e["included"]),
        }


class ScheduleFreeTail:
    """Anchored-averaging state for tail.mode schedule_free (prereg §0.4)."""

    def __init__(self, tail: TailConfig, named_params: Dict[str, Tensor]):
        self.rho = tail.rho
        self._params = named_params
        self.xbar = {n: p.detach().float().clone() for n, p in named_params.items()}
        self._stash = {n: torch.empty_like(p) for n, p in named_params.items()}
        self.t = 0
        self._at_xbar = False

    @torch.no_grad()
    def to_y(self) -> None:
        """params: z -> y = (1-rho)*z + rho*xbar (gradient evaluation point)."""
        for n, p in self._params.items():
            self._stash[n].copy_(p)
            p.lerp_(self.xbar[n].to(p.dtype), self.rho)

    @torch.no_grad()
    def to_z(self) -> None:
        """params: restore the stashed iterate z before the optimizer step."""
        for n, p in self._params.items():
            p.copy_(self._stash[n])

    @torch.no_grad()
    def update_average(self) -> None:
        """xbar <- xbar + (z - xbar)/t after the optimizer step (equal weight)."""
        self.t += 1
        for n, p in self._params.items():
            self.xbar[n].lerp_(p.float(), 1.0 / self.t)

    @torch.no_grad()
    def swap_in_xbar(self) -> None:
        """Write xbar into params (for validation-at-xbar); swap_back undoes."""
        assert not self._at_xbar
        self._at_xbar = True
        for n, p in self._params.items():
            self._stash[n].copy_(p)
            p.copy_(self.xbar[n].to(p.dtype))

    @torch.no_grad()
    def swap_back(self) -> None:
        assert self._at_xbar
        self._at_xbar = False
        for n, p in self._params.items():
            p.copy_(self._stash[n])

    def artifact(self) -> Dict[str, Any]:
        return {
            "xbar": {n: t.detach().cpu() for n, t in self.xbar.items()},
            "final_z": {n: p.detach().float().cpu() for n, p in self._params.items()},
            "t": self.t,
            "rho": self.rho,
        }


class ChunkBuffer:
    """Single-chunk-granularity view of RecordDataGenerator for batch_ramp.

    The generator selects BOS-aligned chunk starts in groups of
    ``record_world_size`` (the record's joint selection — the group structure
    is what keeps the token stream identical to the record's); this buffer
    pulls whole groups as needed and yields exactly ``k`` chunks per optimizer
    step. Requires device_count == 1 (validated in config).
    """

    def __init__(self, gen) -> None:
        self._gen = gen
        self._starts: List[int] = []

    def next_chunks(self, k: int):
        for _ in range(k):
            if not self._starts:
                self._starts = list(self._gen._chunk_starts())
            yield self._gen._materialize(self._starts.pop(0))
