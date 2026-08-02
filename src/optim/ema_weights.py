"""Weight-EMA tracking for the T1 EMA-as-anytime-anneal experiment.

Theory context (docs/litreview/j-theory-theorem-sweep.md, theme T1): iterate
averaging is equivalent to LR decay (Sandler et al. arXiv:2301.02312; Defazio
et al. arXiv:2405.15682), and averaging cancels the lag-1-anticorrelated
oscillation component our Phase-1 measurements found in the majority of
per-direction gradient projections. The testable prediction: an EMA of the
weights evaluated mid-run reaches the accuracy the schedule only reaches
after its anneal.

This module is deliberately harness-agnostic and CPU-testable: it sees a list
of named tensors, keeps fp32 shadows per decay factor, and can temporarily
swap a shadow into the live tensors with bit-exact restore.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


def validate_gammas(gammas: Sequence[float]) -> List[float]:
    """Validate an EMA decay-factor list: non-empty, each strictly in (0, 1),
    no duplicates. Returns the values as floats. Raises SystemExit on invalid
    input (config-validation convention of this repo: fail the run, loudly,
    before any CUDA work)."""
    if not isinstance(gammas, (list, tuple)) or not gammas:
        raise SystemExit("recipe.ema.gammas must be a non-empty list of floats")
    out: List[float] = []
    for g in gammas:
        if not isinstance(g, (int, float)) or isinstance(g, bool):
            raise SystemExit(f"recipe.ema.gammas entries must be numbers, got {g!r}")
        g = float(g)
        if not 0.0 < g < 1.0:
            raise SystemExit(
                f"recipe.ema.gammas entries must be strictly inside (0, 1), got {g}"
            )
        out.append(g)
    if len(set(out)) != len(out):
        raise SystemExit(f"recipe.ema.gammas contains duplicates: {sorted(out)}")
    return out


class WeightEMA:
    """fp32 exponential moving averages of a fixed set of live tensors.

    One shadow set per decay factor gamma; ``update()`` applies
    ``shadow = gamma * shadow + (1 - gamma) * live`` to every tracked tensor.
    Shadows are initialized to the live values at construction time, so the
    EMA is exact from step one (no zero-init bias, no correction needed).

    ``applied(gamma)`` is a context manager that copies the gamma-shadow into
    the live tensors (cast to each tensor's dtype) and restores the exact
    previous live values on exit — the live training state is untouched by an
    eval performed inside the context.
    """

    def __init__(
        self, named_tensors: Iterable[Tuple[str, torch.Tensor]], gammas: Sequence[float]
    ):
        self.gammas: List[float] = validate_gammas(gammas)
        self._tensors: List[Tuple[str, torch.Tensor]] = list(named_tensors)
        if not self._tensors:
            raise SystemExit("WeightEMA needs at least one tensor to track")
        seen = set()
        for name, t in self._tensors:
            if name in seen:
                raise SystemExit(f"WeightEMA got duplicate tensor name {name!r}")
            seen.add(name)
            if not t.is_floating_point():
                raise SystemExit(
                    f"WeightEMA tracks floating tensors only; {name!r} is {t.dtype}"
                )
        self._shadows: Dict[float, List[torch.Tensor]] = {
            g: [t.detach().to(torch.float32).clone() for _, t in self._tensors]
            for g in self.gammas
        }

    @torch.no_grad()
    def update(self) -> None:
        for g, shadows in self._shadows.items():
            for shadow, (_, live) in zip(shadows, self._tensors):
                shadow.mul_(g).add_(live.detach().to(torch.float32), alpha=1.0 - g)

    @contextmanager
    def applied(self, gamma: float):
        if gamma not in self._shadows:
            raise SystemExit(f"WeightEMA has no shadow for gamma={gamma}")
        backups = [t.detach().clone() for _, t in self._tensors]
        try:
            with torch.no_grad():
                for shadow, (_, live) in zip(self._shadows[gamma], self._tensors):
                    live.copy_(shadow.to(live.dtype))
            yield
        finally:
            with torch.no_grad():
                for backup, (_, live) in zip(backups, self._tensors):
                    live.copy_(backup)
