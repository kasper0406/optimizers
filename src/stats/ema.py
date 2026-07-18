"""Bias-corrected exponential moving averages.

The EMA recursion is ``raw_t = beta * raw_{t-1} + (1 - beta) * x_t`` with
``raw_0 = 0``.  The raw value is biased toward zero for small t; the standard
Adam-style correction divides by ``1 - beta**t`` so that for a constant input
the corrected estimate equals the input exactly at every t >= 1.

Effective sample size (ESS) of the corrected estimator at step t uses the
Kish formula on the (normalized) EMA weights w_i = (1-beta) beta^{t-i}:

    ess(t) = (sum w)^2 / (sum w^2)
           = (1 - beta^t)^2 (1 + beta) / ((1 - beta) (1 - beta^{2t}))

ess(1) = 1 and ess(t) -> (1 + beta) / (1 - beta) as t -> inf
(19 for beta = 0.9, 199 for beta = 0.99).  For i.i.d. inputs with variance
sigma^2 the corrected mean estimator has variance sigma^2 / ess(t); this
identity is verified empirically in tests/test_stats_ema.py.

All state broadcasts: ``update`` accepts scalars or numpy arrays of a fixed
shape (one independent EMA per element, sharing the step counter).
"""

from __future__ import annotations

import numpy as np

__all__ = ["BiasCorrectedEma", "ema_effective_sample_size"]


def ema_effective_sample_size(t: int, beta: float) -> float:
    """Kish effective sample size of a bias-corrected EMA after t updates."""
    if t <= 0:
        return 0.0
    if beta <= 0.0:
        return 1.0
    bt = beta**t
    return ((1.0 - bt) ** 2 * (1.0 + beta)) / ((1.0 - beta) * (1.0 - bt * bt))


class BiasCorrectedEma:
    """Scalar/array EMA with Adam-style bias correction."""

    def __init__(self, beta: float):
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        self.beta = float(beta)
        self.t = 0
        self._raw = 0.0

    def update(self, x) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.t += 1
        self._raw = self.beta * self._raw + (1.0 - self.beta) * x

    @property
    def raw(self):
        """Uncorrected EMA value (biased toward 0 for small t)."""
        return self._raw

    @property
    def value(self):
        """Bias-corrected estimate; 0 before any update."""
        if self.t == 0:
            return self._raw
        return self._raw / (1.0 - self.beta**self.t)

    @property
    def ess(self) -> float:
        """Effective sample size of the current estimate."""
        return ema_effective_sample_size(self.t, self.beta)

    def reset(self) -> None:
        self.t = 0
        self._raw = 0.0
