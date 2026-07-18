"""AdamW behind the MatrixOptimizer interface (WP0.4).

Wraps the ``torch.optim.AdamW`` update rule (decoupled weight decay,
Loshchilov & Hutter 2019) in the three-hook interface so instrumentation
(WP1.1) can attach at the same points as for the Muon family. The math is
kept exactly equal to ``torch.optim.AdamW`` (amsgrad=False, maximize=False):

    m_t = b1*m_{t-1} + (1-b1)*g_t
    v_t = b2*v_{t-1} + (1-b2)*g_t^2
    W  *= (1 - lr*wd)
    W  -= lr * (m_t / (1-b1^t)) / (sqrt(v_t) / sqrt(1-b2^t) + eps)

- ``pre_step`` updates (m, v) and returns the bias-corrected Adam direction.
- ``shape_spectrum`` is the identity (AdamW does no spectral shaping).
- ``post_step`` is the base-class default (decoupled WD, then -lr * update),
  which is exactly AdamW's decoupled decay.

The unit test (tests/test_optim_adamw.py) asserts bit-level agreement with
``torch.optim.AdamW`` over multiple steps, plus a hand-computed numpy
expectation.

One deviation from torch: moment buffers are kept in float32 even for
half-precision parameters (torch keeps them in the param dtype). This makes
AdamW usable on airbench's half-precision CifarNet; for float32 params the
behavior is identical to torch.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch

from src.optim.interface import MatrixOptimizer


class AdamW(MatrixOptimizer):
    """AdamW expressed through the per-matrix hook interface."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        super().__init__(
            params, lr=lr, weight_decay=weight_decay, betas=tuple(betas), eps=eps
        )

    def pre_step(
        self, G: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        beta1, beta2 = group["betas"]
        t = state["step"]  # interface increments before pre_step; t starts at 1
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(G, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(G, dtype=torch.float32)
        g = G.to(torch.float32)
        m = state["exp_avg"].mul_(beta1).add_(g, alpha=1.0 - beta1)
        v = state["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**t
        bias_correction2 = 1.0 - beta2**t
        # torch.optim.AdamW: denom = sqrt(v)/sqrt(bc2) + eps; step = lr/bc1 * m/denom
        denom = (v.sqrt() / (bias_correction2**0.5)).add_(group["eps"])
        return (m / denom).div_(bias_correction1).to(G.dtype)

    def shape_spectrum(
        self, O: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        return O
