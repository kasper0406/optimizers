"""AdaMuon behind the MatrixOptimizer interface (WP0.4).

Matched to Algorithm 1 of the AdaMuon paper (arXiv:2507.11005, "The AdaMuon
Optimizer", verified against the published PDF):

    M_t = beta * M_{t-1} + G_t                       # Muon-style momentum
    O_t = NewtonSchulz(Sign(M_t), T)                 # sign-stabilized orthog.
    V_t = beta * V_{t-1} + (1 - beta) * O_t (.) O_t  # element-wise 2nd moment
    O^_t = O_t (/) (sqrt(V_t) + eps * 1)
    gamma_t = 0.2 * sqrt(m*n) / ||O^_t||_F           # RMS-aligned rescaling
    W_{t+1} = W_t - eta * (gamma_t * O^_t + lambda * W_t)

Paper specifics honored here:
- The algorithm box uses a *single* beta for both moments (default 0.95 in
  their experiments); ``beta2`` may override the second-moment decay.
- No bias correction on V_t or M_t: the paper's Appendix B shows the RMS
  alignment cancels any constant multiplicative bias.
- Newton-Schulz uses the classic quintic coefficients a=3.4445, b=-4.7750,
  c=2.0315 with T=5 (paper Sec. 2.1) -- identical to the vendored airbench
  reference (vendor/airbench/airbench94_muon.py:43).
- No nesterov variant appears in the paper, so none is offered.
- Final update: base-class post_step gives W*(1-eta*lambda) - eta*update,
  which equals the paper's W - eta*(gamma*O^ + lambda*W) exactly.

Ablation switches (each defaults to the paper's behavior, all False reduces
AdaMuon to reference Muon with nesterov=False -- property-tested):
- ``sign_stabilize``: feed Sign(M_t) vs M_t into Newton-Schulz.
- ``adaptive``: apply the element-wise second-moment normalization.
- ``rms_align``: apply the gamma_t rescaling.

Second-moment state is kept in float32 regardless of parameter dtype
(V is accumulated from the NS output, which the reference NS computes in
bfloat16; float32 accumulation avoids compounding rounding).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import torch

from src.optim.interface import MatrixOptimizer
from src.optim.newton_schulz import zeropower_via_newtonschulz5


class AdaMuon(MatrixOptimizer):
    """AdaMuon: sign-stabilized Muon + element-wise second moment + RMS alignment."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        beta2: Optional[float] = None,
        eps: float = 1e-8,
        ns_steps: int = 5,
        sign_stabilize: bool = True,
        adaptive: bool = True,
        rms_align: bool = True,
        ns_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if beta2 is not None and not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            beta2=momentum if beta2 is None else beta2,
            eps=eps,
            ns_steps=ns_steps,
        )
        self.sign_stabilize = sign_stabilize
        self.adaptive = adaptive
        self.rms_align = rms_align
        self.ns_dtype = ns_dtype

    # ------------------------------------------------------------------ hooks

    def pre_step(
        self, G: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        # Algorithm 1: M_t = beta * M_{t-1} + G_t
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(G)
        buf = state["momentum_buffer"]
        buf.mul_(group["momentum"]).add_(G)
        return buf

    def shape_spectrum(
        self, O: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        M2 = O.reshape(len(O), -1) if O.ndim > 2 else O

        # Algorithm 1: O_t = NewtonSchulz(Sign(M_t), T)
        ns_in = torch.sign(M2) if self.sign_stabilize else M2
        ortho = zeropower_via_newtonschulz5(
            ns_in, steps=group["ns_steps"], dtype=self.ns_dtype
        ).to(torch.float32)

        if self.adaptive:
            # Algorithm 1: V_t = beta*V_{t-1} + (1-beta)*O_t(.)O_t (no bias corr.)
            if "second_moment" not in state:
                state["second_moment"] = torch.zeros_like(ortho)
            V = state["second_moment"]
            V.mul_(group["beta2"]).addcmul_(ortho, ortho, value=1.0 - group["beta2"])
            # Algorithm 1: O^_t = O_t (/) (sqrt(V_t) + eps*1)
            out = ortho / (V.sqrt() + group["eps"])
        else:
            out = ortho

        if self.rms_align:
            # Algorithm 1: gamma_t = 0.2*sqrt(mn)/||O^_t||_F
            m, n = M2.shape
            gamma = 0.2 * math.sqrt(m * n) / out.norm()
            out = out * gamma

        return out.reshape(O.shape).type_as(O)
