"""Reference Muon behind the MatrixOptimizer interface (WP0.4).

Update rule (matched to vendor/airbench/airbench94_muon.py:56-85, the
vendored reference; modded-nanogpt's historical Muon used the same momentum +
quintic Newton-Schulz with identical coefficients):

    buf_t = momentum * buf_{t-1} + G_t                      (airbench line 80)
    M_t   = G_t + momentum * buf_t   if nesterov else buf_t (airbench line 81)
    O_t   = NewtonSchulz5(M_t.reshape(m, -1)).view_as(M_t)  (airbench line 84)
    W    -= lr * O_t                                        (airbench line 85)

Notes on deliberate differences from the airbench script:
- airbench renormalizes the *weights* each step (line 83,
  ``p.data.mul_(len(p.data)**0.5 / p.data.norm())``). That is part of the
  airbench94 training recipe, not of Muon itself; the airbench harness in
  ``src/optim/airbench_zoo.py`` applies it recipe-side so every zoo optimizer
  gets the identical treatment.
- ``ns_steps`` defaults to 5 (modded-nanogpt-era Muon); airbench uses 3 --
  the airbench smoke config sets ``ns_steps: 3`` explicitly.
- optional ``adjust_lr`` in {None, "spectral_norm", "rms_norm"} ports
  DynMuon's LR adjustment (vendor/DynMuon/dynmuon/dynmuon.py:658-680) so
  Muon-vs-DynMuon comparisons can be run at matched conventions. Default None
  (plain lr, airbench behavior).
- decoupled weight decay uses the base-class post_step convention
  ``W *= (1 - lr*wd)`` with the *unadjusted* lr, matching DynMuon's
  ``muon_update_post_orthogonalize`` (dynmuon.py:629-635). Default wd=0
  (airbench Muon has no weight decay).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import torch

from src.optim.interface import MatrixOptimizer
from src.optim.newton_schulz import zeropower_via_newtonschulz5


def adjusted_lr_for_shape(lr: float, shape: torch.Size, mode: Optional[str]) -> float:
    """LR adjustment ported from vendor/DynMuon/dynmuon/dynmuon.py:658-680.

    Shapes with ndim > 2 are treated as flattened to (shape[0], prod(rest)),
    consistent with how the zoo optimizers flatten before Newton-Schulz.
    """
    if mode is None:
        return lr
    fan_out = shape[0]
    fan_in = math.prod(shape[1:])
    if mode == "spectral_norm":
        # dynmuon.py:671-680 -- adjust from spectral norm 1 to RMS operator norm 1
        return lr * math.sqrt(fan_out / fan_in)
    if mode == "rms_norm":
        # dynmuon.py:658-668 -- constant element-wise RMS norm
        return lr * 0.2 * math.sqrt(max(fan_out, fan_in))
    raise ValueError(f"Unknown adjust_lr mode: {mode!r}")


class AdjustedLRPostStepMixin:
    """post_step shared by Muon-family optimizers: decoupled WD at base lr,
    parameter step at the (optionally) adjusted lr."""

    def post_step(
        self,
        param: torch.Tensor,
        update: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
    ) -> None:
        lr = group["lr"]
        wd = group.get("weight_decay", 0.0)
        if wd != 0.0:
            param.mul_(1.0 - lr * wd)
        alpha = adjusted_lr_for_shape(lr, param.shape, group.get("adjust_lr"))
        param.add_(update, alpha=-alpha)


class Muon(AdjustedLRPostStepMixin, MatrixOptimizer):
    """Reference Muon: SGD-momentum + Newton-Schulz orthogonalization."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
        adjust_lr: Optional[str] = None,
        ns_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        # Validation mirrors vendor/airbench/airbench94_muon.py:58-63.
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        if adjust_lr not in (None, "spectral_norm", "rms_norm"):
            raise ValueError(f"Invalid adjust_lr: {adjust_lr!r}")
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            adjust_lr=adjust_lr,
        )
        self.ns_dtype = ns_dtype

    # ------------------------------------------------------------------ hooks

    def pre_step(
        self, G: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        # airbench lines 77-81
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(G)
        buf = state["momentum_buffer"]
        buf.mul_(group["momentum"]).add_(G)
        if group["nesterov"]:
            return G.add(buf, alpha=group["momentum"])
        return buf

    def shape_spectrum(
        self, O: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        # airbench line 84: flatten >2D params (conv filters) to 2D
        M2 = O.reshape(len(O), -1) if O.ndim > 2 else O
        out = zeropower_via_newtonschulz5(
            M2, steps=group["ns_steps"], eps=group["eps"], dtype=self.ns_dtype
        )
        return out.reshape(O.shape).type_as(O)
