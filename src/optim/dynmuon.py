"""DynMuon behind the MatrixOptimizer interface (WP0.4).

DynMuon (vendor/DynMuon, arXiv "DynMuon: A Dynamic Spectral Shaping View of
Muon") replaces Muon's fixed orthogonalization with a *globally scheduled*
spectral shaping exponent p: the update maps singular values sigma -> sigma^p,
with p annealed from p_max to p_min over training by a logistic schedule.

Everything here is matched to the vendored reference:

- Momentum: ``muon_update_pre_orthogonalize``
  (vendor/DynMuon/dynmuon/dynmuon.py:570-599):
      M_t = mu * M_{t-1} + G_t;   U_t = mu * M_t + G_t if nesterov else M_t
- Schedule: ``Logistic_Scheduler`` (dynmuon.py:44-55):
      q = step/total_steps; u = (q - tau_ratio)/max(width_ratio, 1e-8)
      p(step) = p_min + (p_max - p_min) / (1 + exp(u))
  The reference evaluates it at the 1-based step count (dynmuon.py:186-192);
  here ``state["step"]`` is that same 1-based count.
- Spectral transform: ``dynmuon_spectral_transform`` (dynmuon.py:32-39):
      p >= 0.25       -> identity (raw momentum)
      0 <= p < 0.25   -> Newton-Schulz orthogonalization (their coefficient
                         schedule, newton_schulz_triton.py:306-333)
      p < 0           -> sigma -> sigma^p shaping; reference runtime uses the
                         polynomial ``fast_spectral`` (dynmuon.py:717-772,
                         ``spectral_impl="poly"``, the default here); the
                         exact SVD rule (dynmuon.py:684-711) is available as
                         ``spectral_impl="svd"``.
- LR adjustment: default ``adjust_lr="spectral_norm"`` = lr*sqrt(fan_out/
  fan_in) (dynmuon.py:671-680), applied to the step but not the decay.
- Weight decay: X *= (1 - base_lr*wd) then X -= adjusted_lr*U
  (``muon_update_post_orthogonalize``, dynmuon.py:629-635; cautious_wd not
  ported -- reference default is cautious_wd=False).

Defaults (lr=0.01, mu=0.95, wd=0.01, nesterov=False, total_steps=20000,
p_max=1.0, p_min=-0.25, tau_ratio=0.02, width_ratio=0.08) mirror
``DynMuon.__init__`` (dynmuon.py:61-81).

The reference casts the momentum to bfloat16 before the spectral transform
(dynmuon.py:595-597); the identity branch therefore also passes through
bfloat16, which is reproduced here via ``ns_dtype``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import torch

from src.optim.interface import MatrixOptimizer
from src.optim.muon import AdjustedLRPostStepMixin
from src.optim.newton_schulz import (
    dynmuon_fast_spectral,
    dynmuon_newtonschulz,
    spectral_power_via_svd,
)


class LogisticPScheduler:
    """Port of vendor/DynMuon/dynmuon/dynmuon.py:44-55 (``Logistic_Scheduler``)."""

    def __init__(
        self,
        p_max: float = 1.0,
        p_min: float = -0.25,
        tau_ratio: float = 0.02,
        width_ratio: float = 0.08,
    ) -> None:
        self.p_max = p_max
        self.p_min = p_min
        self.tau_ratio = tau_ratio
        self.width_ratio = width_ratio

    def get_p(self, step: int, total_steps: int = 10000) -> float:
        q_t = step / float(total_steps)
        u = (q_t - self.tau_ratio) / max(self.width_ratio, 1e-8)
        anneal = 1.0 / (1.0 + math.exp(u))
        return self.p_min + (self.p_max - self.p_min) * anneal


class DynMuon(AdjustedLRPostStepMixin, MatrixOptimizer):
    """DynMuon: momentum + globally-scheduled spectral shaping sigma -> sigma^p."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = False,
        total_steps: int = 20000,
        p_max: float = 1.0,
        p_min: float = -0.25,
        tau_ratio: float = 0.02,
        width_ratio: float = 0.08,
        adjust_lr: Optional[str] = "spectral_norm",
        spectral_impl: str = "poly",
        eps: float = 1e-8,
        ns_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {momentum}")
        if adjust_lr not in (None, "spectral_norm", "rms_norm"):
            # matches the reference's accepted values (dynmuon.py:89-92)
            raise ValueError(f"Invalid adjust_lr: {adjust_lr!r}")
        if spectral_impl not in ("poly", "svd"):
            raise ValueError(f"Invalid spectral_impl: {spectral_impl!r}")
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            eps=eps,
            adjust_lr=adjust_lr,
        )
        self.total_steps = int(total_steps)
        self.spectral_impl = spectral_impl
        self.ns_dtype = ns_dtype
        self.scheduler = LogisticPScheduler(
            p_max=p_max, p_min=p_min, tau_ratio=tau_ratio, width_ratio=width_ratio
        )

    # ------------------------------------------------------------------ hooks

    def pre_step(
        self, G: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        # dynmuon.py:570-599 (muon_update_pre_orthogonalize)
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(G)
        buf = state["momentum_buffer"]
        buf.mul_(group["momentum"]).add_(G)
        if group["nesterov"]:
            return buf.mul(group["momentum"]).add_(G)
        return buf

    def shape_spectrum(
        self, O: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        p = self.scheduler.get_p(state["step"], self.total_steps)
        state["p"] = p  # logged by instrumentation; occupancy plots use this

        M2 = O.reshape(len(O), -1) if O.ndim > 2 else O
        eps = group["eps"]
        # dynmuon.py:32-39 (dynmuon_spectral_transform)
        if p >= 0.25:
            out = M2.to(self.ns_dtype)
        elif p >= 0.0:
            out = dynmuon_newtonschulz(M2, eps=eps, dtype=self.ns_dtype)
        elif self.spectral_impl == "poly":
            out = dynmuon_fast_spectral(M2, p=p, eps=eps, dtype=self.ns_dtype)
        else:
            out = spectral_power_via_svd(M2, p=p, eps=eps)
        return out.reshape(O.shape).type_as(O)
