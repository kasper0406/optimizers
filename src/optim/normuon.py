"""NorMuon behind the MatrixOptimizer interface (WP0.4).

Matched to Algorithm 1 of the NorMuon paper (arXiv:2510.05491, "NorMuon:
Making Muon more efficient and scalable", verified against the published
PDF, p. 6), for W in R^{m x n}:

    M_t = beta1 * M_{t-1} + (1 - beta1) * G_t         # EMA momentum
    O_t = NS5(M_t)                                    # Newton-Schulz
    v_t = beta2 * v_{t-1} + (1-beta2) * mean_cols(O_t (.) O_t)   # v in R^m
    O^_t = O_t (/) (sqrt(v_t) + eps)                  # row-wise normalization
    eta^ = 0.2 * eta * sqrt(m*n) / ||O^_t||_F         # norm-preserving scale
    W_{t+1} = W_t - eta * lambda * W_t - eta^ * O^_t

Reference-matching notes:
- NS5 uses the classic quintic coefficients (3.4445, -4.7750, 2.0315), 5
  steps -- identical to the airbench reference NS.
- The second moment is *neuron-wise*: one scalar per row (output neuron),
  v in R^m, i.e. mean over columns of O_t^2 (Algorithm 1 line 7). The
  modded-nanogpt production implementation
  (vendor/modded-nanogpt/train_gpt.py:928-943,
  ``_apply_normuon_variance_reduction``) picks the reduced axis adaptively
  and fuses the same normalize-then-restore-norm computation; this class
  follows the paper's fixed row-wise definition.
- Epsilon placement follows the algorithm box (added after the square root,
  Adam-style); default beta values (beta1, beta2) = (0.95, 0.95) follow the
  paper's pretraining experiments (Sec. 4).
- No bias correction appears in Algorithm 1 and none is applied.

Ablation switches (paper behavior by default; both False reduces NorMuon to
reference Muon with nesterov=False -- property-tested, using the fact that
Newton-Schulz is scale-invariant so the (1-beta1) EMA factor drops out):
- ``normalize``: apply the neuron-wise normalization (lines 7-9).
- ``rms_align``: apply the eta^ rescaling (line 10); when False the update
  is applied at plain lr like Muon.

Per-neuron second-moment state is float32 (m scalars per matrix).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable

import torch

from src.optim.interface import MatrixOptimizer
from src.optim.newton_schulz import zeropower_via_newtonschulz5


class NorMuon(MatrixOptimizer):
    """NorMuon: Muon + neuron-wise post-orthogonalization normalization."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        beta1: float = 0.95,
        beta2: float = 0.95,
        eps: float = 1e-8,
        ns_steps: int = 5,
        normalize: bool = True,
        rms_align: bool = True,
        ns_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid betas: ({beta1}, {beta2})")
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            ns_steps=ns_steps,
        )
        self.normalize = normalize
        self.rms_align = rms_align
        self.ns_dtype = ns_dtype

    # ------------------------------------------------------------------ hooks

    def pre_step(
        self, G: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        # Algorithm 1 line 5: M_t = beta1*M_{t-1} + (1-beta1)*G_t
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(G)
        buf = state["momentum_buffer"]
        buf.mul_(group["beta1"]).add_(G, alpha=1.0 - group["beta1"])
        return buf

    def shape_spectrum(
        self, O: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        M2 = O.reshape(len(O), -1) if O.ndim > 2 else O

        # Algorithm 1 line 6: O_t = NS5(M_t)
        ortho = zeropower_via_newtonschulz5(
            M2, steps=group["ns_steps"], dtype=self.ns_dtype
        ).to(torch.float32)

        if self.normalize:
            # Algorithm 1 line 7: v_t = beta2*v_{t-1} + (1-beta2)*mean_cols(O^2)
            if "neuron_second_moment" not in state:
                state["neuron_second_moment"] = torch.zeros(
                    (M2.shape[0], 1), device=M2.device, dtype=torch.float32
                )
            v = state["neuron_second_moment"]
            v.mul_(group["beta2"]).add_(
                ortho.square().mean(dim=1, keepdim=True), alpha=1.0 - group["beta2"]
            )
            # Algorithm 1 lines 8-9: row-wise normalization
            ortho = ortho / (v.sqrt() + group["eps"])

        return ortho.reshape(O.shape).type_as(O)

    def post_step(
        self,
        param: torch.Tensor,
        update: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
    ) -> None:
        lr = group["lr"]
        wd = group.get("weight_decay", 0.0)
        # Algorithm 1 line 11: W - eta*lambda*W (decoupled decay at base lr)
        if wd != 0.0:
            param.mul_(1.0 - lr * wd)
        if self.rms_align:
            # Algorithm 1 line 10: eta^ = 0.2*eta*sqrt(mn)/||O^_t||_F
            norm = update.to(torch.float32).norm()
            effective_lr = 0.2 * lr * math.sqrt(update.numel()) / float(norm)
        else:
            effective_lr = lr
        param.add_(update, alpha=-effective_lr)
