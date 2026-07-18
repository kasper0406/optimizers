"""Optimizer interface with per-matrix hooks (WP0.0).

Every optimizer in this repo (Muon, DynMuon, AdaMuon, NorMuon, AdamW wrapper,
Routed Muon) implements :class:`MatrixOptimizer`. The step loop decomposes each
per-parameter update into three hooks so that instrumentation (WP1.1) and
routing (WP2.1) can attach at well-defined points:

    for each parameter W with gradient G:
        state = self.state[W]                      # per-matrix state dict
        M = self.pre_step(G, state, group)         # e.g. momentum update
        O = self.shape_spectrum(M, state, group)   # e.g. Newton-Schulz, gains
        self.post_step(W, O, state, group)         # apply update (lr, wd)

Contract notes:

- ``pre_step`` receives the *raw* gradient G (pre-momentum). Instrumentation
  that needs raw-gradient projections (plan section 1.1) hooks here; the
  returned matrix M is whatever the optimizer shapes (for Muon-family, the
  momentum buffer).
- ``shape_spectrum`` maps M to the update direction O. Stock Muon runs
  Newton-Schulz here; Routed Muon additionally applies per-tracked-direction
  gains; elementwise optimizers do their elementwise math here.
- ``post_step`` applies O to the parameter. The base implementation performs
  decoupled weight decay followed by ``W -= lr * O``. Override for anything
  fancier, and for per-step logging.
- Per-matrix state lives in ``self.state[W]`` (single-writer invariant: in a
  distributed port, all state for a matrix lives on its owner rank).

The class subclasses ``torch.optim.Optimizer`` so trainers get the standard
contract for free: ``zero_grad()``, ``state_dict()`` / ``load_state_dict()``
(checkpoint resumability is required by the plan), and param groups.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional

import torch


class MatrixOptimizer(torch.optim.Optimizer, ABC):
    """Abstract base optimizer with per-matrix hooks.

    Args:
        params: iterable of parameters or param-group dicts (torch convention).
        lr: learning rate.
        weight_decay: decoupled weight decay coefficient (applied in the
            default ``post_step`` before the update).
        **extra_defaults: additional per-group hyperparameters made available
            to hooks via ``group``.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        **extra_defaults: Any,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, weight_decay=weight_decay, **extra_defaults)
        super().__init__(params, defaults)

    # ------------------------------------------------------------------ hooks

    @abstractmethod
    def pre_step(
        self, G: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        """Consume the raw gradient G; return the matrix M to be shaped.

        Typical implementation: update the momentum buffer in ``state`` and
        return it. Must not modify G in place (instrumentation may still need
        the raw gradient this step).
        """

    @abstractmethod
    def shape_spectrum(
        self, O: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        """Map the pre-step output to the final update direction.

        For Muon-family optimizers this is where orthogonalization
        (Newton-Schulz) and any spectral shaping / per-direction gains happen.
        The returned tensor is the update direction O passed to ``post_step``
        (sign convention: the parameter moves along ``-lr * O``).
        """

    def post_step(
        self,
        param: torch.Tensor,
        update: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
    ) -> None:
        """Apply the update to the parameter. Default: decoupled WD + SGD step.

        ``param -= lr * wd * param`` (if wd > 0), then ``param -= lr * update``.
        Override for per-parameter scaling, logging, etc.
        """
        lr = group["lr"]
        wd = group.get("weight_decay", 0.0)
        if wd != 0.0:
            param.mul_(1.0 - lr * wd)
        param.add_(update, alpha=-lr)

    # -------------------------------------------------------------- step loop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        """Run the three-hook loop over every parameter with a gradient."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        f"{type(self).__name__} does not support sparse gradients"
                    )
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                state["step"] += 1
                shaped = self.pre_step(grad, state, group)
                update = self.shape_spectrum(shaped, state, group)
                self.post_step(param, update, state, group)
        return loss


class NoOpOptimizer(MatrixOptimizer):
    """Optimizer that exercises the full hook path but changes nothing.

    ``pre_step`` passes the gradient through, ``shape_spectrum`` maps it to a
    zero update, and the default ``post_step`` applies that zero update (weight
    decay is forced to 0). Used by the WP0.0 smoke config to verify the
    config -> trainer -> optimizer -> results-JSON path end to end.
    """

    def __init__(self, params: Iterable, lr: float = 0.0, **kwargs: Any) -> None:
        kwargs.pop("weight_decay", None)  # a no-op must not decay weights
        super().__init__(params, lr=lr, weight_decay=0.0, **kwargs)

    def pre_step(
        self, G: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        return G

    def shape_spectrum(
        self, O: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        return torch.zeros_like(O)
