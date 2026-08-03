"""Explicit central-flow term: apply the implicit curvature penalty that an
edge-of-stability oscillation would generate.

Theory (Cohen et al., arXiv:2410.24206; docs/litreview/j-theory-theorem-sweep.md
§6): a large-LR optimizer oscillating along direction ``v_i`` with variance
``sigma_i^2`` follows, on time average, gradient flow plus the drift

    -(sigma_i^2 / 2) * grad_w lambda_i(w),   lambda_i(w) = v_i^T H(w) v_i.

:class:`CentralFlowTerm` caches that drift (computed by
:mod:`src.optim.sharpgrad`) and subtracts it from the parameters, so a
small-LR / non-oscillating optimizer can be driven along the *same* effective
trajectory as the oscillating one -- the interventional test of whether the
central flow is the mechanism behind self-stabilization / progressive
sharpening in our runs.

Composition contract
--------------------
This is deliberately **not** a ``torch.optim.Optimizer``: it composes with any
base optimizer (Muon, AdamW, plain GD) and owns no learning rate, no momentum
and no parameter groups.  The harness drives it:

    term = CentralFlowTerm()
    for step in range(T):
        ...                                    # base optimizer step
        if step % M == 0:                      # refresh cadence
            term.refresh(loss_closure, params, directions, weights, step=step)
        term.apply(params, beta=lr)            # every step

with

* ``directions``  -- the tracked (detached) oscillation directions, in the
  ``src.optim.sharpgrad`` layout: ``directions[i][j]`` is direction ``i``'s
  block for ``params[j]``, ``None`` for uninvolved parameters.  They are
  unit-normalized internally by ``sharpgrad``.
* ``weights``     -- ``w_i = sigma_i^2 / 2``, the measured oscillation
  variance along ``v_i`` (halved), i.e. the central-flow coefficient.
* ``beta``        -- the flow-time increment of one step, i.e. the base
  optimizer's learning rate.  ``apply`` performs
  ``params[j] -= beta * cached_penalty_grads[j]``, so the total displacement
  per step is ``-lr * sum_i (sigma_i^2 / 2) * grad lambda_i``.

Refresh cost is one forward plus a third-order backward chain; it is
amortized over ``M`` steps, and ``apply`` is a plain fused ``add_`` (no
autograd, no HVP in the per-step path -- the same invariant the routed
optimizer is held to).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from src.optim.sharpgrad import Direction, LossClosure, directional_curvature_and_grad

__all__ = ["CentralFlowTerm"]


class CentralFlowTerm:
    """Cached central-flow penalty drift ``grad_w sum_i w_i lambda_i(w)``.

    State: ``penalty_grads`` (list of tensors matching the params of the last
    refresh), ``curvatures`` (the ``lambda_i`` of that refresh) and
    ``step_of_refresh`` (the step index the caller passed, or an internal
    refresh counter if it passed none).
    """

    def __init__(self) -> None:
        self.penalty_grads: Optional[List[torch.Tensor]] = None
        self.curvatures: Optional[List[float]] = None
        self.step_of_refresh: Optional[int] = None
        self.n_refreshes: int = 0
        self._shapes: Optional[List[tuple]] = None

    # ------------------------------------------------------------------ api

    def refresh(
        self,
        loss_fn: LossClosure,
        params: Sequence[torch.Tensor],
        directions: Sequence[Direction],
        weights: Optional[Sequence[float]] = None,
        step: Optional[int] = None,
    ) -> None:
        """Recompute and cache the penalty gradient at the current params.

        ``weights[i]`` is ``sigma_i^2 / 2`` (see the composition contract).
        ``step`` is recorded for logging; when omitted, an internal refresh
        counter is recorded instead.
        """
        plist = list(params)
        curvatures, penalty_grads = directional_curvature_and_grad(
            loss_fn, plist, directions, weights=weights, create_graph_chain=True
        )
        assert penalty_grads is not None  # create_graph_chain=True
        self.curvatures = curvatures
        self.penalty_grads = penalty_grads
        self._shapes = [tuple(p.shape) for p in plist]
        self.step_of_refresh = self.n_refreshes if step is None else int(step)
        self.n_refreshes += 1

    @torch.no_grad()
    def apply(self, params: Sequence[torch.Tensor], beta: float) -> None:
        """``params[j] -= beta * penalty_grads[j]`` for the cached gradient."""
        if self.penalty_grads is None:
            raise RuntimeError(
                "CentralFlowTerm.apply called before refresh(): there is no "
                "cached penalty gradient to apply"
            )
        plist = list(params)
        if len(plist) != len(self.penalty_grads):
            raise ValueError(
                f"CentralFlowTerm.apply got {len(plist)} params but the cached "
                f"penalty gradient has {len(self.penalty_grads)} entries"
            )
        assert self._shapes is not None
        for j, (p, g, shape) in enumerate(zip(plist, self.penalty_grads, self._shapes)):
            if tuple(p.shape) != shape:
                raise ValueError(
                    f"CentralFlowTerm.apply: params[{j}] has shape "
                    f"{tuple(p.shape)}, refreshed with {shape}"
                )
            p.sub_(g.to(dtype=p.dtype, device=p.device), alpha=float(beta))

    def stats(self) -> Dict[str, Any]:
        """Logging record of the cached state (JSON-serializable)."""
        if self.penalty_grads is None:
            return {
                "n_directions": 0,
                "curvatures": [],
                "penalty_grad_norm": 0.0,
                "refresh_step": None,
            }
        sq = 0.0
        for g in self.penalty_grads:
            sq += float((g.detach().to(torch.float64) ** 2).sum())
        assert self.curvatures is not None
        return {
            "n_directions": len(self.curvatures),
            "curvatures": [float(c) for c in self.curvatures],
            "penalty_grad_norm": sq**0.5,
            "refresh_step": self.step_of_refresh,
        }
