"""Gradient of directional curvature — the primitive behind the central-flow
implicit-sharpness penalty.

Theory context (docs/litreview/j-theory-theorem-sweep.md §6; Cohen et al.,
"Understanding Optimization in Deep Learning with Central Flows",
arXiv:2410.24206): an optimizer oscillating at the edge of stability follows,
on time average, gradient flow PLUS an implicit curvature penalty.  An
oscillation of variance sigma_i^2 along a (fixed) direction v_i contributes a
drift

    dw/dt |_penalty = -(sigma_i^2 / 2) * grad_w lambda_i(w),
    lambda_i(w) = v_i^T H(w) v_i,   H = grad^2 L(w),

i.e. the time-averaged trajectory descends the *directional curvature* on top
of descending the loss.  This module computes that gradient explicitly, so a
non-oscillating (small-LR) optimizer can be given the drift term that a
large-LR oscillating optimizer generates implicitly.

The quantity is third order in the loss: differentiate the Hessian-vector
product a second time, with v_i held as a DETACHED constant.  Concretely, for
each direction i,

    g      = dL/dw                              (create_graph=True)
    gv_i   = <g, v_i>                           (scalar, still in the graph)
    Hv_i   = d gv_i / dw                        (create_graph=True)
    lam_i  = <Hv_i, v_i>                        (scalar, still in the graph)
    pen    = d (sum_i w_i lam_i) / dw           (third-order grad)

Relation to ``src/instrument/hvp.py``: that module computes the same
``lambda = v^T H v`` for *measurement only* (Phase-1 calibration, explicitly
forbidden in any update path), on a detached fp32 functional copy of the
airbench model.  This module is the *update-path* counterpart for the
central-flow experiment: it also returns ``grad_w lambda``, and it
differentiates the caller's own loss closure directly.  The two share the
double-backward convention (unit-norm direction, contraction of Hv with v)
but not the code, because their graph requirements differ (measurement never
needs the third-order chain).

Conventions
-----------
* ``params``: a list of tensors that all require grad (asserted).  Tensors
  with ``requires_grad=False`` must be excluded by the caller.
* ``directions``: a list of direction-sets.  ``directions[i][j]`` is the
  tensor of direction ``i`` for ``params[j]``, with the same shape as
  ``params[j]``; ``None`` means "this direction does not touch that
  parameter" (equivalent to zeros, but skipped rather than multiplied).
  Every direction-set must have exactly ``len(params)`` entries.
* **Direction normalization (documented choice): directions are normalized
  INTERNALLY to unit global L2 norm** over the concatenation of their
  per-parameter blocks.  ``lambda_i`` is therefore always the Rayleigh
  quotient ``v^T H v / ||v||^2`` and is invariant to the scale of the passed
  direction; callers may pass unnormalized directions (e.g. raw power-
  iteration output) without changing the result.  An all-zero direction is a
  caller error and raises ``ValueError``.
* Dtype: the autograd passes run in the parameters' own dtype (the loss
  closure decides that); the scalar contractions ``<g, v>`` and ``<Hv, v>``
  are accumulated in at least float32 (half/bfloat16 are promoted to float32,
  float64 is preserved).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "LossClosure",
    "Direction",
    "directional_curvature",
    "directional_curvature_and_grad",
]

# A closure that runs the forward pass and returns a scalar loss tensor.
LossClosure = Callable[[], torch.Tensor]
# One direction: one entry per parameter, ``None`` for uninvolved parameters.
Direction = Sequence[Optional[torch.Tensor]]


def _accum_dtype(dtype: torch.dtype) -> torch.dtype:
    """Scalar-accumulation dtype: promote half/bfloat16 to float32, keep
    float32/float64 as they are (float64 tests need the extra precision)."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _check_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    plist = list(params)
    if not plist:
        raise ValueError("directional curvature needs at least one parameter")
    for j, p in enumerate(plist):
        if not isinstance(p, torch.Tensor):
            raise ValueError(f"params[{j}] is not a tensor: {type(p)!r}")
        if not p.requires_grad:
            raise ValueError(
                f"params[{j}] (shape {tuple(p.shape)}) has requires_grad=False; "
                "exclude non-differentiable parameters before calling"
            )
        if not p.is_floating_point():
            raise ValueError(f"params[{j}] must be floating point, got {p.dtype}")
    return plist


def _global_accum_dtype(params: Sequence[torch.Tensor]) -> torch.dtype:
    """Widest scalar-accumulation dtype over a (possibly mixed) param list."""
    acc = torch.float32
    for p in params:
        acc = torch.promote_types(acc, _accum_dtype(p.dtype))
    return acc


def _prepare_direction(
    direction: Direction,
    params: List[torch.Tensor],
    index: int,
    acc_dtype: torch.dtype,
) -> List[Optional[torch.Tensor]]:
    """Validate, detach and unit-normalize one direction-set.

    Returns per-parameter blocks in the scalar-accumulation dtype (``None``
    kept as ``None``), scaled so the global L2 norm over all blocks is 1.
    """
    blocks = list(direction)
    if len(blocks) != len(params):
        raise ValueError(
            f"directions[{index}] has {len(blocks)} entries but there are "
            f"{len(params)} params (use None for uninvolved params)"
        )
    prepared: List[Optional[torch.Tensor]] = []
    sq_norm = 0.0
    for j, (v, p) in enumerate(zip(blocks, params)):
        if v is None:
            prepared.append(None)
            continue
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"directions[{index}][{j}] is not a tensor: {type(v)!r}")
        if tuple(v.shape) != tuple(p.shape):
            raise ValueError(
                f"directions[{index}][{j}] has shape {tuple(v.shape)}, expected "
                f"{tuple(p.shape)} (params[{j}])"
            )
        v = v.detach().to(device=p.device, dtype=acc_dtype)
        sq_norm += float((v * v).sum())
        prepared.append(v)
    if sq_norm <= 0.0:
        raise ValueError(
            f"directions[{index}] has zero norm; a direction must be nonzero"
        )
    scale = 1.0 / (sq_norm**0.5)
    return [None if v is None else v * scale for v in prepared]


def _scalar_dot(
    tensors: Sequence[Optional[torch.Tensor]],
    direction: Sequence[Optional[torch.Tensor]],
    acc_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """<tensors, direction> accumulated in ``acc_dtype`` (graph preserved)."""
    total: Optional[torch.Tensor] = None
    for t, v in zip(tensors, direction):
        if t is None or v is None:
            continue
        term = (t.to(acc_dtype) * v).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), dtype=acc_dtype, device=device)
    return total


def directional_curvature_and_grad(
    loss_fn: LossClosure,
    params: Sequence[torch.Tensor],
    directions: Sequence[Direction],
    weights: Optional[Sequence[float]] = None,
    create_graph_chain: bool = True,
) -> Tuple[List[float], Optional[List[torch.Tensor]]]:
    """Directional curvatures and the gradient of their weighted sum.

    Parameters
    ----------
    loss_fn:
        Closure ``() -> scalar tensor``.  Called exactly once; it must build a
        differentiable graph w.r.t. ``params``.
    params:
        Leaf parameters, all with ``requires_grad=True`` (asserted).
    directions:
        ``directions[i][j]`` is direction ``i``'s block for ``params[j]``
        (``None`` = uninvolved).  Normalized internally to unit global norm.
    weights:
        Per-direction ``w_i`` (default 1.0 each), the ``sigma_i^2 / 2``
        coefficients of the central flow when used as a penalty.
    create_graph_chain:
        Build the third-order chain.  ``True`` (default) returns the penalty
        gradient; ``False`` returns ``None`` for it and skips the extra graph
        (measurement-only; see :func:`directional_curvature`).

    Returns
    -------
    ``(curvatures, penalty_grads)`` where ``curvatures[i] = lambda_i =
    v_i^T H v_i`` (Python floats, ``v_i`` unit) and ``penalty_grads[j] =
    d/d params[j] sum_i w_i lambda_i(w)`` (tensors matching ``params``,
    detached, zeros where the parameter does not enter).  ``penalty_grads``
    is ``None`` when ``create_graph_chain=False``.
    """
    plist = _check_params(params)
    dirs = list(directions)
    if not dirs:
        raise ValueError("directional curvature needs at least one direction")
    if weights is None:
        wlist = [1.0] * len(dirs)
    else:
        wlist = [float(w) for w in weights]
        if len(wlist) != len(dirs):
            raise ValueError(
                f"weights has {len(wlist)} entries but there are {len(dirs)} directions"
            )
    acc_dtype = _global_accum_dtype(plist)
    device = plist[0].device
    prepared = [_prepare_direction(d, plist, i, acc_dtype) for i, d in enumerate(dirs)]

    loss = loss_fn()
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("loss_fn must return a scalar tensor")
    grads = torch.autograd.grad(loss, plist, create_graph=True, allow_unused=True)

    curvatures: List[float] = []
    total: Optional[torch.Tensor] = None
    for i, direction in enumerate(prepared):
        gv = _scalar_dot(grads, direction, acc_dtype, device)
        if not gv.requires_grad:
            # The loss is (locally) independent of this direction: H v = 0.
            curvatures.append(0.0)
            continue
        hv = torch.autograd.grad(
            gv,
            plist,
            create_graph=create_graph_chain,
            retain_graph=True,
            allow_unused=True,
        )
        lam = _scalar_dot(hv, direction, acc_dtype, device)
        curvatures.append(float(lam.detach()))
        if create_graph_chain and lam.requires_grad:
            term = lam if wlist[i] == 1.0 else wlist[i] * lam
            total = term if total is None else total + term

    if not create_graph_chain:
        return curvatures, None

    if total is None or not total.requires_grad:
        # Quadratic loss (constant Hessian) or no direction reaching a param:
        # the third-order term vanishes identically.
        return curvatures, [torch.zeros_like(p) for p in plist]

    raw = torch.autograd.grad(total, plist, allow_unused=True)
    penalty = [
        torch.zeros_like(p) if g is None else g.detach() for g, p in zip(raw, plist)
    ]
    return curvatures, penalty


def directional_curvature(
    loss_fn: LossClosure,
    params: Sequence[torch.Tensor],
    directions: Sequence[Direction],
) -> List[float]:
    """Measurement-only ``lambda_i = v_i^T H v_i`` (no third-order graph).

    Same conventions as :func:`directional_curvature_and_grad` (directions
    normalized internally to unit global norm); cheaper because the second
    backward is taken with ``create_graph=False``.
    """
    curvatures, _ = directional_curvature_and_grad(
        loss_fn, params, directions, weights=None, create_graph_chain=False
    )
    return curvatures
