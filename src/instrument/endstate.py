"""Program #13 endpoint tooling: full-network top Hessian eigenvalue.

Pre-registered definition (reports/endstate-prereg.md §2): O4 = top
eigenvalue of the full-network Hessian of the fp32 functional loss
(the documented detached-fp32 / functional_call pattern from
src/instrument/hvp.py — the fp16 training graph is never differentiated
twice), on a fixed batch, via power iteration with a fixed probe seed.

The forward that builds the differentiable graph runs ONCE; all HVP
matvecs reuse it (BatchNorm running-stat updates land on the discarded
fp32 copies during that single forward, so every matvec sees the same
loss surface).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch

from src.instrument.hvp import default_ce_loss, fp32_functional_loss, fp32_overrides


def _flat(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def power_iteration_top_eig(
    hvp_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    iters: int = 20,
    seed: int = 20260722,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[float, torch.Tensor]:
    """Top |eigenvalue| of a symmetric operator given a matvec.

    Returns (rayleigh, v). The Rayleigh quotient carries the sign of the
    dominant-|.|_ eigenvalue.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(dim, generator=g, dtype=dtype)
    if device is not None:
        v = v.to(device)
    v = v / v.norm()
    for _ in range(iters):
        hv = hvp_fn(v)
        n = hv.norm()
        if n == 0:
            return 0.0, v
        v = hv / n
    hv = hvp_fn(v)
    return float(torch.dot(v, hv).item()), v


def endpoint_lambda1(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    iters: int = 20,
    seed: int = 20260722,
    loss_fn=None,
) -> float:
    """Full-network top Hessian eigenvalue at the model's current weights."""
    loss_fn = loss_fn or default_ce_loss()
    params = [p for p in model.parameters() if p.requires_grad]
    overrides, leaves = fp32_overrides(
        model, grad_param_ids={id(p) for p in params}
    )
    leaf_list = [leaves[id(p)] for p in params]
    loss = fp32_functional_loss(model, overrides, inputs, labels, loss_fn)
    grads = torch.autograd.grad(loss, leaf_list, create_graph=True)
    flat_grad = _flat(list(grads))
    dim = flat_grad.numel()

    def hvp_fn(v: torch.Tensor) -> torch.Tensor:
        hv = torch.autograd.grad(
            torch.dot(flat_grad, v), leaf_list, retain_graph=True
        )
        return _flat([h.detach() for h in hv])

    lam, _ = power_iteration_top_eig(
        hvp_fn, dim, iters=iters, seed=seed, device=flat_grad.device
    )
    return lam
