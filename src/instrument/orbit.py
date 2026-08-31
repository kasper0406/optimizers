"""Loss-invariant symmetry orbits: conv->BatchNorm channel rescaling.

WHAT THIS MEASURES
------------------
For a Conv2d whose output feeds a BatchNorm2d, multiplying output channel c's
filter ``W_c`` (and that channel's conv bias, if any) by any ``alpha_c > 0``
leaves the TRAIN-MODE loss invariant: BatchNorm's batch statistics normalize
the scale straight back out, exactly up to the BN ``eps``.  The set of such
rescalings is a closed-form, loss-invariant symmetry ORBIT of the parameter
space.

Along the orbit the loss is constant but the GRADIENT is not.  Because
``L(alpha . W) = L(W)`` identically, differentiating in ``W_c`` gives

    grad_{W_c} L(alpha . W) = (1 / alpha_c) * grad_{W_c} L(W)

so each output-channel row of the reshaped gradient matrix scales as
``1/alpha_c`` while the loss does not move.  Gradient "size" therefore varies
along a level set, and the amount of that variation is exactly the quantity
symmetry-teleportation theory says decides whether teleportation can
accelerate anything (Zhao et al. arXiv:2205.10637 / arXiv:2305.13404; sharpest
conditions in Mishkin et al. arXiv:2403.03362).

For Muon the relevant notion of gradient size is not the Euclidean norm but
the per-matrix NUCLEAR norm: one Muon step's first-order loss decrease is
``<G, polar(G)> = ||G||_*`` (sum of singular values of ``G`` reshaped to
``[out_channels, -1]``).  Both norms are reported; the nuclear one is the one
a teleport-inside-Muon step would be buying.

This module is harness-agnostic and CPU-testable: it knows about
``torch.nn`` modules and nothing about airbench, CUDA, or the training loop.
The gate experiment that uses it is
``src.optim.airbench_zoo.run_airbench_teleport_gate`` (flow-first program item
4, docs/litreview/j-theory-theorem-sweep.md section 6).

RESTORING STATE
---------------
:func:`apply_channel_scales` is in-place and its exact inverse is
:func:`invert_channel_scales` (apply ``1/alpha``).  Multiplicative inversion is
only approximate in floating point (and in half precision it is visibly
lossy), so experiments that need the ORIGINAL state back must restore from
``src.optim.train_snapshot`` instead; the inverse here exists for tests and
for callers that only need approximate undo.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

# Modules that may sit between a Conv2d and its BatchNorm2d without breaking
# the symmetry: every one of these commutes with a POSITIVE per-channel scale
# (max/avg pooling, identity, dropout at eval-scaling).  Anything else (an
# activation, another conv, a flatten) breaks it, and the conv is then not a
# member of a conv->BN pair.
SCALE_COMMUTING_MODULES = (
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.Identity,
    nn.Dropout,
    nn.Dropout2d,
)

ConvBnPair = Tuple[nn.Conv2d, nn.BatchNorm2d, str]


# --------------------------------------------------------------- orbit finding


def find_conv_bn_pairs(model: nn.Module) -> List[ConvBnPair]:
    """Every ``(conv, bn, qualified_name)`` whose conv output feeds that BN.

    The walk is structural: inside each container module the immediate
    children are scanned in REGISTRATION order, and a ``Conv2d`` is paired with
    the next child that is not in :data:`SCALE_COMMUTING_MODULES`, if that
    child is a ``BatchNorm2d`` with matching ``num_features``.  For
    ``nn.Sequential`` registration order is exactly forward order; for hand
    written containers it is the order the submodules were assigned in
    ``__init__``.

    That covers the vendored CifarNet (vendor/airbench/airbench94_muon.py):

    * ``CifarNet.whiten`` is followed by ``layers`` (a Sequential starting with
      GELU), so the whitening conv is correctly EXCLUDED -- its output does not
      reach a BatchNorm before a nonlinearity;
    * each ``ConvGroup`` registers ``conv1, pool, norm1, conv2, norm2, activ``
      and runs ``conv1 -> pool -> norm1 -> activ -> conv2 -> norm2 -> activ``,
      so both convs pair (``conv1`` across the intervening MaxPool2d, which
      commutes with a positive channel scale) and the trailing shared GELU
      pairs with nothing.

    Six pairs result for CifarNet: ``layers.{1,2,3}.conv{1,2}``.  Convs not
    followed by a BatchNorm are excluded.  A module registered twice is paired
    once (deduplicated by identity).

    Only DIRECT siblings are considered; a conv followed by a nested container
    that happens to start with a BatchNorm is deliberately not paired.  Any
    mis-identification is caught empirically downstream: the gate experiment
    records the realized loss change of every orbit move, which is ~0 only if
    the move really was on the orbit.
    """
    pairs: List[ConvBnPair] = []
    seen: set = set()
    for prefix, container in model.named_modules():
        children = list(container.named_children())
        for i, (name, child) in enumerate(children):
            if not isinstance(child, nn.Conv2d) or id(child) in seen:
                continue
            j = i + 1
            while j < len(children) and isinstance(
                children[j][1], SCALE_COMMUTING_MODULES
            ):
                j += 1
            if j >= len(children):
                continue
            bn = children[j][1]
            if not isinstance(bn, nn.BatchNorm2d):
                continue
            if bn.num_features != child.out_channels:
                continue
            seen.add(id(child))
            pairs.append((child, bn, f"{prefix}.{name}" if prefix else name))
    return pairs


def pair_names(pairs: Sequence[ConvBnPair]) -> List[str]:
    """Qualified conv names, in pair order (results-JSON provenance)."""
    return [name for _conv, _bn, name in pairs]


# ------------------------------------------------------------- orbit movement


def _check_scales(pairs: Sequence[ConvBnPair], scales: Sequence[torch.Tensor]) -> None:
    if len(scales) != len(pairs):
        raise ValueError(
            f"expected {len(pairs)} scale vectors (one per pair), got {len(scales)}"
        )
    for (conv, _bn, name), s in zip(pairs, scales):
        s = torch.as_tensor(s)
        if s.ndim != 1 or s.numel() != conv.weight.shape[0]:
            raise ValueError(
                f"scales for {name} must be 1-D of length {conv.weight.shape[0]}, "
                f"got shape {tuple(s.shape)}"
            )
        if not bool(torch.all(s > 0)):
            raise ValueError(f"scales for {name} must be strictly positive")


@torch.no_grad()
def apply_channel_scales(
    pairs: Sequence[ConvBnPair], scales: Sequence[torch.Tensor]
) -> None:
    """Move along the orbit IN PLACE: ``W_c *= alpha_c`` (and bias_c, if any).

    ``scales`` is one positive 1-D tensor per pair, of length
    ``conv.out_channels``.  Scales are cast to each conv's device/dtype, so the
    caller can keep them on CPU in float32 while the model is half precision on
    CUDA.  The BatchNorm's own parameters are untouched -- it is the batch
    statistics, not the affine parameters, that absorb the scale.
    """
    _check_scales(pairs, scales)
    for (conv, _bn, _name), s in zip(pairs, scales):
        s = torch.as_tensor(s)
        w = conv.weight
        view = (-1,) + (1,) * (w.ndim - 1)
        w.mul_(s.to(device=w.device, dtype=w.dtype).view(view))
        if conv.bias is not None:
            conv.bias.mul_(s.to(device=conv.bias.device, dtype=conv.bias.dtype))


@torch.no_grad()
def invert_channel_scales(
    pairs: Sequence[ConvBnPair], scales: Sequence[torch.Tensor]
) -> None:
    """Approximate inverse of :func:`apply_channel_scales` (applies 1/alpha).

    Floating-point round-trip only; experiments restore the exact pre-move
    state from ``src.optim.train_snapshot`` instead (see module docstring).
    """
    _check_scales(pairs, scales)
    apply_channel_scales(pairs, [1.0 / torch.as_tensor(s).float() for s in scales])


def sample_log_uniform_scales(
    pairs: Sequence[ConvBnPair],
    spread: float,
    generator: Optional[torch.Generator] = None,
) -> List[torch.Tensor]:
    """Per-channel scales, log-uniform in ``[1/spread, spread]``.

    Drawn on CPU in float32 (so a CPU ``torch.Generator`` makes the draw
    reproducible regardless of where the model lives);
    :func:`apply_channel_scales` casts them to the model's device/dtype.
    """
    spread = float(spread)
    if not spread > 1.0:
        raise ValueError(f"spread must be > 1, got {spread}")
    log_spread = math.log(spread)
    out = []
    for conv, _bn, _name in pairs:
        n = int(conv.weight.shape[0])
        u = torch.rand(n, generator=generator, dtype=torch.float32)
        out.append(torch.exp((2.0 * u - 1.0) * log_spread))
    return out


# ----------------------------------------------------------------- grad sizes


def grad_norms(
    model: nn.Module, pairs: Sequence[ConvBnPair]
) -> Dict[str, Any]:
    """Gradient size at the current point: Euclidean total + per-pair norms.

    Returns

    ``total``
        Euclidean (l2) norm over ALL parameters that currently carry a
        gradient -- the whole-model gradient size teleportation theory talks
        about.
    ``fro`` / ``nuclear``
        Per pair, the Frobenius and NUCLEAR norms of that conv weight's
        gradient reshaped to ``[out_channels, -1]``.  The nuclear norm is
        Muon's one-step loss-decrease potential ``<G, polar(G)>``.
    ``fro_sum`` / ``nuclear_sum``
        Sums over pairs (the nuclear sum is the gate's objective).

    Gradients are cast to float32 before any norm is taken: the airbench model
    runs in half precision and an fp16 sum of squares over a 256x2304 matrix
    both overflows and rounds badly.  A pair whose conv weight has no gradient
    contributes 0.0.
    """
    # One device-to-host sync for the whole-model term (this runs inside a
    # tight probe loop; a per-parameter .item() would sync dozens of times).
    parts = [
        p.grad.detach().float().pow(2).sum()
        for p in model.parameters()
        if p.grad is not None
    ]
    total_sq = (
        float(torch.stack([t.to(parts[0].device) for t in parts]).sum().item())
        if parts
        else 0.0
    )
    fro: List[float] = []
    nuclear: List[float] = []
    for conv, _bn, _name in pairs:
        g = conv.weight.grad
        if g is None:
            fro.append(0.0)
            nuclear.append(0.0)
            continue
        gm = g.detach().float().reshape(g.shape[0], -1)
        fro.append(float(gm.norm().item()))
        nuclear.append(float(torch.linalg.svdvals(gm).sum().item()))
    return {
        "total": math.sqrt(total_sq),
        "fro": fro,
        "nuclear": nuclear,
        "fro_sum": float(sum(fro)),
        "nuclear_sum": float(sum(nuclear)),
    }


# -------------------------------------------------------------- orbit search

# Objective: scales -> float, or None when the proposal is INFEASIBLE (e.g. it
# moved the loss off the level set by more than the caller's tolerance).
ScaleObjective = Callable[[List[torch.Tensor]], Optional[float]]


def refine_scales(
    objective: ScaleObjective,
    init_scales: Sequence[torch.Tensor],
    iters: int,
    step_size: float,
    spread_limit: float,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Maximize ``objective`` over orbit scales by accept-if-better random search.

    Deliberately derivative-free.  The gate's objective is a constrained one
    (maximize the nuclear-norm sum SUBJECT TO staying on the level set), and it
    is evaluated by a full forward+backward of the real model; a reparameterized
    gradient ascent on log-scales would have to differentiate through that
    constraint, which buys nothing for a go/no-go measurement.  Instead:
    start from ``init_scales``, propose Gaussian perturbations of the
    LOG-scales with standard deviation ``step_size``, clamp the log-scales to
    ``[-log(spread_limit), +log(spread_limit)]``, and keep a proposal iff it is
    feasible and strictly improves.

    Fully deterministic given ``generator`` (all proposals are drawn on CPU).
    Returns ``{"scales", "value", "n_accepted", "n_infeasible", "n_iters"}``;
    ``value`` is ``-inf`` iff no evaluated point (including the start) was
    feasible.
    """
    iters = int(iters)
    if iters < 0:
        raise ValueError(f"iters must be >= 0, got {iters}")
    spread_limit = float(spread_limit)
    if not spread_limit > 1.0:
        raise ValueError(f"spread_limit must be > 1, got {spread_limit}")
    log_limit = math.log(spread_limit)

    cur_log = [
        torch.as_tensor(s).detach().float().clone().log().clamp(-log_limit, log_limit)
        for s in init_scales
    ]
    best_value = objective([t.exp() for t in cur_log])
    if best_value is None:
        best_value = float("-inf")
    best_log = [t.clone() for t in cur_log]

    n_accepted = 0
    n_infeasible = 0
    for _ in range(iters):
        proposal = [
            (t + step_size * torch.randn(t.shape, generator=generator)).clamp(
                -log_limit, log_limit
            )
            for t in best_log
        ]
        value = objective([t.exp() for t in proposal])
        if value is None:
            n_infeasible += 1
            continue
        if value > best_value:
            best_value = value
            best_log = proposal
            n_accepted += 1
    return {
        "scales": [t.exp() for t in best_log],
        "value": float(best_value),
        "n_accepted": n_accepted,
        "n_infeasible": n_infeasible,
        "n_iters": iters,
    }


def ascend_scales(
    objective: ScaleObjective,
    pairs: Sequence[ConvBnPair],
    steps: int,
    lr: float,
    spread_limit: float,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Spec-named entry point for :func:`refine_scales`, starting at alpha = 1.

    ``steps`` is the number of random-search iterations and ``lr`` the
    log-scale proposal standard deviation (see :func:`refine_scales` for why
    the search is derivative-free rather than a gradient ascent).
    """
    init = [torch.ones(int(conv.weight.shape[0])) for conv, _bn, _name in pairs]
    return refine_scales(
        objective,
        init,
        iters=steps,
        step_size=lr,
        spread_limit=spread_limit,
        generator=generator,
    )
