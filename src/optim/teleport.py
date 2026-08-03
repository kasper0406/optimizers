"""Teleport moves for Muon: closed-form nuclear-norm ascent along conv->BN orbits.

WHY
---
``src/instrument/orbit.py`` established (and the WP-J gate measured) that the
conv->BatchNorm per-channel rescaling orbit is loss-invariant in TRAIN mode
while the gradient's size varies a lot along it: at fixed loss, refined orbit
points raised the nuclear-norm sum by **+21-23%** over the base point at every
probed training state (reports/wpj-mech-round1.md section 1; worst realized
|dloss|/L across 1,152 draws was 1.2e-5).  The nuclear norm is the right notion
of "size" here because one Muon step's first-order loss decrease is
``<G, polar(G)> = ||G||_*`` of the gradient reshaped to ``[out_channels, -1]``.
So a move along the orbit is a free (loss-preserving) increase of the next
Muon step's descent potential -- a *teleport move*.

THE MATH (no model forward needed)
----------------------------------
Scaling output channel ``c`` of the conv weight by ``alpha_c > 0`` leaves the
train-mode loss unchanged and transforms the gradient EXACTLY as

    G_c -> G_c / alpha_c

(verified to 1e-5 in the GPU gate runs, and by
``tests/test_orbit.py::test_channel_gradient_scales_as_one_over_alpha``).  The
objective is therefore a closed-form function of the base-point gradient alone:

    maximize   F(alpha) = || diag(1/alpha) G ||_*     over  alpha in
                                        [1/spread, spread]^out

No forward/backward pass is needed per candidate -- only an SVD of an
``[out, fan_in]`` matrix.  That replaces the gate's 64+64 model evaluations per
teleport with a handful of SVDs.

Parameterize ``t_c = log alpha_c`` and ``A(t) = diag(exp(-t)) G``.  The nuclear
norm's subgradient is ``d||A||_* / dA = U V^T`` from the thin SVD
``A = U S V^T``, and the chain rule through ``A_cj = exp(-t_c) G_cj`` gives

    dF/dt_c = - sum_j A_cj (U V^T)_cj = - ( A o (U V^T) ) row-c sum

(``o`` is the elementwise product).  :func:`nuclear_ascent` does projected
gradient ASCENT on ``t`` with that gradient, clamping ``|t_c| <= log(spread)``.

GAUGE NOTE (read before interpreting ``achieved_ratio``)
-------------------------------------------------------
Summing the gradient over channels gives ``sum_c dF/dt_c = -<A, U V^T> =
-||A||_*``, i.e. the ascent direction ALWAYS has a negative uniform component:
shrinking every channel by the same factor grows the gradient by that factor
and the nuclear norm with it.  That component is gauge-trivial for Muon --
``polar(cG) = polar(G)`` for ``c > 0``, so a uniform rescale changes the
orthogonalized update by nothing at all.  It is also exactly what the airbench
recipe's per-step weight renormalization (``p *= sqrt(numel)/||p||``, applied
harness-side in ``src/optim/airbench_zoo.py``) removes: a uniform positive
rescale is itself a member of the orbit's symmetry group, so the
renormalization is gauge-COMPATIBLE with a teleport -- it undoes the uniform
part of the move and leaves the relative channel pattern, which is the only
part that changes the Muon step.

AS OF THE ROUND-2 DECISION (2026-08-03) THE ASCENT IS GAUGE-FIXED: the
uniform component is projected out of every ascent step and ``t`` is
re-centered, so ``geomean(alpha) ~= 1`` (up to clamp boundary effects) and
``achieved_ratio`` IS the relative, Muon-spendable ratio. The factorization
below is retained for interpreting non-centered alpha patterns:

``achieved_ratio`` factors exactly as

    ratio = (1 / geomean(alpha)) * ratio_relative,   ratio_relative computed
                                                     from alpha / geomean(alpha)

and only ``ratio_relative`` is descent potential a Muon step can actually
spend.  The uniform factor is bounded by ``spread``.  This module returns the
raw (spec'd) objective ratio; callers comparing teleport strength across
configs should divide out ``geomean(alpha)`` themselves.  See
``tests/test_teleport.py::test_ratio_factors_into_gauge_and_relative_parts``.

COMPOSITION CONTRACT
--------------------
This module is pure math on tensors: it does not touch modules, optimizers or
the training loop.  The harness composes it with ``src/instrument/orbit.py``
like this, at a teleport step, with the base-point gradients already computed:

    grads = [conv.weight.grad.detach().float().reshape(conv.weight.shape[0], -1)
             for conv, _bn, _name in pairs]
    alphas, ratios = teleport_alphas(pairs, grads, spread, iters, step_size)

    orbit.apply_channel_scales(pairs, alphas)      # WEIGHTS move to the new
                                                   # orbit point (and biases)
    for (conv, _bn, _name), a in zip(pairs, alphas):
        transport_gradlike(conv.weight.grad, a)    # gradient at the new point
        transport_gradlike(opt.state[conv.weight]["momentum_buffer"], a)
        if conv.bias is not None and conv.bias.grad is not None:
            transport_gradlike(conv.bias.grad, a)  # bias grad follows 1/alpha too

    optimizer.step()                               # steps coherently AT the
                                                   # new point

Both ``p.grad`` and the Muon momentum buffer are transported: the buffer is an
accumulation of past gradients and lives in the gradient's transformation law
(``G -> G/alpha`` under ``W -> alpha W``), so leaving it untransported would
make the momentum a stale mixture of two different orbit points.  Transporting
it keeps the whole optimizer state expressed in the new gauge.

Weights move by ``alpha`` (``apply_channel_scales``); gradient-like tensors
move by ``1/alpha`` (:func:`transport_gradlike`).  Note that
``apply_channel_scales`` is in-place and only approximately invertible in
floating point -- see ``src/optim/train_snapshot.py`` for exact restore.

Harness-agnostic and CPU-testable, like the rest of ``src/instrument/orbit.py``.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Sized, Tuple

import torch

# Dtypes the ascent will work in directly; anything else (fp16/bf16 gradients
# from the airbench half-precision model) is promoted to float32 before the
# SVD, matching ``orbit.grad_norms``' cast rule.
_WORK_DTYPES = (torch.float32, torch.float64)


# ------------------------------------------------------------------ internals


def _work_dtype(t: torch.Tensor) -> torch.dtype:
    return t.dtype if t.dtype in _WORK_DTYPES else torch.float32


def _nuclear_value_and_grad(A: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """``(||A||_*, dF/dt)`` at the point whose scaled matrix is ``A``.

    The gradient is the log-scale ascent gradient of the module docstring,
    ``dF/dt_c = -(A o U V^T) row-c sum``, evaluated from the thin SVD.  Raises
    ``torch.linalg.LinAlgError`` if the SVD does not converge (callers handle
    it; cuSOLVER does fail on ill-conditioned early-training matrices --
    reports/wpj-mech-round1.md section 3).
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return float(S.sum().item()), -(A * (U @ Vh)).sum(dim=1)


# --------------------------------------------------------------------- ascent


@torch.no_grad()
def nuclear_ascent(
    grad_mat: torch.Tensor, spread: float, iters: int, step_size: float
) -> Tuple[torch.Tensor, float]:
    """Maximize ``||diag(1/alpha) G||_*`` over the orbit box, in closed form.

    Parameters
    ----------
    grad_mat
        The base-point gradient of ONE conv weight, detached and reshaped to
        ``[out_channels, -1]``.  fp16/bf16 inputs are promoted to float32 for
        the SVD; float64 is kept (the finite-difference test needs it).
    spread
        Orbit box half-width: every returned ``alpha_c`` lies in
        ``[1/spread, spread]``.  Must be ``> 1`` (same contract as
        ``orbit.sample_log_uniform_scales``).
    iters
        Number of ascent steps.  ``0`` means "do not teleport" and returns the
        identity move.  ``iters + 1`` objective values are computed: the point
        reached by the last step is evaluated too, so it can be selected.
    step_size
        Length of each ascent step in log-scale space.  The step is
        NORMALIZED (``t += step_size * g / (||g|| + 1e-12)``) rather than
        proportional to the gradient: ``F`` is homogeneous of degree 1 in ``G``,
        so a raw gradient step's size would scale with the gradient magnitude
        and would need per-layer, per-step retuning.  With a normalized step,
        ``step_size`` is measured in the same units as the ``log(spread)``
        clamp for every layer and every training state.

    Returns
    -------
    ``(alphas, achieved_ratio)``
        ``alphas`` is 1-D of length ``grad_mat.shape[0]``, all inside
        ``[1/spread, spread]``, on ``grad_mat``'s device in the working dtype.
        ``achieved_ratio`` is ``||diag(1/alpha) G||_* / ||G||_*`` -- read the
        module's GAUGE NOTE before comparing ratios across configs.

    Best-seen semantics: fixed-step projected ascent is NOT monotone (a step
    can overshoot, and the clamp can cut a step short), so every evaluated
    point is scored and the ARGMAX is returned.  ``t = 0`` is the first
    candidate, hence ``achieved_ratio >= 1.0`` always and the move is never
    worse than not teleporting.

    Degenerate inputs return the identity move ``(ones, 1.0)``: an all-zero
    gradient, a non-finite gradient (fp16 momentum/gradient buffers transiently
    do contain non-finite values -- reports/wpj-mech-round1.md section 3),
    ``iters == 0``, and an SVD that fails to converge.
    """
    if grad_mat.ndim != 2:
        raise ValueError(
            f"grad_mat must be 2-D [out, fan_in], got shape {tuple(grad_mat.shape)}"
        )
    spread = float(spread)
    if not spread > 1.0:
        raise ValueError(f"spread must be > 1, got {spread}")
    iters = int(iters)
    if iters < 0:
        raise ValueError(f"iters must be >= 0, got {iters}")
    step_size = float(step_size)
    if step_size < 0.0:
        raise ValueError(f"step_size must be >= 0, got {step_size}")

    dtype = _work_dtype(grad_mat)
    G = grad_mat.detach().to(dtype)
    ones = torch.ones(G.shape[0], dtype=dtype, device=G.device)

    if iters == 0 or G.numel() == 0:
        return ones, 1.0
    if not bool(torch.isfinite(G).all()):
        return ones, 1.0
    if float(G.abs().max().item()) == 0.0:
        return ones, 1.0

    log_spread = math.log(spread)
    t = torch.zeros_like(ones)
    best_t = t.clone()
    base_value = None
    best_value = None

    try:
        for i in range(iters + 1):
            A = torch.exp(-t).unsqueeze(1) * G
            value, g_t = _nuclear_value_and_grad(A)
            if not math.isfinite(value):
                break
            if base_value is None:
                base_value = value
            if best_value is None or value > best_value:
                best_value = value
                best_t = t.clone()
            if i == iters:
                break
            # GAUGE FIX (round-2 decision, 2026-08-03): the raw ascent
            # direction always carries a uniform component (sum_c dF/dt_c =
            # -||A||_*) that is Muon-trivial (polar(cG) = polar(G)) and that
            # the recipe's renormalization removes anyway. Project it out and
            # re-center t, so the whole clamp budget is spent on the relative
            # channel pattern -- the only part that changes the Muon step --
            # and ``achieved_ratio`` reports the SPENDABLE ratio directly.
            g_t = g_t - g_t.mean()
            g_norm = g_t.norm()
            if not bool(torch.isfinite(g_norm)) or float(g_norm.item()) == 0.0:
                break
            t = t + step_size * g_t / (g_norm + 1e-12)
            t = t - t.mean()  # re-center (clamp below may reintroduce a
            # bounded mean when the box is hit asymmetrically; acceptable)
            t = t.clamp_(-log_spread, log_spread)
    except torch.linalg.LinAlgError:
        return ones, 1.0

    if base_value is None or base_value <= 0.0:
        return ones, 1.0
    return torch.exp(best_t), float(best_value / base_value)


# ------------------------------------------------------------------ transport


@torch.no_grad()
def transport_gradlike(tensor: torch.Tensor, alphas: torch.Tensor) -> None:
    """Scale a gradient-like tensor's channel (leading) dimension by ``1/alpha``, in place.

    "Gradient-like" = anything that transforms as the gradient does under the
    orbit move ``W -> alpha . W``: the gradient itself, and any accumulation of
    past gradients such as Muon's momentum buffer or an Adam-family first
    moment.  (Second moments would follow ``1/alpha^2``; this function is not
    for those.)  Weights move the other way and are handled by
    ``orbit.apply_channel_scales``.

    Works for any tensor whose leading dimension is the output channel: 4-D
    conv weights/gradients ``[out, in, kh, kw]`` and 1-D bias-shaped tensors
    alike -- the ``1/alpha`` vector is broadcast over all trailing dimensions,
    so no reshape or view is taken and the caller's tensor keeps its storage.

    Dtype: the reciprocal is computed in float32 (or float64 if either side is
    float64) and only THEN cast to ``tensor``'s dtype.  Computing ``1/alpha``
    in fp16 would round twice and can overflow outright -- ``alpha = 1/spread``
    with a large spread inverts to a value fp16 represents poorly, and the
    airbench model's gradients are fp16.
    """
    if tensor.ndim < 1:
        raise ValueError("transport_gradlike needs a tensor with a channel dimension")
    if not tensor.is_floating_point():
        raise ValueError(f"transport_gradlike needs a float tensor, got {tensor.dtype}")
    a = torch.as_tensor(alphas).detach()
    if a.ndim != 1 or a.numel() != tensor.shape[0]:
        raise ValueError(
            f"alphas must be 1-D of length {tensor.shape[0]}, got shape {tuple(a.shape)}"
        )
    if not bool(torch.all(a > 0)):
        raise ValueError("alphas must be strictly positive")

    calc_dtype = (
        torch.float64
        if torch.float64 in (a.dtype, tensor.dtype)
        else torch.float32
    )
    inv = 1.0 / a.to(device=tensor.device, dtype=calc_dtype)
    view = (-1,) + (1,) * (tensor.ndim - 1)
    tensor.mul_(inv.to(tensor.dtype).view(view))


# ------------------------------------------------------------- per-pair sugar


def teleport_alphas(
    pairs: Sized,
    grads_2d: Sequence[torch.Tensor],
    spread: float,
    iters: int,
    step_size: float,
) -> Tuple[List[torch.Tensor], List[float]]:
    """:func:`nuclear_ascent` per conv->BN pair, in pair order.

    ``grads_2d[i]`` is pair ``i``'s base-point gradient reshaped to
    ``[out_channels, -1]``.  The returned ``alphas`` list is exactly the
    ``scales`` argument ``orbit.apply_channel_scales(pairs, scales)`` expects,
    and ``ratios[i]`` is pair ``i``'s predicted nuclear-norm ratio.

    ``pairs`` is used ONLY for its length -- the per-pair objective depends on
    nothing but that pair's gradient matrix, so the pairs never need to be
    inspected.  Checking the length here is what keeps this function's output
    a drop-in for ``apply_channel_scales``' one-vector-per-pair contract.

    The pairs are independent: BatchNorm isolates each conv's channel scaling
    from every other layer's gradient, so per-pair maximization is the joint
    maximization, and permuting the input order permutes the output.
    """
    grads = list(grads_2d)
    if len(grads) != len(pairs):
        raise ValueError(
            f"expected {len(pairs)} gradient matrices (one per pair), got {len(grads)}"
        )
    alphas: List[torch.Tensor] = []
    ratios: List[float] = []
    for grad in grads:
        alpha, ratio = nuclear_ascent(grad, spread, iters, step_size)
        alphas.append(alpha)
        ratios.append(ratio)
    return alphas, ratios
