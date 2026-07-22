"""Muon + DistAdam, copied VERBATIM from the pinned record.

Record source (see src/nanogpt/__init__.py): the 2025-07-12_BosAlign
validation script, ``0c5449cc-....txt`` lines 108-286.

**The record's optimizer *behaviour* is the object of study and may not be
changed.** Exactly one edit has been made to the code below, PORT CHANGE O1 in
``DistAdam.step``: the parameter is written through ``p.detach()`` instead of
through the ``Parameter`` object. Same storage, same writes, same values — it
only removes an autograd *view* relationship that made every ``compile: true``
run abort at the first ``opt.step()``. Full rationale is inline at the change
and in ``docs/nanogpt-port.md`` §2 (O1); regression tests are
``tests/test_nanogpt_port.py::test_dist_adam_param_slice_is_not_a_no_grad_view``
and ``::test_training_loop_end_to_end_on_cpu_with_compile``.

The instantiation values used by the record (RECORD:644-645) are

    DistAdam(scalar_params + head_params + embed_params,
             lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, ...)
    Muon(hidden_matrix_params, lr=0.05, momentum=0.95, weight_decay=0.0, ...)

— note these differ from the class-signature defaults on this page
(RECORD:151, RECORD:207); the *instantiation* values are the record.
They live in ``src/nanogpt/config.py`` and are applied in
``src/nanogpt/train.py``.

Gradient accumulation is invisible here: the port accumulates into
``p.grad`` before calling ``step()``, exactly as one larger backward would,
so these step functions see the same gradient the record's did (see the
accumulation math in ``src/nanogpt/config.py``).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor

# --------------------------------------------------------------------------
# RECORD:108-134 — verbatim
# --------------------------------------------------------------------------


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# --------------------------------------------------------------------------
# RECORD:137-203 — verbatim
# --------------------------------------------------------------------------


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}

        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params,))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        # PORT CHANGE P6 (program #8): optional passive tempo probe. Set
        # ``opt.tempo_probe = TempoProbe(...)`` externally; None (default,
        # via getattr) leaves the record step byte-identical in behavior.
        probe = getattr(self, "tempo_probe", None)
        if probe is not None:
            probe.begin_step()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [torch.zeros_like(params[-1])] * self.world_size
            for base_i in range(0, len(params), self.world_size):
                if base_i + self.rank < len(params):
                    grad = params[base_i + self.rank].grad
                # This gives strange dynamo warnings
                reduce_scatter_futures.append(dist.reduce_scatter(grad, grad_pad[base_i:base_i + self.world_size], op=dist.ReduceOp.AVG, async_op=True).get_future())

        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * self.world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), self.world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    grad = p.grad
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                    eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    if probe is not None:
                        # P6: must run before the in-place ops below — line
                        # `grad.lerp_` mutates p.grad, and the probe wants the
                        # raw averaged gradient + the pre-update buffer.
                        probe.observe(p, grad, momentum_buffer)
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_reduce_futures.append(dist.all_gather(params_pad[base_i:base_i + self.world_size], params_pad[base_i + self.rank], async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()


# --------------------------------------------------------------------------
# RECORD:206-286 — verbatim (DistributedAdam implementation by @vagrawal)
# --------------------------------------------------------------------------


class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01, rank: int = 0, world_size: int = 1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        self.rank = rank
        self.world_size = world_size

        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(
                params=group_params,
            ))
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // self.world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // self.world_size
                # ---- PORT CHANGE O1 -----------------------------------------
                # RECORD:249  `p_slice = p[rank * rank_size:(rank + 1) * rank_size]`
                # RECORD:262  `... dist.all_gather_into_tensor(p, p_slice, ...)`
                # The port writes both through `p_view = p.detach()` instead of
                # through the Parameter object. Nothing else on either line moves.
                #
                # WHY. RECORD:249 slices a *Parameter* (requires_grad=True) inside
                # `@torch.no_grad()`, so `p_slice` is an autograd *differentiable
                # view* carrying creation_meta NO_GRAD_MODE, while `p` itself is a
                # leaf with `._base is None`. The lines below mutate the view in
                # place (`p_slice.mul_`, `p_slice.add_`), bumping the base's version
                # counter, and RECORD:262 then hands *both* the base and the mutated
                # view to a collective. Because `step` is `@torch.compile`d
                # (RECORD:223) that collective is a dynamo graph break, and resuming
                # the trace re-wraps `p_slice` as a graph input, which reads
                # `.is_leaf` -> the `grad_fn` getter -> the view-rebase check ->
                #   "A view was created in no_grad mode and its base ... has been
                #    modified inplace with grad mode enabled."
                # That aborted every `compile: true` run at the first `opt.step()`,
                # on GPU and (see tests) on CPU alike. Handing the collective a
                # base/view pair with mixed `._base` states also trips AOTAutograd's
                # `merge_view_inputs` ("mixed autograd ._base states").
                #
                # WHY IT IS BEHAVIOUR-PRESERVING. `p.detach()` shares `p`'s storage,
                # sizes, strides *and* version counter; the slice of it covers the
                # same elements at the same addresses. Every in-place write below,
                # and the collective's write, therefore lands in exactly the same
                # memory with exactly the same arithmetic, and `p` (the Parameter)
                # observes exactly the same values. Only the autograd *metadata* of
                # the two handles changes: `p.detach()`'s base does not require grad,
                # so neither handle is a differentiable view and no rebase check
                # applies. Nothing here is differentiated anyway — the whole method
                # is `@torch.no_grad()` and `p` is a leaf whose `.grad` is read, not
                # written. This is the record's own idiom for writing into a
                # parameter's storage: RECORD:632 `dist.broadcast(param.detach(), 0)`,
                # RECORD:346 / 372 / 417 `...weight.detach().zero_()`.
                p_view = p.detach()
                p_slice = p_view[self.rank * rank_size:(self.rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]

                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)

                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)

                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t

                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)

                idx += 1
                # PORT CHANGE O1 (RECORD:262): `p` -> `p_view` (== `p.detach()`),
                # same storage, same write. See the block above.
                all_reduce_futures.append(dist.all_gather_into_tensor(p_view, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()
