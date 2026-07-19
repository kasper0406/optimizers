"""Curvature probes (Hessian-vector products) for the airbench harness.

PHASE-1 VALIDATION ONLY (plan "Distributed scalability", invariant 3):
these probes exist solely to calibrate the trajectory-derived implied
eta*lambda estimator (the eta-lambda calibration plot, plan section 1.2
plot 3).  They are FORBIDDEN in any routing or update path: no optimizer
update, gain, or gating decision may consume ``lambda_hvp``.  This module
lives under ``src/instrument`` and must never be imported from ``src/optim``
update code.

What is computed
----------------
For a tracked pair (u_i, v_i) of weight matrix W, the tracker requests
(once per pair per refresh, ``InstrumentationHub`` -> ``MatrixTracker._refresh``)

    lambda_i = vec(D)^T H vec(D),   D = u_i v_i^T reshaped to W's shape,
                                    ||D||_F = 1,

with H the Hessian of the CURRENT batch's training loss restricted to that
matrix.  Realized as a double-backward: g = dL/dW with ``create_graph=True``,
then lambda = <d(g . D)/dW, D>.

Numerical safety on the half-precision airbench model
-----------------------------------------------------
The vendored CifarNet runs its convolutions in fp16 (BatchNorms in fp32).
Double-backward through fp16 ops is exact in dtype but numerically fragile
(fp16 rounding in both backward passes, overflow risk on the sum-reduced
loss).  The probe therefore NEVER differentiates through the training
model's fp16 graph: it re-evaluates the loss functionally in FULL fp32 via
``torch.func.functional_call`` on detached fp32 copies of every parameter
and floating-point buffer (integer buffers, e.g. BatchNorm
``num_batches_tracked``, are cloned unchanged), with the input batch cast
to fp32, autocast off.  The loss is the same sum-reduced label-smoothed
cross-entropy as the training loop.

Read-only guarantee: the training model's parameters, buffers (BatchNorm
running stats update in-place onto the fp32 COPIES, which are discarded),
gradients, dtypes, and RNG streams (the probe's forward consumes no RNG)
are untouched.

Deviations from the training loss surface (documented, all benign for a
curvature probe): (1) fp32 instead of the mixed fp16/fp32 training compute;
(2) the forward uses the model's default ``whiten_bias_grad=True`` -- the
whiten bias is not a tracked matrix, so the probed Hessian block is
identical; (3) BatchNorm sees the probe batch in train mode exactly as the
training forward did.

Cost
----
Per training step on which a refresh fires: ONE fp32 forward + ONE
create-graph backward (built lazily on the first probe of the step, cached
and shared across all pairs and matrices of that step -- ``set_batch``
invalidates), plus ONE Hessian-vector backward per tracked pair.  At the
WP1.2 airbench settings (6 tracked matrices x (k1+k2)=32 pairs,
T_refresh=50, 200 steps => 4 refresh steps) that is 4 fp32 graphs + 768 HVP
backwards per run, off the per-step path.  ``n_graph_builds`` is exported to
the results metrics for cost accounting.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch

__all__ = ["AirbenchHvpProbe"]

# loss_fn(outputs_fp32, labels) -> scalar loss tensor
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class AirbenchHvpProbe:
    """Per-refresh HVP callback for :class:`src.instrument.tracker.InstrumentationHub`.

    Matches the ``HvpFn`` contract: ``probe(param, D) -> float`` returns
    vec(D)^T H vec(D) for the current batch's loss, H restricted to
    ``param``.  The harness calls :meth:`set_batch` once per training step
    with the exact (augmented, normalized) batch the step trained on.

    Parameters
    ----------
    model:
        The training model (used read-only; must NOT be torch.compile'd --
        double-backward through compiled graphs is unsupported, see
        ``run_airbench_instrumented``).
    params:
        The tracked parameters (the airbench filter matrices).  Probes for
        any other parameter raise ``KeyError``.
    loss_fn:
        Optional override of the loss (tests use analytic quadratics).
        Default: sum-reduced cross-entropy with ``label_smoothing``,
        matching the airbench training loop.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        params,
        *,
        loss_fn: Optional[LossFn] = None,
        label_smoothing: float = 0.2,
    ) -> None:
        self.model = model
        self.params: List[torch.Tensor] = list(params)
        if loss_fn is None:
            ls = float(label_smoothing)

            def loss_fn(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.cross_entropy(
                    outputs, labels, label_smoothing=ls, reduction="sum"
                )

        self.loss_fn: LossFn = loss_fn
        self._batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._leaves: Optional[Dict[int, torch.Tensor]] = None
        self._grads: Optional[Dict[int, torch.Tensor]] = None
        self.n_graph_builds = 0  # cost accounting (results metrics)

    # ------------------------------------------------------------------ api

    def set_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> None:
        """Register the current step's batch; invalidates the cached graph."""
        self._batch = (inputs, labels)
        self._leaves = None
        self._grads = None

    def clear(self) -> None:
        """Drop the batch and any retained autograd graph."""
        self._batch = None
        self._leaves = None
        self._grads = None

    def __call__(self, param: torch.Tensor, direction: torch.Tensor) -> float:
        """vec(D)^T H vec(D) for the current batch, H restricted to ``param``."""
        if self._grads is None:
            self._build_graph()
        assert self._leaves is not None and self._grads is not None
        leaf = self._leaves.get(id(param))
        g = self._grads.get(id(param))
        if leaf is None or g is None:
            raise KeyError(
                "AirbenchHvpProbe: probed parameter is not one of the tracked "
                f"params (shape {tuple(param.shape)})"
            )
        # The tracker hands D in param's dtype (fp16 on airbench); the
        # contraction runs in fp32.  ||D||_F = 1 by construction upstream.
        D = direction.detach().to(device=g.device, dtype=torch.float32)
        D = D.reshape(g.shape)
        with torch.enable_grad():
            g_dot_d = (g * D).sum()
            (hv,) = torch.autograd.grad(g_dot_d, leaf, retain_graph=True)
        return float((hv.detach() * D).sum())

    # ------------------------------------------------------------ internals

    def _build_graph(self) -> None:
        """fp32 functional forward + create-graph backward on the current
        batch; cached until the next :meth:`set_batch`/:meth:`clear`."""
        if self._batch is None:
            raise RuntimeError(
                "AirbenchHvpProbe: set_batch(inputs, labels) must be called "
                "with the current step's batch before probing"
            )
        inputs, labels = self._batch
        tracked_ids = {id(p) for p in self.params}
        with torch.enable_grad():
            overrides: Dict[str, torch.Tensor] = {}
            leaves: Dict[int, torch.Tensor] = {}
            for name, p in self.model.named_parameters():
                t = p.detach().to(torch.float32)
                if id(p) in tracked_ids:
                    t.requires_grad_(True)
                    leaves[id(p)] = t
                overrides[name] = t
            missing = [
                tuple(p.shape) for p in self.params if id(p) not in leaves
            ]
            if missing:
                raise ValueError(
                    "AirbenchHvpProbe: tracked params not found among "
                    f"model.named_parameters(): shapes {missing}"
                )
            for name, b in self.model.named_buffers():
                # fp32 copies for float buffers (BN running stats update onto
                # the copies, keeping the training model untouched); integer
                # buffers (num_batches_tracked) cloned unchanged.
                overrides[name] = (
                    b.detach().to(torch.float32)
                    if b.is_floating_point()
                    else b.detach().clone()
                )
            outputs = torch.func.functional_call(
                self.model, overrides, (inputs.detach().to(torch.float32),)
            )
            loss = self.loss_fn(outputs.float(), labels)
            grads = torch.autograd.grad(
                loss, [leaves[id(p)] for p in self.params], create_graph=True
            )
        self._leaves = leaves
        self._grads = {id(p): g for p, g in zip(self.params, grads)}
        self.n_graph_builds += 1
