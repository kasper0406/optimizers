"""Directional smoothness along the trajectory, in the SPECTRAL norm (Muon's
own geometry) and in the Euclidean norm, side by side.

PRE-REGISTERED QUESTION
-----------------------
    Does the GENERALIZED (spectral-norm) directional smoothness of the
    training loss, measured along the actual optimizer trajectory, equilibrate
    at c / lr for some constant c -- the Muon analogue of gradient descent's
    2 and Adam's ~38 -- while the EUCLIDEAN directional smoothness along the
    same trajectory does not?  The reported signature is the dimensionless
    product

        lr * D_smooth_spectral(t)

    plateauing at a constant, on the same runs and the same steps on which the
    Euclidean quantity (lr * D_smooth_frobenius, and the HVP-measured
    eta*lambda of the tracked directions) does not.  Because "equilibrates at
    c / lr" is an lr-scaling claim, the plateau must be tested across an lr
    ladder: the constant is only a constant if it is the SAME constant at
    0.5x, 1x and 2x the record learning rate.

    This module states the question and produces the measurement.  It draws no
    conclusion, evaluates no gate, and carries no threshold.

WHY
---
Islamov, Crawshaw, Cohen & Gower, "Non-Euclidean Gradient Descent Operates at
the Edge of Stability" (arXiv:2603.05002, ICML 2026 oral) prove that steepest
descent under a norm ||.|| operates at the edge of stability with respect to
the directional smoothness measured in THAT norm -- spectral norm for
Muon-style polar-factor updates -- with the Euclidean sharpness decoupled from
stability entirely.  Their result is full-batch, momentum-free and
deterministic; they name the momentum + minibatch regime as the open problem.
That open regime is exactly what this repo measures (reports/wp12-phase1-
measurement.md: HVP-measured Euclidean eta*lambda ~ 65 with training stable,
no divergence out to 6x the record lr).  The missing half of the measurement
is the quantity their theory says actually governs stability; this module adds
it, on the same runs that produce the Euclidean numbers.

WHAT IS COMPUTED
----------------
Every ``t_meas`` steps, for each Muon-managed weight matrix W with the ACTUAL
applied update D (see below):

    remainder = L(W + D) - L(W) - <grad L(W), D>
    D_smooth_spectral   = 2 * remainder / ||D||_2^2      (2 = largest sing. val.)
    D_smooth_frobenius  = 2 * remainder / ||D||_F^2      (the Euclidean version)

all three loss/gradient terms evaluated on the SAME minibatch, one matrix at a
time (every other parameter held at its pre-step value), plus ``||D||_2``,
``||D||_F``, ``lr``, and the products ``lr * D_smooth_*``.

D is measured, not modelled: it is ``W_after_step - W_before_step``, the exact
parameter change the optimizer applied on that step.  For the airbench recipe
that includes the vendored Muon's in-``step()`` weight renormalization and the
Nesterov momentum term -- i.e. D is the real trajectory increment, which is
what the directional-smoothness definition asks for, and is NOT identical to
``-lr * O_t`` for the raw orthogonalized momentum ``O_t``.  ``||D||_2`` is
therefore also measured rather than assumed to equal ``lr``.

HONEST CAVEATS (these are measurement properties, not defects to hide)
---------------------------------------------------------------------
1. **Minibatch estimate.** All three terms use the current training
   minibatch's loss, not the full-batch loss.  The theory being probed is
   full-batch; this is a stochastic estimate of the directional smoothness of
   the minibatch loss surface.  Because the SAME batch is used for L(W),
   L(W+D) and the gradient, the estimate is a consistent second-order
   remainder of ONE well-defined function (no cross-batch cancellation
   error), but it is not the full-batch quantity and must never be reported
   as one.
2. **Loss reduction.** The airbench training loss is SUM-reduced over the
   batch, so ``D_smooth`` inherits that scale.  ``batch_size`` and
   ``loss_reduction`` are recorded in the log so any analysis can convert to
   the per-example convention; the equilibration constant c is
   reduction-dependent and must be quoted with its convention.
3. **fp32 re-evaluation.** As in :mod:`src.instrument.hvp`, the loss is
   re-evaluated in full fp32 through ``torch.func.functional_call`` on
   detached copies (helpers imported from that module, not reimplemented),
   never through the fp16 training graph.  The training model's parameters,
   buffers, gradients, dtypes and RNG stream are untouched.
4. **One matrix at a time.** The perturbation applied in ``L(W + D)`` is the
   update to a single matrix; the other parameters (including the SGD-managed
   head and biases) stay at their pre-step values.  This is the per-matrix
   directional smoothness, matching the per-matrix geometry Muon actually
   descends in; it is not the smoothness along the full joint step.
5. **Sign.** ``remainder`` can be negative (a locally concave slice); the
   ratio is reported as measured, never clipped.

COST
----
Per measured step, with the default ``grad_source="recompute"``: one fp32
forward + one (single) backward for L(W) and its gradient, then one fp32
forward per tracked matrix for L(W + D).  At the airbench settings (6 filter
matrices, 200 steps, ``t_meas=5``) that is 40 measured steps x (1 fwd+bwd + 6
fwd) = 40 backward and 280 forward passes per run, entirely off the per-step
training path.  ``grad_source="training"`` reuses the already-computed
training gradient instead and saves the backward, at the cost of pairing an
fp16-graph gradient with an fp32 loss difference (see :meth:`after_step`).

NOT AN OPTIMIZER INPUT
----------------------
Like the HVP probe, this is instrumentation: nothing here may be consumed by
any optimizer update, gain, or gating decision.  The module lives under
``src/instrument`` and must never be imported from ``src/optim``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from src.instrument.hvp import (
    LossFn,
    default_ce_loss,
    fp32_functional_loss,
    fp32_overrides,
)

__all__ = ["SmoothnessProbe", "GRAD_SOURCES"]

GRAD_SOURCES = ("recompute", "training")


class SmoothnessProbe:
    """Trajectory directional-smoothness probe (see module docstring).

    Usage per training step::

        probe.set_batch(inputs, labels)     # the step's actual batch
        loss.backward()
        probe.before_step(step)             # snapshot W (and, optionally, grads)
        optimizer.step()
        probe.after_step(step, lr)          # measure; no-op if not a measured step

    ``before_step``/``after_step`` are no-ops on steps that are not due, so the
    caller does not need to know the cadence.

    Parameters
    ----------
    model:
        The training model.  Used read-only.  Must not be ``torch.compile``'d
        when ``grad_source="recompute"`` (backward through a compiled
        functional_call graph is not supported here) -- the airbench config
        enforces ``recipe.compile: false``.
    named_params:
        ``(name, param)`` pairs for the matrices to measure (the Muon-managed
        ones).  Names must be ``model.named_parameters()`` names.
    t_meas:
        Measure every ``t_meas`` steps (steps 1, 1+t_meas, ...).
    loss_fn:
        ``(outputs_fp32, labels) -> scalar``.  Default: the airbench
        sum-reduced label-smoothed cross-entropy (``hvp.default_ce_loss``).
    grad_source:
        ``"recompute"`` (default) evaluates the gradient of the same fp32
        functional loss used for L(W) and L(W+D) -- all three terms then come
        from one function, which matters because the measured quantity is
        their second-order *remainder*.  ``"training"`` instead reuses the
        gradient already sitting in ``param.grad`` (no extra backward); it is
        cheaper but pairs an fp16-graph gradient with an fp32 loss difference,
        so the remainder inherits that inconsistency.  Recorded in the log.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        named_params: Iterable[Tuple[str, torch.Tensor]],
        *,
        t_meas: int = 5,
        loss_fn: Optional[LossFn] = None,
        label_smoothing: float = 0.2,
        grad_source: str = "recompute",
        loss_reduction: str = "sum",
    ) -> None:
        if int(t_meas) < 1:
            raise ValueError(f"t_meas must be >= 1, got {t_meas}")
        if grad_source not in GRAD_SOURCES:
            raise ValueError(
                f"grad_source must be one of {GRAD_SOURCES}, got {grad_source!r}"
            )
        self.model = model
        self.named_params: List[Tuple[str, torch.Tensor]] = list(named_params)
        if not self.named_params:
            raise ValueError("SmoothnessProbe requires at least one tracked matrix")
        known = {n for n, _ in model.named_parameters()}
        unknown = [n for n, _ in self.named_params if n not in known]
        if unknown:
            raise ValueError(
                f"SmoothnessProbe: names not in model.named_parameters(): {unknown}"
            )
        self.t_meas = int(t_meas)
        self.loss_fn: LossFn = (
            default_ce_loss(label_smoothing) if loss_fn is None else loss_fn
        )
        self.grad_source = grad_source
        self.loss_reduction = str(loss_reduction)

        self._batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._before: Optional[Dict[str, torch.Tensor]] = None
        self._train_grads: Optional[Dict[str, torch.Tensor]] = None
        self.n_forward = 0
        self.n_backward = 0
        self.n_measured_steps = 0
        self.batch_size: Optional[int] = None
        self.records: Dict[str, Dict[str, List[float]]] = {
            name: {
                key: []
                for key in (
                    "step",
                    "lr",
                    "loss_base",
                    "loss_perturbed",
                    "inner_product",
                    "remainder",
                    "spec_norm_D",
                    "fro_norm_D",
                    "d_smooth_spectral",
                    "d_smooth_frobenius",
                    "lr_times_d_smooth_spectral",
                    "lr_times_d_smooth_frobenius",
                )
            }
            for name, _ in self.named_params
        }

    # ------------------------------------------------------------------ api

    def is_due(self, step: int) -> bool:
        """True on steps 1, 1 + t_meas, 1 + 2*t_meas, ... (1-based steps)."""
        return (int(step) - 1) % self.t_meas == 0

    def set_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> None:
        """Register the current step's (augmented, normalized) batch."""
        self._batch = (inputs, labels)
        self.batch_size = int(len(labels))

    def clear(self) -> None:
        self._batch = None
        self._before = None
        self._train_grads = None

    @torch.no_grad()
    def before_step(self, step: int) -> bool:
        """Snapshot the pre-step parameters (all of them).  No-op if not due.

        Every parameter is snapshotted, not just the tracked matrices: the
        baseline loss L(W) must be evaluated at the pre-step values of the
        whole model, and the non-Muon parameters (head, biases) move on the
        same ``optimizer.step()``.
        """
        if not self.is_due(step):
            return False
        self._before = {
            name: p.detach().to(torch.float32).clone()
            for name, p in self.model.named_parameters()
        }
        if self.grad_source == "training":
            self._train_grads = {
                name: p.grad.detach().to(torch.float32).clone()
                for name, p in self.named_params
                if p.grad is not None
            }
        return True

    def after_step(self, step: int, lr: float) -> Optional[Dict[str, Dict[str, float]]]:
        """Measure directional smoothness for this step.  No-op if not due.

        Returns the per-matrix record dict (also appended to
        :attr:`records`), or ``None`` on a non-measured step.
        """
        if self._before is None:
            return None
        if self._batch is None:
            raise RuntimeError(
                "SmoothnessProbe: set_batch(inputs, labels) must be called with "
                "the current step's batch before measuring"
            )
        before = self._before
        self._before = None
        train_grads = self._train_grads
        self._train_grads = None
        inputs, labels = self._batch

        # The actual applied update, per tracked matrix, in the parameter's
        # own shape (fp32).  Measured, never modelled -- see module docstring.
        deltas: Dict[str, torch.Tensor] = {}
        for name, p in self.named_params:
            deltas[name] = p.detach().to(torch.float32) - before[name]

        tracked_ids = {id(p) for _, p in self.named_params}
        if self.grad_source == "recompute":
            with torch.enable_grad():
                overrides, leaves = fp32_overrides(
                    self.model, grad_param_ids=tracked_ids, param_values=before
                )
                loss0 = fp32_functional_loss(
                    self.model, overrides, inputs, labels, self.loss_fn
                )
                grads = torch.autograd.grad(
                    loss0, [leaves[id(p)] for _, p in self.named_params]
                )
            self.n_forward += 1
            self.n_backward += 1
            loss_base = float(loss0.detach())
            grad_by_name = {
                name: g.detach()
                for (name, _), g in zip(self.named_params, grads)
            }
        else:
            with torch.no_grad():
                overrides, _ = fp32_overrides(self.model, param_values=before)
                loss0 = fp32_functional_loss(
                    self.model, overrides, inputs, labels, self.loss_fn
                )
            self.n_forward += 1
            loss_base = float(loss0)
            if train_grads is None:
                raise RuntimeError(
                    "SmoothnessProbe(grad_source='training'): no gradients were "
                    "present on the tracked params at before_step()"
                )
            grad_by_name = dict(train_grads)

        out: Dict[str, Dict[str, float]] = {}
        for name, p in self.named_params:
            D = deltas[name]
            g = grad_by_name.get(name)
            if g is None:
                continue
            with torch.no_grad():
                # Fresh overrides per forward: BatchNorm updates its running
                # stats in place onto the copies, so reusing them would move
                # the surface between L(W) and L(W + D).
                ov, _ = fp32_overrides(self.model, param_values=before)
                ov[name] = ov[name] + D
                loss1 = fp32_functional_loss(
                    self.model, ov, inputs, labels, self.loss_fn
                )
                self.n_forward += 1
                inner = float((g.reshape(-1) * D.reshape(-1)).sum())
                # Muon-family flattening convention: >2-D params are treated
                # as len(p) x -1 matrices (identical to src.optim / tracker).
                D2 = D.reshape(len(D), -1)
                fro = float(torch.linalg.norm(D2))
                spec = float(torch.linalg.matrix_norm(D2, ord=2))
            loss_pert = float(loss1)
            remainder = loss_pert - loss_base - inner
            d_spec = 2.0 * remainder / (spec * spec) if spec > 0 else float("nan")
            d_fro = 2.0 * remainder / (fro * fro) if fro > 0 else float("nan")
            rec = {
                "step": float(step),
                "lr": float(lr),
                "loss_base": loss_base,
                "loss_perturbed": loss_pert,
                "inner_product": inner,
                "remainder": remainder,
                "spec_norm_D": spec,
                "fro_norm_D": fro,
                "d_smooth_spectral": d_spec,
                "d_smooth_frobenius": d_fro,
                "lr_times_d_smooth_spectral": float(lr) * d_spec,
                "lr_times_d_smooth_frobenius": float(lr) * d_fro,
            }
            for key, value in rec.items():
                self.records[name][key].append(value)
            out[name] = rec
        self.n_measured_steps += 1
        return out

    # ------------------------------------------------------------------- log

    def to_log(self) -> Dict[str, Any]:
        """Serializable smoothness block (see src.instrument.schema)."""
        return {
            "t_meas": self.t_meas,
            "grad_source": self.grad_source,
            "loss_reduction": self.loss_reduction,
            "batch_size": self.batch_size,
            "n_measured_steps": self.n_measured_steps,
            "n_forward": self.n_forward,
            "n_backward": self.n_backward,
            "matrices": {
                name: {key: list(values) for key, values in series.items()}
                for name, series in self.records.items()
            },
        }


def smoothness_from_config(
    instr_cfg: Dict[str, Any],
    model: torch.nn.Module,
    named_params: Sequence[Tuple[str, torch.Tensor]],
    *,
    label_smoothing: float = 0.2,
) -> Optional[SmoothnessProbe]:
    """Build a :class:`SmoothnessProbe` from an ``instrumentation:`` block.

    Reads the optional ``smoothness:`` sub-block::

        instrumentation:
          smoothness:
            enabled: true
            t_meas: 5
            grad_source: recompute   # or: training

    Returns ``None`` when the block is absent or ``enabled: false`` (default
    off, like every other measurement add-on).
    """
    cfg = instr_cfg.get("smoothness")
    if cfg is None:
        return None
    if isinstance(cfg, bool):
        cfg = {"enabled": cfg}
    if not isinstance(cfg, dict):
        raise ValueError(
            "instrumentation.smoothness must be a mapping (enabled, t_meas, "
            f"grad_source) or a bool; got {type(cfg).__name__}"
        )
    if not bool(cfg.get("enabled", True)):
        return None
    return SmoothnessProbe(
        model,
        named_params,
        t_meas=int(cfg.get("t_meas", 5)),
        grad_source=str(cfg.get("grad_source", "recompute")),
        label_smoothing=label_smoothing,
    )
