"""Per-matrix instrumentation: tracked pairs -> projections -> WP0.5 stats.

Orchestrates, per Muon-managed weight matrix (plan section 1.1):

* a :class:`~src.instrument.subspace.TrackedSubspace` (top-k1 singular pairs
  of the momentum matrix M_t + k2 bulk probes, refreshed every ``t_refresh``
  steps via warm-started subspace iteration);
* per tracked pair per step, the scalar projection s_i(t) = u_i^T G_t v_i on
  the RAW PRE-MOMENTUM gradient, fed into the WP0.5-validated statistics
  stack (``src.stats``) at BOTH betas (0.9 and 0.99 by default);
* per matrix per step, the top-singular-value estimate of M_t from the
  tracked block and ||G_t||_F;
* on each refresh, innovation detection: a direction whose subspace pair
  rotated (alignment below ``align_min``) gets its statistics and classifier
  confidence reset through the src.stats reset API (fresh
  ``RegimeClassifier`` state; the classifier restarts in SIGNAL and must
  re-earn a label with n_min fresh observations);
* optionally, once per tracked pair per refresh, a curvature probe
  lambda_i ~= vec(u_i v_i^T)^T H vec(u_i v_i^T) through a trainer-provided
  HVP callback.

HVP POLICY (distributed invariant 3, plan "Distributed scalability"):
HVPs are for **Phase-1 validation only** -- they calibrate the trajectory-
derived implied eta*lambda estimator.  They are FORBIDDEN in any routing or
update path: no optimizer update, gain, or gating decision may consume
``lambda_hvp``.  WP2.x CI greps for HVP usage in the update path; keep it
that way.

All statistics (EMAs, autocorrelation, t-stats, implied eta*lambda,
classification) come from ``src.stats`` -- the WP0.5-tested code.  This
module computes projections and bookkeeping only; it deliberately contains
no statistical formulas.

Call order per training step (see :class:`InstrumentationHub`):

    loss.backward()
    optimizer.step()          # updates momentum buffers in optimizer.state
    hub.after_step()          # reads param.grad (raw G) + momentum buffer
    optimizer.zero_grad()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from src.instrument.subspace import TrackedSubspace
from src.stats import RegimeClassifier

__all__ = [
    "DirectionTrack",
    "MatrixTracker",
    "InstrumentationHub",
    "hub_from_config",
    "HvpFn",
]

# hvp_fn(param, direction_matrix) -> float
#   direction_matrix D = u v^T reshaped to the parameter's original shape,
#   ||D||_F = 1; returns vec(D)^T H vec(D) with H restricted to that matrix.
HvpFn = Callable[[torch.Tensor, torch.Tensor], float]


class DirectionTrack:
    """One tracked direction: a WP0.5 RegimeClassifier per beta + log buffers."""

    def __init__(self, index: int, kind: str, betas: Sequence[float], classifier_kwargs: Dict[str, Any]):
        self.index = index
        self.kind = kind  # "top" | "bulk"
        self.betas = [float(b) for b in betas]
        self._classifier_kwargs = dict(classifier_kwargs)
        self.classifiers: Dict[float, RegimeClassifier] = {
            b: RegimeClassifier(beta=b, **self._classifier_kwargs) for b in self.betas
        }
        # Log buffers.
        self.s_values: List[float] = []  # every observed step
        self.reset_steps: List[int] = []  # subspace-rotation resets (global step)
        self.refresh_alignment: List[Tuple[int, float]] = []  # (step, align)
        self.lambda_hvp: List[Tuple[int, float]] = []  # (step, lambda)
        self.sigma: List[Tuple[int, float]] = []  # (step, sigma estimate at refresh)
        # Per-beta snapshot buffers: step -> stat scalars.
        self.snapshots: Dict[float, Dict[str, List[float]]] = {
            b: {
                "step": [],
                "regime": [],
                "mu": [],
                "var": [],
                "rho": [],
                "t_stat": [],
                "amplitude_ratio": [],
                "implied_eta_lambda": [],
                "ess": [],
                "n_since_reset": [],
            }
            for b in self.betas
        }

    def observe(self, s: float) -> None:
        self.s_values.append(float(s))
        for clf in self.classifiers.values():
            clf.update(float(s))

    def reset(self, step: int) -> None:
        """Innovation reset: rebuild classifier state via the src.stats API.

        A rotated subspace pair means the scalar stream changed identity; the
        classifier is rebuilt fresh (regime -> SIGNAL, n_min clock restarts),
        matching the WP0.5 confidence-reset semantics.
        """
        self.classifiers = {
            b: RegimeClassifier(beta=b, **self._classifier_kwargs) for b in self.betas
        }
        self.reset_steps.append(int(step))

    def snapshot(self, step: int) -> None:
        for b, clf in self.classifiers.items():
            st = clf.stats
            snap = self.snapshots[b]
            snap["step"].append(int(step))
            snap["regime"].append(clf.regime.value)
            snap["mu"].append(float(st.mean))
            snap["var"].append(float(st.var))
            snap["rho"].append(float(st.rho_corrected))
            snap["t_stat"].append(float(st.t_stat))
            snap["amplitude_ratio"].append(float(st.amplitude_ratio))
            snap["implied_eta_lambda"].append(float(st.implied_eta_lambda))
            snap["ess"].append(float(st.ess))
            snap["n_since_reset"].append(int(clf.n_since_reset))


class MatrixTracker:
    """Instrumentation for one weight matrix (flattened to 2-D)."""

    def __init__(
        self,
        name: str,
        shape: Tuple[int, int],
        *,
        k1: int = 16,
        k2: int = 16,
        t_refresh: int = 50,
        subspace_iters: int = 2,
        betas: Sequence[float] = (0.9, 0.99),
        classifier_kwargs: Dict[str, Any],
        align_min: float = 0.9,
        snapshot_every: int = 1,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        m, n = shape
        # Shrink the tracked blocks for small matrices (k1 + k2 <= min(m, n)).
        max_rank = min(m, n)
        k1_eff = min(k1, max_rank)
        k2_eff = min(k2, max_rank - k1_eff)
        self.name = name
        self.shape = (int(m), int(n))
        self.t_refresh = int(t_refresh)
        self.align_min = float(align_min)
        self.snapshot_every = int(snapshot_every)
        self.subspace = TrackedSubspace(
            m,
            n,
            k1=k1_eff,
            k2=k2_eff,
            iters=subspace_iters,
            generator=generator,
            device=device,
        )
        self.directions: List[DirectionTrack] = [
            DirectionTrack(i, kind, betas, classifier_kwargs)
            for i, kind in enumerate(self.subspace.kinds())
        ]
        self.step_count = 0
        # Per-matrix per-step logs.
        self.steps: List[int] = []
        self.grad_fro_norm: List[float] = []
        self.top_sigma_m: List[float] = []
        self.refresh_steps: List[int] = []

    # ------------------------------------------------------------------ step

    def observe(
        self,
        G: torch.Tensor,
        M: torch.Tensor,
        *,
        hvp_fn: Optional[HvpFn] = None,
        param: Optional[torch.Tensor] = None,
    ) -> None:
        """One instrumented step: G = raw pre-momentum gradient, M = momentum.

        Both are 2-D (callers flatten >2-D params as len(p) x -1, matching the
        Muon-family optimizers).  ``hvp_fn``/``param`` enable the per-refresh
        curvature probe (Phase-1 validation only -- see module docstring).
        """
        self.step_count += 1
        step = self.step_count

        if (step - 1) % self.t_refresh == 0:
            self._refresh(M, step, hvp_fn=hvp_fn, param=param)

        # Raw-gradient projections for every tracked pair (plan section 1.1).
        s = self.subspace.project(G)
        for track, s_i in zip(self.directions, s.tolist()):
            track.observe(s_i)

        # Per-matrix per-step scalars.
        sigma_top = float(self.subspace.project(M)[: self.subspace.k1].abs().max())
        self.steps.append(step)
        self.grad_fro_norm.append(float(torch.linalg.norm(G.detach().float())))
        self.top_sigma_m.append(sigma_top)

        if step % self.snapshot_every == 0 or step == 1:
            for track in self.directions:
                track.snapshot(step)

    # --------------------------------------------------------------- refresh

    def _refresh(
        self,
        M: torch.Tensor,
        step: int,
        *,
        hvp_fn: Optional[HvpFn],
        param: Optional[torch.Tensor],
    ) -> None:
        result = self.subspace.refresh(M)
        self.refresh_steps.append(step)
        U, V = self.subspace.all_u(), self.subspace.all_v()
        for i, track in enumerate(self.directions):
            align = float(result.alignment[i])
            track.refresh_alignment.append((step, align))
            track.sigma.append((step, float(result.sigma[i])))
            if not result.first and align < self.align_min:
                # Innovation: the tracked pair rotated -> reset confidence.
                track.reset(step)
            if hvp_fn is not None and param is not None:
                # Curvature along u_i v_i^T -- once per pair per refresh.
                # Phase-1 validation ONLY; never available to routing.
                D = torch.outer(U[:, i], V[:, i]).reshape(param.shape)
                lam = float(hvp_fn(param, D.to(dtype=param.dtype, device=param.device)))
                track.lambda_hvp.append((step, lam))

    # ------------------------------------------------------------------- log

    def to_log(self) -> Dict[str, Any]:
        """Serializable per-matrix log record (schema.py documents the shape)."""
        return {
            "shape": list(self.shape),
            "k1": self.subspace.k1,
            "k2": self.subspace.k2,
            "t_refresh": self.t_refresh,
            "align_min": self.align_min,
            "snapshot_every": self.snapshot_every,
            "steps": list(self.steps),
            "grad_fro_norm": list(self.grad_fro_norm),
            "top_sigma_m": list(self.top_sigma_m),
            "refresh_steps": list(self.refresh_steps),
            "directions": [
                {
                    "index": t.index,
                    "kind": t.kind,
                    "s": list(t.s_values),
                    "reset_steps": list(t.reset_steps),
                    "refresh_alignment": {
                        "step": [s for s, _ in t.refresh_alignment],
                        "value": [a for _, a in t.refresh_alignment],
                    },
                    "sigma": {
                        "step": [s for s, _ in t.sigma],
                        "value": [v for _, v in t.sigma],
                    },
                    "lambda_hvp": {
                        "step": [s for s, _ in t.lambda_hvp],
                        "value": [v for _, v in t.lambda_hvp],
                    },
                    "per_beta": {
                        _beta_key(b): {k: list(v) for k, v in t.snapshots[b].items()}
                        for b in t.betas
                    },
                }
                for t in self.directions
            ],
        }


def _beta_key(beta: float) -> str:
    """Stable string key for a beta value ('0.9', '0.99')."""
    return format(beta, "g")


class InstrumentationHub:
    """Multi-matrix instrumentation attached to a trainer loop.

    Tracks every parameter with ndim >= 2 (flattening >2-D to len(p) x -1,
    the Muon-family convention) whose flattened min dimension is at least
    ``min_dim``.  Reads the raw gradient from ``param.grad`` (the interface
    contract forbids pre_step from modifying G in place) and the momentum
    matrix from ``optimizer.state[param][momentum_key]`` after
    ``optimizer.step()``; falls back to the raw gradient itself when the
    optimizer keeps no momentum buffer (e.g. AdamW -- the tracked subspace
    then follows the gradient's own top directions).
    """

    def __init__(
        self,
        named_params: Iterable[Tuple[str, torch.Tensor]],
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        k1: int = 16,
        k2: int = 16,
        t_refresh: int = 50,
        subspace_iters: int = 2,
        betas: Sequence[float] = (0.9, 0.99),
        classifier_kwargs: Dict[str, Any],
        align_min: float = 0.9,
        snapshot_every: int = 1,
        seed: int = 1000,
        min_dim: int = 2,
        momentum_key: str = "momentum_buffer",
        hvp_fn: Optional[HvpFn] = None,
    ) -> None:
        self.optimizer = optimizer
        self.momentum_key = momentum_key
        self.hvp_fn = hvp_fn
        self.betas = [float(b) for b in betas]
        self._params: List[Tuple[str, torch.Tensor]] = []
        self.trackers: Dict[str, MatrixTracker] = {}
        for name, p in named_params:
            if p.ndim < 2:
                continue
            m = p.shape[0]
            n = int(p.numel() // m)
            if min(m, n) < max(min_dim, 2):
                continue
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed) + _stable_hash(name))
            self._params.append((name, p))
            self.trackers[name] = MatrixTracker(
                name,
                (m, n),
                k1=k1,
                k2=k2,
                t_refresh=t_refresh,
                subspace_iters=subspace_iters,
                betas=self.betas,
                classifier_kwargs=classifier_kwargs,
                align_min=align_min,
                snapshot_every=snapshot_every,
                generator=gen,
                device=p.device,  # linear algebra stays on-device; only
                # k scalars per matrix per step cross to the host stats.
            )
        if not self.trackers:
            raise ValueError("InstrumentationHub found no matrix parameters to track")

    # ------------------------------------------------------------------ step

    @torch.no_grad()
    def after_step(self) -> None:
        """Observe one training step. Call after optimizer.step(), before
        zero_grad (param.grad must still hold the raw gradient)."""
        for name, p in self._params:
            if p.grad is None:
                continue
            G = p.grad
            G2 = G.reshape(len(G), -1) if G.ndim > 2 else G
            M2 = self._momentum_matrix(p, G2)
            self.trackers[name].observe(
                G2.float(),
                M2.float(),
                hvp_fn=self.hvp_fn,
                param=p,
            )

    def _momentum_matrix(self, p: torch.Tensor, G2: torch.Tensor) -> torch.Tensor:
        if self.optimizer is not None:
            state = self.optimizer.state.get(p, {})
            buf = state.get(self.momentum_key)
            if buf is not None:
                return buf.reshape(len(buf), -1) if buf.ndim > 2 else buf
        return G2

    # ------------------------------------------------------------------- log

    def to_log(self) -> Dict[str, Any]:
        """Full instrumentation log (see src.instrument.schema)."""
        from src.instrument.schema import INSTRUMENTATION_SCHEMA_VERSION

        return {
            "instrumentation_schema_version": INSTRUMENTATION_SCHEMA_VERSION,
            "betas": [_beta_key(b) for b in self.betas],
            "hvp_enabled": self.hvp_fn is not None,
            "matrices": {name: tr.to_log() for name, tr in self.trackers.items()},
        }


def _stable_hash(name: str) -> int:
    """Deterministic (process-independent) small hash for per-matrix seeding."""
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) % 1_000_003
    return h


def hub_from_config(
    instr_cfg: Dict[str, Any],
    named_params: Iterable[Tuple[str, torch.Tensor]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    hvp_fn: Optional[HvpFn] = None,
) -> InstrumentationHub:
    """Build a hub from a config file's ``instrumentation:`` block.

    Expected keys (see configs/dev/instrumented_*.yaml): k1, k2, t_refresh,
    subspace_iters, betas, align_min, snapshot_every, seed, min_dim,
    classifier (dict of RegimeClassifier thresholds -- these have NO
    scientific defaults; dev configs carry placeholder values, Phase-1 values
    are pre-registered by the human in criteria/).
    """
    cfg = dict(instr_cfg)
    classifier_kwargs = dict(cfg.get("classifier", {}))
    if not classifier_kwargs:
        raise ValueError(
            "instrumentation config must provide a 'classifier' block "
            "(tau_sig, tau_noise, rho_osc, n_min, ...); there are no "
            "scientific defaults"
        )
    hvp_requested = bool(cfg.get("hvp", False))
    if hvp_requested and hvp_fn is None:
        raise ValueError(
            "config requests HVP probes but no hvp_fn callback was provided "
            "by the trainer (HVPs are Phase-1 validation only)"
        )
    return InstrumentationHub(
        named_params,
        optimizer,
        k1=int(cfg.get("k1", 16)),
        k2=int(cfg.get("k2", 16)),
        t_refresh=int(cfg.get("t_refresh", 50)),
        subspace_iters=int(cfg.get("subspace_iters", 2)),
        betas=tuple(cfg.get("betas", (0.9, 0.99))),
        classifier_kwargs=classifier_kwargs,
        align_min=float(cfg.get("align_min", 0.9)),
        snapshot_every=int(cfg.get("snapshot_every", 1)),
        seed=int(cfg.get("seed", 1000)),
        min_dim=int(cfg.get("min_dim", 2)),
        momentum_key=str(cfg.get("momentum_key", "momentum_buffer")),
        hvp_fn=hvp_fn if hvp_requested else None,
    )
