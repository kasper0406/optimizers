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
  confidence reset through the src.stats reset API
  (``BatchRegimeClassifier.reset_directions``: the direction restarts in
  SIGNAL and must re-earn a label with n_min fresh observations);
* optionally, once per tracked pair per refresh, a curvature probe
  lambda_i ~= vec(u_i v_i^T)^T H vec(u_i v_i^T) through a trainer-provided
  HVP callback;
* optionally, a THIRD tier of k3 FROZEN random probe directions
  (:class:`FrozenProbeBank`, default off) whose statistics accumulate over the
  entire run with an UNBOUNDED window -- see that class's docstring for the
  pre-registered question it exists to answer.

HVP POLICY (distributed invariant 3, plan "Distributed scalability"):
HVPs are for **Phase-1 validation only** -- they calibrate the trajectory-
derived implied eta*lambda estimator.  They are FORBIDDEN in any routing or
update path: no optimizer update, gain, or gating decision may consume
``lambda_hvp``.  WP2.x CI greps for HVP usage in the update path; keep it
that way.

All statistics (EMAs, autocorrelation, t-stats, implied eta*lambda,
classification) come from ``src.stats`` -- the WP0.5-tested code, in its
array mode (``BatchRegimeClassifier``, equivalence-tested against the scalar
WP0.5 path in tests/test_stats_batch_equivalence.py).  All k tracked
directions of a (matrix, beta) advance in O(1) Python calls per step; this
module computes projections and bookkeeping only and deliberately contains
no statistical formulas.

Per-step data movement: the k projections plus the two per-matrix scalars
are packed into one small device tensor per matrix; the hub concatenates
them across matrices and performs a SINGLE device->host ``.cpu()`` transfer
per training step (linear algebra stays on the tensors' device).

Call order per training step (see :class:`InstrumentationHub`):

    loss.backward()
    hub.capture_grads()       # only needed if optimizer.step() mutates p.grad
    optimizer.step()          # updates momentum buffers in optimizer.state
    hub.after_step()          # raw G (captured or param.grad) + momentum buffer
    optimizer.zero_grad()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.instrument.subspace import TrackedSubspace
from src.stats import BatchRegimeClassifier
from src.stats.batch import DirectionView

__all__ = [
    "DirectionTrack",
    "MatrixTracker",
    "InstrumentationHub",
    "hub_from_config",
    "HvpFn",
    "FrozenProbeAccumulator",
    "FrozenProbeBank",
]

# hvp_fn(param, direction_matrix) -> float
#   direction_matrix D = u v^T reshaped to the parameter's original shape,
#   ||D||_F = 1; returns vec(D)^T H vec(D) with H restricted to that matrix.
HvpFn = Callable[[torch.Tensor, torch.Tensor], float]


class DirectionTrack:
    """One tracked direction: log buffers + read-only views into the
    per-(matrix, beta) BatchRegimeClassifiers (``classifiers[beta]`` keeps
    the scalar-classifier-shaped API: .regime, .stats, .n_since_reset)."""

    def __init__(self, index: int, kind: str, views: Dict[float, DirectionView]):
        self.index = index
        self.kind = kind  # "top" | "bulk"
        self.betas = list(views)
        self.classifiers: Dict[float, DirectionView] = dict(views)
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


class FrozenProbeAccumulator:
    """Unbounded-window accumulation of k projection streams.

    Keeps, for k streams simultaneously and in O(k) work per step: the running
    count n, sum, sum of squares, and the running lagged cross-sums
    ``S_j = sum_t x_t x_{t-j}`` for j = 1..``max_lag``.  Nothing is ever
    forgotten, decayed, or reset -- that is the point of this tier.

    From those it reports, at any time:

    * ``t_naive = mean / sqrt(c_0 / n)`` -- the i.i.d. t-statistic of the mean
      projection.  Wrong (anti-conservative) when the stream is positively
      autocorrelated, which per-direction gradient projections generally are;
    * ``t_nw`` -- the same statistic with the variance of the mean replaced by
      a **Newey-West / Bartlett-kernel long-run variance** estimate

          sigma_LR^2 = c_0 + 2 * sum_{j=1..L} (1 - j/(L+1)) c_j,
          c_j = S_j / (n - j) - mean^2,
          L   = min(max_lag, floor(4 (n/100)^(2/9)))      [Newey-West 1994 rule]

      so ``t_nw = mean / sqrt(sigma_LR^2 / n)``.  The Bartlett weights make
      sigma_LR^2 positive-semidefinite by construction; on the rare stream
      where the truncated sum still comes out non-positive the estimator falls
      back to ``c_0`` and sets the ``nw_floored`` flag;
    * ``ess = n * c_0 / sigma_LR^2`` -- the effective sample size implied by
      that correction (n for white noise, smaller for positively correlated
      streams, LARGER for negatively correlated ones -- the negative-rho
      population documented in reports/wp12-phase1-measurement.md makes
      ``t_nw`` *larger* than ``t_naive`` there, which is why both are logged
      and neither is called "the" t-statistic).

    ``c_j`` uses the divisor ``n - j`` and subtracts the full-sample mean; the
    resulting O(1/n) endpoint bias is negligible at run lengths where the
    long-integration question is being asked, and is documented rather than
    corrected.
    """

    def __init__(self, k: int, *, max_lag: int = 8) -> None:
        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if int(max_lag) < 0:
            raise ValueError(f"max_lag must be >= 0, got {max_lag}")
        self.k = int(k)
        self.max_lag = int(max_lag)
        self.n = 0
        self._sum = np.zeros(self.k, dtype=np.float64)
        self._lag_sums = np.zeros((self.max_lag + 1, self.k), dtype=np.float64)
        self._lag_counts = np.zeros(self.max_lag + 1, dtype=np.int64)
        self._hist: List[np.ndarray] = []  # most recent max_lag observations

    def update(self, x: np.ndarray) -> None:
        """Absorb one observation vector (k,).  Never resets."""
        x = np.asarray(x, dtype=np.float64).reshape(self.k)
        self.n += 1
        self._sum += x
        self._lag_sums[0] += x * x
        self._lag_counts[0] += 1
        for j in range(1, min(self.max_lag, len(self._hist)) + 1):
            self._lag_sums[j] += x * self._hist[-j]
            self._lag_counts[j] += 1
        self._hist.append(x)
        if self.max_lag and len(self._hist) > self.max_lag:
            del self._hist[0]
        elif self.max_lag == 0:
            self._hist.clear()

    # ---------------------------------------------------------------- stats

    def lag_truncation(self) -> int:
        """Newey-West (1994) automatic Bartlett bandwidth, capped at max_lag."""
        if self.n < 3 or self.max_lag == 0:
            return 0
        rule = int(np.floor(4.0 * (self.n / 100.0) ** (2.0 / 9.0)))
        return int(max(0, min(self.max_lag, rule, self.n - 2)))

    def stats(self) -> Dict[str, np.ndarray]:
        """Current cumulative statistics for every stream (arrays of shape (k,))."""
        n = self.n
        if n == 0:
            nan = np.full(self.k, np.nan)
            return {
                "n": 0,
                "lag_truncation": 0,
                "mean": nan.copy(),
                "var": nan.copy(),
                "sigma_lr2": nan.copy(),
                "t_naive": nan.copy(),
                "t_nw": nan.copy(),
                "ess": nan.copy(),
                "nw_floored": np.zeros(self.k, dtype=bool),
            }
        mean = self._sum / n
        c0 = self._lag_sums[0] / n - mean * mean
        c0 = np.maximum(c0, 0.0)
        L = self.lag_truncation()
        sigma_lr2 = c0.copy()
        for j in range(1, L + 1):
            cnt = self._lag_counts[j]
            if cnt <= 0:
                continue
            c_j = self._lag_sums[j] / cnt - mean * mean
            sigma_lr2 = sigma_lr2 + 2.0 * (1.0 - j / (L + 1.0)) * c_j
        floored = sigma_lr2 <= 0.0
        sigma_lr2 = np.where(floored, c0, sigma_lr2)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_naive = np.where(c0 > 0, mean / np.sqrt(c0 / n), 0.0)
            t_nw = np.where(sigma_lr2 > 0, mean / np.sqrt(sigma_lr2 / n), 0.0)
            ess = np.where(sigma_lr2 > 0, n * c0 / sigma_lr2, float(n))
        return {
            "n": n,
            "lag_truncation": L,
            "mean": mean,
            "var": c0,
            "sigma_lr2": sigma_lr2,
            "t_naive": t_naive,
            "t_nw": t_nw,
            "ess": ess,
            "nw_floored": floored,
        }


class FrozenProbeBank:
    """The third instrumentation tier: k3 FROZEN random probe directions.

    PRE-REGISTERED QUESTION
    -----------------------
        Along a frozen (never-refreshed, never-reset) random direction, does
        the cumulative t-statistic of the gradient projection grow like
        sqrt(t) -- the signature of a persistent, non-zero-mean component,
        since drift accumulates like T while zero-mean noise accumulates like
        sqrt(T) -- or does it stay flat/bounded at the white-noise scale?  And
        does ANY frozen probe cross the conventional |t| >= 4 by the end of a
        run?

        The contrast is with the tracked (top-k1) and bulk (k2) tiers, whose
        t-statistics are structurally capped: those use EMA statistics (a
        bounded effective window, |t| <= SNR * sqrt(ess) with ess set by beta)
        and are additionally RESET whenever the refreshed subspace rotates.  A
        per-direction persistent signal too weak to clear a bounded window can
        still be detectable at long integration; this tier is the only place
        in the instrumentation where the window is unbounded.

        Descriptive measurement only: no threshold here decides anything.

    DESIGN NOTES
    ------------
    * The k3 direction pairs (u_j, v_j) are drawn ONCE at construction from
      the supplied ``torch.Generator`` (unit-norm Gaussian columns, drawn on
      CPU for device-independent determinism) and are never touched again --
      not on refresh, not on innovation detection, not on reset.  A run's
      probe set is reproducible from its instrumentation seed alone.
    * They are deliberately NOT orthogonalized against the tracked top block
      or against each other: orthogonalizing against a block that is refreshed
      every T_refresh steps would make the probes rotate with it, which is
      exactly the property this tier exists to avoid.  Two random probes in a
      high-dimensional matrix space are near-orthogonal anyway, and the
      analysis treats probes as (weakly dependent) repeated measurements
      rather than an orthogonal basis.
    * The raw projection series is logged, optionally decimated by
      ``decimate`` (default 1 = keep every step) for size control; the
      cumulative statistics are always computed from EVERY observation, never
      from the decimated series.
    """

    def __init__(
        self,
        m: int,
        n: int,
        k3: int,
        *,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        max_lag: int = 8,
        decimate: int = 1,
    ) -> None:
        if int(k3) < 1:
            raise ValueError(f"k3 must be >= 1, got {k3}")
        if int(decimate) < 1:
            raise ValueError(f"decimate must be >= 1, got {decimate}")
        self.k3 = int(k3)
        self.decimate = int(decimate)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        U = torch.randn(int(m), self.k3, generator=generator, dtype=dtype)
        V = torch.randn(int(n), self.k3, generator=generator, dtype=dtype)
        self.U = (U / torch.linalg.norm(U, dim=0, keepdim=True)).to(self.device)
        self.V = (V / torch.linalg.norm(V, dim=0, keepdim=True)).to(self.device)
        self.acc = FrozenProbeAccumulator(self.k3, max_lag=max_lag)
        # Log buffers.
        self.raw_steps: List[int] = []
        self.raw: List[List[float]] = [[] for _ in range(self.k3)]
        self.snapshot_steps: List[int] = []
        self.snapshots: Dict[str, List[List[float]]] = {
            key: [[] for _ in range(self.k3)]
            for key in ("t_naive", "t_nw", "ess", "mean", "var")
        }
        self.snapshot_n: List[int] = []
        self.snapshot_lag_truncation: List[int] = []

    def project(self, G: torch.Tensor) -> torch.Tensor:
        """Frozen-direction projections p_j = u_j^T G v_j, shape (k3,)."""
        G = G.detach().to(dtype=self.dtype, device=self.device)
        return ((self.U.T @ G) * self.V.T).sum(dim=1)

    def observe(self, x: np.ndarray, step: int, *, snapshot: bool) -> None:
        """Absorb one step's projections; optionally append a stat snapshot."""
        self.acc.update(x)
        if (step - 1) % self.decimate == 0:
            self.raw_steps.append(int(step))
            for j in range(self.k3):
                self.raw[j].append(float(x[j]))
        if snapshot:
            st = self.acc.stats()
            self.snapshot_steps.append(int(step))
            self.snapshot_n.append(int(st["n"]))
            self.snapshot_lag_truncation.append(int(st["lag_truncation"]))
            for key in self.snapshots:
                arr = st[key]
                for j in range(self.k3):
                    self.snapshots[key][j].append(float(arr[j]))

    def to_log(self) -> Dict[str, Any]:
        final = self.acc.stats()
        return {
            "k3": self.k3,
            "max_lag": self.acc.max_lag,
            "decimate": self.decimate,
            "n_observations": int(final["n"]),
            "lag_truncation": int(final["lag_truncation"]),
            "snapshot_steps": list(self.snapshot_steps),
            "snapshot_n": list(self.snapshot_n),
            "snapshot_lag_truncation": list(self.snapshot_lag_truncation),
            "raw_steps": list(self.raw_steps),
            "probes": [
                {
                    "index": j,
                    "s": list(self.raw[j]),
                    "t_naive": list(self.snapshots["t_naive"][j]),
                    "t_nw": list(self.snapshots["t_nw"][j]),
                    "ess": list(self.snapshots["ess"][j]),
                    "mean": list(self.snapshots["mean"][j]),
                    "var": list(self.snapshots["var"][j]),
                    "final": {
                        "n": int(final["n"]),
                        "mean": float(final["mean"][j]),
                        "var": float(final["var"][j]),
                        "t_naive": float(final["t_naive"][j]),
                        "t_nw": float(final["t_nw"][j]),
                        "ess": float(final["ess"][j]),
                        "nw_floored": bool(final["nw_floored"][j]),
                    },
                }
                for j in range(self.k3)
            ],
        }


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
        k3: int = 0,
        frozen_max_lag: int = 8,
        frozen_decimate: int = 1,
        frozen_generator: Optional[torch.Generator] = None,
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
        self.betas = [float(b) for b in betas]
        k_total = self.subspace.k_total
        # One BatchRegimeClassifier per beta covering ALL tracked directions
        # of this matrix (array mode; O(1) Python calls per step per beta).
        self.classifiers: Dict[float, BatchRegimeClassifier] = {
            b: BatchRegimeClassifier(beta=b, k=k_total, **dict(classifier_kwargs))
            for b in self.betas
        }
        self.directions: List[DirectionTrack] = [
            DirectionTrack(
                i, kind, {b: self.classifiers[b].view(i) for b in self.betas}
            )
            for i, kind in enumerate(self.subspace.kinds())
        ]
        # Third tier (default off): frozen random probes, unbounded window.
        # Its directions are drawn from a SEPARATE generator so that enabling
        # the tier cannot perturb the RNG stream the tracked subspace consumes
        # (runs with and without frozen probes keep identical tracked pairs).
        self.frozen: Optional[FrozenProbeBank] = None
        if int(k3) > 0:
            self.frozen = FrozenProbeBank(
                m,
                n,
                int(k3),
                generator=frozen_generator,
                device=device,
                max_lag=int(frozen_max_lag),
                decimate=int(frozen_decimate),
            )
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

        Convenience wrapper around :meth:`prepare` + :meth:`finish` with a
        per-matrix device->host transfer; the hub fuses transfers across
        matrices instead (one ``.cpu()`` per training step).
        """
        packed = self.prepare(G, M, hvp_fn=hvp_fn, param=param)
        self.finish(packed.cpu().numpy())

    def prepare(
        self,
        G: torch.Tensor,
        M: torch.Tensor,
        *,
        hvp_fn: Optional[HvpFn] = None,
        param: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Device-side half of one instrumented step.

        Runs the (possibly due) subspace refresh and packs everything that
        must cross to the host -- the k per-direction projections s_i =
        u_i^T G v_i, the top-sigma estimate of M, and ||G||_F -- into one
        (k + 2,) tensor on G's device.  No host synchronization here.
        """
        self.step_count += 1
        step = self.step_count

        if (step - 1) % self.t_refresh == 0:
            self._refresh(M, step, hvp_fn=hvp_fn, param=param)

        # Raw-gradient projections for every tracked pair (plan section 1.1).
        s = self.subspace.project(G)
        sigma_top = self.subspace.project(M)[: self.subspace.k1].abs().max()
        gnorm = torch.linalg.norm(G.detach().float())
        parts = [s, sigma_top.reshape(1), gnorm.reshape(1)]
        if self.frozen is not None:
            # Frozen-tier projections ride the same single device->host
            # transfer; the frozen directions are NOT refreshed above.
            parts.append(self.frozen.project(G))
        return torch.cat(parts)

    def finish(self, packed: np.ndarray) -> None:
        """Host-side half: unpack the :meth:`prepare` payload and advance the
        batched statistics/classifiers for every beta in O(1) numpy calls."""
        step = self.step_count
        k = self.subspace.k_total
        s_np = np.asarray(packed[:k], dtype=np.float64)
        sigma_top = float(packed[k])
        gnorm = float(packed[k + 1])

        for clf in self.classifiers.values():
            clf.update(s_np)
        for i, track in enumerate(self.directions):
            track.s_values.append(float(s_np[i]))

        # Per-matrix per-step scalars.
        self.steps.append(step)
        self.grad_fro_norm.append(gnorm)
        self.top_sigma_m.append(sigma_top)

        snapshot_due = step % self.snapshot_every == 0 or step == 1
        if self.frozen is not None:
            self.frozen.observe(
                np.asarray(packed[k + 2 : k + 2 + self.frozen.k3], dtype=np.float64),
                step,
                snapshot=snapshot_due,
            )
        if snapshot_due:
            self._snapshot_all(step)

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
        # Refresh happens every t_refresh steps; the (k,)-sized transfers
        # here are off the per-step path.
        alignment = result.alignment.cpu().numpy()
        sigma = result.sigma.cpu().numpy()
        U, V = self.subspace.all_u(), self.subspace.all_v()
        rotated: List[int] = []
        for i, track in enumerate(self.directions):
            align = float(alignment[i])
            track.refresh_alignment.append((step, align))
            track.sigma.append((step, float(sigma[i])))
            if not result.first and align < self.align_min:
                # Innovation: the tracked pair rotated -> reset confidence.
                rotated.append(i)
                track.reset_steps.append(step)
            if hvp_fn is not None and param is not None:
                # Curvature along u_i v_i^T -- once per pair per refresh.
                # Phase-1 validation ONLY; never available to routing.
                D = torch.outer(U[:, i], V[:, i]).reshape(param.shape)
                lam = float(hvp_fn(param, D.to(dtype=param.dtype, device=param.device)))
                track.lambda_hvp.append((step, lam))
        if rotated:
            for clf in self.classifiers.values():
                clf.reset_directions(rotated)
        # NOTE: self.frozen is deliberately untouched here. The frozen tier is
        # never refreshed and never reset by innovation detection -- that is
        # the property that makes its integration window unbounded.

    # -------------------------------------------------------------- snapshot

    def _snapshot_all(self, step: int) -> None:
        """Append one stat snapshot for every direction x beta (batched:
        each per-beta stat array is computed once per matrix)."""
        for b, clf in self.classifiers.items():
            st = clf.stats
            mu = st.mean
            var = st.var
            rho = st.rho_corrected
            t_stat = st.t_stat
            amp = st.amplitude_ratio
            iel = st.implied_eta_lambda
            ess = st.ess
            n_obs = st.n_obs
            regimes = clf.regimes
            for i, track in enumerate(self.directions):
                snap = track.snapshots[b]
                snap["step"].append(int(step))
                snap["regime"].append(regimes[i].value)
                snap["mu"].append(float(mu[i]))
                snap["var"].append(float(var[i]))
                snap["rho"].append(float(rho[i]))
                snap["t_stat"].append(float(t_stat[i]))
                snap["amplitude_ratio"].append(float(amp[i]))
                snap["implied_eta_lambda"].append(float(iel[i]))
                snap["ess"].append(float(ess[i]))
                snap["n_since_reset"].append(int(n_obs[i]))

    # ------------------------------------------------------------------- log

    def to_log(self) -> Dict[str, Any]:
        """Serializable per-matrix log record (schema.py documents the shape)."""
        record: Dict[str, Any] = {
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
        if self.frozen is not None:
            record["frozen_probes"] = self.frozen.to_log()
        return record


def _beta_key(beta: float) -> str:
    """Stable string key for a beta value ('0.9', '0.99')."""
    return format(beta, "g")


class InstrumentationHub:
    """Multi-matrix instrumentation attached to a trainer loop.

    Tracks every parameter with ndim >= 2 (flattening >2-D to len(p) x -1,
    the Muon-family convention) whose flattened min dimension is at least
    ``min_dim``.  Reads the raw gradient from ``param.grad`` (the interface
    contract forbids pre_step from modifying G in place) -- or from the
    snapshot taken by :meth:`capture_grads` when the optimizer's ``step()``
    mutates ``p.grad`` in place -- and the momentum matrix from
    ``optimizer.state[param][momentum_key]`` after ``optimizer.step()``;
    falls back to the raw gradient itself when the optimizer keeps no
    momentum buffer (e.g. AdamW -- the tracked subspace then follows the
    gradient's own top directions).
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
        k3: int = 0,
        frozen_max_lag: int = 8,
        frozen_decimate: int = 1,
    ) -> None:
        self.optimizer = optimizer
        self.k3 = int(k3)
        self.momentum_key = momentum_key
        self.hvp_fn = hvp_fn
        self.betas = [float(b) for b in betas]
        self._params: List[Tuple[str, torch.Tensor]] = []
        self._captured: Dict[str, torch.Tensor] = {}
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
            frozen_gen = None
            if self.k3 > 0:
                # Disjoint stream: enabling frozen probes must not shift the
                # tracked subspace's RNG consumption.
                frozen_gen = torch.Generator(device="cpu")
                frozen_gen.manual_seed(int(seed) + _stable_hash(name) + FROZEN_SEED_OFFSET)
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
                k3=self.k3,
                frozen_max_lag=int(frozen_max_lag),
                frozen_decimate=int(frozen_decimate),
                frozen_generator=frozen_gen,
            )
        if not self.trackers:
            raise ValueError("InstrumentationHub found no matrix parameters to track")

    # ------------------------------------------------------------------ step

    @torch.no_grad()
    def capture_grads(self) -> None:
        """Snapshot raw pre-momentum gradients (call between backward() and
        optimizer.step()).  Only needed when the optimizer mutates ``p.grad``
        in place during ``step()``; :meth:`after_step` prefers the snapshot
        and falls back to ``param.grad`` otherwise."""
        captured: Dict[str, torch.Tensor] = {}
        for name, p in self._params:
            if p.grad is None:
                continue
            G = p.grad
            G2 = G.reshape(len(G), -1) if G.ndim > 2 else G
            captured[name] = G2.detach().float().clone()
        self._captured = captured

    @torch.no_grad()
    def after_step(self) -> None:
        """Observe one training step. Call after optimizer.step(), before
        zero_grad (param.grad -- or the capture_grads() snapshot -- must
        hold the raw gradient).  All per-direction scalars cross the
        device->host boundary in ONE batched .cpu() transfer."""
        prepared: List[Tuple[MatrixTracker, torch.Tensor]] = []
        for name, p in self._params:
            G2 = self._captured.pop(name, None)
            if G2 is None:
                if p.grad is None:
                    continue
                G = p.grad
                G2 = (G.reshape(len(G), -1) if G.ndim > 2 else G).float()
            M2 = self._momentum_matrix(p, G2)
            packed = self.trackers[name].prepare(
                G2,
                M2.float(),
                hvp_fn=self.hvp_fn,
                param=p,
            )
            prepared.append((self.trackers[name], packed))
        self._captured = {}
        if not prepared:
            return
        if len(prepared) == 1:
            fused = prepared[0][1].cpu().numpy()
            prepared[0][0].finish(fused)
            return
        fused = torch.cat([packed for _, packed in prepared]).cpu().numpy()
        offset = 0
        for tracker, packed in prepared:
            n = packed.numel()
            tracker.finish(fused[offset : offset + n])
            offset += n

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
            "frozen_probes_enabled": self.k3 > 0,
            "matrices": {name: tr.to_log() for name, tr in self.trackers.items()},
        }


FROZEN_SEED_OFFSET = 7919  # keeps the frozen-probe RNG stream disjoint


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
    frozen = _frozen_probe_settings(cfg.get("frozen_probes"))
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
        k3=frozen["k3"],
        frozen_max_lag=frozen["max_lag"],
        frozen_decimate=frozen["decimate"],
    )


FROZEN_PROBE_DEFAULTS = {"k3": 16, "max_lag": 8, "decimate": 1}


def _frozen_probe_settings(spec: Any) -> Dict[str, int]:
    """Parse ``instrumentation.frozen_probes`` (default OFF).

    Accepted forms::

        frozen_probes: false            # or key absent -> disabled
        frozen_probes: true             # enabled with the defaults
        frozen_probes: 8                # enabled with k3 = 8
        frozen_probes: {enabled: true, k3: 16, max_lag: 8, decimate: 1}
    """
    out = {"k3": 0, "max_lag": FROZEN_PROBE_DEFAULTS["max_lag"],
           "decimate": FROZEN_PROBE_DEFAULTS["decimate"]}
    if spec is None or spec is False:
        return out
    if spec is True:
        out["k3"] = FROZEN_PROBE_DEFAULTS["k3"]
        return out
    if isinstance(spec, int):
        out["k3"] = int(spec)
        return out
    if not isinstance(spec, dict):
        raise ValueError(
            "instrumentation.frozen_probes must be a bool, an int (k3), or a "
            f"mapping; got {type(spec).__name__}"
        )
    if not bool(spec.get("enabled", True)):
        return out
    out["k3"] = int(spec.get("k3", FROZEN_PROBE_DEFAULTS["k3"]))
    out["max_lag"] = int(spec.get("max_lag", FROZEN_PROBE_DEFAULTS["max_lag"]))
    out["decimate"] = int(spec.get("decimate", FROZEN_PROBE_DEFAULTS["decimate"]))
    if out["k3"] < 0:
        raise ValueError(f"frozen_probes.k3 must be >= 0, got {out['k3']}")
    return out
