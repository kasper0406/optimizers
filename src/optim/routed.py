"""Routed Muon v0 (WP2.1, plan section 2.1): two-tier, COSMOS-shaped.

Update rule per Muon-managed matrix W (flattened to 2-D, m x n):

    Bulk tier (identical to src/optim/muon.py):
        buf_t = momentum * buf_{t-1} + G_t
        M_t   = G_t + momentum * buf_t   (nesterov; else buf_t)
        O_t   = NewtonSchulz5(M_t)       (same coefficients, same ns_dtype)

    Tracked tier (k singular pairs of M_t, warm-started subspace iteration,
    refreshed every t_refresh steps):
        s_i(t) = u_i^T G_t v_i           (RAW pre-momentum gradient)
        all k streams feed ONE src.stats.BatchRegimeClassifier (per matrix,
        single beta) -> per-direction regime + statistics
        O_t <- O_t + sum_i (g(i) - 1) * (u_i^T O_t v_i) * u_i v_i^T

    Routing gain g(i) by regime:
        SIGNAL       (|t| >= tau_sig, rho not strongly negative): g = 1
                     (stock Muon; also every direction's starting prior).
        NOISE        (|t| < tau_noise, |rho| small): g = g_noise (default
                     0.25 -- conservative floor, misclassification asymmetry:
                     a starved weak-signal direction is exactly what Muon
                     exists to feed).
        OSCILLATING  (rho <= -rho_osc) AND amplitude non-decaying
                     (amplitude_ratio >= 1 - amp_decay_margin):
                     adaptive mode (default, g_osc_const is None):
                       g = clip(1 / (eta_lambda_implied - 1), g_osc_min, 1)
                         = clip(1 / amplitude_ratio,          g_osc_min, 1)
                       targeting critical damping eta*lambda -> 1, with
                       eta_lambda_implied = 1 + amplitude_ratio read from the
                       trajectory (src.stats machinery).
                     constant mode (Gate-1 amendment A2, g_osc_const set):
                       g = g_osc_const exactly (fixed attenuation), because
                       the Gate-1 record found the adaptive values match the
                       pure-noise null of the ratio statistic (~0.53
                       near-constant); the constant arm adjudicates
                       "adaptive vs constant attenuation".
                     In BOTH modes a decaying oscillation
                     (amplitude_ratio < 1 - amp_decay_margin) is left alone
                     (g = 1): it is already self-damping.

Confidence (plan section 2.1): every direction starts in SIGNAL (= stock
behavior) and only leaves it once the classifier's n_min effective-sample
gate passes -- that logic lives entirely in the WP0.5-validated
src.stats.BatchRegimeClassifier (no reimplementation here). A subspace
refresh whose alignment for a direction falls below align_min (innovation:
the pair rotated) resets that direction's statistics and confidence through
``BatchRegimeClassifier.reset_directions`` -> back to SIGNAL, n_min clock
restarts.

Ablation / scoping switches:
    enable_noise_channel / enable_oscillation_channel -- Gate-1 scoping
        (e.g. oscillation-only branch) as a config change; with BOTH off the
        optimizer is bit-for-bit stock Muon (the correction is skipped
        entirely, not added-as-zero).
    rho_ignored (ablation 4c) -- magnitude-only gate: the classifier is
        built with an unreachable oscillation threshold (rho is clipped to
        [-1, 1], so rho <= -2 never fires); signal/noise gating from the
        t-statistic alone. Isolates the autocorrelation channel.
    random_gating (ablation 4d, placebo) -- the per-step gain vector is
        randomly permuted across the k tracked directions by a dedicated
        numpy RNG seeded from the optimizer ``seed``: the same fraction of
        directions is gated with the same gain values, but assignment is
        random.

Distributed invariants (plan "Distributed scalability", enforced by
tests/test_optim_routed_invariants.py):
    * ALL routing state is per-matrix, owned alongside that matrix's
      momentum buffer in ``optimizer.state[param]`` (single-writer pattern);
      no cross-matrix state, no module-level mutable state.
    * Projections are k rank-1 contractions ((U^T G) * V^T summed over the
      shared dim -- k scalars per matrix per step); no full-gradient gathers.
    * The routing path is trajectory-only: the oscillation gain reads
      eta*lambda from amplitude ratios. No curvature probes of any kind in
      src/optim (CI greps for the forbidden estimator names).
    * Subspace refresh is 2-3 warm-started power iterations (sharded-matvec
      friendly, Dion-style).

The small power-iteration core below is copy-adapted from
src/instrument/subspace.py (TrackedSubspace, top block only -- credit where
due); it is deliberately NOT imported: src.optim must never depend on
src.instrument.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from src.optim.interface import MatrixOptimizer
from src.optim.muon import Muon
from src.optim.newton_schulz import zeropower_via_newtonschulz5
from src.stats import BatchRegimeClassifier, Regime

__all__ = ["RoutedMuon"]

# rho is clipped to [-1, 1] in src.stats; a threshold of 2 makes the
# oscillation branch (rho <= -rho_osc) unreachable -- the rho_ignored mode.
_RHO_OSC_UNREACHABLE = 2.0


class _TrackedTier:
    """Per-matrix tracked tier: warm-started top-k subspace + classifier.

    Power-iteration core copy-adapted from src/instrument/subspace.py
    (TrackedSubspace: warm start, QR orthonormalization, k x k SVD rotation
    onto singular pairs, joint sign canonicalization, signed alignment
    score) restricted to the top block -- no bulk probes, no logging. Kept
    local so src.optim has no dependency on src.instrument.

    All state (U, V, classifier EMAs, gating RNG) is owned by this object,
    which lives in ``optimizer.state[param]["routing"]`` -- the per-matrix
    single-writer invariant.
    """

    def __init__(
        self,
        m: int,
        n: int,
        *,
        k: int,
        iters: int,
        align_min: float,
        classifier: BatchRegimeClassifier,
        seed: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.m, self.n, self.k = int(m), int(n), int(k)
        self.iters = int(iters)
        self.align_min = float(align_min)
        self.classifier = classifier
        self.device = device
        self.dtype = dtype
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(int(seed))
        # Dedicated placebo RNG (random_gating), seeded from the same seed.
        self.gating_rng = np.random.default_rng(int(seed))

        self.U: Optional[torch.Tensor] = None  # (m, k)
        self.V: Optional[torch.Tensor] = None  # (n, k)
        self.sigma: Optional[torch.Tensor] = None  # (k,)
        self.n_refreshes = 0
        # Introspection (tests / debugging): what the last step applied.
        self.last_gains: Optional[np.ndarray] = None
        self.last_regimes: Optional[List[Regime]] = None
        self.last_alignment: Optional[np.ndarray] = None
        # Routing telemetry (Gate-1 amendment A5): cheap per-step counters,
        # accumulated per matrix; read out via RoutedMuon.routing_stats().
        self.n_rotation_resets = 0  # refresh-alignment confidence resets
        self._n_innovation_resets = 0  # classifier z-detector resets (mirror)
        self.last_stats: Optional[Dict[str, Any]] = None
        self.cum = {
            "n_steps": 0,
            "direction_steps": 0,
            "signal_direction_steps": 0,
            "noise_direction_steps": 0,
            "oscillating_direction_steps": 0,
            "treated_direction_steps": 0,
            "in_confidence_window_direction_steps": 0,
            "gain_sum": 0.0,
            "gain_min": None,  # min/max over all direction-steps so far
            "gain_max": None,
        }

    # ------------------------------------------------------------- internals

    def _randn(self, rows: int, cols: int) -> torch.Tensor:
        # Draw on CPU with the instance generator (device-independent
        # determinism), then move to the compute device. [subspace.py]
        x = torch.randn(rows, cols, generator=self._gen, dtype=self.dtype)
        return x.to(self.device)

    @staticmethod
    def _orthonormalize(X: torch.Tensor) -> torch.Tensor:
        Q, _ = torch.linalg.qr(X, mode="reduced")
        return Q

    # --------------------------------------------------------------- refresh

    def refresh(self, M: torch.Tensor) -> None:
        """Warm-started subspace refresh against the momentum matrix M;
        resets classifier confidence for rotated directions (innovation:
        signed alignment <u_new, u_old><v_new, v_old> < align_min)."""
        M = M.detach().to(dtype=self.dtype, device=self.device)
        first = self.U is None
        old_u, old_v = self.U, self.V

        U = self._randn(self.m, self.k) if first else self.U.clone()
        U = self._orthonormalize(U)
        n_iters = self.iters if not first else max(self.iters, 8)
        for _ in range(n_iters):
            V = self._orthonormalize(M.T @ U)
            U = self._orthonormalize(M @ V)

        # Rotate onto singular-pair estimates: B = P S Q^T -> U P, V Q.
        B = U.T @ M @ V  # (k, k)
        P, S, Qh = torch.linalg.svd(B)
        self.U = U @ P
        self.V = V @ Qh.T
        self.sigma = S  # sorted descending

        if first:
            self.last_alignment = np.ones(self.k)
        else:
            # Joint sign canonicalization (s_i invariant under joint flips).
            sign = torch.sign((self.U * old_u).sum(dim=0))
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            self.U = self.U * sign
            self.V = self.V * sign
            align = (self.U * old_u).sum(dim=0) * (self.V * old_v).sum(dim=0)
            align_np = align.detach().cpu().numpy()
            self.last_alignment = align_np
            rotated = np.nonzero(align_np < self.align_min)[0]
            if rotated.size:
                # Innovation -> confidence reset -> start-in-SIGNAL.
                self.classifier.reset_directions(rotated.tolist())
                self.n_rotation_resets += int(rotated.size)
        self.n_refreshes += 1

    # ------------------------------------------------------------ projections

    def project(self, G: torch.Tensor) -> torch.Tensor:
        """s_i = u_i^T G v_i for all k pairs: k rank-1 contractions (the
        distributed-safe primitive -- k scalars, never a gradient gather)."""
        assert self.U is not None and self.V is not None
        G = G.detach().to(dtype=self.dtype, device=self.device)
        return ((self.U.T @ G) * self.V.T).sum(dim=1)

    # -------------------------------------------------------------- telemetry

    def record_step(self, step: int, gains: np.ndarray) -> None:
        """Accumulate per-step routing telemetry (Gate-1 amendment A5).

        Cheap: a handful of numpy reductions over the (k,) gain vector and
        the classifier's (k,) label/count arrays; no tensors, no copies of
        anything large. Called once per matrix per step from
        RoutedMuon.shape_spectrum after the applied gains are known
        (post-random_gating: these ARE the applied gains).
        """
        clf = self.classifier
        regimes = clf.regimes  # (k,) labels, one list build per step
        n_signal = sum(r is Regime.SIGNAL for r in regimes)
        n_noise = sum(r is Regime.NOISE for r in regimes)
        n_osc = self.k - n_signal - n_noise
        n_treated = int(np.count_nonzero(gains != 1.0))
        n_in_window = int(np.count_nonzero(clf.n_since_reset < clf.n_min))
        g_sum = float(gains.sum())
        g_min = float(gains.min())
        g_max = float(gains.max())
        n_innov_resets = sum(len(r) for r in clf.reset_steps)

        self.last_stats = {
            "step": int(step),
            "k": self.k,
            "n_signal": n_signal,
            "n_noise": n_noise,
            "n_oscillating": n_osc,
            "n_treated": n_treated,
            "treated_fraction": n_treated / self.k,
            "n_in_confidence_window": n_in_window,
            "gain_sum": g_sum,
            "gain_min": g_min,
            "gain_max": g_max,
        }
        cum = self.cum
        cum["n_steps"] += 1
        cum["direction_steps"] += self.k
        cum["signal_direction_steps"] += n_signal
        cum["noise_direction_steps"] += n_noise
        cum["oscillating_direction_steps"] += n_osc
        cum["treated_direction_steps"] += n_treated
        cum["in_confidence_window_direction_steps"] += n_in_window
        cum["gain_sum"] += g_sum
        cum["gain_min"] = g_min if cum["gain_min"] is None else min(cum["gain_min"], g_min)
        cum["gain_max"] = g_max if cum["gain_max"] is None else max(cum["gain_max"], g_max)
        # Stored, not accumulated (already cumulative at the source).
        self._n_innovation_resets = n_innov_resets

    def stats_snapshot(self) -> Dict[str, Any]:
        """JSON-able per-matrix telemetry: last step + cumulative + resets."""
        cum = dict(self.cum)
        ds = max(cum["direction_steps"], 1)
        cum["treated_fraction"] = cum["treated_direction_steps"] / ds
        return {
            "shape": [self.m, self.n],
            "k": self.k,
            "n_refreshes": self.n_refreshes,
            "n_rotation_resets": self.n_rotation_resets,
            "n_innovation_resets": self._n_innovation_resets,
            "n_confidence_resets": self.n_rotation_resets + self._n_innovation_resets,
            "last": None if self.last_stats is None else dict(self.last_stats),
            "cumulative": cum,
        }


class RoutedMuon(Muon):
    """Routed Muon v0: stock Muon bulk tier + per-direction routed gains.

    Muon hyperparameters (lr, momentum, nesterov, ns_steps, eps, adjust_lr,
    ns_dtype, weight_decay) behave exactly as in :class:`src.optim.muon.Muon`.

    Routing hyperparameters (plan section 2.1 defaults; only g_noise,
    rho_osc, k are meant to be swept):

        k                    tracked singular pairs per matrix (16; 16-64).
        t_refresh            subspace refresh cadence in steps (50).
        beta                 EMA timescale of the routing statistics (0.99).
        tau_sig, tau_noise   t-statistic thresholds for SIGNAL / NOISE
                             (dev placeholders 4.0 / 2.0, matching the
                             instrumented dev configs; Phase-1-informed
                             values arrive via config).
        rho_osc              oscillation threshold on rho (dev placeholder
                             0.5).
        g_noise              noise-channel gain in [0, 0.5] (0.25).
        n_min                confidence gate: observations required since
                             the last reset before any regime change (50).
        align_min            refresh alignment below this = innovation ->
                             confidence reset (0.9).
        subspace_iters       warm-started power iterations per refresh (2).
        g_osc_min            lower clip of the oscillation gain (0.1;
                             adaptive mode only).
        g_osc_const          Gate-1 amendment A2: when set (None default),
                             the oscillation channel applies THIS fixed gain
                             to oscillation-classified, non-decaying
                             directions instead of the adaptive
                             clip(1/(eta*lambda_implied - 1), g_osc_min, 1)
                             formula. The amp_decay_margin escape (decaying
                             oscillation -> g = 1) applies unchanged in both
                             modes.
        amp_decay_margin     amplitude_ratio < 1 - margin counts as decaying
                             -> oscillation gain not applied (0.05); applies
                             in adaptive AND constant (g_osc_const) mode.
        enable_noise_channel / enable_oscillation_channel
                             channel switches (both True); both False =
                             bit-for-bit stock Muon.
        rho_ignored          ablation 4c: magnitude-only gate.
        random_gating        ablation 4d: placebo (random assignment of the
                             same gain multiset), dedicated RNG from seed.
        seed                 base seed for per-matrix subspace/gating RNGs.
        classifier_extra     optional dict of extra BatchRegimeClassifier
                             kwargs (innovation detectors z_reset/z_quiet
                             etc.; default off).

    Matrices whose flattened min dimension is < 2 get the stock Muon path
    (nothing to track). k is clamped to min(m, n) per matrix.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
        adjust_lr: Optional[str] = None,
        ns_dtype: torch.dtype = torch.bfloat16,
        *,
        k: int = 16,
        t_refresh: int = 50,
        beta: float = 0.99,
        tau_sig: float = 4.0,
        tau_noise: float = 2.0,
        rho_osc: float = 0.5,
        g_noise: float = 0.25,
        n_min: int = 50,
        align_min: float = 0.9,
        subspace_iters: int = 2,
        g_osc_min: float = 0.1,
        g_osc_const: Optional[float] = None,
        amp_decay_margin: float = 0.05,
        enable_noise_channel: bool = True,
        enable_oscillation_channel: bool = True,
        rho_ignored: bool = False,
        random_gating: bool = False,
        seed: int = 1000,
        classifier_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if t_refresh < 1:
            raise ValueError(f"t_refresh must be >= 1, got {t_refresh}")
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if not 0.0 <= g_noise <= 1.0:
            raise ValueError(f"g_noise must be in [0, 1], got {g_noise}")
        if not 0.0 < g_osc_min <= 1.0:
            raise ValueError(f"g_osc_min must be in (0, 1], got {g_osc_min}")
        if g_osc_const is not None and not 0.0 < g_osc_const <= 1.0:
            raise ValueError(
                f"g_osc_const must be in (0, 1] or None, got {g_osc_const}"
            )
        if n_min < 1:
            raise ValueError(f"n_min must be >= 1, got {n_min}")
        if tau_noise > tau_sig:
            # Same contract as src.stats.RegimeClassifier, checked eagerly
            # (tiers are created lazily on the first step).
            raise ValueError("tau_noise must be <= tau_sig")
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            adjust_lr=adjust_lr,
            ns_dtype=ns_dtype,
        )
        # Routing hyperparameters are optimizer-level (constructor/config
        # params, plan section 2.1); per-matrix STATE lives in
        # self.state[param]["routing"].
        self.k = int(k)
        self.t_refresh = int(t_refresh)
        self.beta = float(beta)
        self.tau_sig = float(tau_sig)
        self.tau_noise = float(tau_noise)
        self.rho_osc = float(rho_osc)
        self.g_noise = float(g_noise)
        self.n_min = int(n_min)
        self.align_min = float(align_min)
        self.subspace_iters = int(subspace_iters)
        self.g_osc_min = float(g_osc_min)
        self.g_osc_const = None if g_osc_const is None else float(g_osc_const)
        self.amp_decay_margin = float(amp_decay_margin)
        self.enable_noise_channel = bool(enable_noise_channel)
        self.enable_oscillation_channel = bool(enable_oscillation_channel)
        self.rho_ignored = bool(rho_ignored)
        self.random_gating = bool(random_gating)
        self.seed = int(seed)
        self.classifier_extra = dict(classifier_extra or {})
        self._n_tiers = 0  # deterministic per-matrix seed offsets

    # ------------------------------------------------------------------ hooks

    def pre_step(
        self, G: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        # Stash the RAW pre-momentum gradient for this step's projections
        # (plan section 1.1: momentum filtering would mask the
        # autocorrelation structure). Reference only -- consumed and dropped
        # inside shape_spectrum, never serialized, G never modified.
        state["_routed_raw_grad"] = G.reshape(len(G), -1) if G.ndim > 2 else G
        return super().pre_step(G, state, group)

    def shape_spectrum(
        self, O: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]
    ) -> torch.Tensor:
        G2 = state.pop("_routed_raw_grad", None)
        M2 = O.reshape(len(O), -1) if O.ndim > 2 else O

        # Bulk tier: vanilla Newton-Schulz on M_t, identical to muon.py.
        ns_out = zeropower_via_newtonschulz5(
            M2, steps=group["ns_steps"], eps=group["eps"], dtype=self.ns_dtype
        )
        stock = ns_out.reshape(O.shape).type_as(O)

        if G2 is None or M2.ndim != 2 or min(M2.shape) < 2:
            return stock  # nothing trackable: stock Muon path

        tier = state.get("routing")
        if tier is None:
            tier = state["routing"] = self._make_tier(M2)

        # Tracked tier: refresh cadence anchored at the first step
        # ((step - 1) % t_refresh == 0), matching the plan's amortized
        # power-iteration schedule.
        if (state["step"] - 1) % self.t_refresh == 0:
            tier.refresh(M2)

        # Raw-gradient projections -> one BatchRegimeClassifier update for
        # all k directions of this matrix.
        s = tier.project(G2)
        regimes = tier.classifier.update(s.cpu().numpy().astype(np.float64))
        gains = self._compute_gains(tier)
        tier.last_gains = gains
        tier.last_regimes = regimes
        tier.record_step(state["step"], gains)  # A5 telemetry (cheap)

        if np.all(gains == 1.0):
            # No deviation from stock Muon (also guarantees the bit-for-bit
            # equivalence when both channels are disabled: the correction is
            # skipped, not added-as-zero).
            return stock

        # O_t <- O_t + sum_i (g_i - 1) * (u_i^T O_t v_i) * u_i v_i^T
        # -- a rank-<=k correction computed on the owner rank; the bulk of
        # O_t is untouched.
        O32 = ns_out.to(tier.dtype)
        proj = ((tier.U.T @ O32) * tier.V.T).sum(dim=1)  # (k,) u_i^T O v_i
        coeff = (
            torch.as_tensor(gains - 1.0, dtype=tier.dtype, device=O32.device)
            * proj
        )
        corrected = O32 + (tier.U * coeff.unsqueeze(0)) @ tier.V.T
        return corrected.to(stock.dtype).reshape(O.shape)

    # -------------------------------------------------------------- internals

    def _make_tier(self, M2: torch.Tensor) -> _TrackedTier:
        m, n = M2.shape
        k_eff = min(self.k, min(m, n))
        # rho_ignored (ablation 4c): oscillation branch made unreachable so
        # gating is magnitude-only (t-statistic); everything else identical.
        rho_osc = _RHO_OSC_UNREACHABLE if self.rho_ignored else self.rho_osc
        classifier = BatchRegimeClassifier(
            beta=self.beta,
            k=k_eff,
            tau_sig=self.tau_sig,
            tau_noise=self.tau_noise,
            rho_osc=rho_osc,
            n_min=self.n_min,
            **self.classifier_extra,
        )
        tier_seed = self.seed * 1_000_003 + self._n_tiers
        self._n_tiers += 1
        return _TrackedTier(
            m,
            n,
            k=k_eff,
            iters=self.subspace_iters,
            align_min=self.align_min,
            classifier=classifier,
            seed=tier_seed,
            device=M2.device,
        )

    def _compute_gains(self, tier: _TrackedTier) -> np.ndarray:
        """Per-direction routing gains from the classifier's current labels.

        SIGNAL (and any direction still inside its confidence window --
        the classifier holds SIGNAL until n_min clears) -> 1.
        """
        clf = tier.classifier
        gains = np.ones(tier.k, dtype=np.float64)
        regimes = clf.regimes

        if self.enable_noise_channel:
            noise = np.array([r is Regime.NOISE for r in regimes])
            gains[noise] = self.g_noise

        if self.enable_oscillation_channel:
            osc = np.array([r is Regime.OSCILLATING for r in regimes])
            if osc.any():
                non_decaying = ~clf.stats.is_decaying(self.amp_decay_margin)
                active = osc & non_decaying
                if self.g_osc_const is not None:
                    # Constant mode (Gate-1 amendment A2): fixed attenuation
                    # for oscillation-classified, non-decaying directions.
                    gains[active] = self.g_osc_const
                else:
                    amp = clf.stats.amplitude_ratio  # eta*lambda_implied - 1
                    # g_osc = 1 / (eta*lambda_implied - 1), clipped: critical
                    # damping target eta*lambda -> 1 from the amplitude ratio.
                    g_osc = 1.0 / np.maximum(amp, 1e-12)
                    gains[active] = np.clip(g_osc[active], self.g_osc_min, 1.0)

        if self.random_gating:
            # Placebo (ablation 4d): same gain multiset -- same gated
            # fraction, same magnitudes -- randomly assigned to directions.
            gains = tier.gating_rng.permutation(gains)
        return gains

    # -------------------------------------------------------------- telemetry

    def routing_stats(self) -> Dict[str, Any]:
        """Per-matrix + aggregate routing telemetry (Gate-1 amendment A5).

        JSON-able dict; all counters are accumulated per matrix inside the
        step (``_TrackedTier.record_step``), so this call is pure read-out
        and can be invoked at any cadence (the airbench harness samples the
        aggregate every 10 steps and stores the full dict at end of run).

        Shape::

            {"per_matrix": {"matrix0_24x27": {"shape", "k", "n_refreshes",
                            "n_rotation_resets", "n_innovation_resets",
                            "n_confidence_resets", "last": {...},
                            "cumulative": {...}}, ...},
             "aggregate": {"last": {...} | None, "cumulative": {...} | None}}

        ``last`` holds the most recent step's per-channel counts, treated
        count/fraction, gain sum/min/max, and in-confidence-window count;
        ``cumulative`` the direction-step totals of the same quantities plus
        reset counts. Aggregate entries are None until the first routed step.
        """
        per_matrix: Dict[str, Any] = {}
        tiers = []
        i = 0
        for group in self.param_groups:
            for p in group["params"]:
                tier = self.state.get(p, {}).get("routing")
                if tier is None:
                    continue
                per_matrix[f"matrix{i}_{tier.m}x{tier.n}"] = tier.stats_snapshot()
                tiers.append(tier)
                i += 1

        stepped = [t for t in tiers if t.last_stats is not None]
        agg_last: Optional[Dict[str, Any]] = None
        agg_cum: Optional[Dict[str, Any]] = None
        if stepped:
            k_tot = sum(t.k for t in stepped)
            n_treated = sum(t.last_stats["n_treated"] for t in stepped)
            agg_last = {
                "step": max(t.last_stats["step"] for t in stepped),
                "n_matrices": len(stepped),
                "k_total": k_tot,
                "n_signal": sum(t.last_stats["n_signal"] for t in stepped),
                "n_noise": sum(t.last_stats["n_noise"] for t in stepped),
                "n_oscillating": sum(
                    t.last_stats["n_oscillating"] for t in stepped
                ),
                "n_treated": n_treated,
                "treated_fraction": n_treated / k_tot,
                "n_in_confidence_window": sum(
                    t.last_stats["n_in_confidence_window"] for t in stepped
                ),
                "gain_sum": sum(t.last_stats["gain_sum"] for t in stepped),
                "gain_min": min(t.last_stats["gain_min"] for t in stepped),
                "gain_max": max(t.last_stats["gain_max"] for t in stepped),
            }
            ds = sum(t.cum["direction_steps"] for t in stepped)
            treated_ds = sum(t.cum["treated_direction_steps"] for t in stepped)
            agg_cum = {
                "n_matrices": len(stepped),
                "direction_steps": ds,
                "signal_direction_steps": sum(
                    t.cum["signal_direction_steps"] for t in stepped
                ),
                "noise_direction_steps": sum(
                    t.cum["noise_direction_steps"] for t in stepped
                ),
                "oscillating_direction_steps": sum(
                    t.cum["oscillating_direction_steps"] for t in stepped
                ),
                "treated_direction_steps": treated_ds,
                "treated_fraction": treated_ds / max(ds, 1),
                "in_confidence_window_direction_steps": sum(
                    t.cum["in_confidence_window_direction_steps"]
                    for t in stepped
                ),
                "gain_sum": sum(t.cum["gain_sum"] for t in stepped),
                "gain_min": min(t.cum["gain_min"] for t in stepped),
                "gain_max": max(t.cum["gain_max"] for t in stepped),
                "n_refreshes": sum(t.n_refreshes for t in stepped),
                "n_rotation_resets": sum(t.n_rotation_resets for t in stepped),
                "n_innovation_resets": sum(
                    t._n_innovation_resets for t in stepped
                ),
                "n_confidence_resets": sum(
                    t.n_rotation_resets + t._n_innovation_resets
                    for t in stepped
                ),
            }
        return {"per_matrix": per_matrix, "aggregate": {"last": agg_last, "cumulative": agg_cum}}
