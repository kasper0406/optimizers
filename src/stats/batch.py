"""Array-mode (batched) regime classification for k directions at once.

Motivation (WP1.1 overhead): the scalar :class:`~src.stats.classifier.
RegimeClassifier` costs ~76 us per ``update`` call in Python; at airbench
shapes (6 matrices x 32 tracked directions x 2 betas) that is 20-30 ms of
serial CPU work per training step vs ~29 ms GPU steps.  This module updates
a whole k-vector of directions in O(1) Python calls per step (numpy
vectorized) instead of O(k).

SEMANTICS CONTRACT: :class:`BatchRegimeClassifier` implements *exactly* the
logic, thresholds, confidence gating, hysteresis, and innovation/reset
semantics of the WP0.5-validated scalar pair
(:class:`~src.stats.direction_stats.DirectionStats`,
:class:`~src.stats.classifier.RegimeClassifier`) -- same formulas, same
guards, same numerical constants, evaluated element-wise.  No metric
definition changes (CLAUDE.md ground rule 3).  The scalar implementations
remain untouched and canonical; equivalence is enforced by
``tests/test_stats_batch_equivalence.py`` (identical label sequences, stats,
t-stats, and reset steps across all regimes and both betas).

Differences from the scalar API (bookkeeping only, no semantics):

* one shared step counter for the batch (all k directions of a matrix
  observe every step -- exactly the tracker's usage pattern);
* per-direction observation counts (``n_obs``) diverge after resets, so the
  batched statistics keep a per-element step counter instead of the shared
  counter of ``BiasCorrectedEma``;
* ``reset_steps`` is a list of per-direction lists;
* :meth:`BatchRegimeClassifier.reset_directions` re-creates the
  "fresh classifier" state for a subset of directions (what the tracker's
  subspace-rotation innovation reset did by rebuilding scalar classifiers).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .classifier import Regime

__all__ = [
    "ArrayDirectionStats",
    "BatchRegimeClassifier",
    "DirectionView",
    "DirectionStatsView",
]

# Stable code <-> Regime mapping (SIGNAL is the start/reset prior).
_REGIMES: tuple = (Regime.SIGNAL, Regime.NOISE, Regime.OSCILLATING)
_CODE = {r: i for i, r in enumerate(_REGIMES)}
_SIGNAL, _NOISE, _OSC = (
    _CODE[Regime.SIGNAL],
    _CODE[Regime.NOISE],
    _CODE[Regime.OSCILLATING],
)


class ArrayDirectionStats:
    """Element-wise :class:`DirectionStats` over a fixed-size batch of k
    directions, with a per-element observation counter (directions reset
    independently) and masked updates.

    Every formula, guard, and numerical constant mirrors
    ``direction_stats.py`` exactly; see the module docstring for the
    equivalence contract.
    """

    def __init__(
        self,
        beta: float,
        k: int,
        *,
        var_floor: float = 1e-30,
        ratio_clip: float = 10.0,
        abs_floor: float = 1e-300,
        adjust_ess_for_autocorr: bool = True,
        kendall_min_ess: float = 4.0,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.beta = float(beta)
        self.k = int(k)
        self.var_floor = float(var_floor)
        self.ratio_clip = float(ratio_clip)
        self.abs_floor = float(abs_floor)
        self.adjust_ess_for_autocorr = bool(adjust_ess_for_autocorr)
        self.kendall_min_ess = float(kendall_min_ess)

        self._mu_raw = np.zeros(k)
        self._q_raw = np.zeros(k)
        self._a_raw = np.zeros(k)  # lag-1 product EMA
        self._ratio_raw = np.zeros(k)  # |s_t| / |s_{t-1}| EMA
        self._t = np.zeros(k, dtype=np.int64)  # per-element mu/q counter
        self._t_lag = np.zeros(k, dtype=np.int64)  # lag EMAs start 1 later
        self._prev = np.zeros(k)
        self._has_prev = np.zeros(k, dtype=bool)
        self._all_have_prev = False  # fast-path flag: every element has _prev
        # Derived-quantity memo, cleared on every state mutation.  Purely a
        # recomputation cache: values are exactly what the properties would
        # compute.  Callers must treat returned arrays as read-only.
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ update

    def update(self, s: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """One masked EMA update; elements outside ``mask`` are untouched."""
        s = np.asarray(s, dtype=np.float64)
        beta = self.beta
        self._cache.clear()

        if mask is None and self._all_have_prev:
            # Fast path (the every-step case once all elements have a
            # predecessor): identical arithmetic to the masked path below --
            # beta * raw + (1 - beta) * x, evaluated in the same order --
            # just without the mask selects.
            self._t += 1
            self._t_lag += 1
            self._mu_raw *= beta
            self._mu_raw += (1.0 - beta) * s
            self._q_raw *= beta
            self._q_raw += (1.0 - beta) * (s * s)
            self._a_raw *= beta
            self._a_raw += (1.0 - beta) * (s * self._prev)
            denom = np.maximum(np.abs(self._prev), self.abs_floor)
            ratio = np.clip(np.abs(s) / denom, 0.0, self.ratio_clip)
            self._ratio_raw *= beta
            self._ratio_raw += (1.0 - beta) * ratio
            self._prev[:] = s
            return

        m = np.ones(self.k, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        self._t = self._t + m
        self._mu_raw = np.where(m, beta * self._mu_raw + (1.0 - beta) * s, self._mu_raw)
        self._q_raw = np.where(
            m, beta * self._q_raw + (1.0 - beta) * (s * s), self._q_raw
        )

        lag_m = m & self._has_prev
        self._t_lag = self._t_lag + lag_m
        self._a_raw = np.where(
            lag_m, beta * self._a_raw + (1.0 - beta) * (s * self._prev), self._a_raw
        )
        denom = np.maximum(np.abs(self._prev), self.abs_floor)
        ratio = np.clip(np.abs(s) / denom, 0.0, self.ratio_clip)
        self._ratio_raw = np.where(
            lag_m, beta * self._ratio_raw + (1.0 - beta) * ratio, self._ratio_raw
        )

        self._prev = np.where(m, s, self._prev)
        self._has_prev = self._has_prev | m
        self._all_have_prev = bool(self._has_prev.all())

    def reset(self, mask: Optional[np.ndarray] = None) -> None:
        """Rebuild-from-scratch for the masked elements (DirectionStats.reset)."""
        m = np.ones(self.k, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        self._cache.clear()
        for arr in (self._mu_raw, self._q_raw, self._a_raw, self._ratio_raw, self._prev):
            arr[m] = 0.0
        self._t[m] = 0
        self._t_lag[m] = 0
        self._has_prev[m] = False
        self._all_have_prev = False

    # --------------------------------------------------------------- estimates

    def _corrected(self, raw: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Adam-style bias correction; value == raw (0) before any update."""
        denom = 1.0 - self.beta**t
        safe = np.where(t > 0, denom, 1.0)
        return np.where(t > 0, raw / safe, raw)

    def _memo(self, name: str, fn: Callable[[], np.ndarray]) -> np.ndarray:
        value = self._cache.get(name)
        if value is None:
            value = self._cache[name] = fn()
        return value

    @property
    def n_obs(self) -> np.ndarray:
        """Per-direction observations since construction/reset."""
        return self._t.copy()

    @property
    def mean(self) -> np.ndarray:
        return self._memo("mean", lambda: self._corrected(self._mu_raw, self._t))

    @property
    def second_moment(self) -> np.ndarray:
        return self._memo("q", lambda: self._corrected(self._q_raw, self._t))

    @property
    def var(self) -> np.ndarray:
        def compute():
            mu = self.mean
            return np.maximum(self.second_moment - mu * mu, 0.0)

        return self._memo("var", compute)

    @property
    def autocov(self) -> np.ndarray:
        def compute():
            mu = self.mean
            return self._corrected(self._a_raw, self._t_lag) - mu * mu

        return self._memo("autocov", compute)

    @property
    def rho(self) -> np.ndarray:
        def compute():
            var = self.var
            safe = np.maximum(var, self.var_floor)
            raw = self.autocov / safe
            raw = np.where(var > self.var_floor, raw, 0.0)
            return np.clip(raw, -1.0, 1.0)

        return self._memo("rho", compute)

    @property
    def ess(self) -> np.ndarray:
        """Kish ESS of the bias-corrected EMA, per element (ema.py formula)."""

        def compute():
            beta = self.beta
            t = self._t
            if beta <= 0.0:  # pragma: no cover - beta in (0,1) enforced upstream
                return np.where(t > 0, 1.0, 0.0)
            bt = beta**t.astype(np.float64)
            num = (1.0 - bt) ** 2 * (1.0 + beta)
            den = (1.0 - beta) * (1.0 - bt * bt)
            safe = np.where(t > 0, den, 1.0)
            return np.where(t > 0, num / safe, 0.0)

        return self._memo("ess", compute)

    @property
    def rho_corrected(self) -> np.ndarray:
        def compute():
            rho = self.rho
            ess = self.ess
            apply = ess >= self.kendall_min_ess
            safe_ess = np.where(apply, ess, 1.0)
            corrected = np.clip(rho + (1.0 + 3.0 * rho) / safe_ess, -1.0, 1.0)
            return np.where(apply, corrected, rho)

        return self._memo("rho_corrected", compute)

    @property
    def ess_adjusted(self) -> np.ndarray:
        def compute():
            ess = self.ess
            if not self.adjust_ess_for_autocorr:
                return ess
            rho_pos = np.clip(self.rho_corrected, 0.0, 0.95)
            adjusted = ess * (1.0 - rho_pos) / (1.0 + rho_pos)
            return np.where(ess <= 0.0, ess, adjusted)

        return self._memo("ess_adjusted", compute)

    @property
    def t_stat(self) -> np.ndarray:
        def compute():
            ess = np.maximum(self.ess_adjusted, 1.0)
            se = np.sqrt(np.maximum(self.var, self.var_floor) / ess)
            return np.where(self._t == 0, 0.0, self.mean / se)

        return self._memo("t_stat", compute)

    @property
    def amplitude_ratio(self) -> np.ndarray:
        return self._memo(
            "amplitude_ratio", lambda: self._corrected(self._ratio_raw, self._t_lag)
        )

    @property
    def implied_eta_lambda(self) -> np.ndarray:
        return 1.0 + self.amplitude_ratio

    def is_decaying(self, margin: float) -> np.ndarray:
        return self.amplitude_ratio < 1.0 - float(margin)


class BatchRegimeClassifier:
    """k independent RegimeClassifiers advanced with O(1) numpy calls/step.

    Constructor parameters are identical to :class:`RegimeClassifier` plus
    ``k``.  ``update`` takes a (k,) sample vector and returns the (k,) list
    of :class:`Regime` labels after incorporating it.
    """

    def __init__(
        self,
        *,
        beta: float,
        k: int,
        tau_sig: float,
        tau_noise: float,
        rho_osc: float,
        n_min: int,
        z_reset: float | None = None,
        innov_needed: int = 2,
        innov_window: int = 4,
        z_quiet: float | None = None,
        quiet_window: int = 6,
        warmup_detect: int = 5,
        stats_kwargs: dict | None = None,
    ):
        if tau_noise > tau_sig:
            raise ValueError("tau_noise must be <= tau_sig")
        self.beta = float(beta)
        self.k = int(k)
        self.tau_sig = float(tau_sig)
        self.tau_noise = float(tau_noise)
        self.rho_osc = float(rho_osc)
        self.n_min = int(n_min)
        self.z_reset = None if z_reset is None else float(z_reset)
        self.innov_needed = int(innov_needed)
        self.innov_window = int(innov_window)
        self.z_quiet = None if z_quiet is None else float(z_quiet)
        self.quiet_window = int(quiet_window)
        self.warmup_detect = int(warmup_detect)
        self._stats_kwargs = dict(stats_kwargs or {})

        self.stats = ArrayDirectionStats(self.beta, self.k, **self._stats_kwargs)
        self.regime_codes = np.full(self.k, _SIGNAL, dtype=np.int64)
        self.step = 0
        self.reset_steps: List[List[int]] = [[] for _ in range(self.k)]

        self._buflen = max(self.innov_window, self.quiet_window)
        # Rolling window, oldest row first, newest row last; a direction's
        # valid entries are the trailing ``_valid[i]`` rows (cleared on reset).
        self._win_s = np.zeros((self._buflen, self.k))
        self._win_z = np.full((self._buflen, self.k), np.nan)
        self._valid = np.zeros(self.k, dtype=np.int64)
        self._rows = np.arange(self._buflen)[:, None]  # (buflen, 1)
        # Saturated-window fast path: once every direction has buflen valid
        # entries (and until the next reset) the per-direction "recent rows"
        # masks are constant; precompute them.
        self._window_full = False
        self._recent_innov_full = self._rows >= self._buflen - min(
            self.innov_window, self._buflen
        )
        self._recent_quiet_full = self._rows >= self._buflen - min(
            self.quiet_window, self._buflen
        )

    # ------------------------------------------------------------------- api

    @property
    def regimes(self) -> List[Regime]:
        return [_REGIMES[c] for c in self.regime_codes]

    @property
    def n_since_reset(self) -> np.ndarray:
        return self.stats.n_obs

    def update(self, s: np.ndarray) -> List[Regime]:
        """Feed one (k,) sample vector; returns the per-direction regimes
        *after* incorporating it (RegimeClassifier.update, element-wise)."""
        s = np.asarray(s, dtype=np.float64).reshape(self.k)
        self.step += 1

        z_abs = self._standardized_innovation(s)

        # Slide the window (buflen x k copies -- tiny) and append (s, z).
        self._win_s[:-1] = self._win_s[1:]
        self._win_z[:-1] = self._win_z[1:]
        self._win_s[-1] = s
        self._win_z[-1] = z_abs
        if not self._window_full:
            self._valid = np.minimum(self._valid + 1, self._buflen)
            self._window_full = bool((self._valid == self._buflen).all())


        reset_mask = self._should_reset(z_abs)
        if reset_mask.any():
            self._reset_and_replay(reset_mask)
            keep = ~reset_mask
            if keep.any():
                self.stats.update(s, keep)
        else:
            self.stats.update(s)

        self._classify()
        return self.regimes

    def reset_directions(self, indices: Sequence[int]) -> None:
        """Fresh-classifier state for a subset of directions (the tracker's
        subspace-rotation innovation reset: statistics rebuilt from scratch,
        regime reverts to SIGNAL, n_min clock restarts, empty window).

        Bookkeeping of *which* step the rotation happened at stays with the
        caller (as it did when scalar classifiers were rebuilt wholesale),
        so this does not append to ``reset_steps``.
        """
        idx = np.asarray(list(indices), dtype=np.int64)
        if idx.size == 0:
            return
        mask = np.zeros(self.k, dtype=bool)
        mask[idx] = True
        self.stats.reset(mask)
        self._valid[mask] = 0
        self._window_full = False
        self.regime_codes[mask] = _SIGNAL

    # ------------------------------------------------------------ internals

    def _standardized_innovation(self, s: np.ndarray) -> np.ndarray:
        """|z| vs current (pre-update) statistics; NaN while warming up or
        when the variance sits at/below the floor (classifier.py exactly)."""
        var = self.stats.var
        eligible = (self.stats._t >= self.warmup_detect) & (var > self.stats.var_floor)
        z = np.abs(s - self.stats.mean) / np.sqrt(np.maximum(var, self.stats.var_floor))
        return np.where(eligible, z, np.nan)

    def _should_reset(self, z_now: np.ndarray) -> np.ndarray:
        # A NaN incoming z can never trigger (scalar: early return False).
        can = ~np.isnan(z_now)
        if not can.any() or (self.z_reset is None and self.z_quiet is None):
            return np.zeros(self.k, dtype=bool)
        nn = ~np.isnan(self._win_z)
        fire = np.zeros(self.k, dtype=bool)
        if self.z_reset is not None:
            # Last min(innov_window, valid) window entries per direction.
            if self._window_full:
                recent = self._recent_innov_full
            else:
                recent = self._rows >= self._buflen - np.minimum(self.innov_window, self._valid)[None, :]
            hits = recent & nn & (self._win_z >= self.z_reset)
            fire |= hits.sum(axis=0) >= self.innov_needed
        if self.z_quiet is not None:
            if self._window_full:
                recent = self._recent_quiet_full
            else:
                recent = self._rows >= self._buflen - np.minimum(self.quiet_window, self._valid)[None, :]
            usable = recent & nn
            count = usable.sum(axis=0)
            full = count >= self.quiet_window  # all quiet_window entries, no NaN
            sq = np.where(usable, np.square(np.nan_to_num(self._win_z)), 0.0).sum(axis=0)
            rms = np.sqrt(sq / np.maximum(count, 1))
            fire |= full & (rms < self.z_quiet)
        return fire & can

    def _replay_start_rows(self, reset_mask: np.ndarray) -> np.ndarray:
        """Global window-row index from which to replay, per direction
        (classifier._replay_start translated to the shared buffer: a
        direction's window list starts at global row buflen - valid)."""
        first_valid_row = self._buflen - self._valid  # (k,)
        # Fallback (quiet trigger): the whole quiet window belongs to the new
        # regime -> list index max(0, valid - quiet_window).
        start = np.maximum(first_valid_row, self._buflen - self.quiet_window)
        if self.z_reset is not None:
            valid_rows = self._rows >= first_valid_row[None, :]
            cand = valid_rows & ~np.isnan(self._win_z) & (self._win_z >= self.z_reset)
            has = cand.any(axis=0)
            first_hit = cand.argmax(axis=0)  # first True row (0 if none)
            start = np.where(has, first_hit, start)
        return np.where(reset_mask, start, self._buflen)  # buflen = no replay

    def _reset_and_replay(self, reset_mask: np.ndarray) -> None:
        start = self._replay_start_rows(reset_mask)
        first_valid_row = self._buflen - self._valid
        self.stats.reset(reset_mask)
        for j in range(self._buflen):
            m_j = reset_mask & (j >= start) & (j >= first_valid_row)
            if m_j.any():
                self.stats.update(self._win_s[j], m_j)
        self._valid[reset_mask] = 0
        self._window_full = False
        self.regime_codes[reset_mask] = _SIGNAL  # confidence reset -> prior
        for d in np.nonzero(reset_mask)[0]:
            self.reset_steps[d].append(self.step)

    def view(self, index: int) -> "DirectionView":
        """Read-only single-direction view (scalar-classifier-shaped API)."""
        return DirectionView(self, index)

    def _classify(self) -> None:
        gate = self.stats._t >= self.n_min
        if not gate.any():
            return
        rho = self.stats.rho_corrected
        t_abs = np.abs(self.stats.t_stat)
        proposed = np.where(
            rho <= -self.rho_osc,
            _OSC,
            np.where(
                t_abs >= self.tau_sig,
                _SIGNAL,
                np.where(t_abs < self.tau_noise, _NOISE, self.regime_codes),
            ),
        )
        self.regime_codes = np.where(gate, proposed, self.regime_codes).astype(np.int64)


class DirectionStatsView:
    """Read-only scalar view of one direction inside ArrayDirectionStats.

    Exposes the DirectionStats property surface (scalars) for logging and
    tests; it holds no state of its own.
    """

    def __init__(self, stats: ArrayDirectionStats, index: int):
        self._stats = stats
        self._i = int(index)

    @property
    def n_obs(self) -> int:
        return int(self._stats._t[self._i])

    def _scalar(self, name: str) -> float:
        return float(getattr(self._stats, name)[self._i])

    @property
    def mean(self) -> float:
        return self._scalar("mean")

    @property
    def second_moment(self) -> float:
        return self._scalar("second_moment")

    @property
    def var(self) -> float:
        return self._scalar("var")

    @property
    def autocov(self) -> float:
        return self._scalar("autocov")

    @property
    def rho(self) -> float:
        return self._scalar("rho")

    @property
    def rho_corrected(self) -> float:
        return self._scalar("rho_corrected")

    @property
    def ess(self) -> float:
        return self._scalar("ess")

    @property
    def ess_adjusted(self) -> float:
        return self._scalar("ess_adjusted")

    @property
    def t_stat(self) -> float:
        return self._scalar("t_stat")

    @property
    def amplitude_ratio(self) -> float:
        return self._scalar("amplitude_ratio")

    @property
    def implied_eta_lambda(self) -> float:
        return self._scalar("implied_eta_lambda")

    def is_decaying(self, margin: float) -> bool:
        return bool(self.amplitude_ratio < 1.0 - float(margin))


class DirectionView:
    """Read-only scalar-classifier-shaped view of one batch direction."""

    def __init__(self, clf: BatchRegimeClassifier, index: int):
        self._clf = clf
        self._i = int(index)
        self.stats = DirectionStatsView(clf.stats, index)

    @property
    def regime(self) -> Regime:
        return _REGIMES[int(self._clf.regime_codes[self._i])]

    @property
    def n_since_reset(self) -> int:
        return int(self._clf.stats._t[self._i])

    @property
    def reset_steps(self) -> List[int]:
        return list(self._clf.reset_steps[self._i])
