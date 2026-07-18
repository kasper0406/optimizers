"""Regime classifier for a single tracked direction.

Regimes (plan §1.2 / §2.1):
  * SIGNAL      -- persistent mean: |t-stat| >= tau_sig, rho not strongly negative
  * NOISE       -- |t-stat| < tau_noise and rho not strongly negative
  * OSCILLATING -- rho <= -rho_osc (checked first; an oscillating direction has
                   near-zero mean, so the t-stat channels do not see it)

Confidence logic (plan §2.1):
  * Every direction starts in SIGNAL (= stock Muon behavior) and only leaves
    it (or changes regime at all) once n_min observations have accumulated
    since the last confidence reset.  Between gate passes the previous regime
    is held.  Inside the gate, the band tau_noise <= |t| < tau_sig is
    hysteresis: the current regime is kept.
  * Innovation detection resets confidence: statistics are rebuilt from
    scratch, the regime reverts to SIGNAL, and the direction must re-earn a
    classification with n_min fresh observations.

Innovation detectors (both optional; enabled by passing their threshold):
  * jump: the standardized innovation z = (s - mean) / sqrt(var) exceeds
    z_reset in magnitude for >= innov_needed of the last innov_window
    samples ("m of last w" rather than consecutive so that an alternating
    oscillation, whose every other sample is close to the stale mean, still
    triggers).
  * quiet (variance collapse): the RMS of z over the last quiet_window
    samples falls below z_quiet -- e.g. a large-amplitude oscillation dying
    into small noise never produces a large |z|, only a suspiciously small
    one.

On reset the detector window is replayed from the first sample that
satisfies the triggering condition, so the samples that revealed the new
regime are not thrown away; the n_min clock counts them.

ALL decision thresholds (tau_sig, tau_noise, rho_osc, n_min, z_reset,
z_quiet, window shapes) are constructor parameters with no scientific
defaults -- the required ones must be passed explicitly.  The synthetic
validation tests pick values appropriate to each synthetic scenario;
production values are human-authored later in criteria/.
"""

from __future__ import annotations

import enum
import math
from collections import deque

from .direction_stats import DirectionStats

__all__ = ["Regime", "RegimeClassifier"]


class Regime(enum.Enum):
    SIGNAL = "signal"
    NOISE = "noise"
    OSCILLATING = "oscillating"


class RegimeClassifier:
    def __init__(
        self,
        *,
        beta: float,
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

        self.stats = DirectionStats(beta, **self._stats_kwargs)
        self.regime = Regime.SIGNAL
        self.step = 0  # total samples seen
        self.reset_steps: list[int] = []  # step index at which each reset fired
        buflen = max(self.innov_window, self.quiet_window)
        self._window: deque[tuple[float, float]] = deque(maxlen=buflen)  # (s, |z|)

    # ------------------------------------------------------------------ api

    @property
    def n_since_reset(self) -> int:
        return self.stats.n_obs

    def update(self, s: float) -> Regime:
        """Feed one sample; returns the regime *after* incorporating it."""
        s = float(s)
        self.step += 1

        z_abs = self._standardized_innovation(s)
        self._window.append((s, z_abs))

        if self._should_reset(z_abs):
            self._reset_and_replay()
        else:
            self.stats.update(s)

        self._classify()
        return self.regime

    # ------------------------------------------------------------ internals

    def _standardized_innovation(self, s: float) -> float:
        """|z| of the incoming sample vs current (pre-update) statistics."""
        if self.stats.n_obs < self.warmup_detect:
            return math.nan
        var = float(self.stats.var)
        floor = self.stats.var_floor
        if var <= floor:
            return math.nan
        return abs(s - float(self.stats.mean)) / math.sqrt(var)

    def _should_reset(self, z_abs: float) -> bool:
        if math.isnan(z_abs):
            return False
        if self.z_reset is not None:
            recent = list(self._window)[-self.innov_window :]
            zs = [z for _, z in recent if not math.isnan(z)]
            if sum(z >= self.z_reset for z in zs) >= self.innov_needed:
                return True
        if self.z_quiet is not None:
            recent = list(self._window)[-self.quiet_window :]
            zs = [z for _, z in recent if not math.isnan(z)]
            if len(zs) >= self.quiet_window:
                rms = math.sqrt(sum(z * z for z in zs) / len(zs))
                if rms < self.z_quiet:
                    return True
        return False

    def _replay_start(self) -> int:
        """Index into the window buffer from which to replay after a reset.

        Jump trigger: first sample in the window with |z| >= z_reset.
        Quiet trigger: the whole quiet window belongs to the new regime.
        """
        window = list(self._window)
        if self.z_reset is not None:
            for i, (_, z) in enumerate(window):
                if not math.isnan(z) and z >= self.z_reset:
                    return i
        return max(0, len(window) - self.quiet_window)

    def _reset_and_replay(self) -> None:
        start = self._replay_start()
        replay = [s for s, _ in list(self._window)[start:]]
        self.stats = DirectionStats(self.beta, **self._stats_kwargs)
        for s in replay:
            self.stats.update(s)
        self._window.clear()
        self.reset_steps.append(self.step)
        self.regime = Regime.SIGNAL  # confidence reset -> stock behavior

    def _classify(self) -> None:
        if self.stats.n_obs < self.n_min:
            return  # not enough confidence to leave the current regime
        rho = float(self.stats.rho_corrected)
        t_abs = abs(float(self.stats.t_stat))
        if rho <= -self.rho_osc:
            self.regime = Regime.OSCILLATING
        elif t_abs >= self.tau_sig:
            self.regime = Regime.SIGNAL
        elif t_abs < self.tau_noise:
            self.regime = Regime.NOISE
        # else: hysteresis band -- keep current regime
