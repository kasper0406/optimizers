"""Per-direction temporal statistics of a scalar projection stream s_i(t).

Implements the plan §1.1 quantities for one tracked direction:

    mu   = EMA[s]                       (mean)
    q    = EMA[s^2]                     -> var = q - mu^2
    a    = EMA[s_t * s_{t-1}]           -> lag-1 autocov c = a - mu^2
                                        -> autocorr rho = c / var
    ratio = EMA[|s_t| / |s_{t-1}|]      -> implied eta*lambda = 1 + ratio
                                           (for s(t) = A (-r)^t: ratio = r,
                                            eta*lambda = 1 + r)

plus derived quantities:

    ess        Kish effective sample size of the EMA window
    ess_adj    ess shrunk by (1 - rho) / (1 + rho) for rho > 0 (positive
               autocorrelation reduces the information content of the window;
               negative rho is conservatively NOT credited)
    t_stat     mean / sqrt(var / ess_adj) -- t-statistic of the mean
    rho_corrected  first-order small-sample (Kendall/Marriott-Pope) bias
               correction rho + (1 + 3 rho) / ess, applied once ess is large
               enough for the first-order expansion to make sense

All EMAs are bias-corrected (see ema.py).  The lag-product and ratio EMAs
keep their own step counters (they start one sample later than mu/q).

Numerical guards (these are numerical, not scientific, constants):
  * var is clamped to >= 0 (q - mu^2 can go slightly negative in floats);
  * rho is defined as 0 when var <= var_floor and clipped to [-1, 1];
  * the amplitude ratio denominator is floored at abs_floor and the ratio
    clipped to [0, ratio_clip] so a near-zero previous sample cannot inject
    an infinite ratio.

No decision thresholds live here; classification thresholds are constructor
parameters of RegimeClassifier (classifier.py).
"""

from __future__ import annotations

import numpy as np

from .ema import BiasCorrectedEma, ema_effective_sample_size

__all__ = ["DirectionStats"]


class DirectionStats:
    def __init__(
        self,
        beta: float,
        *,
        var_floor: float = 1e-30,
        ratio_clip: float = 10.0,
        abs_floor: float = 1e-300,
        adjust_ess_for_autocorr: bool = True,
        kendall_min_ess: float = 4.0,
    ):
        self.beta = float(beta)
        self.var_floor = float(var_floor)
        self.ratio_clip = float(ratio_clip)
        self.abs_floor = float(abs_floor)
        self.adjust_ess_for_autocorr = bool(adjust_ess_for_autocorr)
        self.kendall_min_ess = float(kendall_min_ess)

        self._mu = BiasCorrectedEma(beta)
        self._q = BiasCorrectedEma(beta)
        self._a = BiasCorrectedEma(beta)  # lag-1 product EMA
        self._ratio = BiasCorrectedEma(beta)  # |s_t| / |s_{t-1}| EMA
        self._prev = None

    # ------------------------------------------------------------------ update

    def update(self, s) -> None:
        s = np.asarray(s, dtype=np.float64)
        self._mu.update(s)
        self._q.update(s * s)
        if self._prev is not None:
            self._a.update(s * self._prev)
            denom = np.maximum(np.abs(self._prev), self.abs_floor)
            ratio = np.clip(np.abs(s) / denom, 0.0, self.ratio_clip)
            self._ratio.update(ratio)
        self._prev = s

    @property
    def n_obs(self) -> int:
        """Number of observations since construction/reset."""
        return self._mu.t

    # ------------------------------------------------------------- estimates

    @property
    def mean(self):
        return self._mu.value

    @property
    def mean_raw(self):
        """Uncorrected EMA mean (exposed for the small-t bias tests)."""
        return self._mu.raw

    @property
    def second_moment(self):
        return self._q.value

    @property
    def var(self):
        mu = self._mu.value
        return np.maximum(self._q.value - mu * mu, 0.0)

    @property
    def autocov(self):
        mu = self._mu.value
        return self._a.value - mu * mu

    @property
    def rho(self):
        """Lag-1 autocorrelation c / var, guarded and clipped to [-1, 1]."""
        var = self.var
        safe = np.maximum(var, self.var_floor)
        raw = self.autocov / safe
        raw = np.where(var > self.var_floor, raw, 0.0)
        return np.clip(raw, -1.0, 1.0)

    @property
    def rho_corrected(self):
        """rho with first-order small-sample bias correction (clipped)."""
        rho = self.rho
        ess = self.ess
        if ess < self.kendall_min_ess:
            return rho
        return np.clip(rho + (1.0 + 3.0 * rho) / ess, -1.0, 1.0)

    # ------------------------------------------------- sample size & t-stat

    @property
    def ess(self) -> float:
        return ema_effective_sample_size(self._mu.t, self.beta)

    @property
    def ess_adjusted(self):
        """ESS shrunk for positive autocorrelation (conservative)."""
        ess = self.ess
        if not self.adjust_ess_for_autocorr or ess <= 0.0:
            return ess
        rho_pos = np.clip(self.rho_corrected, 0.0, 0.95)
        return ess * (1.0 - rho_pos) / (1.0 + rho_pos)

    @property
    def t_stat(self):
        """t-statistic of the mean: mean / sqrt(var / ess_adjusted)."""
        if self._mu.t == 0:
            return 0.0
        ess = np.maximum(self.ess_adjusted, 1.0)
        se = np.sqrt(np.maximum(self.var, self.var_floor) / ess)
        return self.mean / se

    # --------------------------------------------------- oscillation channel

    @property
    def amplitude_ratio(self):
        """Bias-corrected EMA of |s_t / s_{t-1}| (r for s = A (-r)^t)."""
        return self._ratio.value

    @property
    def implied_eta_lambda(self):
        """Implied eta*lambda from the amplitude ratio: 1 + r."""
        return 1.0 + self._ratio.value

    def is_decaying(self, margin: float):
        """Amplitude-decay flag: ratio < 1 - margin.

        margin is a caller-supplied threshold (no scientific default here).
        r < 1 -> decaying oscillation; r >= 1 -> non-decaying (critical or
        growing).
        """
        return self.amplitude_ratio < 1.0 - float(margin)

    def reset(self) -> None:
        self._mu.reset()
        self._q.reset()
        self._a.reset()
        self._ratio.reset()
        self._prev = None
