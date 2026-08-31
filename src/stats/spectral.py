"""Two-channel (DC / alternating) spectral readout of a projection stream.

EXPLORATORY TIER.  Everything in this module is offline analysis of already
recorded per-direction projection streams s_i(t); it decides nothing and is
not on any training path.  The confirmatory surface remains the frozen-probe
tier (``src.instrument.tracker.FrozenProbeBank``, re-run on GPU); the
quantities here were designed after peeking at existing sidecars and are
therefore descriptive only (CLAUDE.md ground rules 1 and 6).

WHY TWO CHANNELS
----------------
Under Muon the per-direction projections carry a large *negative* lag-1
autocorrelation -- a period-2 zig-zag; pooled over the tracked tier the
estimator reads rho_1 ~ -0.34 on `top` directions, which corresponds to a
true phi of about -0.358 once the finite-segment bias below is inverted.
It is NOT lr-invariant: over the measured lr ladder the pooled `top` value
runs about -0.19 to -0.44 (``reports/channel-audit-phase-a.md`` §0b, A1).
That regime is exactly the one in which the plain (DC) mean and the
demodulated (alternating) mean measure different things:

    dc  channel:  y_t = s_t              -> mean = the persistent component
    alt channel:  y_t = (-1)^t s_t       -> mean = the period-2 amplitude

A pure period-2 oscillation s_t = A (-1)^t is invisible to the DC mean (it
cancels) and appears at full strength in the ALT mean; a persistent drift is
the other way round.  Both channel means are read out with the SAME
Newey-West machinery so the two numbers are directly comparable, and both
are calibrated against an AR(1) surrogate null (:func:`ar1_surrogate_null`)
because a zig-zag stream inflates the effective sample size (ESS > n) and a
naive |t| is not on the N(0, 1) scale.

That orthogonality is EXACT ONLY AT EVEN n.  At odd n a period-2 component
leaves A/n in the DC mean, and -- worse -- its Bartlett-truncated long-run
variance collapses toward zero (c_j = (-1)^j A^2 makes
c_0 + 2(0.75 c_1 + 0.5 c_2 + 0.25 c_3) ~ 0 at L = 3), so both ends of the
ratio inflate |t_dc| together.  Measured on a pure period-2 signal plus unit
noise (6000 reps, A/sigma = 10): at n = 45 the median |t_dc| is 1.655 and
P(|t_dc| >= 2) = 0.410, against 0.727 and 0.107 at n = 44, which is the
null.  The leak is below 0.01 in |t| at the amplitudes the tracked tier
actually shows (A/sigma <~ 0.5), but it is a property of the WINDOW, not of
the data: a caller who is free to choose the post-burn-in length should
choose an even one.

MIRROR CONTRACT
---------------
:func:`channel_t` reproduces, for a single 1-D series, exactly the estimator
of :meth:`src.instrument.tracker.FrozenProbeAccumulator.stats` -- same
Bartlett kernel, same Newey-West (1994) automatic bandwidth rule, same
divisors, same flooring semantics, same ``t_naive`` / ``t_nw`` / ``ess``
definitions:

    c_0        = S_0 / n - mean^2                       (divisor n)
    c_j        = S_j / (n - j) - mean^2                 (divisor n - j)
    L          = min(max_lag, floor(4 (n/100)^(2/9)), n - 2)
    sigma_LR^2 = c_0 + 2 sum_{j=1..L} (1 - j/(L+1)) c_j
    t_naive    = mean / sqrt(c_0 / n)
    t_nw       = mean / sqrt(sigma_LR^2 / n)
    ess        = n c_0 / sigma_LR^2

with sigma_LR^2 falling back to c_0 (and ``nw_floored`` set) when the
truncated sum is non-positive.  No metric definition is changed here
(CLAUDE.md ground rule 3); the accumulator stays canonical and the
equivalence is enforced by ``tests/test_stats_spectral.py``.

BURN-IN AND THE FINITE-SEGMENT BIAS CORRECTION
----------------------------------------------
Tracked directions are reset whenever the subspace is refreshed, so their
streams are a concatenation of short segments whose first few observations
carry a refresh transient (the direction is momentum-anchored right after a
rotation).  Every function here therefore takes ``burn_in``: the first
``burn_in`` observations of the supplied series are dropped BEFORE any
statistic is formed.  Burn-in is load-bearing, not cosmetic -- a transient
confined to the head of a segment biases the whole lag ladder, and the size
of the bias does not shrink with the number of segments.

On a segment of post-burn-in length n the sample autocorrelation is computed
about the segment's own sample mean, and that mean subtraction alone makes
every lag biased low by about 1/n: for a zero-mean stationary process

    E[c_j] ~= gamma_j - G/n ,  E[c_0] ~= gamma_0 - G/n ,  G = sum_k gamma_k

so

    E[rho_hat_j] ~= rho_j - (g/n) (1 - rho_j) ,  g = G / gamma_0 ,

which for white noise (g = 1, rho_j = 0) is -1/n per lag.  The correction
applied by :func:`lag_ladder` is that process-independent mean-subtraction
term

    rho_j = c_j / c_0 + 1/n                     (bias_correct=True)

i.e. a flat +1/n added to every lag j >= 1.  This is the same first-order
family as the Kendall/Marriott-Pope ``(1 + 3 rho) / ess`` correction used by
:class:`src.stats.direction_stats.DirectionStats`, restricted to the term
that does not require knowing the process; it is reported alongside the
uncorrected ladder (``rho_raw``) so the correction is always auditable.

HOW ACCURATE THE +1/n CORRECTION ACTUALLY IS (measured, not asserted)
--------------------------------------------------------------------
Both statements below are simulations of THIS estimator, because the
intervals the audit prints are narrower than the residual and a first-order
formula is not good enough to interpret them.

* **White noise is not corrected exactly.**  This estimator divides c_j by
  n - j and c_0 by n, so the ratio of expectations at rho_j = 0 is
  -1/(n - 1), not -1/n, and the flat +1/n leaves ~ +1/(n(n-1)) behind.
  Measured raw lag-1 mean at n = 45 over 400k reps: -0.02303, against
  -1/45 = -0.02222 and -1/44 = -0.02273.
* **The first-order residual understates the real one by 2-3x at the
  operating point.**  Substituting the correction into the expansion above
  leaves ``(1 - g (1 - rho_j)) / n`` -- note where the bracket goes; for an
  AR(1) at phi = -0.34, g = (1 + phi)/(1 - phi) = 0.49 and n = 45, that
  predicts +0.0076.  Direct simulation (300k reps, n = 45, MEDIAN over
  segments, which is how the audit pools) measures +0.0141 at phi = -0.343
  and +0.0067 at phi = -0.172; mean-pooled the residuals are +0.0213 and
  +0.0103.  Inverting the map, a reported median rho_1 of -0.343
  corresponds to a true phi of about -0.358, and -0.172 to about -0.179, so
  a top-minus-bulk difference of -0.171 is really about -0.179.  **Any
  interval on rho_1 narrower than ~0.015 at n = 45 is an interval on the
  estimator, not on phi**; report it as such or invert the map above.

WHAT THE NULL-CALIBRATED RATIO DOES *NOT* IDENTIFY
--------------------------------------------------
``|t_nw|`` divided by the median ``|t_nw|`` of an AR(1)(phi_hat) surrogate
is a scale correction, not an identification.  Its numerator is exactly the
zero-frequency content of the window, while its denominator integrates only
lags 1..L with Bartlett weights, and L is the Newey-West bandwidth of that
window (L = 3 at n = 45, L = 2 at n = 25).  A ZERO-MEAN component whose
correlation time exceeds ~L steps is therefore invisible to sigma_LR^2 and
fully present in Var(ybar); an AR(1) fitted to rho_1 alone has correlation
time < 1 step by construction and cannot calibrate it away.

Measured: a zero-mean sum of a fast AR(1) (phi = -0.40, unit variance) and a
slow AR(1) (phi = 0.97, variance 0.15), read at n = 45 after burn-in 5 and
calibrated against the AR(1)(rho_1_hat) null this module supplies, returns
rho_1 = -0.332, dc median |t| = 1.78, **dc ratio = 2.71**, alt ratio = 1.04.
A stream with no mean at all reproduces a "large DC excess, alternating
channel at the null" reading.  ``ratio >> 1`` on the DC channel therefore
says *power at zero frequency beyond what an AR(1)(rho_1) window absorbs*,
never *a persistent mean*.

Separating the two needs a LONGER window, which is what
:func:`segmented_channel_t` is for: over k concatenated segments a genuinely
persistent mean grows |t| like sqrt(k), while within-segment low-frequency
power does not, because L grows with N and starts absorbing it.  Both the
confound and the discriminator are pinned in ``tests/test_stats_spectral.py``.

Everything is deterministic: NumPy only (no torch, no GPU), seeded RNGs
only, no timestamps, sorted dict keys on output.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

__all__ = [
    "CHANNELS",
    "DEFAULT_MAX_LAG",
    "ar1_segmented_null",
    "ar1_streams",
    "ar1_surrogate_null",
    "block_bootstrap_ci",
    "channel_t",
    "demodulate",
    "lag_ladder",
    "newey_west_bandwidth",
    "segment_mean_persistence",
    "segmented_channel_t",
]

CHANNELS = ("alt", "dc")
DEFAULT_MAX_LAG = 8  # matches FrozenProbeAccumulator's default
DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
VAR_FLOOR = 1e-30  # numerical guard, matches direction_stats.py


# --------------------------------------------------------------------- utils


def _series(x, name: str = "x") -> np.ndarray:
    """Validate and copy the input as a 1-D float64 array."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    return a


def _drop_burn_in(x: np.ndarray, burn_in: int) -> np.ndarray:
    burn_in = int(burn_in)
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")
    return x[burn_in:]


def demodulate(x) -> np.ndarray:
    """Alternating demodulation ``(-1)^t x_t`` with t indexing the input.

    The sign is tied to the position in the SUPPLIED series, so demodulation
    commutes with nothing: apply it before dropping burn-in (as
    :func:`channel_t` does) and an odd ``burn_in`` flips the sign of the
    channel mean, never its magnitude.
    """
    a = _series(x)
    sign = np.where(np.arange(a.size) % 2 == 0, 1.0, -1.0)
    return a * sign


def newey_west_bandwidth(n: int, max_lag: int = DEFAULT_MAX_LAG) -> int:
    """Newey-West (1994) automatic Bartlett bandwidth, capped at ``max_lag``.

    Mirrors :meth:`FrozenProbeAccumulator.lag_truncation` exactly:
    ``L = min(max_lag, floor(4 (n/100)^(2/9)), n - 2)``, and 0 when n < 3 or
    ``max_lag == 0``.
    """
    n, max_lag = int(n), int(max_lag)
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")
    if n < 3 or max_lag == 0:
        return 0
    rule = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return int(max(0, min(max_lag, rule, n - 2)))


def _lag_sums(y: np.ndarray, max_lag: int) -> np.ndarray:
    """``S_j = sum_t y_t y_{t-j}`` for j = 0..max_lag (NaN where no pairs)."""
    n = y.size
    out = np.full(max_lag + 1, np.nan)
    out[0] = float(np.dot(y, y))
    for j in range(1, max_lag + 1):
        if n - j < 1:
            break
        out[j] = float(np.dot(y[j:], y[:-j]))
    return out


# ---------------------------------------------------------------- lag ladder


def lag_ladder(
    x,
    max_lag: int = 4,
    burn_in: int = 0,
    *,
    bias_correct: bool = True,
) -> Dict[str, object]:
    """Per-lag autocorrelations rho_1..rho_max_lag of a 1-D series.

    The first ``burn_in`` observations are dropped first; every statistic is
    then formed on the remaining segment of length n about that segment's own
    sample mean, with the autocovariance definition of
    :meth:`FrozenProbeAccumulator.stats` (``c_0 = S_0/n - mean^2``,
    ``c_j = S_j/(n-j) - mean^2``).

    The reported ladder carries the finite-segment mean-subtraction bias
    correction documented in the module docstring::

        rho_j = c_j / c_0 + 1/n            (bias_correct=True, the default)
        rho_j = c_j / c_0                  (bias_correct=False)

    a flat +1/n on every lag j >= 1; the uncorrected ladder is returned too
    (``rho_raw``) and ``bias_correction`` records the exact offset used.

    Degenerate segments follow the repo convention
    (:class:`DirectionStats`): a segment whose variance is at or below
    ``VAR_FLOOR`` reports rho = 0 at every lag and no bias correction (the
    correlation is undefined, not zero-with-error).  Lags with no available
    pair (j >= n) report NaN, as does a segment shorter than 2.

    Returns a dict with keys ``bias_correction``, ``burn_in``, ``lags``,
    ``max_lag``, ``mean``, ``n``, ``rho``, ``rho_raw``, ``var``.
    """
    max_lag = int(max_lag)
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")
    y = _drop_burn_in(_series(x), burn_in)
    n = int(y.size)
    lags = np.arange(1, max_lag + 1)
    out = {
        "bias_correction": 0.0,
        "burn_in": int(burn_in),
        "lags": lags,
        "max_lag": max_lag,
        "mean": float("nan"),
        "n": n,
        "rho": np.full(max_lag, np.nan),
        "rho_raw": np.full(max_lag, np.nan),
        "var": float("nan"),
    }
    if n < 2:
        return out

    mean = float(y.sum() / n)
    S = _lag_sums(y, max_lag)
    c0 = max(float(S[0] / n - mean * mean), 0.0)
    out["mean"] = mean
    out["var"] = c0
    if c0 <= VAR_FLOOR:
        out["rho"] = np.zeros(max_lag)
        out["rho_raw"] = np.zeros(max_lag)
        return out

    raw = np.full(max_lag, np.nan)
    for j in lags:
        if n - j < 1 or not np.isfinite(S[j]):
            continue
        raw[j - 1] = (S[j] / (n - j) - mean * mean) / c0
    correction = 1.0 / n if bias_correct else 0.0
    out["bias_correction"] = float(correction)
    out["rho_raw"] = raw
    out["rho"] = raw + correction
    return out


# ------------------------------------------------------------ channel t-stat


def channel_t(
    x,
    channel: str = "dc",
    burn_in: int = 0,
    *,
    max_lag: int = DEFAULT_MAX_LAG,
) -> Dict[str, object]:
    """Newey-West t-statistic of a channel mean of a 1-D series.

    ``channel='dc'`` reads the plain series, ``channel='alt'`` the
    demodulated series ``(-1)^t x_t`` (t indexing the supplied array, see
    :func:`demodulate`).  The channel transform is applied FIRST and the
    first ``burn_in`` observations of the transformed series are dropped
    second, so a given absolute step keeps its demodulation sign no matter
    what burn-in the caller chooses.

    The estimator is the mirror of
    :meth:`FrozenProbeAccumulator.stats` (see the module docstring) applied
    to the post-burn-in channel series: Bartlett kernel, Newey-West (1994)
    automatic bandwidth capped at ``max_lag``, ``c_j`` with divisor ``n - j``
    and the full-segment mean subtracted, and a fallback to ``c_0`` with
    ``nw_floored`` set when the truncated long-run variance is non-positive.

    Returns a dict with keys ``burn_in``, ``channel``, ``ess``,
    ``lag_truncation``, ``mean``, ``n``, ``nw_floored``, ``sigma_lr2``,
    ``t_naive``, ``t_nw``, ``var``.  An empty post-burn-in segment reports
    NaN statistics with ``n = 0``.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {sorted(CHANNELS)}, got {channel!r}")
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")

    a = _series(x)
    z = demodulate(a) if channel == "alt" else a
    y = _drop_burn_in(z, burn_in)
    n = int(y.size)
    out = {
        "burn_in": int(burn_in),
        "channel": channel,
        "ess": float("nan"),
        "lag_truncation": 0,
        "mean": float("nan"),
        "n": n,
        "nw_floored": False,
        "sigma_lr2": float("nan"),
        "t_naive": float("nan"),
        "t_nw": float("nan"),
        "var": float("nan"),
    }
    if n == 0:
        return out

    mean = float(y.sum() / n)
    S = _lag_sums(y, max_lag)
    c0 = max(float(S[0] / n - mean * mean), 0.0)
    L = newey_west_bandwidth(n, max_lag)
    sigma_lr2 = c0
    for j in range(1, L + 1):
        cnt = n - j
        if cnt <= 0 or not np.isfinite(S[j]):
            continue
        c_j = float(S[j] / cnt - mean * mean)
        sigma_lr2 += 2.0 * (1.0 - j / (L + 1.0)) * c_j
    floored = bool(sigma_lr2 <= 0.0)
    if floored:
        sigma_lr2 = c0

    out["ess"] = float(n * c0 / sigma_lr2) if sigma_lr2 > 0.0 else float(n)
    out["lag_truncation"] = int(L)
    out["mean"] = mean
    out["nw_floored"] = floored
    out["sigma_lr2"] = float(sigma_lr2)
    out["t_naive"] = float(mean / np.sqrt(c0 / n)) if c0 > 0.0 else 0.0
    out["t_nw"] = float(mean / np.sqrt(sigma_lr2 / n)) if sigma_lr2 > 0.0 else 0.0
    out["var"] = c0
    return out


# ------------------------------------------------ pooled (slot-level) t-stat


def segmented_channel_t(
    segments,
    channel: str = "dc",
    burn_in: int = 0,
    *,
    max_lag: int = DEFAULT_MAX_LAG,
) -> Dict[str, object]:
    """Channel-mean Newey-West t over SEVERAL segments of one direction.

    This is the estimator the channel-audit pre-registration §5.1 registers
    as its unit of analysis: a direction's post-burn-in segments pooled into
    one statistic, with **lag products never crossing a segment boundary**.
    Each supplied segment is demodulated (``channel='alt'``) and burn-in
    dropped exactly as :func:`channel_t` does it, and then

        N     = sum_i n_i
        mean  = (sum_i sum_t y^(i)_t) / N
        S_j   = sum_i sum_{t >= j} y^(i)_t y^(i)_{t-j}
        P_j   = sum_i max(n_i - j, 0)                (pairs, P_0 = N)
        c_0   = S_0 / N   - mean^2                   (floored at 0)
        c_j   = S_j / P_j - mean^2
        L     = min(max_lag, floor(4 (N/100)^(2/9)), N - 2)

    with the same Bartlett kernel, the same ``c_0`` fallback and the same
    ``t_naive`` / ``t_nw`` / ``ess`` definitions as :func:`channel_t`, which
    it reproduces exactly on a single segment.

    WHY IT EXISTS.  The per-segment ``ratio`` cannot tell a persistent mean
    from zero-mean low-frequency power (module docstring): at n = 45 the
    bandwidth is L = 3, so anything slower than ~3 steps sits entirely in
    the numerator.  Pooling k segments is the discriminator, because the two
    alternatives scale differently: a mean that persists across refreshes
    grows |t| like sqrt(k), while within-segment low-frequency power does
    not, since L rises with N and starts absorbing it.  Report the growth
    factor against sqrt(k), not the pooled |t| alone.

    Segments shorter than ``burn_in + 1`` contribute nothing and are counted
    in ``n_segments_empty``.  Returns a dict with keys ``burn_in``,
    ``channel``, ``ess``, ``lag_truncation``, ``mean``, ``n``,
    ``n_segments``, ``n_segments_empty``, ``nw_floored``, ``pair_counts``,
    ``segment_lengths``, ``segment_means``, ``sigma_lr2``, ``t_naive``,
    ``t_nw``, ``var``.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {sorted(CHANNELS)}, got {channel!r}")
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")

    kept: List[np.ndarray] = []
    n_empty = 0
    for k, seg in enumerate(segments):
        a = _series(seg, name=f"segments[{k}]")
        z = demodulate(a) if channel == "alt" else a
        y = _drop_burn_in(z, burn_in)
        if y.size:
            kept.append(y)
        else:
            n_empty += 1
    lengths = [int(y.size) for y in kept]
    n = int(sum(lengths))
    out: Dict[str, object] = {
        "burn_in": int(burn_in),
        "channel": channel,
        "ess": float("nan"),
        "lag_truncation": 0,
        "mean": float("nan"),
        "n": n,
        "n_segments": len(kept),
        "n_segments_empty": n_empty,
        "nw_floored": False,
        "pair_counts": [0] * (max_lag + 1),
        "segment_lengths": lengths,
        "segment_means": [],
        "sigma_lr2": float("nan"),
        "t_naive": float("nan"),
        "t_nw": float("nan"),
        "var": float("nan"),
    }
    if n == 0:
        return out

    S = np.zeros(max_lag + 1)
    P = np.zeros(max_lag + 1, dtype=np.int64)
    seg_means: List[float] = []
    total = 0.0
    for y in kept:
        total += float(y.sum())
        seg_means.append(float(y.sum() / y.size))
        s = _lag_sums(y, max_lag)
        for j in range(max_lag + 1):
            if np.isfinite(s[j]):
                S[j] += float(s[j])
                P[j] += y.size - j
    P[0] = n
    mean = total / n
    c0 = max(float(S[0] / n - mean * mean), 0.0)
    L = newey_west_bandwidth(n, max_lag)
    sigma_lr2 = c0
    for j in range(1, L + 1):
        if P[j] <= 0:
            continue
        sigma_lr2 += 2.0 * (1.0 - j / (L + 1.0)) * float(S[j] / P[j] - mean * mean)
    floored = bool(sigma_lr2 <= 0.0)
    if floored:
        sigma_lr2 = c0

    out["ess"] = float(n * c0 / sigma_lr2) if sigma_lr2 > 0.0 else float(n)
    out["lag_truncation"] = int(L)
    out["mean"] = mean
    out["nw_floored"] = floored
    out["pair_counts"] = [int(v) for v in P]
    out["segment_means"] = seg_means
    out["sigma_lr2"] = float(sigma_lr2)
    out["t_naive"] = float(mean / np.sqrt(c0 / n)) if c0 > 0.0 else 0.0
    out["t_nw"] = float(mean / np.sqrt(sigma_lr2 / n)) if sigma_lr2 > 0.0 else 0.0
    out["var"] = c0
    return out


# ---------------------------------------------------------- AR(1) null model


def ar1_streams(
    rng: np.random.Generator, phi: float, n: int, reps: int
) -> np.ndarray:
    """``reps`` x ``n`` zero-mean unit-noise AR(1) draws, stationary start.

    Same recursion and same stationary initialization as
    :func:`src.stats.generators.ar1` (x_0 drawn from
    N(0, 1 / (1 - phi^2)) and then iterated), vectorized over reps so the
    whole null costs one RNG stream.

    Public because the surrogate generator is part of the tested surface:
    callers that need synthetic controls (the channel-audit estimator
    controls, for one) must draw them from here rather than re-implement the
    recursion, so a control cannot silently drift away from the null it is
    supposed to calibrate (CLAUDE.md WP1.1: the stats module is the tested
    code, no reimplementation).
    """
    if not -1.0 < float(phi) < 1.0:
        raise ValueError(f"ar1_streams requires |phi| < 1, got {phi}")
    phi, n, reps = float(phi), int(n), int(reps)
    eps = rng.standard_normal((reps, n))
    x = rng.standard_normal(reps) / np.sqrt(1.0 - phi * phi)
    out = np.empty((reps, n), dtype=np.float64)
    for t in range(n):
        x = phi * x + eps[:, t]
        out[:, t] = x
    return out


_ar1_streams = ar1_streams  # backwards-compatible private alias


def _summary(values: np.ndarray, quantiles: Sequence[float]) -> Dict[str, object]:
    """median / mean / quantiles of a 1-D sample, NaN-safe on empty input."""
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "quantiles": [float(q) for q in np.quantile(v, list(quantiles))],
    }


def ar1_surrogate_null(
    phi: float,
    n: int,
    reps: int,
    seed: int,
    *,
    burn_in: int = 0,
    max_lag: int = DEFAULT_MAX_LAG,
    channels: Sequence[str] = CHANNELS,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> Dict[str, object]:
    """Null distribution of :func:`channel_t` on zero-mean AR(1) streams.

    Draws ``reps`` independent AR(1) streams of length ``n`` with parameter
    ``phi`` (zero mean by construction: there is NOTHING to detect, in either
    channel) and pushes each through :func:`channel_t` with the same
    ``burn_in`` and ``max_lag`` the observed streams are read with.  The
    resulting distribution is what an observed |t| must be compared against:
    a zig-zag null (phi < 0) inflates ess above n and shifts the whole |t|
    distribution, so an uncalibrated |t| threshold is not interpretable.

    At phi = 0 the |t_nw| median tends to the half-normal median 0.674 only
    ASYMPTOTICALLY; the Bartlett truncation costs a few percent at short n
    and the finite-n value is what a ratio is actually divided by.  Measured
    (20000 reps, max_lag = 8): **0.737 at n = 25, 0.723 at n = 45, 0.696 at
    n = 100, 0.685 at n = 400** -- i.e. +9%, +7%, +3%, +2% over 0.674.  Use
    the null this function returns, never 0.674, as a denominator.

    ``reps`` is a Monte-Carlo budget and the median it produces has Monte-
    Carlo error of its own: measured over 24 independent seeds at
    n = 50 / burn_in = 5 / reps = 2000, the relative sd of
    ``median |t_nw|`` is 2.8% (dc) and 3.1% (alt) at phi = 0, and 3.1% /
    3.3% at phi = -0.343, falling like 1/sqrt(reps).  A ratio
    formed against this median inherits that error in its DENOMINATOR;
    quote it (the per-rep ``samples`` support an i.i.d. bootstrap:
    ``block_bootstrap_ci(samples['abs_t_nw'], 1, ...)``) rather than
    printing the ratio bare.

    The null is computed by calling :func:`channel_t` itself, never a
    shortcut formula, so any change to the estimator moves the null with it.

    Returns a dict with keys ``burn_in``, ``max_lag``, ``n``, ``phi``,
    ``quantile_levels``, ``reps``, ``seed`` and one entry per channel name;
    each channel entry holds ``abs_t_naive``, ``abs_t_nw``, ``ess``,
    ``ess_over_n``, ``t_nw`` summaries (median / mean / quantiles),
    ``nw_floored_frac``, and the per-rep ``samples`` (``abs_t_nw``,
    ``ess``, ``t_naive``, ``t_nw``) for exceedance/p-value calibration.
    """
    phi, n, reps, seed = float(phi), int(n), int(reps), int(seed)
    if not -1.0 < phi < 1.0:
        raise ValueError(f"ar1_surrogate_null requires |phi| < 1, got {phi}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    channels = tuple(channels)
    for ch in channels:
        if ch not in CHANNELS:
            raise ValueError(f"unknown channel {ch!r}")

    rng = np.random.default_rng(seed)
    streams = _ar1_streams(rng, phi, n, reps)

    out: Dict[str, object] = {
        "burn_in": int(burn_in),
        "max_lag": int(max_lag),
        "n": n,
        "phi": phi,
        "quantile_levels": [float(q) for q in quantiles],
        "reps": reps,
        "seed": seed,
    }
    for ch in channels:
        t_nw = np.empty(reps)
        t_naive = np.empty(reps)
        ess = np.empty(reps)
        floored = np.empty(reps, dtype=bool)
        for r in range(reps):
            st = channel_t(streams[r], ch, burn_in, max_lag=max_lag)
            t_nw[r] = st["t_nw"]
            t_naive[r] = st["t_naive"]
            ess[r] = st["ess"]
            floored[r] = st["nw_floored"]
        n_eff = max(n - int(burn_in), 1)
        out[ch] = {
            "abs_t_naive": _summary(np.abs(t_naive), quantiles),
            "abs_t_nw": _summary(np.abs(t_nw), quantiles),
            "ess": _summary(ess, quantiles),
            "ess_over_n": _summary(ess / n_eff, quantiles),
            "nw_floored_frac": float(np.mean(floored)),
            "samples": {
                "abs_t_nw": np.abs(t_nw),
                "ess": ess,
                "t_naive": t_naive,
                "t_nw": t_nw,
            },
            "t_nw": _summary(t_nw, quantiles),
        }
    return out


def segment_mean_persistence(means) -> Dict[str, float]:
    """Two boundary-crossing summaries of a slot's SEGMENT MEANS.

    ``means`` is the ordered sequence of per-segment channel means of one
    direction slot (``segmented_channel_t(...)['segment_means']``).  Both
    statistics are taken ABOUT ZERO, not about the slot's own mean, because
    the question is whether the mean itself persists across a subspace
    refresh -- demeaning would remove exactly the quantity being tested:

        coherence = max(#(m > 0), #(m < 0)) / k
        acf1      = sum_i m_i m_{i+1} / sum_i m_i^2

    A mean that survives refreshes drives coherence to 1 and acf1 to its own
    ceiling (k - 1)/k -- the numerator has k - 1 terms against the
    denominator's k, so 1 is unreachable and 0.833 is the k = 6 maximum.
    Independent zero-mean segments give coherence ~ E[max(B, k-B)]/k
    (measured 0.667 at k = 6, -> 0.5 as k -> inf) and acf1 ~ 0.  Both are
    arithmetic on already-estimated means, not a second estimator, and they
    carry no null of their own: compare
    against :func:`ar1_segmented_null`, which reports the same two numbers on
    zero-mean streams of the same shape.

    Returns ``{'acf1': ..., 'coherence': ..., 'k': ...}``; NaN where k < 2
    (acf1 also NaN when every mean is exactly 0).
    """
    m = np.asarray(means, dtype=np.float64).ravel()
    k = int(m.size)
    out = {"acf1": float("nan"), "coherence": float("nan"), "k": k}
    if k < 1:
        return out
    pos = int(np.sum(m > 0.0))
    neg = int(np.sum(m < 0.0))
    out["coherence"] = float(max(pos, neg) / k)
    if k >= 2:
        denom = float(np.dot(m, m))
        if denom > 0.0:
            out["acf1"] = float(np.dot(m[1:], m[:-1]) / denom)
    return out


def ar1_segmented_null(
    phi: float,
    seg_len,
    n_segments: int,
    reps: int,
    seed: int,
    *,
    burn_in: int = 0,
    max_lag: int = DEFAULT_MAX_LAG,
    channels: Sequence[str] = CHANNELS,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> Dict[str, object]:
    """Null distribution of :func:`segmented_channel_t` over k segments.

    The slot-level companion of :func:`ar1_surrogate_null`.  Each rep is one
    synthetic direction slot: ``n_segments`` INDEPENDENT zero-mean AR(1)
    draws of length ``seg_len``, each with its own stationary start.  The
    independence is the point, not a shortcut -- the tracked subspace is
    re-derived at every refresh, so consecutive segments of a slot are not
    consecutive observations of one trajectory, and a null that chained them
    would smuggle in the very cross-segment memory the slot-level statistic
    exists to detect.

    Zero mean by construction, so ``median |t_nw|`` here is the reference the
    observed slot-level ``|t_nw|`` is divided by, and the segment-level null
    at the same phi is the reference for the segment-level one; the ratio of
    the two growth factors is what separates a persistent mean (grows like
    sqrt(n_segments)) from within-segment low-frequency power (does not).

    ``seg_len`` is either one length (repeated ``n_segments`` times) or the
    explicit sequence of segment lengths, which is what a real slot supplies:
    a run whose last refresh window is truncated has a ragged shape such as
    (50, 50, 50, 42), and matching the null to the cell's MODAL or MEDIAN
    length instead of its actual shape is how a null ends up drawn at a
    length that does not occur in the data at all.  When a sequence is given
    it wins and ``n_segments`` is ignored.

    Returns the same shape as :func:`ar1_surrogate_null` plus ``n_segments``
    and ``seg_len`` (always the resolved list); ``n`` is the pooled
    post-burn-in length.
    """
    phi, seed = float(phi), int(seed)
    reps = int(reps)
    if np.ndim(seg_len) == 0:
        lengths = [int(seg_len)] * int(n_segments)
    else:
        lengths = [int(v) for v in seg_len]
    n_segments = len(lengths)
    if not -1.0 < phi < 1.0:
        raise ValueError(f"ar1_segmented_null requires |phi| < 1, got {phi}")
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")
    if min(lengths) < 1:
        raise ValueError(f"every seg_len must be >= 1, got {lengths}")
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    channels = tuple(channels)
    for ch in channels:
        if ch not in CHANNELS:
            raise ValueError(f"unknown channel {ch!r}")

    rng = np.random.default_rng(seed)
    per_segment = [ar1_streams(rng, phi, L, reps) for L in lengths]
    n_kept = int(sum(max(L - int(burn_in), 0) for L in lengths))

    out: Dict[str, object] = {
        "burn_in": int(burn_in),
        "max_lag": int(max_lag),
        "n": int(n_kept),
        "n_segments": n_segments,
        "phi": phi,
        "quantile_levels": [float(q) for q in quantiles],
        "reps": reps,
        "seed": seed,
        "seg_len": lengths,
    }
    for ch in channels:
        t_nw = np.empty(reps)
        t_naive = np.empty(reps)
        ess = np.empty(reps)
        coherence = np.empty(reps)
        acf1 = np.empty(reps)
        floored = np.empty(reps, dtype=bool)
        for r in range(reps):
            st = segmented_channel_t(
                [blk[r] for blk in per_segment], ch, burn_in, max_lag=max_lag
            )
            t_nw[r] = st["t_nw"]
            t_naive[r] = st["t_naive"]
            ess[r] = st["ess"]
            floored[r] = st["nw_floored"]
            per = segment_mean_persistence(st["segment_means"])
            coherence[r] = per["coherence"]
            acf1[r] = per["acf1"]
        out[ch] = {
            "abs_t_naive": _summary(np.abs(t_naive), quantiles),
            "abs_t_nw": _summary(np.abs(t_nw), quantiles),
            "ess": _summary(ess, quantiles),
            "ess_over_n": _summary(ess / max(n_kept, 1), quantiles),
            "nw_floored_frac": float(np.mean(floored)),
            "samples": {
                "abs_t_nw": np.abs(t_nw),
                "ess": ess,
                "segment_mean_acf1": acf1,
                "segment_mean_coherence": coherence,
                "t_naive": t_naive,
                "t_nw": t_nw,
            },
            "segment_mean_acf1": _summary(acf1[np.isfinite(acf1)], quantiles)
            if np.any(np.isfinite(acf1))
            else _summary(np.array([np.nan]), quantiles),
            "segment_mean_coherence": _summary(coherence, quantiles),
            "t_nw": _summary(t_nw, quantiles),
        }
    return out


# ------------------------------------------------------- block bootstrap CI


def block_bootstrap_ci(
    values,
    block: int,
    reps: int,
    seed: int,
    *,
    level: float = 95.0,
    statistic: Callable[..., np.ndarray] = np.median,
    max_elems: int = 2_000_000,
) -> Dict[str, object]:
    """Circular-block-bootstrap CI for a median over dependent statistics.

    ``values`` are per-segment (or per-direction, or per-step) statistics
    that are NOT independent: neighbouring refresh segments of the same run
    share a direction, a learning rate and a phase of training, so an i.i.d.
    bootstrap understates the spread of their median.  The circular block
    bootstrap resamples ``ceil(n / block)`` contiguous blocks of length
    ``block`` with wrap-around, concatenates them, truncates back to n, and
    recomputes ``statistic``; ``block = 1`` degenerates to the i.i.d.
    bootstrap, and the CI widens as ``block`` grows on positively dependent
    input.

    ``statistic`` must accept an ``axis`` keyword (``np.median`` by default,
    which is the intended use); it is applied along the resampled axis.
    ``level`` is the two-sided coverage in percent.  Deterministic given
    ``seed``.

    ``max_elems`` bounds the resample index array (reps x n) in MEMORY only:
    the reps are drawn in chunks of ``max(1, max_elems // n)`` from the same
    generator, in the same order, so the result is bit-identical to the
    unchunked computation at any ``max_elems``.  It is deliberately not a
    rep-count reduction -- a 95% interval from 50 draws is the 2nd order
    statistic at each end, seed-unstable and ~5% too narrow, which is not a
    trade a caller should make silently on its largest pool.

    Returns a dict with keys ``block``, ``ci_hi``, ``ci_lo``, ``level``,
    ``n``, ``point``, ``reps``, ``se``, ``seed``.
    """
    v = _series(values, name="values")
    n = int(v.size)
    block, reps, seed = int(block), int(reps), int(seed)
    if n < 1:
        raise ValueError("values must be non-empty")
    if not 1 <= block <= n:
        raise ValueError(f"block must be in [1, {n}], got {block}")
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    if not 0.0 < level < 100.0:
        raise ValueError(f"level must be in (0, 100), got {level}")

    if max_elems < 1:
        raise ValueError(f"max_elems must be >= 1, got {max_elems}")

    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)[None, None, :]
    per_chunk = int(max(1, min(reps, max_elems // n)))
    rng = np.random.default_rng(seed)
    chunks = []
    drawn = 0
    while drawn < reps:
        size = min(per_chunk, reps - drawn)
        starts = rng.integers(0, n, size=(size, n_blocks))
        idx = (starts[:, :, None] + offsets) % n
        idx = idx.reshape(size, n_blocks * block)[:, :n]
        chunks.append(np.asarray(statistic(v[idx], axis=1), dtype=np.float64))
        drawn += size
    draws = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    tail = (100.0 - level) / 2.0
    lo, hi = np.percentile(draws, [tail, 100.0 - tail])
    return {
        "block": block,
        "ci_hi": float(hi),
        "ci_lo": float(lo),
        "level": float(level),
        "n": n,
        "point": float(statistic(v)),
        "reps": reps,
        "se": float(np.std(draws, ddof=1)) if reps > 1 else float("nan"),
        "seed": seed,
    }
