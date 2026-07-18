"""Synthetic scalar-stream generators for the WP0.5 validation suite.

Pure NumPy, no GPU, no training.  Every generator takes an explicit seed
(tests use development seeds >= 1000 per repo seed discipline).
"""

from __future__ import annotations

import numpy as np

__all__ = ["ar1", "drifting_mean", "oscillation", "gaussian_noise", "concat_segments"]


def ar1(
    n: int,
    rho: float,
    *,
    noise_scale: float = 1.0,
    mean: float = 0.0,
    seed: int,
) -> np.ndarray:
    """Stationary AR(1): s_t - mean = rho (s_{t-1} - mean) + noise_scale * eps_t.

    Initialized from the stationary distribution
    N(mean, noise_scale^2 / (1 - rho^2)); lag-1 autocorrelation is exactly rho.
    """
    if not -1.0 < rho < 1.0:
        raise ValueError("ar1 requires |rho| < 1")
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * noise_scale
    s = np.empty(n, dtype=np.float64)
    x = rng.standard_normal() * noise_scale / np.sqrt(1.0 - rho * rho)
    for t in range(n):
        x = rho * x + eps[t]
        s[t] = mean + x
    return s


def drifting_mean(
    n: int,
    snr: float,
    *,
    noise_scale: float = 1.0,
    drift_amplitude: float = 0.2,
    period: int = 5000,
    seed: int,
) -> np.ndarray:
    """Slowly drifting mean plus i.i.d. Gaussian noise.

    m(t) = snr * noise_scale * (1 + drift_amplitude * sin(2 pi t / period)),
    s(t) = m(t) + noise_scale * eps_t.

    The instantaneous SNR |m(t)| / noise_scale therefore stays inside
    [snr * (1 - drift_amplitude), snr * (1 + drift_amplitude)] -- the bounds
    the analytic t-statistic expectation in the tests is computed from.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    m = snr * noise_scale * (1.0 + drift_amplitude * np.sin(2.0 * np.pi * t / period))
    return m + noise_scale * rng.standard_normal(n)


def oscillation(n: int, r: float, *, amplitude: float = 1.0, t0: int = 0) -> np.ndarray:
    """Pure oscillation s(t) = A * (-r)^t for t = t0 .. t0 + n - 1.

    Deterministic (no seed).  |s_t / s_{t-1}| = r exactly, so the implied
    eta*lambda ground truth is 1 + r.
    """
    t = np.arange(t0, t0 + n, dtype=np.float64)
    return amplitude * np.power(-float(r), t)


def gaussian_noise(
    n: int, *, mean: float = 0.0, noise_scale: float = 1.0, seed: int
) -> np.ndarray:
    """i.i.d. N(mean, noise_scale^2) samples (rho = 0)."""
    rng = np.random.default_rng(seed)
    return mean + noise_scale * rng.standard_normal(n)


def concat_segments(*segments: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Concatenate regime segments; returns (stream, switch_indices).

    switch_indices[k] is the index of the first sample of segment k+1, i.e.
    the step at which the k-th mid-stream regime switch occurs.
    """
    stream = np.concatenate(segments)
    boundaries = np.cumsum([len(s) for s in segments])[:-1]
    return stream, [int(b) for b in boundaries]
