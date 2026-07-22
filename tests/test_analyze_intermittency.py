"""Synthetic-signal tests for the intermittency scan (WP0.5 discipline):
planted heavy-tailed directions must light up; Gaussian AR(1) must not."""

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_intermittency import (
    direction_stats,
    robust_pooled_z,
    simulate_null,
    split_half_counts,
)

REFRESH = [1 + 50 * j for j in range(16)]


def ar1(rho, n, rng):
    s = np.empty(n)
    s[0] = rng.standard_normal()
    c = math.sqrt(1 - rho * rho)
    for t in range(1, n):
        s[t] = rho * s[t - 1] + c * rng.standard_normal()
    return s


def spiky(n, rng, rate=0.02, amp=8.0):
    s = rng.standard_normal(n)
    hits = rng.random(n) < rate
    s[hits] += amp * rng.choice([-1.0, 1.0], size=hits.sum())
    return s


def test_gaussian_ar1_stays_in_null_band():
    rng = np.random.default_rng(0)
    null = simulate_null(500, seed=1)
    thr = np.quantile(null["g2"], 0.99)
    hits = 0
    for _ in range(100):
        z = robust_pooled_z(ar1(-0.4, 800, rng), REFRESH)
        if direction_stats(z)["g2"] > thr:
            hits += 1
    assert hits <= 6  # ~1% expected; generous bound


def test_planted_spikes_light_up():
    rng = np.random.default_rng(2)
    null = simulate_null(500, seed=1)
    thr_g2 = np.quantile(null["g2"], 0.99)
    thr_p4 = np.quantile(null["p4"], 0.99)
    hits_g2 = hits_p4 = 0
    for _ in range(50):
        z = robust_pooled_z(spiky(800, rng), REFRESH)
        st = direction_stats(z)
        hits_g2 += st["g2"] > thr_g2
        hits_p4 += st["p4"] > thr_p4
    assert hits_g2 >= 45
    assert hits_p4 >= 45


def test_mad_standardization_does_not_mask_spikes():
    """A single huge spike must still exceed |z|>4 after standardization
    (the sd-based z would shrink it toward the threshold)."""
    rng = np.random.default_rng(3)
    s = rng.standard_normal(800)
    s[400] = 50.0
    z = robust_pooled_z(s, REFRESH)
    assert np.max(np.abs(z)) > 20


def test_split_half_counts_partition_windows():
    rng = np.random.default_rng(4)
    s = rng.standard_normal(800)
    # plant spikes only in even-indexed windows (0-based): steps [1,51), [101,151), ...
    for j in range(0, 16, 2):
        s[(j * 50) + 10] = 30.0
    odd, even = split_half_counts(s, REFRESH)
    # counts[0] collects j%2==0 windows: 8 planted spikes there, none in the
    # other half; MAD-z background exceedances (~2-5 per half) hit both.
    assert odd - even >= 5
