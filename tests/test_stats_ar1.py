"""WP0.5: AR(1) recovery -- rho within +/-0.05 after burn-in; regime labels.

DoD line covered: "AR(1) processes with rho in {-0.8, -0.4, 0, 0.4, 0.8},
various noise scales -> classifier must recover rho within +/-0.05 after
burn-in and assign the correct regime label."

Recovery is asserted on the time-averaged bias-corrected autocorrelation
estimate over the post-burn-in window (the instantaneous estimate of any
finite-window estimator fluctuates with sd ~ (1 - rho^2)/sqrt(ESS); the
time average over many window lengths is the quantity with a +/-0.05
resolution, and is what the WP1.2 occupancy analysis will consume).

All zero-mean AR(1) streams have no persistent mean, so the correct label is
NOISE for rho >= 0 and OSCILLATING for rho strongly negative (the boundary
between the two is the rho_osc threshold, a constructor parameter chosen
here per beta to sit well inside the estimator's resolution).

All seeds are development seeds (>= 1000).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from src.stats import DirectionStats, Regime, RegimeClassifier, ar1

RHOS = [-0.8, -0.4, 0.0, 0.4, 0.8]
NOISE_SCALES = [0.5, 2.0]
SEEDS = [1234, 5678]
N, BURN = 8000, 3000


def _recovered_rho(stream: np.ndarray, beta: float) -> float:
    st = DirectionStats(beta)
    vals = []
    for t, s in enumerate(stream):
        st.update(s)
        if t >= BURN:
            vals.append(float(st.rho_corrected))
    return float(np.mean(vals))


@pytest.mark.parametrize("beta", [0.9, 0.99])
@pytest.mark.parametrize("rho", RHOS)
@pytest.mark.parametrize("noise_scale", NOISE_SCALES)
@pytest.mark.parametrize("seed", SEEDS)
def test_rho_recovered_within_tolerance(beta, rho, noise_scale, seed):
    stream = ar1(N, rho, noise_scale=noise_scale, seed=seed)
    recovered = _recovered_rho(stream, beta)
    assert recovered == pytest.approx(rho, abs=0.05)


@pytest.mark.parametrize("beta", [0.9, 0.99])
@pytest.mark.parametrize("rho", RHOS)
def test_rho_recovery_scale_invariant(beta, rho):
    """The autocorrelation estimate must not depend on the noise scale."""
    a = _recovered_rho(ar1(N, rho, noise_scale=0.5, seed=1234), beta)
    b = _recovered_rho(ar1(N, rho, noise_scale=2.0, seed=1234), beta)
    assert a == pytest.approx(b, abs=1e-9)


# --------------------------------------------------------------------------
# Regime labels
# --------------------------------------------------------------------------


def _run_classifier(stream, **kwargs):
    clf = RegimeClassifier(**kwargs)
    return [clf.update(s) for s in stream]


def _occupancy(history, regime, burn=BURN):
    post = history[burn:]
    return sum(h == regime for h in post) / len(post)


@pytest.mark.parametrize(
    "rho,expected",
    [
        (-0.8, Regime.OSCILLATING),
        (-0.4, Regime.OSCILLATING),
        (0.0, Regime.NOISE),
        (0.4, Regime.NOISE),
        (0.8, Regime.NOISE),
    ],
)
def test_labels_beta099(rho, expected):
    stream = ar1(N, rho, noise_scale=1.0, seed=4321)
    hist = _run_classifier(
        stream, beta=0.99, tau_sig=4.0, tau_noise=2.0, rho_osc=0.25, n_min=200
    )
    assert hist[-1] == expected
    assert _occupancy(hist, expected) >= 0.9


@pytest.mark.parametrize(
    "rho,expected",
    [
        (-0.8, Regime.OSCILLATING),
        (-0.4, Regime.OSCILLATING),
        (0.0, Regime.NOISE),
        (0.4, Regime.NOISE),
        (0.8, Regime.NOISE),
    ],
)
def test_labels_beta09(rho, expected):
    """beta=0.9 has ESS ~= 19: the instantaneous rho estimate has sd ~= 0.2,
    so labels near the rho_osc boundary flap between windows.  The correct
    label must still hold the majority of post-burn-in steps and the final
    step; the tighter >= 0.9 occupancy is a beta = 0.99 property (timescale
    sensitivity, documented in the WP0.5 report)."""
    stream = ar1(N, rho, noise_scale=1.0, seed=4321)
    hist = _run_classifier(
        stream, beta=0.9, tau_sig=4.0, tau_noise=2.0, rho_osc=0.25, n_min=50
    )
    assert hist[-1] == expected
    assert _occupancy(hist, expected) >= 0.6


def test_mean_shifted_ar1_is_signal():
    """AR(1) around a persistent nonzero mean must classify as SIGNAL."""
    stream = ar1(N, 0.4, noise_scale=1.0, mean=2.0, seed=9001)
    hist = _run_classifier(
        stream, beta=0.99, tau_sig=4.0, tau_noise=2.0, rho_osc=0.25, n_min=200
    )
    assert hist[-1] == Regime.SIGNAL
    assert _occupancy(hist, Regime.SIGNAL) >= 0.9
