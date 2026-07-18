"""WP0.5: pure oscillation s(t) = A (-r)^t -- implied eta*lambda and flags.

DoD line covered: "Pure oscillation s(t) = A (-r)^t for r in {0.8, 1.0, 1.1}
-> implied eta*lambda from amplitude ratio within +/-0.1 of ground truth;
classified oscillating; amplitude-decay flag correct."

Ground truth: |s_t / s_{t-1}| = r exactly, and for the quadratic-model
oscillation x_{t+1} = (1 - eta*lambda) x_t the amplitude ratio r maps to
eta*lambda = 1 + r.  Decay flag: r < 1 decaying, r >= 1 non-decaying.

The r = 0.8 stream is run for 100 steps only so the second-moment EMA
(~ A^2 r^{2t} ~ 1e-20 at t = 100) stays far above the var_floor guard;
decayed-to-zero oscillations are a var_floor question, not a WP0.5 one.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

from src.stats import Regime, RegimeClassifier, oscillation

DECAY_MARGIN = 0.05  # threshold parameter for the decay flag, chosen here
N_MIN = 30

CASES = [
    # (r, n_steps, expected_decaying)
    (0.8, 100, True),
    (1.0, 300, False),
    (1.1, 200, False),
]


def _run(beta, r, n):
    stream = oscillation(n, r, amplitude=1.0)
    clf = RegimeClassifier(
        beta=beta, tau_sig=4.0, tau_noise=2.0, rho_osc=0.5, n_min=N_MIN
    )
    history = [clf.update(s) for s in stream]
    return clf, history


@pytest.mark.parametrize("beta", [0.9, 0.99])
@pytest.mark.parametrize("r,n,expected_decaying", CASES)
def test_implied_eta_lambda_within_tolerance(beta, r, n, expected_decaying):
    clf, _ = _run(beta, r, n)
    true_eta_lambda = 1.0 + r
    assert float(clf.stats.implied_eta_lambda) == pytest.approx(
        true_eta_lambda, abs=0.1
    )


@pytest.mark.parametrize("beta", [0.9, 0.99])
@pytest.mark.parametrize("r,n,expected_decaying", CASES)
def test_classified_oscillating(beta, r, n, expected_decaying):
    _, history = _run(beta, r, n)
    # Starts in SIGNAL (confidence default), must be OSCILLATING for every
    # step once the n_min confidence gate has passed.
    assert history[-1] == Regime.OSCILLATING
    assert all(h == Regime.OSCILLATING for h in history[N_MIN:])
    # Before the confidence gate the direction stays in SIGNAL (stock Muon).
    assert all(h == Regime.SIGNAL for h in history[: N_MIN - 1])


@pytest.mark.parametrize("beta", [0.9, 0.99])
@pytest.mark.parametrize("r,n,expected_decaying", CASES)
def test_amplitude_decay_flag(beta, r, n, expected_decaying):
    clf, _ = _run(beta, r, n)
    assert bool(clf.stats.is_decaying(DECAY_MARGIN)) is expected_decaying
