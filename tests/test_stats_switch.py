"""WP0.5: mid-stream regime switches -- confidence reset & re-classification.

DoD line covered: "Regime switches mid-stream -> confidence reset logic
re-classifies within N_min steps."

Scenario (one stream, three mid-stream switches):
    noise(400)  ->  signal mean 5 (300)  ->  oscillation +/-6 (300)  ->  noise(400)

For every segment boundary (including t = 0, where the classifier starts in
its SIGNAL prior) the classifier must reach the segment's correct regime
within n_min steps of the boundary, and hold it at the segment's end.

The noise -> signal jump must be caught by the jump (|z| >= z_reset)
innovation detector; the oscillation -> noise collapse produces no large
|z| and must be caught by the quiet (RMS z < z_quiet) detector -- both are
asserted via the recorded reset steps.

All seeds are development seeds (>= 1000); the oscillation segment is
deterministic.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

from src.stats import Regime, RegimeClassifier, concat_segments, gaussian_noise, oscillation

N_MIN = 15


def _scenario():
    segments = [
        gaussian_noise(400, seed=3111),
        gaussian_noise(300, mean=5.0, seed=3112),
        oscillation(300, 1.0, amplitude=6.0),
        gaussian_noise(400, seed=3113),
    ]
    expected = [Regime.NOISE, Regime.SIGNAL, Regime.OSCILLATING, Regime.NOISE]
    stream, switches = concat_segments(*segments)
    return stream, [0] + switches, expected


def _run(beta):
    stream, boundaries, expected = _scenario()
    clf = RegimeClassifier(
        beta=beta,
        tau_sig=4.0,
        tau_noise=2.5,
        rho_osc=0.5,
        n_min=N_MIN,
        z_reset=3.0,
        innov_needed=2,
        innov_window=4,
        z_quiet=0.4,
        quiet_window=6,
    )
    history = [clf.update(s) for s in stream]
    return clf, history, boundaries, expected


@pytest.mark.parametrize("beta", [0.9, 0.99])
def test_reclassified_within_n_min_of_each_switch(beta):
    _, history, boundaries, expected = _run(beta)
    ends = boundaries[1:] + [len(history)]
    for boundary, end, want in zip(boundaries, ends, expected):
        first_correct = next(
            (i for i in range(boundary, len(history)) if history[i] == want), None
        )
        assert first_correct is not None, f"never reached {want} after {boundary}"
        delay = first_correct - boundary
        assert delay <= N_MIN, (
            f"segment at {boundary}: reached {want.value} only after "
            f"{delay} steps (> n_min = {N_MIN})"
        )
        # ... and holds the correct regime at the end of its segment.
        assert history[end - 1] == want


@pytest.mark.parametrize("beta", [0.9, 0.99])
def test_confidence_resets_fire_at_the_detectable_switches(beta):
    clf, _, boundaries, _ = _run(beta)
    resets = clf.reset_steps
    # noise -> signal at t=400: mean jump, caught by the jump detector.
    assert any(boundaries[1] < r <= boundaries[1] + N_MIN for r in resets)
    # oscillation -> noise at t=1000: variance collapse (no large |z|),
    # caught by the quiet detector.
    assert any(boundaries[3] < r <= boundaries[3] + N_MIN for r in resets)


@pytest.mark.parametrize("beta", [0.9, 0.99])
def test_reset_reverts_to_signal_prior(beta):
    """Immediately after a confidence reset the regime is SIGNAL (stock
    behavior) until n_min fresh observations accumulate."""
    stream, _, _ = _scenario()
    clf = RegimeClassifier(
        beta=beta, tau_sig=4.0, tau_noise=2.5, rho_osc=0.5, n_min=N_MIN,
        z_reset=3.0, innov_needed=2, innov_window=4, z_quiet=0.4, quiet_window=6,
    )
    regimes = []
    for s in stream:
        regimes.append(clf.update(s))
    # Find the reset at the oscillation -> noise boundary (t = 1000) and
    # check the regime right at that step was reverted to SIGNAL if the gate
    # had not yet re-passed.
    reset_at_1000 = [r for r in clf.reset_steps if 1000 < r <= 1000 + N_MIN]
    assert reset_at_1000
    step = reset_at_1000[0]  # 1-indexed step of the reset
    assert regimes[step - 1] == Regime.SIGNAL
