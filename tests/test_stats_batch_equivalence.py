"""Scalar-vs-array equivalence for the batched regime classifier (Task A of
the stats vectorization authorized after the WP1.1 overhead measurement).

The scalar WP0.5-validated pair (DirectionStats, RegimeClassifier) is the
canonical definition; BatchRegimeClassifier must reproduce it exactly:
identical per-step label sequences, statistics, t-stats, and reset steps
across all regimes, both betas (0.9, 0.99), including mid-stream regime
switches and both innovation detectors (jump and quiet/variance-collapse),
plus the tracker-style subspace-rotation reset (fresh classifier state).

The WP0.5 test files themselves are untouched; this file only adds the
equivalence guarantee for the vectorized path.

All seeds are development seeds (>= 1000).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from src.stats import (
    BatchRegimeClassifier,
    Regime,
    RegimeClassifier,
    ar1,
    concat_segments,
    gaussian_noise,
    oscillation,
)

BETAS = [0.9, 0.99]

# Thresholds in the same style as the WP0.5 suite (synthetic-scenario values,
# not scientific defaults); both innovation detectors enabled.
CLF_KWARGS = dict(
    tau_sig=4.0,
    tau_noise=2.5,
    rho_osc=0.4,
    n_min=15,
    z_reset=3.0,
    innov_needed=2,
    innov_window=4,
    z_quiet=0.4,
    quiet_window=6,
)

STAT_PROPS = (
    "mean",
    "var",
    "rho",
    "rho_corrected",
    "ess",
    "ess_adjusted",
    "t_stat",
    "amplitude_ratio",
    "implied_eta_lambda",
)


def _streams(n_extra_noise: int = 0) -> np.ndarray:
    """A (k, T) bank covering every regime and both detector types.

    Rows: signal, noise, oscillating (decaying / critical / growing-ish via
    AR(1) rho<0), AR(1) at several rhos, and two switch scenarios --
    noise -> signal (jump detector) and oscillation -> noise (quiet
    detector), plus optional extra i.i.d. noise rows to reach a target k.
    """
    T = 900
    rows = [
        gaussian_noise(T, mean=5.0, seed=2000),                      # signal
        gaussian_noise(T, seed=2001),                                # noise
        oscillation(T, 1.0, amplitude=6.0),                          # oscillating (critical)
        oscillation(T, 0.98, amplitude=8.0),                         # decaying osc
        ar1(T, -0.8, seed=2002),                                     # oscillating AR(1)
        ar1(T, 0.8, seed=2003),                                      # correlated noise
        ar1(T, 0.4, mean=2.0, seed=2004),                            # mean-shifted AR(1)
        # noise -> signal at t=400 (jump detector fires)
        concat_segments(
            gaussian_noise(400, seed=2005), gaussian_noise(500, mean=5.0, seed=2006)
        )[0],
        # oscillation -> noise at t=450 (quiet detector fires)
        concat_segments(
            oscillation(450, 1.0, amplitude=6.0), gaussian_noise(450, seed=2007)
        )[0],
        # noise -> signal -> oscillation -> noise (the WP0.5 switch scenario)
        concat_segments(
            gaussian_noise(250, seed=2008),
            gaussian_noise(200, mean=5.0, seed=2009),
            oscillation(200, 1.0, amplitude=6.0),
            gaussian_noise(250, seed=2010),
        )[0],
    ]
    for i in range(n_extra_noise):
        rows.append(gaussian_noise(T, seed=2100 + i))
    return np.stack(rows)


def _run_scalar(bank: np.ndarray, beta: float):
    k, T = bank.shape
    clfs = [RegimeClassifier(beta=beta, **CLF_KWARGS) for _ in range(k)]
    labels = np.empty((T, k), dtype=object)
    stats = {p: np.empty((T, k)) for p in STAT_PROPS}
    n_since = np.empty((T, k), dtype=np.int64)
    for t in range(T):
        for i, clf in enumerate(clfs):
            labels[t, i] = clf.update(bank[i, t])
            for p in STAT_PROPS:
                stats[p][t, i] = float(getattr(clf.stats, p))
            n_since[t, i] = clf.n_since_reset
    resets = [list(clf.reset_steps) for clf in clfs]
    return labels, stats, n_since, resets


def _run_batch(bank: np.ndarray, beta: float):
    k, T = bank.shape
    clf = BatchRegimeClassifier(beta=beta, k=k, **CLF_KWARGS)
    labels = np.empty((T, k), dtype=object)
    stats = {p: np.empty((T, k)) for p in STAT_PROPS}
    n_since = np.empty((T, k), dtype=np.int64)
    for t in range(T):
        labels[t, :] = clf.update(bank[:, t])
        for p in STAT_PROPS:
            stats[p][t, :] = getattr(clf.stats, p)
        n_since[t, :] = clf.n_since_reset
    return labels, stats, n_since, [list(r) for r in clf.reset_steps], clf


@pytest.mark.parametrize("beta", BETAS)
def test_batch_matches_scalar_everything(beta):
    bank = _streams()
    s_labels, s_stats, s_n, s_resets = _run_scalar(bank, beta)
    b_labels, b_stats, b_n, b_resets, _ = _run_batch(bank, beta)

    # Innovation resets actually occur in this scenario (both detectors),
    # otherwise the equivalence would not cover the replay path.
    assert any(s_resets), "scenario must trigger at least one innovation reset"

    # Identical reset step indices per direction.
    assert b_resets == s_resets

    # Identical label sequence at every step for every direction.
    mismatch = np.argwhere(s_labels != b_labels)
    assert mismatch.size == 0, f"label mismatch at (t, i) = {mismatch[:5]}"

    # Identical confidence clocks.
    np.testing.assert_array_equal(b_n, s_n)

    # Statistics agree to floating-point roundoff at every step.
    for p in STAT_PROPS:
        np.testing.assert_allclose(
            b_stats[p], s_stats[p], rtol=1e-9, atol=1e-12, err_msg=f"stat {p}"
        )


@pytest.mark.parametrize("beta", BETAS)
def test_batch_matches_scalar_at_k32(beta):
    """Full k=32 bank (airbench per-matrix tracked-direction count)."""
    bank = _streams(n_extra_noise=22)
    assert bank.shape[0] == 32
    s_labels, s_stats, _, s_resets = _run_scalar(bank, beta)
    b_labels, b_stats, _, b_resets, _ = _run_batch(bank, beta)
    assert b_resets == s_resets
    assert (s_labels == b_labels).all()
    for p in ("t_stat", "rho_corrected", "implied_eta_lambda"):
        np.testing.assert_allclose(b_stats[p], s_stats[p], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("beta", BETAS)
def test_reset_directions_matches_fresh_scalar_classifier(beta):
    """The tracker-style subspace-rotation reset (reset_directions) must be
    equivalent to rebuilding a scalar classifier from scratch mid-stream."""
    bank = _streams()
    k, T = bank.shape
    t_rot = 300
    rotated = [1, 4]  # directions whose tracked pair "rotated" at t_rot

    batch = BatchRegimeClassifier(beta=beta, k=k, **CLF_KWARGS)
    scalars = [RegimeClassifier(beta=beta, **CLF_KWARGS) for _ in range(k)]
    labels_b = np.empty((T, k), dtype=object)
    labels_s = np.empty((T, k), dtype=object)
    for t in range(T):
        if t == t_rot:
            batch.reset_directions(rotated)
            for i in rotated:
                scalars[i] = RegimeClassifier(beta=beta, **CLF_KWARGS)
        labels_b[t, :] = batch.update(bank[:, t])
        for i in range(k):
            labels_s[t, i] = scalars[i].update(bank[i, t])

    assert (labels_b == labels_s).all()
    for i in rotated:
        assert batch.n_since_reset[i] == scalars[i].n_since_reset
        np.testing.assert_allclose(
            float(batch.stats.t_stat[i]), float(scalars[i].stats.t_stat), rtol=1e-9
        )


@pytest.mark.parametrize("beta", BETAS)
def test_no_detectors_configured(beta):
    """Without z_reset/z_quiet the batch path must also match (no resets)."""
    kwargs = dict(tau_sig=4.0, tau_noise=2.0, rho_osc=0.5, n_min=50)
    bank = np.stack(
        [
            gaussian_noise(600, mean=5.0, seed=3001),
            ar1(600, -0.8, seed=3002),
            gaussian_noise(600, seed=3003),
        ]
    )
    k, T = bank.shape
    batch = BatchRegimeClassifier(beta=beta, k=k, **kwargs)
    scalars = [RegimeClassifier(beta=beta, **kwargs) for _ in range(k)]
    for t in range(T):
        got = batch.update(bank[:, t])
        want = [scalars[i].update(bank[i, t]) for i in range(k)]
        assert got == want, f"t={t}"
    assert all(r == [] for r in batch.reset_steps)
    # Sanity: the three planted regimes are actually recovered.
    assert batch.regimes == [Regime.SIGNAL, Regime.OSCILLATING, Regime.NOISE]


def test_batch_validates_thresholds():
    with pytest.raises(ValueError, match="tau_noise"):
        BatchRegimeClassifier(
            beta=0.9, k=4, tau_sig=1.0, tau_noise=2.0, rho_osc=0.5, n_min=10
        )
