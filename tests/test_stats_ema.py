"""WP0.5: bias-corrected EMA machinery -- small-t unbiasedness and ESS.

DoD line covered: "Both beta values (0.9, 0.99); bias-corrected estimates
tested at small t" -- asserting near-unbiasedness in early steps where the
uncorrected EMA is provably far off (raw EMA of a constant c equals
c * (1 - beta^t), i.e. 3% of c at t=3 for beta=0.99).

All random seeds are development seeds (>= 1000).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from src.stats import BiasCorrectedEma, DirectionStats, ema_effective_sample_size

BETAS = [0.9, 0.99]


# --------------------------------------------------------------------------
# Exactness on constant input (deterministic)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("t_small", [1, 2, 3, 5])
def test_constant_input_is_exact_at_small_t(beta, t_small):
    c = 3.0
    ema = BiasCorrectedEma(beta)
    for _ in range(t_small):
        ema.update(c)
    # Bias-corrected estimate is exact at every t >= 1.
    assert ema.value == pytest.approx(c, rel=1e-12)
    # The uncorrected EMA is off by exactly the bias factor (1 - beta^t):
    # at these small t it would badly fail (e.g. 0.0297*c at t=3, beta=0.99).
    assert ema.raw == pytest.approx(c * (1.0 - beta**t_small), rel=1e-12)
    assert abs(ema.raw - c) > 0.25 * c  # uncorrected estimate is far off


@pytest.mark.parametrize("beta", BETAS)
def test_constant_input_variance_is_zero(beta):
    st = DirectionStats(beta)
    for _ in range(5):
        st.update(2.5)
    assert float(st.mean) == pytest.approx(2.5, rel=1e-12)
    assert float(st.var) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Statistical unbiasedness at small t (vectorized over independent streams)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("t_small", [3, 10])
def test_small_t_mean_near_unbiased_iid(beta, t_small):
    rng = np.random.default_rng(1729)
    n_streams = 50_000
    true_mean, sigma = 2.0, 1.0
    ema = BiasCorrectedEma(beta)
    for _ in range(t_small):
        ema.update(true_mean + sigma * rng.standard_normal(n_streams))
    est = np.asarray(ema.value)
    # Corrected: mean over streams within Monte-Carlo error of the truth.
    mc_tol = 5.0 * sigma / np.sqrt(ema.ess * n_streams) + 1e-3
    assert abs(est.mean() - true_mean) < mc_tol
    # Uncorrected: biased by exactly the analytic factor -> clearly wrong.
    raw_expected = true_mean * (1.0 - beta**t_small)
    assert np.asarray(ema.raw).mean() == pytest.approx(raw_expected, abs=0.02)
    assert abs(raw_expected - true_mean) > 0.25 * true_mean


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("t", [5, 12, 50])
def test_effective_sample_size_matches_empirical_variance(beta, t):
    """Var over streams of the corrected mean estimator == sigma^2 / ess(t)."""
    rng = np.random.default_rng(1801)
    n_streams = 50_000
    ema = BiasCorrectedEma(beta)
    for _ in range(t):
        ema.update(rng.standard_normal(n_streams))
    est = np.asarray(ema.value)
    predicted = 1.0 / ema.ess
    assert est.var() == pytest.approx(predicted, rel=0.05)


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("t", [10, 30])
def test_small_t_variance_estimator_analytic_bias(beta, t):
    """For iid input, E[var_hat] = sigma^2 * (1 - 1/ess(t)) exactly."""
    rng = np.random.default_rng(1903)
    n_streams = 50_000
    st = DirectionStats(beta)
    for _ in range(t):
        st.update(rng.standard_normal(n_streams))
    predicted = 1.0 - 1.0 / st.ess
    assert np.asarray(st.var).mean() == pytest.approx(predicted, rel=0.03)


@pytest.mark.parametrize("beta", BETAS)
def test_small_t_autocorr_near_zero_iid(beta):
    rng = np.random.default_rng(2077)
    n_streams = 50_000
    st = DirectionStats(beta)
    for _ in range(12):
        st.update(rng.standard_normal(n_streams))
    assert abs(np.asarray(st.rho).mean()) < 0.15  # raw estimate, small window
    # Bias-corrected rho should be closer to 0 on average than raw.
    assert abs(np.asarray(st.rho_corrected).mean()) <= abs(np.asarray(st.rho).mean())


# --------------------------------------------------------------------------
# ESS formula sanity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("beta", BETAS)
def test_ess_formula(beta):
    assert ema_effective_sample_size(0, beta) == 0.0
    assert ema_effective_sample_size(1, beta) == pytest.approx(1.0)
    values = [ema_effective_sample_size(t, beta) for t in range(1, 2000)]
    assert all(b >= a for a, b in zip(values, values[1:]))  # monotone
    asymptote = (1.0 + beta) / (1.0 - beta)
    assert values[-1] == pytest.approx(asymptote, rel=1e-3)
    assert values[-1] <= asymptote + 1e-9


@pytest.mark.parametrize("beta", BETAS)
def test_ess_below_observation_count(beta):
    # EMA weights are unequal, so ess(t) < t for t >= 2.
    for t in [2, 5, 20, 100]:
        assert ema_effective_sample_size(t, beta) < t
