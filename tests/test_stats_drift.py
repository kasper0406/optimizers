"""WP0.5: drifting mean + noise -- t-statistic vs analytic expectation.

DoD line covered: "Drifting mean + noise at SNR in {0.1, 1, 10} ->
t-statistic crosses signal threshold iff SNR and window predict it
(analytic expectation in test)."

Analytic model (computed here, not eyeballed):
  the stream is s(t) = m(t) + sigma * eps_t with
  m(t) = snr * sigma * (1 + A_d * sin(2 pi t / period)), so the
  instantaneous SNR lies in [snr (1 - A_d), snr (1 + A_d)].  The EMA
  t-statistic has expectation E[t] ~= SNR_inst * sqrt(ESS_inf) with
  ESS_inf = (1 + beta) / (1 - beta), and fluctuates around it with
  ~unit standard deviation.  Hence:

    * sustained crossing predicted  iff  snr (1 - A_d) sqrt(ESS_inf) > 2 tau
      (expectation at least 2 threshold-widths above tau -> |t| > tau for
      >= 90% of post-burn-in steps);
    * sustained crossing excluded   iff  snr (1 + A_d) sqrt(ESS_inf) < tau
      (median of t below tau -> |t| > tau for < 50% of steps).

Each (beta, tau, snr) scenario is required by the test itself to fall into
exactly one of the two branches, so the crossing outcome is an "iff".

All seeds are development seeds (>= 1000).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from src.stats import DirectionStats, drifting_mean

N, BURN = 8000, 2000
DRIFT_AMPLITUDE = 0.2
PERIOD = 4000
SEED = 2024

# tau is a threshold parameter (chosen per beta so that the three SNR values
# probe both sides of the analytic crossing boundary for that window size).
SCENARIOS = [(0.99, 4.0), (0.9, 1.5)]
SNRS = [0.1, 1.0, 10.0]


def _t_stat_trajectory(beta, snr):
    stream = drifting_mean(
        N, snr, drift_amplitude=DRIFT_AMPLITUDE, period=PERIOD, seed=SEED
    )
    st = DirectionStats(beta)
    ts = np.empty(N)
    for t, s in enumerate(stream):
        st.update(s)
        ts[t] = float(st.t_stat)
    return ts


@pytest.mark.parametrize("beta,tau", SCENARIOS)
@pytest.mark.parametrize("snr", SNRS)
def test_t_stat_crosses_iff_analytically_predicted(beta, tau, snr):
    ess_inf = (1.0 + beta) / (1.0 - beta)
    e_t_min = snr * (1.0 - DRIFT_AMPLITUDE) * np.sqrt(ess_inf)
    e_t_max = snr * (1.0 + DRIFT_AMPLITUDE) * np.sqrt(ess_inf)

    predicted_sustained = e_t_min > 2.0 * tau
    predicted_absent = e_t_max < tau
    # The scenario must be analytically unambiguous, otherwise the "iff" is
    # not being tested (guards the test design, not the implementation).
    assert predicted_sustained != predicted_absent

    ts = np.abs(_t_stat_trajectory(beta, snr)[BURN:])
    frac_above = float(np.mean(ts > tau))
    if predicted_sustained:
        assert frac_above > 0.9
    else:
        assert frac_above < 0.5


@pytest.mark.parametrize("beta,tau", SCENARIOS)
@pytest.mark.parametrize("snr", SNRS)
def test_t_stat_magnitude_matches_analytic_expectation(beta, tau, snr):
    """Post-burn-in mean of t within [0.6, 1.4]x the analytic expectation
    E[t] = mean_t SNR(t) * sqrt(ESS_inf) computed from the generator
    parameters (the exact drift profile, averaged over the same window)."""
    ess_inf = (1.0 + beta) / (1.0 - beta)
    t_idx = np.arange(BURN, N)
    snr_t = snr * (1.0 + DRIFT_AMPLITUDE * np.sin(2.0 * np.pi * t_idx / PERIOD))
    analytic_mean_t = float(np.mean(snr_t)) * np.sqrt(ess_inf)

    measured = float(np.mean(_t_stat_trajectory(beta, snr)[BURN:]))
    assert 0.6 * analytic_mean_t <= measured <= 1.4 * analytic_mean_t
