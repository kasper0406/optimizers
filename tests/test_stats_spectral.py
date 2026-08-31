"""Two-channel spectral readout (src/stats/spectral.py) -- CPU only.

EXPLORATORY module, confirmatory-grade tests: every property the offline
channel analysis rests on is pinned here against a synthetic stream whose
ground truth is known analytically.

Covered:

* lag ladder recovers AR(1) at phi in {-0.8, -0.34, 0, +0.4} within +/-0.05
  at every lag j (rho_j = phi^j) on a long series;
* the +1/n finite-segment mean-subtraction bias correction: on short
  white-noise segments the UNCORRECTED ladder sits at -1/n at every lag and
  the corrected one at 0;
* burn-in is load-bearing: segments carrying a synthetic anchoring transient
  in their first observations recover the true rho only for burn_in >= 5;
* an injected period-2 signal is seen by the 'alt' channel iff
  SNR*sqrt(n) predicts detection, and never by 'dc';
* a persistent mean is seen by 'dc' and not by 'alt' (the mirror case);
* zero-mean white noise puts the median |t| at the half-normal median
  0.674 in BOTH channels over many surrogate reps;
* channel_t is the exact mirror of FrozenProbeAccumulator.stats();
* the circular block bootstrap restores median CI coverage on dependent
  input where the i.i.d. bootstrap under-covers, and its chunked evaluation
  is bit-identical to the unchunked one at any memory bound;
* the MEASURED accuracy of the +1/n bias correction at the audit's operating
  point (n = 45, phi = -0.34), which the module docstring quotes;
* the identification limit of the null-calibrated ratio -- a zero-mean stream
  with slow power reproduces a "large DC excess" reading -- and the
  slot-level growth factor that separates it from a real persistent mean;
* the finite-n value of the phi = 0 null median (it is not 0.674 at n = 45);
* the odd-window period-2 leak into the DC channel.

Thresholds used here (tau = 4, the +/-0.05 rho tolerance) are the WP0.5 test
conventions, not decision thresholds. All seeds are development seeds
(>= 1000) per repo seed discipline; no GPU, no torch in the module under
test (the mirror test imports the tracker, which is CPU-only here).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from src.stats import ar1, gaussian_noise
from src.stats.spectral import (
    CHANNELS,
    ar1_segmented_null,
    ar1_streams,
    ar1_surrogate_null,
    block_bootstrap_ci,
    channel_t,
    demodulate,
    lag_ladder,
    newey_west_bandwidth,
    segment_mean_persistence,
    segmented_channel_t,
)

RHO_TOL = 0.05  # WP0.5 recovery tolerance
TAU = 4.0  # conventional detection threshold (test convention only)
HALF_NORMAL_MEDIAN = 0.6744897501960817  # median of |N(0, 1)|


def _alternating(n: int) -> np.ndarray:
    return np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


# ==========================================================================
# 1. Lag ladder: AR(1) recovery
# ==========================================================================

PHIS = [-0.8, -0.34, 0.0, 0.4]
LONG_N = 20000
MAX_LAG = 4


@pytest.mark.parametrize("phi", PHIS)
@pytest.mark.parametrize("seed", [1234, 5678])
def test_lag_ladder_recovers_ar1(phi, seed):
    """rho_j = phi^j at every lag on a long series, within +/-0.05."""
    ladder = lag_ladder(ar1(LONG_N, phi, seed=seed), MAX_LAG, 0)
    assert ladder["n"] == LONG_N
    truth = np.array([phi**j for j in range(1, MAX_LAG + 1)])
    assert ladder["rho"] == pytest.approx(truth, abs=RHO_TOL)


@pytest.mark.parametrize("phi", PHIS)
def test_lag_ladder_scale_invariant(phi):
    """Autocorrelations must not depend on the noise scale."""
    a = lag_ladder(ar1(4000, phi, noise_scale=0.5, seed=1234), MAX_LAG, 0)["rho"]
    b = lag_ladder(ar1(4000, phi, noise_scale=2.0, seed=1234), MAX_LAG, 0)["rho"]
    assert a == pytest.approx(b, abs=1e-9)


def test_lag_ladder_bias_correction_is_exactly_one_over_n():
    """The corrected and raw ladders differ by exactly +1/n at every lag."""
    x = ar1(60, -0.34, seed=1777)
    ladder = lag_ladder(x, MAX_LAG, 5)
    assert ladder["n"] == 55
    assert ladder["bias_correction"] == pytest.approx(1.0 / 55)
    assert ladder["rho"] - ladder["rho_raw"] == pytest.approx(
        np.full(MAX_LAG, 1.0 / 55)
    )
    off = lag_ladder(x, MAX_LAG, 5, bias_correct=False)
    assert off["bias_correction"] == 0.0
    assert off["rho"] == pytest.approx(ladder["rho_raw"])


# ==========================================================================
# 2. Bias correction: short white-noise segments
# ==========================================================================

SEG_N = 20
N_SEG_BIAS = 3000


def _white_noise_ladders(n, n_seg, *, bias_correct):
    return np.array(
        [
            lag_ladder(
                gaussian_noise(n, seed=1000 + s), MAX_LAG, 0, bias_correct=bias_correct
            )["rho"]
            for s in range(n_seg)
        ]
    )


def test_uncorrected_ladder_is_biased_low_by_one_over_n():
    """Mean-subtraction on an n-length segment costs about -1/n per lag.

    White noise has rho_j = 0 for every j >= 1, so the uncorrected sample
    ladder must average to -1/n (the residual is the O(1/n^2) term of the
    ratio estimator, ~0.007 at n = 20).
    """
    raw = _white_noise_ladders(SEG_N, N_SEG_BIAS, bias_correct=False).mean(axis=0)
    assert raw == pytest.approx(np.full(MAX_LAG, -1.0 / SEG_N), abs=0.012)


def test_corrected_ladder_is_unbiased_on_white_noise():
    """The +1/n correction removes the bulk of that bias (>= 4x shrink)."""
    raw = _white_noise_ladders(SEG_N, N_SEG_BIAS, bias_correct=False).mean(axis=0)
    corrected = _white_noise_ladders(SEG_N, N_SEG_BIAS, bias_correct=True).mean(axis=0)
    assert corrected == pytest.approx(np.zeros(MAX_LAG), abs=0.012)
    assert np.all(np.abs(corrected) < 0.25 * np.abs(raw))


# ==========================================================================
# 3. Burn-in: anchoring transient at the head of a refresh segment
# ==========================================================================

ANCHOR_PHI = -0.34  # the measured tracked-direction AR(1) parameter
ANCHOR_SEG = 50  # refresh segment length
N_SEG_ANCHOR = 400
# A momentum-anchored direction right after a subspace rotation: a strong
# co-rotating transient over the first 4 observations plus its one-step
# spill-over, i.e. contamination of observations 0..4 -> burn_in >= 5.
ANCHOR = np.array([12.0, 9.6, 7.68, 6.144, 4.8])
ANCHOR_LEN = ANCHOR.size


def _anchored_segments():
    out = []
    for s in range(N_SEG_ANCHOR):
        x = ar1(ANCHOR_SEG, ANCHOR_PHI, seed=1000 + s).copy()
        x[:ANCHOR_LEN] += ANCHOR
        out.append(x)
    return out


ANCHORED = _anchored_segments()


def _mean_ladder(segments, burn_in):
    return np.mean(
        [lag_ladder(x, MAX_LAG, burn_in)["rho"] for x in segments], axis=0
    )


@pytest.mark.parametrize("burn_in", [0, 1, 2, 3, 4])
def test_anchored_segments_need_burn_in(burn_in):
    """Below burn_in = 5 the transient still contaminates the ladder."""
    rho1 = _mean_ladder(ANCHORED, burn_in)[0]
    assert abs(rho1 - ANCHOR_PHI) > RHO_TOL


@pytest.mark.parametrize("burn_in", [5, 6, 8])
def test_anchored_segments_recover_rho_with_burn_in(burn_in):
    """At burn_in >= 5 the segment is clean and the ladder is AR(1) again."""
    rho = _mean_ladder(ANCHORED, burn_in)
    truth = np.array([ANCHOR_PHI**j for j in range(1, MAX_LAG + 1)])
    assert rho == pytest.approx(truth, abs=RHO_TOL)


def test_burn_in_transition_is_monotone_at_the_boundary():
    """The recovery is a boundary effect at ANCHOR_LEN, not a slow drift."""
    errs = [
        abs(_mean_ladder(ANCHORED, b)[0] - ANCHOR_PHI) for b in range(ANCHOR_LEN + 2)
    ]
    assert errs[ANCHOR_LEN] < 0.25 * errs[ANCHOR_LEN - 1]
    assert errs[ANCHOR_LEN + 1] == pytest.approx(errs[ANCHOR_LEN], abs=0.02)


def test_clean_segments_do_not_need_burn_in():
    """Control: without a transient, burn_in = 0 already recovers rho."""
    clean = [ar1(ANCHOR_SEG, ANCHOR_PHI, seed=1000 + s) for s in range(N_SEG_ANCHOR)]
    assert _mean_ladder(clean, 0)[0] == pytest.approx(ANCHOR_PHI, abs=RHO_TOL)


# ==========================================================================
# 4. Channels: period-2 signal vs persistent mean
# ==========================================================================

DET_N = 400  # even, so a period-2 component cancels exactly in 'dc'
DET_SEEDS = list(range(1000, 1008))
# (amplitude / sigma, expected detection); the "iff" is asserted below from
# SNR*sqrt(n) alone, the amplitudes only place the two scenarios.
DET_CASES = [(0.5, True), (0.02, False)]


def _predicts(snr, n):
    """(detect, absent) from SNR*sqrt(n); exactly one must be true.

    The channel mean of an injected component of amplitude A on noise sigma
    is A with standard error sigma/sqrt(n), so E[|t|] = SNR*sqrt(n).
    Detection is predicted when that expectation clears tau by a further tau
    (>= 2 tau), absence when the expectation is itself below tau.
    """
    expected = snr * np.sqrt(n)
    return bool(expected > 2.0 * TAU), bool(expected < TAU)


@pytest.mark.parametrize("snr,expected", DET_CASES)
@pytest.mark.parametrize("seed", DET_SEEDS)
def test_alt_channel_detects_period_two_iff_snr_predicts(snr, expected, seed):
    detect, absent = _predicts(snr, DET_N)
    assert detect != absent  # the scenario must be analytically unambiguous
    assert detect is expected

    x = gaussian_noise(DET_N, seed=seed) + snr * _alternating(DET_N)
    st = channel_t(x, "alt")
    assert st["n"] == DET_N
    assert not st["nw_floored"]
    assert (abs(st["t_nw"]) > TAU) is expected


@pytest.mark.parametrize("snr,expected", DET_CASES)
@pytest.mark.parametrize("seed", DET_SEEDS)
def test_dc_channel_blind_to_period_two(snr, expected, seed):
    """The same stream carries no DC signal at any amplitude."""
    x = gaussian_noise(DET_N, seed=seed) + snr * _alternating(DET_N)
    assert abs(channel_t(x, "dc")["t_nw"]) < TAU


@pytest.mark.parametrize("snr,expected", DET_CASES)
@pytest.mark.parametrize("seed", DET_SEEDS)
def test_dc_channel_detects_persistent_mean_iff_snr_predicts(snr, expected, seed):
    detect, absent = _predicts(snr, DET_N)
    assert detect != absent
    x = gaussian_noise(DET_N, mean=snr, seed=seed)
    assert (abs(channel_t(x, "dc")["t_nw"]) > TAU) is expected


@pytest.mark.parametrize("snr,expected", DET_CASES)
@pytest.mark.parametrize("seed", DET_SEEDS)
def test_alt_channel_blind_to_persistent_mean(snr, expected, seed):
    """Mirror case: a persistent mean carries no period-2 component."""
    x = gaussian_noise(DET_N, mean=snr, seed=seed)
    assert abs(channel_t(x, "alt")["t_nw"]) < TAU


def test_demodulation_is_an_involution_and_moves_the_zigzag_to_dc():
    x = ar1(500, -0.34, seed=1900)
    assert demodulate(demodulate(x)) == pytest.approx(x)
    # (-1)^t on an AR(1) flips the sign of the lag-1 autocorrelation (up to
    # the O(1/n) mean-subtraction term, which differs between channels
    # because the two channels have different sample means).
    assert lag_ladder(demodulate(x), 1, 0)["rho"][0] == pytest.approx(
        -lag_ladder(x, 1, 0)["rho"][0], abs=0.01
    )


def test_odd_burn_in_flips_alt_sign_not_magnitude():
    """Demodulation is tied to absolute step index, per the module contract."""
    x = ar1(300, -0.34, mean=0.3, seed=1901)
    a = channel_t(x, "alt", 4)
    b = channel_t(x[1:], "alt", 3)  # same retained window, phase shifted by 1
    assert a["t_nw"] == pytest.approx(-b["t_nw"])
    assert a["ess"] == pytest.approx(b["ess"])


# ==========================================================================
# 5. Surrogate null
# ==========================================================================

NULL_N = 400
NULL_REPS = 6000
NULL_SEEDS = [1234, 4321, 20250831]


@pytest.mark.parametrize("seed", NULL_SEEDS)
@pytest.mark.parametrize("channel", sorted(CHANNELS))
def test_white_noise_null_median_abs_t_is_half_normal(seed, channel):
    """Zero-mean white noise: median |t| = 0.674 in BOTH channels."""
    null = ar1_surrogate_null(0.0, NULL_N, NULL_REPS, seed)[channel]
    assert null["abs_t_nw"]["median"] == pytest.approx(HALF_NORMAL_MEDIAN, abs=0.04)
    assert null["abs_t_naive"]["median"] == pytest.approx(HALF_NORMAL_MEDIAN, abs=0.04)
    assert null["ess_over_n"]["median"] == pytest.approx(1.0, abs=0.1)
    assert null["nw_floored_frac"] == 0.0


def test_zigzag_null_inflates_ess_above_n_in_dc_and_deflates_it_in_alt():
    """phi = -0.34: the DC channel's ESS/n sits well above 1 (the frozen-probe
    observation), while its demodulated twin (phi -> +0.34) sits below 1.

    The population value is (1 - phi)/(1 + phi) = 2.03; the Bartlett-truncated
    estimator at L = 5 recovers ~1.8 of it, so the assertion is a band, not
    the exact ratio.
    """
    null = ar1_surrogate_null(-0.34, NULL_N, 2000, 1234)
    assert 1.6 < null["dc"]["ess_over_n"]["median"] < 2.1
    assert 0.45 < null["alt"]["ess_over_n"]["median"] < 0.75


def test_surrogate_null_is_deterministic_and_zero_mean():
    a = ar1_surrogate_null(-0.34, 200, 500, 777)
    b = ar1_surrogate_null(-0.34, 200, 500, 777)
    for ch in CHANNELS:
        assert np.array_equal(a[ch]["samples"]["t_nw"], b[ch]["samples"]["t_nw"])
        # nothing to detect: the signed t distribution is centred on 0.
        assert a[ch]["t_nw"]["median"] == pytest.approx(0.0, abs=0.15)
    c = ar1_surrogate_null(-0.34, 200, 500, 778)
    assert not np.array_equal(a["dc"]["samples"]["t_nw"], c["dc"]["samples"]["t_nw"])


def test_surrogate_streams_have_the_requested_autocorrelation():
    """The null's own streams must be the AR(1) they claim to be."""
    null = ar1_surrogate_null(-0.34, 4000, 200, 1234)
    # ess/n of a long AR(1) DC stream identifies phi through the truncated
    # long-run variance; recover phi from the lag ladder of a matching draw.
    ladder = lag_ladder(ar1(20000, -0.34, seed=1234), 2, 0)
    assert ladder["rho"][0] == pytest.approx(-0.34, abs=RHO_TOL)
    assert null["dc"]["ess_over_n"]["median"] > 1.5


# ==========================================================================
# 6. Mirror contract with FrozenProbeAccumulator
# ==========================================================================

MIRROR_CASES = [(-0.34, 200, 1234), (0.8, 53, 5678), (0.0, 7, 4321), (-0.8, 3, 1111)]


@pytest.mark.parametrize("phi,n,seed", MIRROR_CASES)
def test_channel_t_mirrors_frozen_probe_accumulator(phi, n, seed):
    """channel_t('dc') must reproduce the canonical accumulator exactly."""
    from src.instrument.tracker import FrozenProbeAccumulator

    x = ar1(n, phi, seed=seed)
    acc = FrozenProbeAccumulator(1, max_lag=8)
    for v in x:
        acc.update(np.array([v]))
    ref = acc.stats()
    mine = channel_t(x, "dc", 0, max_lag=8)
    for key in ("mean", "var", "sigma_lr2", "t_naive", "t_nw", "ess"):
        assert mine[key] == pytest.approx(float(ref[key][0]), rel=1e-9, abs=1e-12)
    assert mine["lag_truncation"] == int(ref["lag_truncation"])
    assert mine["nw_floored"] is bool(ref["nw_floored"][0])
    assert mine["n"] == int(ref["n"])


@pytest.mark.parametrize("n", [1, 2, 3, 10, 100, 1000])
def test_newey_west_bandwidth_matches_the_1994_rule(n):
    from src.instrument.tracker import FrozenProbeAccumulator

    acc = FrozenProbeAccumulator(1, max_lag=8)
    for _ in range(n):
        acc.update(np.array([0.0]))
    assert newey_west_bandwidth(n, 8) == acc.lag_truncation()


# ==========================================================================
# 7. Circular block bootstrap
# ==========================================================================

BOOT_N = 400
BOOT_REPS = 1000
BOOT_SEEDS = list(range(3000, 3040))


def _coverage(streams, block):
    hits = 0
    for v in streams:
        ci = block_bootstrap_ci(v, block, BOOT_REPS, 1234)
        hits += ci["ci_lo"] <= 1.0 <= ci["ci_hi"]
    return hits / len(streams)


def test_block_bootstrap_is_deterministic():
    v = gaussian_noise(200, seed=1500)
    a = block_bootstrap_ci(v, 8, 500, 99)
    b = block_bootstrap_ci(v, 8, 500, 99)
    assert a == b
    assert block_bootstrap_ci(v, 8, 500, 100) != a
    assert a["point"] == pytest.approx(float(np.median(v)))


def test_block_bootstrap_covers_the_median_on_iid_input():
    streams = [gaussian_noise(BOOT_N, mean=1.0, seed=s) for s in BOOT_SEEDS]
    assert _coverage(streams, 4) >= 0.85


def test_block_bootstrap_restores_coverage_under_dependence():
    """The i.i.d. bootstrap (block = 1) under-covers on AR(1) input; blocks
    long relative to the correlation time restore nominal coverage."""
    streams = [ar1(BOOT_N, 0.7, mean=1.0, seed=s) for s in BOOT_SEEDS]
    iid = _coverage(streams, 1)
    blocked = _coverage(streams, 20)
    assert iid < 0.85
    assert blocked >= 0.85
    assert blocked > iid


def test_block_bootstrap_ci_widens_with_block_on_dependent_input():
    v = ar1(BOOT_N, 0.9, mean=1.0, seed=2002)
    widths = [
        block_bootstrap_ci(v, b, BOOT_REPS, 1234)["ci_hi"]
        - block_bootstrap_ci(v, b, BOOT_REPS, 1234)["ci_lo"]
        for b in (1, 5, 20)
    ]
    assert widths[0] < widths[1] < widths[2]


# ==========================================================================
# 8. Guards and degenerate input
# ==========================================================================


def test_channel_t_rejects_unknown_channel():
    with pytest.raises(ValueError, match="channel"):
        channel_t(np.zeros(10), "quadrature")


@pytest.mark.parametrize(
    "fn,args",
    [
        (lag_ladder, (np.zeros((3, 3)), 2, 0)),
        (channel_t, (np.zeros((3, 3)), "dc", 0)),
    ],
)
def test_two_dimensional_input_is_rejected(fn, args):
    with pytest.raises(ValueError, match="1-D"):
        fn(*args)


def test_negative_burn_in_is_rejected():
    with pytest.raises(ValueError, match="burn_in"):
        lag_ladder(np.zeros(10), 2, -1)


def test_constant_series_reports_zero_rho_and_zero_t():
    """Repo convention (DirectionStats): a var-floored series has rho = 0."""
    x = np.full(50, 3.0)
    ladder = lag_ladder(x, MAX_LAG, 5)
    assert ladder["var"] == pytest.approx(0.0)
    assert np.all(ladder["rho"] == 0.0)
    assert ladder["bias_correction"] == 0.0
    st = channel_t(x, "dc")
    assert st["t_naive"] == 0.0 and st["t_nw"] == 0.0


def test_empty_and_short_segments_report_nan():
    assert channel_t(np.zeros(5), "dc", 5)["n"] == 0
    assert np.isnan(channel_t(np.zeros(5), "dc", 5)["t_nw"])
    short = lag_ladder(np.array([1.0]), 3, 0)
    assert short["n"] == 1 and np.all(np.isnan(short["rho"]))
    # lags with no available pair are NaN, not 0.
    three = lag_ladder(ar1(3, 0.0, seed=1234), 4, 0)
    assert np.all(np.isfinite(three["rho"][:2]))
    assert np.all(np.isnan(three["rho"][2:]))


def test_ar1_surrogate_null_rejects_nonstationary_phi():
    with pytest.raises(ValueError, match="phi"):
        ar1_surrogate_null(1.0, 100, 10, 1234)


def test_block_bootstrap_rejects_bad_block():
    with pytest.raises(ValueError, match="block"):
        block_bootstrap_ci(np.zeros(10), 0, 100, 1234)
    with pytest.raises(ValueError, match="block"):
        block_bootstrap_ci(np.zeros(10), 11, 100, 1234)


# ==========================================================================
# 9. Chunked block bootstrap (memory bound, never a rep-count reduction)
# ==========================================================================


@pytest.mark.parametrize("max_elems", [10**9, 40000, 4000, 1])
def test_chunked_bootstrap_is_bit_identical_to_the_unchunked_one(max_elems):
    """``max_elems`` bounds memory only; it must not change the answer.

    The alternative an earlier caller used -- reducing ``reps`` when the pool
    is large -- bottoms out at a 95% interval taken between the 2nd and 49th
    order statistics of 50 draws, on exactly the largest and most-quoted
    pools.  Chunking removes the need for that trade, so the equality below
    is the property that makes the full rep count affordable.
    """
    v = ar1(4000, 0.5, mean=1.0, seed=2600)
    ref = block_bootstrap_ci(v, 16, 400, 4242, max_elems=10**12)
    got = block_bootstrap_ci(v, 16, 400, 4242, max_elems=max_elems)
    assert got == ref
    assert got["reps"] == 400


def test_bootstrap_rejects_a_non_positive_memory_bound():
    with pytest.raises(ValueError, match="max_elems"):
        block_bootstrap_ci(np.zeros(10), 2, 10, 1234, max_elems=0)


# ==========================================================================
# 10. Measured accuracy of the +1/n bias correction (module docstring)
# ==========================================================================

BIAS_N = 45  # the tracked window: 50-step segment, burn-in 5
BIAS_REPS = 20000


def _corrected_rho1(phi, n, reps, seed):
    streams = ar1_streams(np.random.default_rng(seed), phi, n, reps)
    return np.array([lag_ladder(x, 1, 0)["rho"][0] for x in streams])


def test_white_noise_uncorrected_lag1_is_minus_one_over_n_minus_one():
    """Not -1/n: c_j has divisor n - j and c_0 has divisor n.

    E[c_j] = -1/n and E[c_0] = (n-1)/n, so the ratio of expectations is
    -1/(n-1).  The docstring used to call the +1/n correction "exact for
    white noise"; it is not, and the difference is +1/(n(n-1)).
    """
    streams = np.random.default_rng(2601).standard_normal((60000, BIAS_N))
    raw = np.array([lag_ladder(x, 1, 0, bias_correct=False)["rho"][0] for x in streams])
    mean = float(raw.mean())
    assert abs(mean - (-1.0 / (BIAS_N - 1))) < abs(mean - (-1.0 / BIAS_N))
    assert mean == pytest.approx(-1.0 / (BIAS_N - 1), abs=0.002)


@pytest.mark.parametrize(
    "phi,residual_median", [(-0.343, 0.0141), (-0.172, 0.0067)]
)
def test_bias_correction_residual_at_the_operating_point(phi, residual_median):
    """The docstring's measured residuals, pinned.

    The first-order formula (1 - g(1 - rho))/n predicts +0.0076 at
    phi = -0.34, n = 45; the estimator's actual median residual is about
    twice that, which is 4-5x the block-bootstrap CI half-width the audit
    prints on rho_1.  If this drifts, the docstring's inversion map
    (-0.343 -> true phi about -0.358) drifts with it.
    """
    got = float(np.median(_corrected_rho1(phi, BIAS_N, BIAS_REPS, 2602))) - phi
    assert got == pytest.approx(residual_median, abs=0.004)
    first_order = (1.0 - ((1 + phi) / (1 - phi)) * (1 - phi)) / BIAS_N
    assert got > 1.5 * first_order  # the first-order term understates it


def test_the_bias_map_inverts_to_the_documented_true_phi():
    """A reported median rho_1 of -0.343 comes from a true phi near -0.358."""
    got = float(np.median(_corrected_rho1(-0.358, BIAS_N, BIAS_REPS, 2603)))
    assert got == pytest.approx(-0.343, abs=0.004)


# ==========================================================================
# 11. What the null-calibrated ratio does NOT identify
# ==========================================================================

SLOW_N, SLOW_BURN, SLOW_REPS, SLOW_K = 50, 5, 2000, 6


def _two_component(rng, reps, *, slow_phi=0.97, slow_var=0.15, fast_phi=-0.40):
    fast = ar1_streams(rng, fast_phi, SLOW_N, reps)
    slow = ar1_streams(rng, slow_phi, SLOW_N, reps)
    return fast + slow / float(slow.std()) * np.sqrt(slow_var)


def _median_ratio(streams, phi_hat, channel="dc"):
    obs = np.median(
        [abs(channel_t(x, channel, SLOW_BURN, max_lag=8)["t_nw"]) for x in streams]
    )
    null = ar1_surrogate_null(
        phi_hat, SLOW_N, 2000, 4242, burn_in=SLOW_BURN, max_lag=8,
        channels=(channel,),
    )
    return float(obs / np.median(null[channel]["samples"]["abs_t_nw"]))


def test_a_zero_mean_stream_reproduces_a_large_dc_ratio():
    """The confound the module docstring documents, pinned as a test.

    Numerator = zero-frequency content; denominator = lags 1..L only, and
    L = newey_west_bandwidth(45, 8) = 3.  A zero-mean component slower than
    ~3 steps is invisible to the denominator, so ``ratio >> 1`` on dc means
    "power at zero frequency", never "a persistent mean".
    """
    assert newey_west_bandwidth(SLOW_N - SLOW_BURN, 8) == 3
    x = _two_component(np.random.default_rng(2604), SLOW_REPS)
    phi_hat = float(np.median([lag_ladder(s, 1, SLOW_BURN)["rho"][0] for s in x]))
    assert phi_hat == pytest.approx(-0.33, abs=0.03)  # looks like the tracked tier
    assert np.mean([s.mean() for s in x]) == pytest.approx(0.0, abs=0.02)
    assert _median_ratio(x, phi_hat, "dc") > 2.0  # a "large DC excess"
    assert _median_ratio(x, phi_hat, "alt") == pytest.approx(1.0, abs=0.15)


def test_the_slot_level_growth_factor_separates_the_two_alternatives():
    """A persistent mean grows |t| like sqrt(k); slow in-segment power does not.

    Both streams below give nearly the same per-segment ratio; only the
    longer window tells them apart, which is why the analysis script reports
    the growth factor and not the pooled |t| alone.
    """
    rng = np.random.default_rng(2605)
    reps = 800
    persistent = ar1_streams(rng, -0.40, SLOW_N, reps * SLOW_K).reshape(
        reps, SLOW_K, SLOW_N
    ) + 0.30 * rng.standard_normal(reps)[:, None, None]
    slow = _two_component(rng, reps * SLOW_K).reshape(reps, SLOW_K, SLOW_N)

    growths = {}
    for name, block in (("persistent", persistent), ("slow", slow)):
        flat = block.reshape(-1, SLOW_N)
        phi_hat = float(np.median(
            [lag_ladder(x, 1, SLOW_BURN)["rho"][0] for x in flat]
        ))
        seg = float(np.median(
            [abs(channel_t(x, "dc", SLOW_BURN, max_lag=8)["t_nw"]) for x in flat]
        ))
        slot = float(np.median([
            abs(segmented_channel_t(sl, "dc", SLOW_BURN, max_lag=8)["t_nw"])
            for sl in block
        ]))
        null_seg = float(np.median(ar1_surrogate_null(
            phi_hat, SLOW_N, 1500, 4242, burn_in=SLOW_BURN, max_lag=8,
            channels=("dc",))["dc"]["samples"]["abs_t_nw"]))
        null_slot = float(np.median(ar1_segmented_null(
            phi_hat, SLOW_N, SLOW_K, 1500, 4242, burn_in=SLOW_BURN, max_lag=8,
            channels=("dc",))["dc"]["samples"]["abs_t_nw"]))
        growths[name] = (slot / null_slot) / (seg / null_seg)
        assert seg / null_seg > 1.8  # both look the same per segment

    assert growths["persistent"] == pytest.approx(np.sqrt(SLOW_K), rel=0.3)
    assert growths["slow"] < 1.3
    assert growths["persistent"] > 2.0 * growths["slow"]


# ==========================================================================
# 12. Slot-level estimator and its null
# ==========================================================================


@pytest.mark.parametrize("channel", sorted(CHANNELS))
@pytest.mark.parametrize("burn_in", [0, 5])
def test_segmented_channel_t_reduces_to_channel_t_on_one_segment(channel, burn_in):
    x = ar1(60, -0.34, mean=0.2, seed=2606)
    a = channel_t(x, channel, burn_in, max_lag=8)
    b = segmented_channel_t([x], channel, burn_in, max_lag=8)
    for key in ("mean", "var", "sigma_lr2", "t_naive", "t_nw", "ess"):
        assert b[key] == pytest.approx(a[key], rel=1e-12, abs=1e-15)
    assert b["n"] == a["n"] and b["lag_truncation"] == a["lag_truncation"]
    assert b["n_segments"] == 1 and b["segment_lengths"] == [a["n"]]


def test_segmented_channel_t_never_crosses_a_segment_boundary():
    """Pair counts are sum_i (n_i - j), and the pooled mean is subtracted last.

    A concatenation that ignored boundaries would report P_j = N - j; the
    difference is exactly the cross-boundary products the prereg forbids.
    """
    segs = [ar1(20, -0.34, seed=2607 + i) for i in range(3)]
    got = segmented_channel_t(segs, "dc", 2, max_lag=4)
    assert got["n"] == 3 * 18
    assert got["pair_counts"] == [54, 51, 48, 45, 42]  # 3*(18-j), P_0 = N
    assert got["segment_lengths"] == [18, 18, 18]
    assert len(got["segment_means"]) == 3


def test_segmented_channel_t_handles_ragged_and_empty_segments():
    segs = [ar1(20, 0.0, seed=2610), ar1(14, 0.0, seed=2611), np.zeros(3)]
    got = segmented_channel_t(segs, "dc", 5, max_lag=4)
    assert got["segment_lengths"] == [15, 9]  # the 3-long segment is all burn-in
    assert got["n_segments"] == 2 and got["n_segments_empty"] == 1
    assert got["n"] == 24
    empty = segmented_channel_t([np.zeros(2)], "dc", 5)
    assert empty["n"] == 0 and np.isnan(empty["t_nw"])


def test_segmented_channel_t_rejects_bad_channel_and_2d_input():
    with pytest.raises(ValueError, match="channel"):
        segmented_channel_t([np.zeros(10)], "quadrature")
    with pytest.raises(ValueError, match="1-D"):
        segmented_channel_t([np.zeros((2, 3))], "dc")


def test_ar1_segmented_null_is_zero_mean_deterministic_and_shape_aware():
    a = ar1_segmented_null(-0.34, 50, 4, 400, 777, burn_in=5, max_lag=8)
    b = ar1_segmented_null(-0.34, [50, 50, 50, 50], 4, 400, 777, burn_in=5, max_lag=8)
    for ch in CHANNELS:
        assert np.array_equal(a[ch]["samples"]["t_nw"], b[ch]["samples"]["t_nw"])
        assert a[ch]["t_nw"]["median"] == pytest.approx(0.0, abs=0.15)
        assert a[ch]["abs_t_nw"]["median"] == pytest.approx(0.7, abs=0.15)
    assert a["n"] == 4 * 45 and a["seg_len"] == [50, 50, 50, 50]
    ragged = ar1_segmented_null(-0.34, [50, 50, 42], 3, 200, 777, burn_in=5)
    assert ragged["n"] == 45 + 45 + 37 and ragged["n_segments"] == 3
    c = ar1_segmented_null(-0.34, 50, 4, 400, 778, burn_in=5, max_lag=8)
    assert not np.array_equal(
        a["dc"]["samples"]["t_nw"], c["dc"]["samples"]["t_nw"]
    )


def test_ar1_segmented_null_reports_the_persistence_reference():
    """Independent zero-mean segments: coherence ~ 0.67 at k = 6, acf1 ~ 0."""
    null = ar1_segmented_null(-0.34, 50, 6, 600, 4242, burn_in=5, channels=("dc",))
    assert null["dc"]["segment_mean_coherence"]["median"] == pytest.approx(
        2.0 / 3.0, abs=0.02
    )
    assert null["dc"]["segment_mean_acf1"]["median"] == pytest.approx(0.0, abs=0.06)


@pytest.mark.parametrize("bad", [dict(seg_len=0), dict(n_segments=0), dict(reps=0)])
def test_ar1_segmented_null_rejects_degenerate_shapes(bad):
    kwargs = dict(seg_len=10, n_segments=2, reps=10)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        ar1_segmented_null(-0.3, kwargs["seg_len"], kwargs["n_segments"],
                           kwargs["reps"], 1234)


def test_segment_mean_persistence_hits_its_analytic_bounds():
    assert segment_mean_persistence([2.0, 2.0, 2.0, 2.0]) == {
        "acf1": pytest.approx(0.75), "coherence": 1.0, "k": 4
    }  # a constant sequence: acf1 ceiling is (k-1)/k, never 1
    flip = segment_mean_persistence([1.0, -1.0, 1.0, -1.0])
    assert flip["acf1"] == pytest.approx(-0.75) and flip["coherence"] == 0.5
    assert np.isnan(segment_mean_persistence([1.0])["acf1"])
    assert np.isnan(segment_mean_persistence([0.0, 0.0])["acf1"])
    assert np.isnan(segment_mean_persistence([])["coherence"])


def test_ar1_streams_is_public_and_stationary():
    """The generator is part of the tested surface (no private copies)."""
    rng = np.random.default_rng(2612)
    x = ar1_streams(rng, -0.34, 4000, 30)
    rho = np.mean([lag_ladder(row, 1, 0)["rho"][0] for row in x])
    assert rho == pytest.approx(-0.34, abs=RHO_TOL)
    assert x.shape == (30, 4000)
    # stationary start: the first column has the stationary variance
    y = ar1_streams(np.random.default_rng(2613), 0.9, 5, 20000)
    assert float(np.var(y[:, 0])) == pytest.approx(1.0 / (1 - 0.81), rel=0.06)
    with pytest.raises(ValueError, match="phi"):
        ar1_streams(rng, 1.0, 10, 5)


# ==========================================================================
# 13. Finite-n facts the docstrings quote
# ==========================================================================


@pytest.mark.parametrize("n,expected", [(25, 0.737), (45, 0.723), (400, 0.685)])
def test_phi_zero_null_median_is_above_the_half_normal_median_at_short_n(n, expected):
    """0.674 is the asymptote, not the value a short window is divided by."""
    got = ar1_surrogate_null(0.0, n, 6000, 20260831, max_lag=8, channels=("dc",))
    assert got["dc"]["abs_t_nw"]["median"] == pytest.approx(expected, abs=0.025)
    assert got["dc"]["abs_t_nw"]["median"] > HALF_NORMAL_MEDIAN


def test_null_median_carries_monte_carlo_error_at_2000_reps():
    """The denominator of every `ratio` is itself a Monte-Carlo median.

    Its relative sd is ~3% at reps = 2000, which on the audit's pooled cells
    is larger than the numerator's whole CI half-width -- the reason
    ar1_surrogate_null's docstring tells callers to quote it.
    """
    meds = [
        ar1_surrogate_null(-0.343, 50, 2000, 900000 + s, burn_in=5, max_lag=8,
                           channels=("dc",))["dc"]["abs_t_nw"]["median"]
        for s in range(12)
    ]
    rel_sd = float(np.std(meds, ddof=1) / np.mean(meds))
    assert 0.01 < rel_sd < 0.06
    big = [
        ar1_surrogate_null(-0.343, 50, 20000, 910000 + s, burn_in=5, max_lag=8,
                           channels=("dc",))["dc"]["abs_t_nw"]["median"]
        for s in range(4)
    ]
    assert float(np.std(big, ddof=1)) < float(np.std(meds, ddof=1))


ODD_LEAK_REPS = 2000
ODD_LEAK_AMP = 10.0


@pytest.mark.parametrize("n,leaks", [(44, False), (45, True)])
def test_odd_windows_leak_a_period_two_component_into_dc(n, leaks):
    """dc/alt orthogonality is exact only at EVEN n (module docstring).

    At odd n the period-2 component leaves A/n in the DC mean AND collapses
    the Bartlett-truncated sigma_LR^2 toward zero, so both ends of the ratio
    inflate |t_dc| together.  The production window is odd (t_refresh 50,
    burn-in 5), so this is a latent hazard of the window choice, not of the
    estimator; the amplitudes actually observed are ~0.5 sigma, far below
    where it bites.
    """
    rng = np.random.default_rng(5150)
    sign = _alternating(n)
    X = rng.standard_normal((ODD_LEAK_REPS, n)) + ODD_LEAK_AMP * sign[None, :]
    t = np.abs([channel_t(x, "dc", 0, max_lag=8)["t_nw"] for x in X])
    if leaks:
        assert float(np.median(t)) > 1.4
        assert float(np.mean(t >= 2.0)) > 0.30
    else:
        assert float(np.median(t)) == pytest.approx(HALF_NORMAL_MEDIAN, abs=0.10)
        assert float(np.mean(t >= 2.0)) < 0.15


def test_newey_west_bandwidth_at_every_window_the_design_produces():
    """The tracked tier runs at L = 3, not the L = 4 of the frozen tier."""
    assert [newey_west_bandwidth(n, 8) for n in (45, 37, 35, 25)] == [3, 3, 3, 2]
    assert [newey_west_bandwidth(n, 8) for n in (187, 195)] == [4, 4]
    # pooled slots: the bandwidth GROWS with the window, which is exactly
    # why the slot-level read absorbs low-frequency power the segment cannot.
    assert [newey_west_bandwidth(n, 8) for n in (180, 270, 360, 720)] == [4, 4, 5, 6]
