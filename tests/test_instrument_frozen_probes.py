"""Frozen-probe tier: unbounded-window cumulative statistics (CPU only).

Covers the two properties the tier is built on:

* the DETECTION property -- on a stream with a persistent non-zero mean the
  cumulative |t| grows like sqrt(t), while on a pure-noise stream it stays at
  the N(0, 1) scale no matter how long the integration runs;
* the FROZEN property -- the probe directions never rotate on a subspace
  refresh and are never reset by innovation detection, unlike the tracked and
  bulk tiers.

Plus the Newey-West correction's sign behaviour and the log/schema contract.
Dev seeds (>= 1000) throughout; synthetic streams come from the WP0.5-
validated ``src.stats.generators``.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from src.instrument.schema import validate_instrumentation
from src.instrument.tracker import (
    FrozenProbeAccumulator,
    FrozenProbeBank,
    MatrixTracker,
)
from src.stats import ar1, gaussian_noise

CLASSIFIER_KWARGS = dict(tau_sig=4.0, tau_noise=2.0, rho_osc=0.5, n_min=50)


def _feed(stream, *, max_lag=8):
    acc = FrozenProbeAccumulator(1, max_lag=max_lag)
    for x in stream:
        acc.update(np.array([x]))
    return acc


def _t_at_checkpoints(stream, ns, key="t_naive", max_lag=8):
    """One pass over ``stream``, reading the cumulative t at each n in ``ns``."""
    acc = FrozenProbeAccumulator(1, max_lag=max_lag)
    out, wanted = [], set(int(n) for n in ns)
    for i, x in enumerate(stream, start=1):
        acc.update(np.array([x]))
        if i in wanted:
            out.append(float(acc.stats()[key][0]))
    return out


def _t_at(stream, n, key="t_naive", max_lag=8):
    acc = FrozenProbeAccumulator(1, max_lag=max_lag)
    for x in stream[:n]:
        acc.update(np.array([x]))
    return float(acc.stats()[key][0])


# --------------------------------------------------------------- detection


def test_persistent_mean_gives_sqrt_t_growth_of_the_cumulative_t():
    """Planted mean mu with unit noise: t_naive(n) = mu * sqrt(n), so
    quadrupling the integration length must double |t|."""
    mu = 0.05
    ns = np.array([500, 1000, 2000, 4000, 8000, 16000])
    streams = [
        gaussian_noise(16000, mean=mu, noise_scale=1.0, seed=s)
        for s in range(1001, 1021)
    ]
    # t at a given n is itself random with unit sd around mu*sqrt(n); average
    # over 20 independent streams so the analytic law is what is being tested.
    mean_t = np.array(
        [np.mean(col) for col in zip(*(_t_at_checkpoints(st, ns) for st in streams))]
    )
    for n, t in zip(ns, mean_t):
        assert t == pytest.approx(mu * np.sqrt(n), abs=0.6), n
    assert mean_t[3] / mean_t[1] == pytest.approx(2.0, rel=0.2)  # 4000 vs 1000
    assert mean_t[5] / mean_t[3] == pytest.approx(2.0, rel=0.2)  # 16000 vs 4000
    # log t vs log n slope ~ 0.5 -- the reported growth-law signature.
    slope = np.polyfit(np.log(ns), np.log(mean_t), 1)[0]
    assert 0.4 < slope < 0.6
    # And it clears the conventional threshold only at long integration.
    assert mean_t[1] < 4.0 <= mean_t[5]


def test_pure_noise_t_stays_at_the_unit_scale_at_every_integration_length():
    """No drift: |t| must not grow with n; across independent streams it is
    distributed like |N(0, 1)|."""
    finals = []
    for i, seed in enumerate(range(1100, 1140)):
        stream = gaussian_noise(16000, mean=0.0, noise_scale=1.0, seed=seed)
        finals.append(_t_at(stream, 16000))
        if i == 0:
            ns = np.array([500, 2000, 8000, 16000])
            ts = np.array([abs(_t_at(stream, int(n))) for n in ns])
            assert ts.max() < 4.0
    finals = np.asarray(finals)
    assert np.abs(finals).max() < 4.0  # 40 draws from |N(0,1)|
    assert 0.5 < finals.std() < 1.8


def test_newey_west_is_conservative_on_positively_correlated_input():
    """AR(1) with rho > 0: the long-run variance exceeds the marginal one, so
    the adjusted |t| must be SMALLER than the naive one (and ess < n)."""
    stream = ar1(8000, 0.6, noise_scale=1.0, mean=0.05, seed=1200)
    acc = _feed(stream)
    st = acc.stats()
    assert abs(float(st["t_nw"][0])) < abs(float(st["t_naive"][0]))
    assert float(st["ess"][0]) < acc.n
    assert st["lag_truncation"] > 0
    assert not bool(st["nw_floored"][0])


def test_newey_west_inflates_t_on_negatively_correlated_input():
    """The mirror case, documented rather than hidden: on the negative-rho
    population this repo actually observes, the correction moves the other
    way (ess > n) -- which is why both t's are logged."""
    stream = ar1(8000, -0.6, noise_scale=1.0, mean=0.05, seed=1201)
    st = _feed(stream).stats()
    assert abs(float(st["t_nw"][0])) > abs(float(st["t_naive"][0]))
    assert float(st["ess"][0]) > 8000


def test_newey_west_and_naive_agree_on_white_noise():
    st = _feed(gaussian_noise(8000, mean=0.03, noise_scale=1.0, seed=1202)).stats()
    assert float(st["t_nw"][0]) == pytest.approx(float(st["t_naive"][0]), rel=0.25)
    assert float(st["ess"][0]) == pytest.approx(8000, rel=0.35)


def test_accumulator_never_forgets_and_reports_moments_exactly():
    rng = np.random.default_rng(1203)
    x = rng.standard_normal((500, 3))
    acc = FrozenProbeAccumulator(3, max_lag=4)
    for row in x:
        acc.update(row)
    st = acc.stats()
    assert st["n"] == 500
    assert np.allclose(st["mean"], x.mean(axis=0))
    assert np.allclose(st["var"], x.var(axis=0))
    assert np.allclose(st["t_naive"], x.mean(axis=0) / np.sqrt(x.var(axis=0) / 500))


def test_accumulator_edge_cases():
    empty = FrozenProbeAccumulator(2).stats()
    assert empty["n"] == 0 and np.isnan(empty["mean"]).all()
    const = _feed(np.full(200, 2.0))  # zero variance -> t defined as 0
    st = const.stats()
    assert float(st["var"][0]) == pytest.approx(0.0, abs=1e-9)
    assert float(st["t_naive"][0]) == 0.0
    zero_lag = _feed(gaussian_noise(300, seed=1204), max_lag=0)
    assert zero_lag.lag_truncation() == 0
    st0 = zero_lag.stats()
    assert float(st0["t_nw"][0]) == pytest.approx(float(st0["t_naive"][0]))
    with pytest.raises(ValueError):
        FrozenProbeAccumulator(0)
    with pytest.raises(ValueError):
        FrozenProbeAccumulator(2, max_lag=-1)


# ------------------------------------------------------------------ frozen


def _bank(seed=1300, k3=4, m=12, n=9, **kw):
    gen = torch.Generator().manual_seed(seed)
    return FrozenProbeBank(m, n, k3, generator=gen, **kw)


def test_bank_directions_are_reproducible_from_the_seed_and_unit_norm():
    a, b = _bank(), _bank()
    assert torch.equal(a.U, b.U) and torch.equal(a.V, b.V)
    assert torch.allclose(torch.linalg.norm(a.U, dim=0), torch.ones(a.k3), atol=1e-6)
    assert torch.allclose(torch.linalg.norm(a.V, dim=0), torch.ones(a.k3), atol=1e-6)
    assert not torch.equal(_bank(seed=1301).U, a.U)


def test_bank_projection_matches_the_explicit_bilinear_form():
    bank = _bank()
    G = torch.randn(12, 9, generator=torch.Generator().manual_seed(1302))
    p = bank.project(G)
    for j in range(bank.k3):
        assert float(p[j]) == pytest.approx(
            float(bank.U[:, j] @ G @ bank.V[:, j]), rel=1e-5
        )


def _run_tracker(steps=120, k3=4, **kw):
    gen = torch.Generator().manual_seed(1400)
    fgen = torch.Generator().manual_seed(1401)
    tr = MatrixTracker(
        "w",
        (12, 9),
        k1=3,
        k2=2,
        t_refresh=25,
        betas=(0.9,),
        classifier_kwargs=CLASSIFIER_KWARGS,
        snapshot_every=10,
        generator=gen,
        k3=k3,
        frozen_generator=fgen,
        **kw,
    )
    rng = np.random.default_rng(1402)
    for t in range(steps):
        # A momentum matrix that ROTATES over time, forcing subspace
        # innovation on the tracked tier.
        ang = 2.0 * np.pi * t / steps
        M = torch.zeros(12, 9)
        M[0, 0] = 10.0 * np.cos(ang)
        M[1, 1] = 10.0 * np.sin(ang)
        M[2, 2] = 3.0
        G = torch.from_numpy(rng.standard_normal((12, 9)).astype(np.float32))
        tr.observe(G, M)
    return tr


def test_frozen_directions_survive_every_refresh_and_every_reset():
    tr = _run_tracker()
    U0, V0 = tr.frozen.U.clone(), tr.frozen.V.clone()
    assert len(tr.refresh_steps) > 1  # refreshes did happen
    assert any(t.reset_steps for t in tr.directions)  # tracked tier did reset
    assert torch.equal(tr.frozen.U, U0)  # frozen tier did not move
    assert torch.equal(tr.frozen.V, V0)
    # The frozen accumulator saw EVERY step -- no reset ever truncated it.
    assert tr.frozen.acc.n == 120
    assert tr.frozen.to_log()["n_observations"] == 120


def test_enabling_frozen_probes_does_not_perturb_the_tracked_tier():
    """Separate RNG stream: the tracked subspace and its statistics must be
    bit-identical with and without the frozen tier."""
    with_frozen = _run_tracker(k3=4)
    without = _run_tracker(k3=0)
    assert without.frozen is None
    for a, b in zip(with_frozen.directions, without.directions):
        assert a.s_values == b.s_values
        assert a.reset_steps == b.reset_steps


def test_tracked_tier_t_is_reset_while_the_frozen_tier_integrates_on():
    tr = _run_tracker()
    beta = tr.betas[0]
    n_tracked = max(
        max(t.snapshots[beta]["n_since_reset"]) for t in tr.directions
    )
    # The tracked tier's window is bounded (EMA + resets); the frozen tier's
    # sample count equals the full run length. This is the structural contrast
    # the pre-registered question is about.
    assert tr.frozen.acc.n == 120
    assert n_tracked <= 120


def test_decimation_shrinks_the_raw_series_but_not_the_statistics():
    dense = _run_tracker(k3=4)
    sparse = _run_tracker(k3=4, frozen_decimate=10)
    assert len(dense.frozen.raw[0]) == 120
    assert len(sparse.frozen.raw[0]) == 12
    assert sparse.frozen.acc.n == dense.frozen.acc.n == 120
    d_final = dense.frozen.to_log()["probes"][0]["final"]
    s_final = sparse.frozen.to_log()["probes"][0]["final"]
    assert d_final["t_nw"] == pytest.approx(s_final["t_nw"], rel=1e-9)


def test_frozen_log_round_trips_through_the_schema_validator():
    tr = _run_tracker()
    log = {
        "instrumentation_schema_version": 2,
        "betas": ["0.9"],
        "hvp_enabled": False,
        "frozen_probes_enabled": True,
        "matrices": {"w": tr.to_log()},
    }
    validate_instrumentation(log)
    fp = log["matrices"]["w"]["frozen_probes"]
    assert fp["k3"] == 4 and len(fp["probes"]) == 4
    assert len(fp["probes"][0]["t_nw"]) == len(fp["snapshot_steps"])
    assert set(fp["probes"][0]["final"]) >= {"n", "mean", "t_naive", "t_nw", "ess"}


def test_schema_rejects_a_malformed_frozen_block():
    from src.instrument.schema import InstrumentationValidationError

    tr = _run_tracker()
    mat = tr.to_log()
    mat["frozen_probes"]["probes"].pop()  # k3 no longer matches
    with pytest.raises(InstrumentationValidationError):
        validate_instrumentation(
            {
                "instrumentation_schema_version": 2,
                "betas": ["0.9"],
                "hvp_enabled": False,
                "matrices": {"w": mat},
            }
        )
