"""Synthetic-signal suite for scripts/analyze_channel_audit_frozen.py -- CPU only.

``reports/channel-audit-preregistration.md`` 6d calls this file a **launch
precondition, not a nicety** (K0(b) names it), and states the standard it has
to meet:

    Every registered criterion must be shown to produce EVERY one of its
    registered branches on synthetic data, and to produce each one ONLY from
    the stream that should produce it.

That is the direct lesson of ``reports/bbp-prereg.md`` amendment A2, where a
criterion shipped that could only ever pass and measured nothing; the
pre-registration's own repairs R1 and R2 are two more instances, both caught
by exactly this exercise (a criterion that fires on its own null is as
defective as one that can never fire).  So every branch below is asserted
twice: that the stream which should produce it does, and that the streams
which should not, do not.

What is pinned here:

* the registered defaults of 6b's table, read off the argument parser, plus
  the per-quantity pooling (median, except tau: mean) and the proposed
  thresholds of section 4 / the Appendix;
* **K1** (section 7): the white-noise control returns ``band_contrast`` and a
  tail contrast within 1.00 +/- 0.10 and ``tau_cal`` within 1.00 +/- 0.10 at
  EVERY K in {8, 16, 32, 64}, and P3 returns UNDECIDED on it -- the single
  most important assertion in 6d, because under the previous DRAFT's estimator
  white noise fired the DECISIVE clause;
* **P1** rows 1-6 of the registered outcome map, each from its own stream,
  plus the repair-R1 regression: a channel-common inflation with NO planted
  signal must still FAIL;
* **P2** rows A-F, each from its own planted (top, bulk) pair;
* **P3** DECISIVE / FAIL / UNDECIDED, the K-stability requirement, and the
  consistency clause in both states, asserted against the estimator's own
  references (0.658 at phi = -0.34, 2.33 at phi = +0.5) and never against the
  analytic value, which a correct pipeline does not return;
* **Rider-1** pass band / flat / ambiguous / vacuity guard and **Rider-2**
  B-invariant / sampling-consistent / mixed;
* **K6** firing and not firing, and P1 reported unread when it does;
* the structural refusals: mixed segment-start parity, ``decimate > 1``, a
  gapped raw-step series;
* the mirror contract against ``FrozenProbeAccumulator`` itself, on the logged
  statistic the sidecar actually carries;
* end-to-end through ``main()`` on JSON sidecar fixtures, byte-identical on a
  rerun (report, markdown and figures), with the descriptive-only framing.

Dev seeds (>= 1000) throughout, no GPU, no torch training, no network.  The
scenarios run at reduced ``--null-reps`` / ``--bootstrap-reps`` for runtime;
the DEFAULTS are asserted separately and are the registered ones.  Tolerances
here are test conventions, not decision thresholds (CLAUDE.md ground rule 1).
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from src.instrument.schema import SIDECAR_SUFFIX
from src.instrument.tracker import FrozenProbeAccumulator
from src.stats.spectral import ar1_streams


def _load_script():
    path = REPO_ROOT / "scripts" / "analyze_channel_audit_frozen.py"
    spec = importlib.util.spec_from_file_location("analyze_channel_audit_frozen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acf = _load_script()

N_STEPS = 200
PHI = -0.384  # the tier-implied phi_hat of prereg section 2
CORE_RUN_SEEDS = (1300, 1301, 1302)
K1_TOL = 0.10  # prereg 7 K1's registered tolerance
FAST = ["--null-reps", "8000", "--bootstrap-reps", "300", "--burn-ins", "5",
        "--no-controls"]


# ==========================================================================
# fixtures: synthetic sidecar-shaped runs with planted components
# ==========================================================================


def _final_block(X, max_lag=8):
    """The ``final`` statistics a real sidecar carries, from the real accumulator.

    ``FrozenProbeAccumulator`` is the class that writes them during training, so
    building the fixture with it keeps the fixture schema-valid AND makes the
    producer's mirror check (prereg 5.1) a genuine cross-estimator comparison
    rather than a self-check.
    """
    acc = FrozenProbeAccumulator(X.shape[0], max_lag=max_lag)
    for t in range(X.shape[1]):
        acc.update(X[:, t])
    st = acc.stats()
    return [
        {"ess": float(st["ess"][j]), "mean": float(st["mean"][j]), "n": int(st["n"]),
         "nw_floored": bool(st["nw_floored"][j]), "t_naive": float(st["t_naive"][j]),
         "t_nw": float(st["t_nw"][j]), "var": float(st["var"][j])}
        for j in range(X.shape[0])
    ]


def _matrix_log(X, Y, n_steps, t_refresh):
    steps = list(range(1, n_steps + 1))
    finals = _final_block(X)
    directions = []
    if Y is not None:
        for j in range(Y.shape[0]):
            directions.append({
                "index": j,
                "kind": "top" if j < Y.shape[0] // 2 else "bulk",
                "lambda_hvp": {"step": [], "value": []},
                "per_beta": {},
                "refresh_alignment": {"step": [], "value": []},
                "reset_steps": [],
                "s": [float(v) for v in Y[j]],
                "sigma": {"step": [], "value": []},
            })
    return {
        "align_min": 0.9,
        "directions": directions,
        "frozen_probes": {
            "decimate": 1,
            "k3": int(X.shape[0]),
            "lag_truncation": 4,
            "max_lag": 8,
            "n_observations": n_steps,
            "probes": [
                {"ess": [], "final": finals[j], "index": j, "mean": [],
                 "s": [float(v) for v in X[j]], "t_naive": [], "t_nw": [], "var": []}
                for j in range(X.shape[0])
            ],
            "raw_steps": steps,
            "snapshot_lag_truncation": [],
            "snapshot_n": [],
            "snapshot_steps": [],
        },
        "grad_fro_norm": [1.0] * n_steps,
        "k1": 0 if Y is None else Y.shape[0] // 2,
        "k2": 0 if Y is None else Y.shape[0] - Y.shape[0] // 2,
        "refresh_steps": list(range(1, n_steps + 1, t_refresh)),
        "shape": [8, 8],
        "snapshot_every": 5,
        "steps": steps,
        "t_refresh": t_refresh,
        "top_sigma_m": [1.0] * n_steps,
    }


def _entry(run, seed, lr, batch, frozen, tracked=None, n_steps=N_STEPS, t_refresh=50):
    """One (metadata, instrumentation log) pair, the shape ``ingest`` consumes."""
    log = {
        "betas": [],
        "frozen_probes_enabled": True,
        "hvp_enabled": False,
        "instrumentation_schema_version": 2,
        "matrices": {
            name: _matrix_log(X, None if tracked is None else tracked[name], n_steps, t_refresh)
            for name, X in sorted(frozen.items())
        },
    }
    meta = {"batch_size": batch, "lr": lr, "problem": None, "run": run,
            "seed": seed, "sidecar": run + SIDECAR_SUFFIX}
    return meta, log


def _plant(rng, probes, *, phi=PHI, n=N_STEPS, broad_alt=0.0, sparse_frac=0.0,
           sparse_alt=0.0, dc=0.0, infl_dc=0.0, infl_alt=0.0, phi_slow=0.9):
    """AR(1) at ``phi`` plus the planted components each scenario needs.

    ``broad_alt`` / ``sparse_alt`` are period-2 means tied to the absolute step
    (a genuine alternating signal, homogeneous or concentrated on a few
    probes); ``dc`` is a persistent mean; ``infl_dc`` / ``infl_alt`` add
    zero-mean SLOW power at the two band ends -- long-lag structure the L = 4
    Newey-West window cannot see, which inflates |t| in a channel WITHOUT
    planting a mean there.  With both set, the stream is the repair-R1 object:
    an instrument inflation common to the two channels and no signal at all.
    """
    X = ar1_streams(rng, phi, n, probes)
    parity = (-1.0) ** np.arange(n)
    if infl_dc or infl_alt:
        scale = np.sqrt(1.0 - phi_slow ** 2)
        if infl_dc:
            X = X + infl_dc * (ar1_streams(rng, phi_slow, n, probes) * scale)
        if infl_alt:
            X = X + infl_alt * (ar1_streams(rng, phi_slow, n, probes) * scale) * parity[None, :]
    if broad_alt:
        X = X + broad_alt * parity[None, :]
    if sparse_frac:
        k = int(round(sparse_frac * probes))
        X[:k] = X[:k] + sparse_alt * parity[None, :]
    if dc:
        X = X + dc
    return X


def core_entries(seed, *, probes=192, matrices=2, tracked=None, **plant_kw):
    """The 3-run B = 2000 core (prereg section 3 rows 1-2), planted."""
    rng = np.random.default_rng(seed)
    out = []
    for s in CORE_RUN_SEEDS:
        frozen = {"m.%d.weight" % i: _plant(rng, probes, **plant_kw) for i in range(matrices)}
        tr = None
        if tracked is not None:
            tr = {
                name: _tracked_plant(rng, tracked["k"], tracked.get("dc_top", 0.0),
                                     tracked.get("dc_bulk", 0.0))
                for name in frozen
            }
        out.append(_entry("airbench_instrumented_seed%d_x" % s, s, 0.24, 2000, frozen, tr))
    return out


def _tracked_plant(rng, k, dc_top, dc_bulk, phi=PHI, n=N_STEPS):
    Y = ar1_streams(rng, phi, n, k)
    half = k // 2
    if dc_top:
        Y[:half] += dc_top
    if dc_bulk:
        Y[half:] += dc_bulk
    return Y


def rider_entries(seed, dc_by_batch, phi_by_batch=None, probes=192):
    """The step-matched batch rider (rows 3-5): seeds 1320/1321, B in {500, 2000, 8000}."""
    rng = np.random.default_rng(seed)
    out = []
    for s in (1320, 1321):
        for batch in sorted(dc_by_batch):
            n = 192 if batch == 8000 else 200
            phi = PHI if phi_by_batch is None else phi_by_batch[batch]
            frozen = {
                "m.%d.weight" % i: _plant(rng, probes, phi=phi, n=n, dc=dc_by_batch[batch])
                for i in range(2)
            }
            out.append(_entry(
                "airbench_instrumented_seed%d_b%d" % (s, batch), s, 0.24, batch,
                frozen, n_steps=n,
            ))
    return out


def report_of(entries, extra=()):
    args = acf.build_parser().parse_args(FAST + list(extra))
    args._selection = None
    return acf.build_report(acf.ingest(entries), args)


@pytest.fixture(scope="module")
def cache():
    return {}


def cached(cache, key, factory):
    if key not in cache:
        cache[key] = report_of(factory())
    return cache[key]


# ==========================================================================
# 1. The registered defaults, pooling and thresholds (prereg 6b, 6d)
# ==========================================================================


def _defaults():
    return vars(acf.build_parser().parse_args([]))


def test_registered_defaults_equal_the_registered_values():
    """prereg 6b's defaults table, read off the parser rather than trusted.

    "A default that silently disagrees with the registration is how a report
    ends up quoting neither quantity."  Every row of that table is here, and a
    drift in EITHER direction fails.
    """
    d = _defaults()
    assert d["max_lag"] == 64  # the ladder
    assert d["tau_lags"] == [8, 16, 32, 64]
    assert d["tau_primary_k"] == 32
    assert d["null_reps"] == 200_000  # section 5.6: 2000 cannot resolve 1e-4
    assert d["null_seed"] == 4242
    assert d["tau_reference_seed"] == 4243  # section 5.9, != the null seed
    assert d["bootstrap_block"] == 64  # one frozen bank
    assert d["bootstrap_block_tracked"] == 4  # the segments of one direction
    assert d["bootstrap_reps"] == 2000
    assert d["bootstrap_seed"] == 4242
    assert d["burn_ins"] == [5, 15, 25] and d["primary_burn_in"] == 5
    assert d["min_seed"] == 1000  # CLAUDE.md ground rule 2
    assert d["synthetic_control"] == "none"
    assert d["out_md"] == Path("reports/channel-audit.md")
    assert d["out_json"] == Path("reports/channel-audit.json")
    assert d["out_figdir"] == Path("reports/figures")
    # the module constants the parser reads from carry the same values
    assert (acf.TAU_LAGS, acf.TAU_PRIMARY_K) == ((8, 16, 32, 64), 32)
    assert (acf.NULL_REPS, acf.NULL_SEED, acf.TAU_REFERENCE_SEED) == (200_000, 4242, 4243)
    assert acf.BOOTSTRAP_BLOCK == 64 and acf.LADDER_MAX_LAG == 64
    assert acf.REGISTERED_BURN_INS == (5, 15, 25) and acf.PRIMARY_BURN_IN == 5


def test_pooling_is_median_everywhere_except_tau():
    """prereg 5.8: median pooling, EXCEPT tau, which is the mean (repair R2).

    Median pooling of a 32-lag sum is biased ~12% low and made the previous
    DRAFT's decisive clause fire on white noise.
    """
    assert acf.POOLING["tau"] == "mean"
    for key in ("ratio_c", "frame_gain", "excess_dc", "phi_hat"):
        assert acf.POOLING[key] == "median"
    assert acf.POOLING["frac_c"] == "rate"


def test_tau_is_actually_mean_pooled_not_just_documented():
    """The registered pooling has to be in the code path, not only the table."""
    rng = np.random.default_rng(7)
    rho = rng.standard_normal((512, 64)) * 0.05
    tau = acf._tau_from_rho(rho, 32)
    assert tau.shape == (512,)
    assert float(np.mean(tau)) != pytest.approx(float(np.median(tau)), abs=1e-9)
    expected = 1.0 + 2.0 * np.sum(rho[:, :32], axis=1)
    assert np.allclose(tau, expected)


def test_proposed_thresholds_are_the_prereg_appendix_values():
    r = acf.REGISTERED
    assert r["p1_band_contrast_pass"] == 1.30 and r["p1_band_contrast_middle_edge"] == 1.15
    assert r["p1_tail_contrast_pass"] == 3.0 and r["p1_frac_alt_floor"] == 0.010
    assert r["p1_min_dc_events"] == 10
    assert r["p2_frame_gain_pass"] == 3.0
    assert r["p2_bulk_tracks_ceiling"] == 1.3 and r["p2_bulk_elevated_floor"] == 2.0
    assert r["p3_consistency_band"] == (0.75, 1.30) and r["p3_decisive_upper"] == 1.0
    assert r["rider1_pass_band"] == (2.8, 5.6) and r["rider1_fail_flat"] == 1.5
    assert r["rider1_vacuity_guard"] == 0.05
    assert r["rider2_invariance_max_over_min"] == 1.3
    assert r["rider2_sampling_ess_tolerance"] == 1.15
    assert r["k2_nw_floored_frac"] == 0.05
    assert r["k3_phi_window"] == (-0.60, -0.15) and r["k3_phi_spread_max"] == 0.35
    assert r["k4_frozen_median_t_dc_max"] == 2.0
    assert r["k6_channel_shape_divergence"] == 0.15
    assert r["t_exceedance"] == 4.0
    assert "FROZEN" in acf.THRESHOLD_STATUS


def test_the_estimators_are_the_tested_module_not_a_copy():
    """prereg 5 preamble / CLAUDE.md WP1.1: no reimplementation."""
    import src.stats.spectral as spectral

    for name in ("ar1_streams", "ar1_surrogate_null", "block_bootstrap_ci",
                 "channel_t", "lag_ladder", "newey_west_bandwidth"):
        assert getattr(acf, name) is getattr(spectral, name), name


def test_registered_pools_are_the_section_3_run_set():
    assert acf.CORE_SEEDS == (1300, 1301, 1302, 1310, 1311)
    assert acf.RIDER_SEEDS == (1320, 1321)
    assert acf.CORE_BATCH == 2000
    assert acf.pool_of(1300, 2000, acf.CORE_SEEDS, acf.RIDER_SEEDS, 2000) == "core"
    assert acf.pool_of(1320, 500, acf.CORE_SEEDS, acf.RIDER_SEEDS, 2000) == "rider"
    assert acf.pool_of(1320, 2000, acf.CORE_SEEDS, acf.RIDER_SEEDS, 2000) == "rider"
    # a core seed at a rider batch is NOT the core pool (single n, single rung)
    assert acf.pool_of(1300, 500, acf.CORE_SEEDS, acf.RIDER_SEEDS, 2000) == "unassigned"
    assert acf.pool_of(1999, 2000, acf.CORE_SEEDS, acf.RIDER_SEEDS, 2000) == "unassigned"


# ==========================================================================
# 2. Structural refusals (prereg 1, 3, 5.2)
# ==========================================================================


def test_mixed_segment_start_parity_raises():
    """prereg 1: the parity assertion, and it must fail LOUDLY.

    ``channel_t`` fixes the demodulation sign from the first element of the
    array it is given, so segments whose absolute starts differ in parity
    carry opposite alternating signs and cancel when pooled.
    """
    entries = core_entries(1)
    ok = acf.ingest(entries)
    assert ok.parity["parity"] == "odd"
    assert ok.parity["n_odd_starts"] == ok.parity["n_segment_starts"]
    # shift one matrix's raw steps by one -> even start
    meta, log = entries[0]
    block = log["matrices"]["m.0.weight"]["frozen_probes"]
    block["raw_steps"] = [s + 1 for s in block["raw_steps"]]
    with pytest.raises(SystemExit, match="MIXED parity"):
        acf.ingest(entries)


def test_mixed_parity_from_an_odd_refresh_cadence_raises():
    entries = core_entries(2, probes=8, tracked={"k": 4})
    for _meta, log in entries:
        log["matrices"]["m.0.weight"]["refresh_steps"] = [1, 20, 40]
    with pytest.raises(SystemExit, match="MIXED parity"):
        acf.ingest(entries)


def test_decimated_and_gapped_series_are_refused():
    """prereg 3: ``decimate: 1`` is load-bearing; a dropped step breaks the band."""
    entries = core_entries(3, probes=8)
    entries[0][1]["matrices"]["m.0.weight"]["frozen_probes"]["decimate"] = 2
    with pytest.raises(SystemExit, match="decimate"):
        acf.ingest(entries)
    entries = core_entries(3, probes=8)
    block = entries[0][1]["matrices"]["m.0.weight"]["frozen_probes"]
    block["raw_steps"] = [s if s < 100 else s + 5 for s in block["raw_steps"]]
    with pytest.raises(SystemExit, match="not consecutive"):
        acf.ingest(entries)


def test_runs_without_the_frozen_tier_are_skipped_not_analyzed():
    entries = core_entries(4, probes=8)
    entries[0][1]["frozen_probes_enabled"] = False
    bank = acf.ingest(entries)
    assert len(bank.runs) == 2 and len(bank.skipped) == 1
    assert "frozen_probes disabled" in bank.skipped[0]["reason"]
    entries = core_entries(4, probes=8)
    for _m, log in entries:
        log["frozen_probes_enabled"] = False
    with pytest.raises(SystemExit, match="no usable frozen-tier series"):
        acf.ingest(entries)


def test_rows_are_ordered_by_run_matrix_probe_so_a_block_is_one_bank():
    """prereg 5.8: block = 64 is "exactly one (run, matrix) frozen bank".

    That is only true if the statistics are ordered by (run, matrix, probe
    index), which is the ordering the bootstrap's circular blocks then cut.
    """
    bank = acf.ingest(core_entries(6, probes=64, matrices=2))
    frozen = [r for r in bank.rows if r["tier"] == "frozen"]
    keys = [(r["run"], r["matrix"], r["index"]) for r in frozen]
    assert keys == sorted(keys)
    for start in range(0, len(frozen), 64):
        block = frozen[start:start + 64]
        assert len({(r["run"], r["matrix"]) for r in block}) == 1
        assert [r["index"] for r in block] == list(range(64))


def test_selection_excludes_tombstones_eval_seeds_and_other_families(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for name in ("airbench_instrumented_seed1300_a", "airbench_instrumented_seed0042_eval",
                 "nanogpt_seed1500_other", "airbench_instrumented_seed1301_b"):
        (results / (name + SIDECAR_SUFFIX)).write_text("{}")
    (results / "INVALID_RUNS.json").write_text(json.dumps({
        "invalid": [{"file": "airbench_instrumented_seed1301_b.json", "reason": "t"}]
    }))
    paths, sel = acf.select_sidecars(
        results, run_prefix="airbench_instrumented_", min_seed=1000
    )
    assert [p.name for p in paths] == ["airbench_instrumented_seed1300_a" + SIDECAR_SUFFIX]
    assert sel["excluded"]["min_seed"] == ["airbench_instrumented_seed0042_eval" + SIDECAR_SUFFIX]
    assert sel["excluded"]["run_prefix"] == ["nanogpt_seed1500_other" + SIDECAR_SUFFIX]
    assert sel["excluded"]["invalid_runs"] == ["airbench_instrumented_seed1301_b" + SIDECAR_SUFFIX]
    assert acf.tombstoned_runs(results) == {"airbench_instrumented_seed1301_b"}


# ==========================================================================
# 3. The mirror contract and the Newey-West bandwidth (prereg 5.1)
# ==========================================================================


@pytest.mark.parametrize("max_lag", [8, 32])
def test_channel_t_reproduces_the_accumulator_the_sidecar_logged(max_lag):
    """prereg 5.1: the accumulator stays canonical and channel_t mirrors it.

    Checked against ``FrozenProbeAccumulator`` itself -- the class that writes
    the ``final`` block of a real sidecar -- at both the published cap (8) and
    the Phase B configs' cap (32), so this is the estimator contract on the
    object the runs will actually produce, not a self-check.
    """
    rng = np.random.default_rng(1234)
    X = ar1_streams(rng, PHI, N_STEPS, 6)
    entries = core_entries(5, probes=6, matrices=1)
    meta, log = entries[0]
    block = log["matrices"]["m.0.weight"]["frozen_probes"]
    block["max_lag"] = max_lag
    finals = _final_block(X, max_lag=max_lag)
    for j, probe in enumerate(block["probes"]):
        probe["s"] = [float(v) for v in X[j]]
        probe["final"] = finals[j]
    bank = acf.ingest([entries[0]])
    check = acf.mirror_check(bank, 64)
    assert check["n_checked"] == 6
    assert check["max_abs_dev_t_nw"] < 1e-9
    assert check["max_abs_dev_ess"] < 1e-9


def test_the_fast_newey_west_cap_is_the_same_bandwidth_as_the_ladder_cap():
    """The 8-lag NW call is bitwise identical to a 64-lag one at every design n."""
    check = acf.nw_bandwidth_check([200, 192, 50, 42], 5, 64)
    assert check["identical"] is True
    assert {row["L_at_max_lag"] for row in check["by_length"]} <= {3, 4}
    for row in check["by_length"]:
        assert row["L_at_max_lag"] == row["L_at_nw_max_lag"]
    # and the guard is real: it would notice a length where they diverge
    assert acf.nw_bandwidth_check([9000], 5, 64)["identical"] is False


# ==========================================================================
# 4. K1 -- the estimator's own controls (prereg 7 K1, 6d's headline)
# ==========================================================================


@pytest.fixture(scope="module")
def controls():
    args = acf.build_parser().parse_args(
        ["--null-reps", "20000", "--bootstrap-reps", "400", "--burn-ins", "5",
         "--control-probes", "3456"]
    )
    return acf.k1_controls(args, 20000)


def test_k1_white_noise_control_returns_one_on_both_contrasts(controls):
    c = controls["white"]
    assert c["band_contrast"] == pytest.approx(1.0, abs=K1_TOL)
    assert c["tail_contrast_at_null_q75"] == pytest.approx(1.0, abs=K1_TOL)
    assert c["contrasts_within_tolerance"] is True
    assert c["outcome_row"] == 6  # a pure null lands in FAIL, by construction


def test_k1_white_noise_control_returns_tau_cal_one_at_every_K(controls):
    """prereg 6d's registered raw references: 0.998 / 1.005 / 1.002 / 1.003."""
    c = controls["white"]
    for k in (8, 16, 32, 64):
        assert c["tau_cal_by_K"]["K%d" % k] == pytest.approx(1.0, abs=K1_TOL), k
    assert c["tau_cal_within_tolerance_every_K"] is True


def test_k1_white_noise_returns_p3_UNDECIDED_not_DECISIVE(controls):
    """6d: "the single most important assertion in the file".

    Under the previous DRAFT's median-pooled tau, white noise produced an
    upper end of 0.9109 < 1.0 and the decisive clause fired -- confirming the
    premise it claims to refute.
    """
    assert controls["white"]["tau_verdict"] == "UNDECIDED"


def test_k1_ar1_control_returns_one_on_both_contrasts_and_DECISIVE_tau(controls):
    c = controls["ar1_phi_m0.34"]
    assert c["band_contrast"] == pytest.approx(1.0, abs=K1_TOL)
    assert c["tail_contrast_at_null_q75"] == pytest.approx(1.0, abs=K1_TOL)
    assert c["contrasts_within_tolerance"] is True
    assert c["outcome_row"] == 6
    assert c["tau_verdict"] == "DECISIVE"
    # 6d registers the reference, not the analytic 0.49
    assert c["tau_cal_by_K"]["K32"] == pytest.approx(0.658, abs=0.05)


def test_the_registered_theta_tail_contrast_is_guarded_on_a_pure_null(controls):
    """The registered denominator guard is not decoration.

    At ``theta = 4 / median_null,dc`` a pure null holds ~1e-4 of its mass, so
    the DC exceedance count is far below the registered 10-event guard and
    ``tail_contrast`` is correctly reported as undefined rather than as a
    ratio of counting noise.  That is why K1's tail leg is read at the null's
    own q75 (see the module's TAIL_PROFILE_QUANTILES note).
    """
    for label in ("ar1_phi_m0.34", "white"):
        assert controls[label]["tail_contrast_denominator_guard"] is True
        assert controls[label]["tail_contrast"] is None


@pytest.mark.parametrize("kind", ["ar1", "white"])
def test_synthetic_control_mode_runs_the_whole_pipeline_from_the_cli(tmp_path, kind):
    out_json, out_md = tmp_path / "c.json", tmp_path / "c.md"
    rc = acf.main([
        "--synthetic-control", kind, "--out-json", str(out_json),
        "--out-md", str(out_md), "--out-figdir", str(tmp_path / "figs"),
        "--null-reps", "4000", "--bootstrap-reps", "150", "--burn-ins", "5",
        "--control-probes", "512", "--control-steps", "200",
    ])
    assert rc == 0
    report = json.loads(out_json.read_text())
    assert report["inputs"]["synthetic_control"] == kind
    assert report["p1"]["available"] and report["p3"]["available"]
    assert report["p2"]["available"]  # the control carries both tiers
    assert report["controls"]["note"].startswith("skipped: this run IS")
    assert out_md.read_text().startswith("# Channel audit")


def test_synthetic_control_refuses_to_write_the_registered_report_paths():
    with pytest.raises(SystemExit, match="CONTROL"):
        acf.main(["--synthetic-control", "white"])


# ==========================================================================
# 5. P1 -- every row of the registered outcome map, from its own stream
# ==========================================================================

P1_STREAMS = {
    # row: (label, plant kwargs)  -- prereg 4's exhaustive 3 x 2 partition
    6: ("no alternating component at all", {}),
    4: ("a weak broad alternating mean", {"broad_alt": 0.08}),
    2: ("a broad alternating mean, no probe-level tail", {"broad_alt": 0.10}),
    1: ("a strong broad alternating mean", {"broad_alt": 0.25}),
    5: ("a sparse alternating mean on 2% of probes", {"sparse_frac": 0.02, "sparse_alt": 0.6}),
    3: ("a weak broad mean plus a sparse tail",
        {"broad_alt": 0.06, "sparse_frac": 0.02, "sparse_alt": 0.6}),
}


@pytest.mark.parametrize("row", sorted(P1_STREAMS))
def test_p1_every_registered_row_is_produced_by_its_own_stream(cache, row):
    label, kw = P1_STREAMS[row]
    rep = cached(cache, "p1row%d" % row, lambda: core_entries(4242, **kw))
    p1 = rep["p1"]
    assert p1["outcome_row"] == row, (label, p1["band_contrast"], p1["frac_alt"])
    assert p1["n_probes_calibrated"] == 3 * 2 * 192
    # K6 leaves every row alone except the strong homogeneous plant of row 1,
    # which is a registered property of that diagnostic, pinned in its own test
    assert p1["read"] is (row != 1)


def test_p1_rows_partition_the_line_exhaustively_and_exclusively(cache):
    """Repair R5: every state has exactly one row, and no two streams share one."""
    rows = {}
    for row in sorted(P1_STREAMS):
        rep = cached(cache, "p1row%d" % row, lambda r=row: core_entries(4242, **P1_STREAMS[r][1]))
        rows[row] = rep["p1"]
    assert sorted(rows) == [1, 2, 3, 4, 5, 6]
    assert len({r["outcome_row"] for r in rows.values()}) == 6
    # the partition is on (band_contrast, P1b), and the labels follow it
    for row, p1 in rows.items():
        band, p1b = p1["band_contrast"], p1["p1b_holds"]
        if row in (1, 2):
            assert band >= acf.REGISTERED["p1_band_contrast_pass"]
        elif row in (3, 4):
            assert acf.REGISTERED["p1_band_contrast_middle_edge"] <= band < acf.REGISTERED["p1_band_contrast_pass"]
        else:
            assert band < acf.REGISTERED["p1_band_contrast_middle_edge"]
        assert p1b is (row in (1, 3, 5))
    assert rows[1]["outcome_row_label"] == "PASS"
    assert rows[6]["outcome_row_label"] == "FAIL"
    for row in (2, 3, 4, 5):
        assert "MIDDLE BAND" in rows[row]["outcome_row_label"]


def test_p1_band_contrast_at_the_registered_planted_amplitude(cache):
    """prereg 4's power table: a homogeneous A = 0.10 reads band_contrast 1.411."""
    rep = cached(cache, "p1row2", lambda: core_entries(4242, broad_alt=0.10))
    assert rep["p1"]["band_contrast"] == pytest.approx(1.411, abs=0.10)


def test_p1_alternating_plant_does_not_move_the_dc_channel(cache):
    """"each branch only from the stream that should produce it", inside P1."""
    null = cached(cache, "p1row6", lambda: core_entries(4242))
    strong = cached(cache, "p1row1", lambda: core_entries(4242, broad_alt=0.25))
    assert strong["p1"]["ratio_dc"] == pytest.approx(null["p1"]["ratio_dc"], abs=0.10)
    assert strong["p1"]["ratio_alt"] > 3.0 * null["p1"]["ratio_alt"]
    assert strong["p1"]["raw"]["frac_dc_abs_t_ge_4"] == pytest.approx(0.0, abs=1e-3)


def test_p1_channel_common_inflation_with_no_signal_still_fails(cache):
    """The direct regression test for repair R1.

    A stream with ~1.4x the null |t| in BOTH channels and zero planted
    alternating mean.  On the previous DRAFT's statistic (raw ``ratio_alt``
    plus an absolute tail floor) exactly this stream passed both P1 clauses;
    on the registered band contrast it must land in row 6.
    """
    rep = cached(cache, "inflated", lambda: core_entries(4242, infl_dc=0.22, infl_alt=0.65))
    p1 = rep["p1"]
    assert p1["ratio_alt"] > 1.25 and p1["ratio_dc"] > 1.25  # the inflation is real
    assert p1["ratio_alt"] / p1["ratio_dc"] == pytest.approx(1.0, abs=0.10)  # and common
    assert p1["band_contrast"] == pytest.approx(1.0, abs=0.10)
    assert p1["outcome_row"] == 6 and p1["outcome_row_label"] == "FAIL"
    # the previous DRAFT's statistic on the same stream: raw ratio_alt >= 1.3
    # is the clause that would have passed, and it is printed, not used
    assert p1["raw"]["median_abs_t_alt"] > 1.0


def test_p1_reports_the_raw_companions_and_the_per_matrix_theta(cache):
    rep = cached(cache, "p1row1", lambda: core_entries(4242, broad_alt=0.25))
    p1 = rep["p1"]
    for key in ("frac_alt_abs_t_ge_4", "frac_dc_abs_t_ge_4", "median_abs_t_alt",
                "median_abs_t_dc"):
        assert p1["raw"][key] is not None
    assert set(p1["theta_by_matrix"]) == {"m.0.weight", "m.1.weight"}
    for entry in p1["theta_by_matrix"].values():
        # theta is |t| >= 4 in dc-null units, and higher on the alt side
        assert entry["alt_raw_threshold"] > acf.REGISTERED["t_exceedance"]
    assert p1["by_batch"] == {}  # no rider runs in this pool


def test_p1_tail_contrast_is_read_when_the_dc_tail_is_populated(cache):
    """The other side of the denominator guard.

    A dc-heavy inflation puts >= 10 probes above ``theta`` on the DC channel,
    so ``tail_contrast`` has a usable denominator and is reported as a number
    rather than as a one-sided bound.
    """
    rep = cached(cache, "dctail",
                 lambda: core_entries(77, infl_dc=0.35, infl_alt=0.35, broad_alt=0.35))
    p1 = rep["p1"]
    assert p1["n_events"]["dc"] >= acf.REGISTERED["p1_min_dc_events"]
    assert p1["tail_contrast_denominator_guard"] is False
    assert p1["tail_contrast"] is not None
    assert p1["tail_contrast"] >= acf.REGISTERED["p1_tail_contrast_pass"]
    assert p1["p1b_tail_holds"] is True and p1["p1b_holds"] is True
    assert p1["outcome_row"] == 1
    # ... and the tail clause fails on the same construction with a weaker plant
    weaker = cached(cache, "dctail_weak",
                    lambda: core_entries(77, infl_dc=0.35, infl_alt=0.35, broad_alt=0.25))
    assert weaker["p1"]["tail_contrast_denominator_guard"] is False
    assert weaker["p1"]["tail_contrast"] < acf.REGISTERED["p1_tail_contrast_pass"]
    assert weaker["p1"]["p1b_tail_holds"] is False


# ==========================================================================
# 6. K6 -- the band contrast's own kill clause (prereg 7 K6)
# ==========================================================================


def test_k6_fires_when_the_two_channels_have_different_shape(cache):
    rep = cached(cache, "k6", lambda: core_entries(4242, sparse_frac=0.10, sparse_alt=0.5))
    shape = rep["diagnostics"]["channel_shape"]
    assert shape["k6_fires"] is True
    assert shape["max_divergence"] > acf.REGISTERED["k6_channel_shape_divergence"]
    assert rep["p1"]["read"] is False
    assert "UNREAD" in rep["p1"]["unread_reason"]
    assert rep["p1"]["outcome_row"] is not None  # still printed, never hidden
    # P2, P3 and the riders are explicitly unaffected by K6
    assert rep["p3"]["available"] is True


def test_k6_does_not_fire_on_the_null_or_on_the_weaker_plants(cache):
    for key, kw in (("p1row6", {}), ("p1row2", {"broad_alt": 0.10}),
                    ("p1row5", {"sparse_frac": 0.02, "sparse_alt": 0.6})):
        rep = cached(cache, key, lambda k=kw: core_entries(4242, **k))
        shape = rep["diagnostics"]["channel_shape"]
        assert shape["k6_fires"] is False, key
        assert shape["max_divergence"] < acf.REGISTERED["k6_channel_shape_divergence"]
        assert rep["p1"]["read"] is True, key


def test_k6_also_fires_on_a_strong_homogeneous_alternating_signal(cache):
    """A disclosed property of the registered diagnostic, pinned not repaired.

    K6's 15% bar is calibrated under the NULL (the AR(1) anchor has the two
    profiles agreeing to 1.07% through q90).  A homogeneous planted alternating
    mean is a location shift of |t_alt|, which compresses that channel's
    quantile-over-median profile, so a signal deep inside P1's PASS row trips
    the clause that reports P1 unread.  Changing K6 would be an amendment
    (prereg 7); the producer therefore reports both numbers and still prints
    the row.  This test exists so the behaviour is a recorded fact rather than
    a surprise at read-out time.
    """
    strong = cached(cache, "p1row1", lambda: core_entries(4242, broad_alt=0.25))
    weak = cached(cache, "p1row2", lambda: core_entries(4242, broad_alt=0.10))
    assert strong["p1"]["outcome_row"] == 1  # the PASS row, on the numbers
    assert strong["diagnostics"]["channel_shape"]["k6_fires"] is True
    assert strong["p1"]["read"] is False
    assert weak["diagnostics"]["channel_shape"]["k6_fires"] is False
    assert "calibrated under the null only" in strong["diagnostics"]["channel_shape"]["note"]


# ==========================================================================
# 7. P2 -- every row A-F from its own planted (top, bulk) pair
# ==========================================================================

P2_STREAMS = {
    "A": (0.45, 0.00),
    "B": (0.45, 0.30),
    "C": (0.45, 0.12),
    "D": (0.12, 0.00),
    "E": (0.12, 0.30),
    "F": (0.12, 0.12),
}


@pytest.mark.parametrize("row", sorted(P2_STREAMS))
def test_p2_every_registered_row_is_produced_by_its_own_plant(cache, row):
    dc_top, dc_bulk = P2_STREAMS[row]
    rep = cached(cache, "p2row" + row, lambda: core_entries(
        11, probes=64, tracked={"k": 16, "dc_top": dc_top, "dc_bulk": dc_bulk}))
    p2 = rep["p2"]
    assert p2["available"] is True
    assert p2["outcome_row"] == row, (p2["frame_gain"], p2["bulk_gain"])
    gain, bulk = p2["frame_gain"], p2["bulk_gain"]
    if row in ("A", "B", "C"):
        assert gain >= acf.REGISTERED["p2_frame_gain_pass"]
    else:
        assert gain < acf.REGISTERED["p2_frame_gain_pass"]
    if row in ("A", "D"):
        assert bulk <= acf.REGISTERED["p2_bulk_tracks_ceiling"]
    elif row in ("B", "E"):
        assert bulk >= acf.REGISTERED["p2_bulk_elevated_floor"]
    else:
        assert acf.REGISTERED["p2_bulk_tracks_ceiling"] < bulk < acf.REGISTERED["p2_bulk_elevated_floor"]


def test_p2_is_one_on_an_unplanted_stream_and_is_a_frame_not_a_length_effect(cache):
    """Both sides are calibrated at their own n, so n = 45 vs n = 195 cancels."""
    rep = cached(cache, "p2null", lambda: core_entries(
        11, probes=64, tracked={"k": 16}))
    p2 = rep["p2"]
    assert p2["frame_gain"] == pytest.approx(1.0, abs=0.25)
    assert p2["bulk_gain"] == pytest.approx(1.0, abs=0.25)
    assert p2["outcome_row"] == "D"  # "no gain", the null's row
    assert p2["median_T_dc"]["frozen"] == pytest.approx(1.0, abs=0.10)
    assert rep["p1"]["outcome_row"] == 6  # a DC plant must not move P1
    # the tracked segments really are the short ones
    assert rep["estimator"]["n_kept"]["min"] == 45


def test_p2_needs_both_tiers_and_says_so_when_it_has_only_one(cache):
    rep = cached(cache, "p1row6", lambda: core_entries(4242))
    assert rep["p2"]["available"] is False
    assert "tracked" in rep["p2"]["reason"]


def test_p2_dc_plant_on_tracked_slots_does_not_create_a_p1_pass(cache):
    rep = cached(cache, "p2rowA", lambda: core_entries(
        11, probes=64, tracked={"k": 16, "dc_top": 0.45, "dc_bulk": 0.0}))
    assert rep["p1"]["outcome_row"] == 6
    assert rep["p3"]["verdict"] == "DECISIVE"  # the frozen stream is still AR(1)


# ==========================================================================
# 8. P3 -- DECISIVE / FAIL / UNDECIDED, and the consistency clause
# ==========================================================================

P3_STREAMS = {
    "ar1_m0.34": (-0.34, "DECISIVE"),
    "ar1_p0.20": (0.20, "FAIL"),
    "ar1_p0.50": (0.50, "FAIL"),
    "white": (0.0, "UNDECIDED"),
}


@pytest.mark.parametrize("label", sorted(P3_STREAMS))
def test_p3_every_registered_branch_from_its_own_stream(cache, label):
    phi, expected = P3_STREAMS[label]
    rep = cached(cache, "p3" + label, lambda: core_entries(31, phi=phi))
    p3 = rep["p3"]
    assert p3["verdict"] == expected, p3["by_K"]
    if expected != "UNDECIDED":
        assert p3["k_stable"] is True
        for k in (8, 16, 32, 64):
            assert p3["verdict_by_K"]["K%d" % k] == expected, k


def test_p3_asserts_the_estimator_reference_not_the_analytic_value(cache):
    """6d: "Assert the reference, not 3.0" -- the estimator is biased at phi > 0."""
    rep = cached(cache, "p3ar1_p0.50", lambda: core_entries(31, phi=0.50))
    assert rep["p3"]["tau_cal"] == pytest.approx(2.33, abs=0.10)
    assert rep["p3"]["tau_cal"] < 3.0  # the true tau; a correct pipeline is below it
    dec = cached(cache, "p3ar1_m0.34", lambda: core_entries(31, phi=-0.34))
    assert dec["p3"]["tau_cal"] == pytest.approx(0.658, abs=0.05)


def test_p3_white_noise_tau_cal_is_one_and_the_reference_seed_is_not_the_null_seed(cache):
    rep = cached(cache, "p3white", lambda: core_entries(31, phi=0.0))
    for k in (8, 16, 32, 64):
        assert rep["p3"]["by_K"]["K%d" % k]["tau_cal"] == pytest.approx(1.0, abs=K1_TOL), k
    # tau_white is a real reference, not a tautology: with a shared seed the
    # ratio would be exactly 1.000 by construction (bbp-prereg A2)
    assert rep["settings"]["tau_reference_seed"] != rep["settings"]["null_seed"]
    assert rep["p3"]["tau_cal"] != 1.0


def test_p3_reports_tau_hat_white_and_ar1_side_by_side_at_every_K(cache):
    rep = cached(cache, "p3ar1_m0.34", lambda: core_entries(31, phi=-0.34))
    for k in (8, 16, 32, 64):
        entry = rep["p3"]["by_K"]["K%d" % k]
        for key in ("tau_ar1", "tau_cal", "tau_cal_ci95", "tau_hat", "tau_white"):
            assert entry[key] is not None, (k, key)
    # the AR(1) reference reproduces the prereg's committed tau_AR1 ladder
    assert rep["p3"]["by_K"]["K8"]["tau_ar1"] < rep["p3"]["by_K"]["K64"]["tau_ar1"]


def test_p3_consistency_clause_holds_on_ar1_and_fails_on_a_long_memory_stream(cache):
    """The clause is a genuine test of the AR(1) SHAPE beyond lag 1.

    phi_hat is fitted from rho_1 alone, so agreement of the 32-lag sum is not
    automatic: a stream with slow power beyond the fitted AR(1) breaks it.
    """
    ar1 = cached(cache, "p3ar1_m0.34", lambda: core_entries(31, phi=-0.34))
    assert ar1["p3"]["consistency_holds"] is True
    assert ar1["p3"]["consistency_ratio"] == pytest.approx(1.0, abs=0.15)
    slow = cached(cache, "inflated", lambda: core_entries(4242, infl_dc=0.22, infl_alt=0.65))
    assert slow["p3"]["consistency_holds"] is False
    assert slow["p3"]["consistency_ratio"] > acf.REGISTERED["p3_consistency_band"][1]


# ==========================================================================
# 9. The riders (prereg 4 Rider-1 / Rider-2)
# ==========================================================================

RIDER1_STREAMS = {
    "PASS": {500: 0.05, 2000: 0.0727, 8000: 0.109},      # excess ~ sqrt(B)
    "FAIL_FLAT": {500: 0.05, 2000: 0.05, 8000: 0.05},    # excess flat in B
    "GUARD_FIRED": {500: 0.0, 2000: 0.05, 8000: 0.109},  # denominator pinned at 0
    "AMBIGUOUS": {500: 0.05, 2000: 0.06, 8000: 0.07},    # scaling, but not sqrt(B)
}


@pytest.mark.parametrize("branch", sorted(RIDER1_STREAMS))
def test_rider1_every_registered_branch_from_its_own_stream(cache, branch):
    rep = cached(cache, "rider1" + branch, lambda: rider_entries(21, RIDER1_STREAMS[branch]))
    r = rep["rider"]
    assert r["available"] is True
    assert r["rider1_branch"] == branch, r["excess_by_batch"]
    if branch == "GUARD_FIRED":
        assert r["vacuity_guard_fired"] is True
        assert r["ratio"] is None  # "excess unmeasurable at B = 500", not a number
        assert "unmeasurable" in r["vacuity_guard_note"]
    else:
        assert r["vacuity_guard_fired"] is False
        assert r["ratio"] is not None
    if branch == "PASS":
        lo, hi = acf.REGISTERED["rider1_pass_band"]
        assert lo <= r["ratio"] <= hi
    if branch == "FAIL_FLAT":
        assert r["ratio"] < acf.REGISTERED["rider1_fail_flat"]
    if branch == "AMBIGUOUS":
        assert acf.REGISTERED["rider1_fail_flat"] <= r["ratio"] < acf.REGISTERED["rider1_pass_band"][0]


def test_rider1_flags_the_b500_nyquist_epoch_harmonic_and_stays_descriptive(cache):
    rep = cached(cache, "rider1PASS", lambda: rider_entries(21, RIDER1_STREAMS["PASS"]))
    by_batch = rep["rider"]["by_batch"]
    assert by_batch["500"]["nyquist_is_epoch_harmonic"] is True
    assert by_batch["2000"]["nyquist_is_epoch_harmonic"] is False
    assert by_batch["8000"]["nyquist_is_epoch_harmonic"] is False
    # the B = 8000 rung is 192 steps, not 200: the shortfall is carried
    assert by_batch["8000"]["n_kept_median"] == 187
    assert by_batch["500"]["n_kept_median"] == 195
    # the descriptive P1 extension over the rider rungs is present and flagged
    for key, entry in rep["p1"]["by_batch"].items():
        assert entry["criterion"] is False and "DESCRIPTIVE" in entry["note"]
    assert rep["p1"]["by_batch"]["500"]["nyquist_is_epoch_harmonic"] is True


def test_rider1_pool_is_the_rider_seeds_and_never_the_core(cache):
    rep = cached(cache, "rider1PASS", lambda: rider_entries(21, RIDER1_STREAMS["PASS"]))
    assert rep["inputs"]["pools"]["core"]["n_runs"] == 0
    assert rep["inputs"]["pools"]["rider"]["n_runs"] == 6
    assert rep["p1"]["available"] is False  # no core pool -> no P1 verdict at all
    core = cached(cache, "p1row6", lambda: core_entries(4242))
    assert core["rider"]["available"] is False
    assert "rider-pool" in core["rider"]["reason"]


RIDER2_STREAMS = {
    "B_INVARIANT": {500: -0.34, 2000: -0.34, 8000: -0.34},
    "SAMPLING_CONSISTENT": {500: -0.34, 2000: -0.20, 8000: -0.02},
    "MIXED": {500: -0.34, 2000: -0.02, 8000: -0.34},
}


@pytest.mark.parametrize("branch", sorted(RIDER2_STREAMS))
def test_rider2_every_registered_branch_from_its_own_stream(cache, branch):
    rep = cached(cache, "rider2" + branch, lambda: rider_entries(
        23, {500: 0.0, 2000: 0.0, 8000: 0.0}, RIDER2_STREAMS[branch], probes=96))
    r = rep["rider"]
    assert r["rider2_branch"] == branch, r["ess_over_n_by_batch"]
    if branch == "B_INVARIANT":
        assert r["ess_over_n_max_over_min"] < acf.REGISTERED["rider2_invariance_max_over_min"]
    if branch == "SAMPLING_CONSISTENT":
        assert r["rider2_ess_monotone_decreasing"] is True
        assert float(r["ess_over_n_by_batch"]["8000"]) == pytest.approx(1.0, abs=0.15)
    if branch == "MIXED":
        assert r["rider2_ess_monotone_decreasing"] is False
        assert r["ess_over_n_max_over_min"] >= acf.REGISTERED["rider2_invariance_max_over_min"]
    # phi_hat by batch is the other registered rider-2 output
    assert sorted(r["phi_by_batch"]) == ["2000", "500", "8000"]


# ==========================================================================
# 10. The section 6b output keys, the sensitivities and the diagnostics
# ==========================================================================


def test_every_registered_output_key_of_the_6b_table_is_present(cache):
    rep = cached(cache, "p2rowA", lambda: core_entries(
        11, probes=64, tracked={"k": 16, "dc_top": 0.45, "dc_bulk": 0.0}))
    for key in ("cells", "channels", "descriptive", "diagnostics", "estimator",
                "ladder", "null", "p1", "p2", "p3", "rider", "runs",
                "sensitivity", "registered", "settings", "inputs"):
        assert key in rep, key
    est = rep["estimator"]
    assert est["burn_in"] == 5 and est["n_kept"]["median"] is not None
    assert est["series_parity"]["parity"] == "odd"
    for key in ("by_matrix", "ci95", "point", "rho_1_raw"):
        assert key in est["phi_hat"], key
    assert len(rep["ladder"]["frozen"]["rho"]) == 64
    assert len(rep["ladder"]["frozen"]["rho_raw"]) == 64
    assert rep["ladder"]["frozen"]["rho"][0] != rep["ladder"]["frozen"]["rho_raw"][0]
    for ch in ("alt", "dc"):
        assert rep["channels"][ch]["t_nw"]["median"] is not None
        assert rep["channels"][ch]["ess"]["median"] is not None
        assert rep["channels"][ch]["T"]["median"] is not None
        assert rep["channels"][ch]["n_nw_floored"] == 0
        assert rep["channels"][ch]["nw_floored"] == {
            "frac": 0.0, "n": 0, "n_probes": 3 * 2 * 64}
    null = rep["null"]["frozen/m.0.weight/b2000"]
    assert set(null) == {"alt", "dc"}
    assert null["dc"]["195"]["samples"]["reps"] == 8000
    assert null["dc"]["195"]["abs_t_nw"]["median"] is not None
    assert rep["descriptive"]["tier_contrast"]["calibrated_nw_median_T_dc"]["top"] is not None
    assert rep["descriptive"]["ess_over_n"]["published_anchor"]["median"] == 1.9495
    assert rep["diagnostics"]["n_nw_floored"]["by_channel"]["alt"]["k2_fires"] is False


def test_the_burn_in_and_phi_sensitivities_are_reported_next_to_the_quantities():
    """prereg 5.3 and 5.5: both sweeps travel with every registered number."""
    rep = report_of(core_entries(4242, probes=48, broad_alt=0.25),
                    extra=["--burn-ins", "5", "15", "25"])
    sens = rep["sensitivity"]
    assert sorted(sens["burn_in"]) == ["15", "25", "5"]
    for b, entry in sens["burn_in"].items():
        for key in ("band_contrast", "p1_outcome_row", "phi_hat", "ratio_alt", "tau_cal"):
            assert key in entry, (b, key)
    assert sens["burn_in"]["5"]["band_contrast"] == rep["p1"]["band_contrast"]
    assert sorted(sens["phi"]) == ["phi_fixed_-0.34", "phi_hat+0.05", "phi_hat-0.05"]
    for entry in sens["phi"].values():
        # reported on BOTH channels (repair R7 disclosed the wrong one)
        assert entry["ratio_alt"] is not None and entry["ratio_dc"] is not None
    lo = sens["phi"]["phi_hat-0.05"]["ratio_alt"]
    hi = sens["phi"]["phi_hat+0.05"]["ratio_alt"]
    assert lo != hi  # the null really was redrawn


def test_k3_and_k4_diagnostics_report_and_can_fire(cache):
    rep = cached(cache, "p1row6", lambda: core_entries(4242))
    k3 = rep["diagnostics"]["k3_phi"]
    assert set(k3["by_matrix"]) == {"m.0.weight", "m.1.weight"}
    assert k3["window"] == [-0.60, -0.15]
    assert k3["k3_fires"] is False  # phi_hat ~ -0.38 in both matrices
    assert rep["diagnostics"]["k4_frame_gain_denominator"]["k4_fires"] is False
    # K3 fires on a matrix outside the window, K4 on an inflated frozen tier
    out = report_of(core_entries(5, probes=48, phi=-0.75))
    assert out["diagnostics"]["k3_phi"]["k3_fires"] is True
    # K4's object: a frozen DC channel that is itself carrying something
    hot = report_of(core_entries(77, probes=64, dc=0.10))
    k4 = hot["diagnostics"]["k4_frame_gain_denominator"]
    assert k4["median_T_dc_frozen"] > acf.REGISTERED["k4_frozen_median_t_dc_max"]
    assert k4["k4_fires"] is True


def test_k2_is_inert_on_these_streams_and_its_trigger_is_still_exercised(cache):
    """prereg 7 K2, registered as expected-inert on dc (published 0 / 864).

    The Bartlett kernel makes the truncated long-run variance PSD in the
    population, so flooring is a finite-sample event that no stationary stream
    plants on demand -- which is exactly why the clause is registered as a
    guard rather than as a live test.  The rate is reported per channel and per
    pool, and the trigger itself is exercised directly on a table that carries
    the flag.
    """
    rep = cached(cache, "p1row6", lambda: core_entries(4242))
    diag = rep["diagnostics"]["n_nw_floored"]
    for ch in ("alt", "dc"):
        assert diag["by_channel"][ch]["n_floored"] == 0
        assert diag["by_channel"][ch]["frac"] == 0.0
        assert diag["by_channel"][ch]["k2_fires"] is False
    assert diag["k2_fires"] is False and diag["bar"] == acf.REGISTERED["k2_nw_floored_frac"]
    # the trigger fires above the registered 5%, and not below it
    n = 100
    table = {ch: {"nw_floored": np.zeros(n, dtype=bool)} for ch in ("alt", "dc")}
    table["alt"]["nw_floored"][:6] = True  # 6% > 5%
    table["dc"]["nw_floored"][:4] = True   # 4% < 5%
    fired = acf.nw_floored_diagnostic([{}] * n, table, range(n))
    assert fired["by_channel"]["alt"]["k2_fires"] is True
    assert fired["by_channel"]["alt"]["frac"] == 0.06
    assert fired["by_channel"]["dc"]["k2_fires"] is False
    assert fired["k2_fires"] is True


# ==========================================================================
# 11. End to end through main(), on JSON sidecars, byte-identically
# ==========================================================================


def _write_fixture(tmp_path, entries):
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    for meta, log in entries:
        (results / (meta["run"] + SIDECAR_SUFFIX)).write_text(
            json.dumps(log, indent=1, sort_keys=True) + "\n"
        )
        (results / (meta["run"] + ".json")).write_text(json.dumps({
            "config": {"contents": {"probe_overrides": {"lr": meta["lr"]},
                                    "train": {"batch_size": meta["batch_size"]}}},
            "metrics": {"instrumentation_sidecar": meta["run"] + SIDECAR_SUFFIX},
            "seed": meta["seed"],
        }, indent=1, sort_keys=True) + "\n")
    return results


def _run_main(tmp_path, results, tag):
    out_json = tmp_path / ("%s.json" % tag)
    out_md = tmp_path / ("%s.md" % tag)
    figdir = tmp_path / ("figs_%s" % tag)
    argv = [
        "--sidecars", str(results), "--out-json", str(out_json),
        "--out-md", str(out_md), "--out-figdir", str(figdir),
        "--null-reps", "3000", "--bootstrap-reps", "150", "--burn-ins", "5",
        "--no-controls",
    ]
    assert acf.main(argv) == 0
    figs = sorted(p.name for p in figdir.glob("*.png"))
    return json.loads(out_json.read_text()), out_md.read_text(), figdir, figs


@pytest.fixture(scope="module")
def end_to_end(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("frozen_e2e")
    entries = core_entries(101, probes=48, tracked={"k": 8, "dc_top": 0.45})
    entries += rider_entries(102, {500: 0.05, 2000: 0.0727, 8000: 0.109}, probes=48)
    results = _write_fixture(tmp, entries)
    return tmp, results, _run_main(tmp, results, "first")


def test_end_to_end_reads_sidecars_and_labels_every_pool(end_to_end):
    _tmp, _results, (report, md, _figdir, figs) = end_to_end
    assert report["inputs"]["n_runs"] == 9
    assert report["inputs"]["pools"]["core"]["n_runs"] == 3
    assert report["inputs"]["pools"]["rider"]["n_runs"] == 6
    assert report["p1"]["available"] and report["p2"]["available"]
    assert report["p3"]["available"] and report["rider"]["available"]
    assert {r["run"] for r in report["runs"]} == {
        m["run"] for m, _l in core_entries(101, probes=1)
    } | {m["run"] for m, _l in rider_entries(102, {500: 0.0, 2000: 0.0, 8000: 0.0}, probes=1)}
    mc = report["diagnostics"]["mirror_check"]
    assert mc["n_checked"] == 64  # the deterministic sample of logged finals
    assert mc["max_abs_dev_t_nw"] < 1e-9 and mc["max_abs_dev_ess"] < 1e-9
    assert figs == ["channel-audit-channel-shape.png", "channel-audit-ladder.png",
                    "channel-audit-rider.png", "channel-audit-tau.png"]


def test_end_to_end_is_byte_identical_on_a_rerun(end_to_end):
    """Module contract: identical inputs -> identical outputs, no timestamps."""
    tmp, results, (first_json, first_md, first_figdir, first_figs) = end_to_end
    second_json, second_md, second_figdir, second_figs = _run_main(tmp, results, "second")
    assert first_md == second_md
    assert json.dumps(first_json, sort_keys=True) == json.dumps(second_json, sort_keys=True)
    assert first_figs == second_figs
    for name in first_figs:
        assert (first_figdir / name).read_bytes() == (second_figdir / name).read_bytes(), name
    assert "20260" not in first_md.split("\n")[0]


def test_the_report_is_descriptive_and_never_adjudicates(end_to_end):
    """prereg 6b: quantities next to thresholds, and nothing resembling a verdict."""
    _tmp, _results, (report, md, _figdir, _figs) = end_to_end
    assert "Descriptive only" in md
    assert "adjudication is HUMAN" in md
    assert "FROZEN" in md and "FROZEN" in report["registered"]["status"]
    assert "LOOKUP, not a verdict" in md
    assert report["registered"]["thresholds"]["p1_band_contrast_pass"] == 1.30
    assert report["registered"]["pooling"]["tau"] == "mean"
    # the numbers are printed next to their proposed bars
    assert "band_contrast" in md and "frame_gain" in md and "tau_cal" in md
    assert "Registered outcome-map row" in md


def test_end_to_end_recovers_every_plant_it_was_given(end_to_end):
    """The fixture plants a tracked-`top` DC mean and a sqrt(B) rider."""
    _tmp, _results, (report, _md, _figdir, _figs) = end_to_end
    assert report["p2"]["outcome_row"] == "A"  # gain, bulk frozen
    assert report["p2"]["frame_gain"] >= acf.REGISTERED["p2_frame_gain_pass"]
    assert report["p1"]["outcome_row"] == 6  # nothing planted in the band
    assert report["p3"]["verdict"] == "DECISIVE"  # the AR(1) stream
    assert report["rider"]["rider1_branch"] == "PASS"
    lo, hi = acf.REGISTERED["rider1_pass_band"]
    assert lo <= report["rider"]["ratio"] <= hi
