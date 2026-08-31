"""Synthetic-signal suite for scripts/analyze_channel_audit.py -- CPU only.

``reports/channel-audit-preregistration.md`` §6d calls a suite like this "a
launch precondition, not a nicety", and states the standard it has to meet:
**every registered criterion must be shown to produce both of its branches on
synthetic data, and each branch only from the stream that should produce it.**
A control that cannot fail measures nothing.  This file covers the Phase A
producer; §6b's frozen-tier producer does not exist yet and gets its own file.

What is pinned here:

* the argument parser's defaults, INCLUDING the three that deliberately differ
  from the registration -- a default that silently drifts back or further away
  is how a report ends up quoting neither quantity (§6b);
* the Phase A / Phase B output-path guard;
* segmentation (``boundaries_from_steps``, ``direction_groups``) and the
  mixed-parity refusal;
* the batched kernel against ``src.stats.spectral`` at the segment level AND
  the slot level, on ragged shapes;
* K1 (zero-mean -> ratio 1 in both channels at phi in {0, -0.34, +0.5}) with
  its FAIL branch, K2a (planted alternating mean), K2b (planted per-slot DC
  mean) and **K3, the zero-mean stream whose per-segment reading is
  indistinguishable from K2b's** -- the identification limit of ``ratio``, and
  the slot-level growth factor that separates them;
* the null mixture over a cell's actual segment lengths (never their median);
* the bootstrap rep count on a large pool (it must not be silently reduced);
* the ratio interval carrying the null's Monte-Carlo error;
* ``tau_nw`` against the null's own ESS/n;
* the Phase A reproduction ledger reporting its obligation as OPEN;
* run selection: tombstones, run family, and the seed floor (ground rule 2).

Dev seeds (>= 1000) throughout, no GPU, no torch, no network.  Tolerances here
are test conventions, not decision thresholds (CLAUDE.md ground rule 1).
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
from src.stats.spectral import (
    ar1_streams,
    channel_t,
    lag_ladder,
    newey_west_bandwidth,
    segment_mean_persistence,
    segmented_channel_t,
)


def _load_script():
    path = REPO_ROOT / "scripts" / "analyze_channel_audit.py"
    spec = importlib.util.spec_from_file_location("analyze_channel_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aca = _load_script()

RATIO_TOL = 0.15  # K1: a null-calibrated ratio must sit at 1 within this
SEED = 20260831  # dev seed, > 1000


# ==========================================================================
# 1. Registered defaults and the Phase A / Phase B path guard
# ==========================================================================


def _defaults():
    return vars(aca.build_parser().parse_args([]))


def test_output_paths_default_to_the_phase_a_names():
    """Prereg §6a registers -phase-a; §6b reserves the bare names."""
    d = _defaults()
    assert d["out_md"] == Path("reports/channel-audit-phase-a.md")
    assert d["out_json"] == Path("reports/channel-audit-phase-a.json")
    assert d["allow_phase_b_path"] is False


@pytest.mark.parametrize("name", ["channel-audit.md", "channel-audit.json"])
def test_phase_b_reserved_paths_are_refused(name):
    with pytest.raises(SystemExit, match="Phase B"):
        aca._check_out_path(Path("reports") / name, allow=False)
    aca._check_out_path(Path("reports") / name, allow=True)  # explicit opt-in
    aca._check_out_path(Path("reports/channel-audit-phase-a.md"), allow=False)


def test_registered_defaults_and_their_disclosed_deviations():
    """The defaults are asserted, not trusted (prereg §6b).

    Three of them differ from the registration on purpose; each is named in
    the script's DEVIATIONS block.  If one of them moves -- in either
    direction -- this test fails and the docstring has to move with it.
    """
    d = _defaults()
    assert d["primary_burn_in"] == 5  # §5.3, registered
    assert d["null_seed"] == 4242 and d["bootstrap_seed"] == 4242  # §5.6/§5.8
    assert d["bootstrap_reps"] == 2000  # §5.8, registered
    assert d["max_lag"] == 8
    assert d["segment_at"] == "refresh"
    assert d["min_seed"] == 1000  # CLAUDE.md ground rule 2
    # deviation (a): registered reps = 200000
    assert d["null_reps"] == 2000
    # deviation (b): registered block = the 4 segments of one direction
    assert d["bootstrap_block"] == 16 and d["bootstrap_block_alt"] == 4
    # deviation (c): burn-in 0 carried on top of the registered sweep
    burn_ins = sorted(int(b) for b in d["burn_ins"].split(","))
    assert burn_ins == [0, 5, 15, 25]
    assert set(aca.REGISTERED_BURN_INS) == {5, 15, 25}
    assert set(aca.REGISTERED_BURN_INS).issubset(burn_ins)


def test_deviations_block_names_every_deviating_default():
    """The docstring must mention each deviating flag by name."""
    doc = aca.__doc__
    assert "DEVIATIONS FROM THE DRAFT PRE-REGISTRATION" in doc
    for token in ("--null-reps", "--bootstrap-block", "Burn-in 0", "200000"):
        assert token in doc, token


def test_the_ar1_generator_is_the_tested_one_not_a_copy():
    """prereg §5 preamble: the stats module is the tested code."""
    import src.stats.spectral as spectral

    assert aca.ar1_streams is spectral.ar1_streams
    assert not hasattr(aca, "_ar1_block")


# ==========================================================================
# 2. Segmentation
# ==========================================================================


def test_boundaries_from_steps_cuts_at_the_refresh_and_keeps_remainders():
    steps = np.arange(0, 25)
    assert aca.boundaries_from_steps(steps, [10, 20]) == ((0, 10), (10, 20), (20, 25))
    # a cut outside the recorded range is ignored, never an empty segment
    assert aca.boundaries_from_steps(steps, [0, 10, 25, 900]) == ((0, 10), (10, 25))
    assert aca.boundaries_from_steps(steps, []) == ((0, 25),)
    for a, b in aca.boundaries_from_steps(steps, [3, 3, 7]):
        assert b > a


def test_direction_groups_refresh_is_one_group_and_reset_splits():
    steps = np.arange(0, 30)
    mat = {
        "refresh_steps": [0, 10, 20],
        "directions": [
            {"reset_steps": [10]},
            {"reset_steps": [10]},
            {"reset_steps": [10, 20]},
        ],
    }
    groups = aca.direction_groups(mat, steps, "refresh")
    assert len(groups) == 1 and groups[0][1] == [0, 1, 2]
    assert groups[0][0] == ((0, 10), (10, 20), (20, 30))
    groups = aca.direction_groups(mat, steps, "reset")
    assert sorted(members for _b, members in groups) == [[0, 1], [2]]


def test_mirror_sampler_covers_the_stream_and_does_not_alias():
    """Not the first N, and not a fixed stride either.

    The block stream is periodic (groups of 1, 2, 4, 8, 16 segments x 4
    burn-ins), so any power-of-two stride aliases against it: a measured
    evenly spaced 64-block sample landed on raw length 50 all 64 times and
    never checked a short segment.  Seeded reservoir sampling has no phase
    to align with.
    """
    s = aca.DeterministicSampler(16, 4242)
    for i in range(2000):
        s.offer(i)
    got = s.sample()
    assert len(got) == 16
    assert got == sorted(got)  # returned in stream order
    assert got[0] > 0 and got[-1] < 1999  # not the head, not the tail
    assert aca.DeterministicSampler(16, 4242).sample() == []
    # deterministic given the seed, and different for a different one
    def draw(seed, n=2000):
        t = aca.DeterministicSampler(16, seed)
        for j in range(n):
            t.offer(j)
        return t.sample()
    assert draw(4242) == got and draw(4243) != got
    # a periodic 6%-rare subpopulation is reached; a power-of-two stride
    # over the same stream can miss it entirely
    big = aca.DeterministicSampler(64, 4242)
    for i in range(2048):
        big.offer(i)
    assert [g for g in big.sample() if g % 32 == 7], "missed a subpopulation"
    stride = [i for i in range(0, 2048, 2048 // 64)]
    assert not [g for g in stride if g % 32 == 7]  # the aliasing it replaces
    # under budget: everything survives
    u = aca.DeterministicSampler(8, 4242)
    for i in range(5):
        u.offer(i)
    assert u.sample() == [0, 1, 2, 3, 4]
    assert aca.DeterministicSampler(0, 4242).sample() == []


# ==========================================================================
# 3. Mirror contract: the batched kernel is not a second estimator
# ==========================================================================


def _parity(n, start=0):
    return np.where((np.arange(start, start + n)) % 2 == 0, 1.0, -1.0)


@pytest.mark.parametrize("burn_in", [0, 3])
@pytest.mark.parametrize("n", [23, 50])
def test_segment_block_matches_src_stats_spectral(burn_in, n):
    rng = np.random.default_rng(1234 + n)
    raw = rng.standard_normal((5, n)) + 0.4
    parity = _parity(n, start=1)
    block = aca.segment_block(raw, parity, burn_in, 4)
    for row in range(raw.shape[0]):
        ladder = lag_ladder(raw[row], 4, burn_in)
        assert block["rho"][row] == pytest.approx(ladder["rho"], abs=1e-12)
        assert block["rho_raw"][row] == pytest.approx(ladder["rho_raw"], abs=1e-12)
        ref_dc = channel_t(raw[row], "dc", burn_in, max_lag=4)
        ref_alt = channel_t(raw[row] * parity, "dc", burn_in, max_lag=4)
        assert block["dc"]["t_nw"][row] == pytest.approx(ref_dc["t_nw"], abs=1e-12)
        assert block["alt"]["t_nw"][row] == pytest.approx(ref_alt["t_nw"], abs=1e-12)
        assert block["dc"]["ess"][row] == pytest.approx(ref_dc["ess"], rel=1e-12)


def test_slot_accumulator_matches_segmented_channel_t_on_ragged_shapes():
    """Ragged shapes are the real case: (50, 50, 50, 42) occurs on disk."""
    rng = np.random.default_rng(4321)
    lengths = [20, 20, 14]
    segs = [rng.standard_normal((4, L)) + 0.3 for L in lengths]
    start = 1
    parities, pos = [], start
    for L in lengths:
        parities.append(_parity(L, start=pos))
        pos += L
    acc = aca.SlotAccumulator(4, 4)
    for raw, par in zip(segs, parities):
        acc.add(aca.segment_block(raw, par, 2, 4))
    got = acc.finish()
    assert got["n"] == sum(L - 2 for L in lengths)
    for row in range(4):
        for ch, series in (
            ("dc", [raw[row] for raw in segs]),
            ("alt", [raw[row] * par for raw, par in zip(segs, parities)]),
        ):
            ref = segmented_channel_t(series, "dc", 2, max_lag=4)
            per = segment_mean_persistence(ref["segment_means"])
            assert got[ch]["t_nw"][row] == pytest.approx(ref["t_nw"], abs=1e-11)
            assert got[ch]["ess"][row] == pytest.approx(ref["ess"], rel=1e-11)
            assert got["lag_truncation"] == ref["lag_truncation"]
            assert got[ch]["segment_mean_coherence"][row] == pytest.approx(
                per["coherence"]
            )
            assert got[ch]["segment_mean_acf1"][row] == pytest.approx(per["acf1"])


def test_raw_ladder_of_a_variance_floored_segment_reads_zero_not_minus_one_over_n():
    """DirectionStats convention: a constant segment has undefined rho.

    Reconstructing the raw ladder downstream as ``rho - 1/n`` would report
    -1/n for a row the estimator deliberately left at exactly 0.
    """
    raw = np.zeros((2, 30))
    raw[1] = np.random.default_rng(7).standard_normal(30)
    block = aca.segment_block(raw, _parity(30), 5, 4)
    assert np.all(block["rho"][0] == 0.0)
    assert np.all(block["rho_raw"][0] == 0.0)
    assert block["rho_raw"][1] == pytest.approx(block["rho"][1] - 1.0 / 25)


# ==========================================================================
# 4. K1 / K2 / K3 -- the estimator controls and their two branches
# ==========================================================================

CTRL = {"reps": 1500, "burn_in": 5, "max_lag": 8, "seg_len": 50, "n_segments": 6}


@pytest.fixture(scope="module")
def controls():
    nulls = aca.NullBank(2000, 4242, CTRL["max_lag"])
    return aca.self_test(
        CTRL["reps"], CTRL["burn_in"], CTRL["max_lag"], CTRL["seg_len"],
        CTRL["n_segments"], nulls,
    )


@pytest.mark.parametrize("label", ["K1a phi=0.00", "K1b phi=-0.34", "K1c phi=+0.50"])
@pytest.mark.parametrize("ch", ["alt", "dc"])
def test_k1_zero_mean_streams_calibrate_to_ratio_one(controls, label, ch):
    """Nothing to detect -> ratio 1 in BOTH channels, whatever the memory."""
    e = controls[label][ch]
    assert e["ratio"] == pytest.approx(1.0, abs=RATIO_TOL)
    assert e["slot_growth_calibrated"] == pytest.approx(1.0, abs=0.25)


def test_k1_tau_branches_on_the_sign_of_phi(controls):
    """tau > 1 iff the stream is positively autocorrelated -- both branches."""
    assert controls["K1a phi=0.00"]["tau_4"] == pytest.approx(1.0, abs=0.1)
    assert controls["K1c phi=+0.50"]["tau_4"] > 2.0  # true 1+2*sum phi^k = 2.94
    assert controls["K1b phi=-0.34"]["tau_4"] < 0.7
    assert controls["K1c phi=+0.50"]["rho_1_hat"] > 0.4
    assert controls["K1b phi=-0.34"]["rho_1_hat"] < -0.28


def test_k2a_planted_alternating_mean_moves_alt_only(controls):
    """The FAIL branch of K1: a planted signal must break ratio = 1."""
    e = controls["K2a phi=-0.34 +alt mean"]
    assert e["alt"]["ratio"] > 2.0
    assert e["dc"]["ratio"] == pytest.approx(1.0, abs=RATIO_TOL)


def test_k2b_planted_per_slot_dc_mean_moves_dc_and_grows_like_sqrt_k(controls):
    e = controls["K2b phi=-0.40 +slot dc mean"]
    assert e["dc"]["ratio"] > 2.0
    assert e["alt"]["ratio"] == pytest.approx(1.0, abs=RATIO_TOL)
    root_k = float(np.sqrt(CTRL["n_segments"]))
    assert e["dc"]["slot_growth_calibrated"] == pytest.approx(root_k, rel=0.25)
    # a mean that survives a refresh makes every segment mean share a sign
    assert e["dc"]["slot_segment_mean_coherence"] > 0.9
    assert e["dc"]["slot_segment_mean_acf1"] > 0.3


def test_k3_zero_mean_slow_power_is_indistinguishable_per_segment(controls):
    """The identification limit of ``ratio`` (spectral module docstring).

    K3 has NO mean in either channel.  At the per-segment window its reading
    is the reported `top` cell: large dc ratio, alt at the null.  If this ever
    stops being true the report's central caveat is stale.
    """
    k2b = controls["K2b phi=-0.40 +slot dc mean"]
    k3 = controls["K3 zero-mean slow power"]
    assert k3["planted_alt_mean"] == 0.0 and k3["planted_slot_dc_mean"] == 0.0
    assert k3["dc"]["ratio"] > 2.0
    assert k3["alt"]["ratio"] == pytest.approx(1.0, abs=RATIO_TOL)
    assert k3["dc"]["ratio"] == pytest.approx(k2b["dc"]["ratio"], rel=0.35)


def test_the_slot_growth_factor_separates_k3_from_k2b(controls):
    """...and the slot-level statistic is what tells them apart."""
    k2b = controls["K2b phi=-0.40 +slot dc mean"]
    k3 = controls["K3 zero-mean slow power"]
    assert k3["dc"]["slot_growth_calibrated"] < 1.3
    assert k2b["dc"]["slot_growth_calibrated"] > 2.0
    assert (k2b["dc"]["slot_growth_calibrated"]
            > 2.0 * k3["dc"]["slot_growth_calibrated"])


# ==========================================================================
# 5. Bootstrap, null mixture and the ratio interval
# ==========================================================================


def test_bootstrap_does_not_reduce_reps_on_a_large_pool():
    """A 95% interval from 50 draws is the 2nd order statistic at each end."""
    v = np.random.default_rng(11).standard_normal(200_000)
    got = aca._bootstrap(v, 16, 300, 4242, max_elems=2_000_000)
    assert got["reps"] == 300
    # chunking is a memory bound only: the answer must not depend on it
    tight = aca._bootstrap(v, 16, 300, 4242, max_elems=200_000)
    assert got == tight


def test_null_mixture_uses_every_length_present_and_never_their_median():
    """b04000 cells hold raw lengths 46 and 50 -- the median 48 occurs never."""
    nulls = aca.NullBank(400, 4242, 8)
    weights = [(46, 300), (50, 300)]
    mix = aca._null_mixture(nulls, -0.34, weights, 5, "dc")
    assert sorted(mix["by_segment_len"]) == ["46", "50"]
    assert "48" not in mix["by_segment_len"]
    assert mix["by_segment_len"]["46"]["weight"] == pytest.approx(0.5)
    assert mix["abs_t"].size == pytest.approx(400, abs=2)


def test_slot_null_is_drawn_at_the_exact_ragged_shape():
    """A truncated last window makes shapes like (50, 50, 50, 42) real.

    Matching the null to the slot's first, modal or median segment length
    instead of its actual shape is the same defect the per-segment mixture
    repairs, one level up.
    """
    nulls = aca.NullBank(200, 4242, 4)
    ragged = nulls.get_slot(-0.34, (50, 50, 50, 42), 4, 5)
    assert ragged["seg_len"] == [50, 50, 50, 42]
    assert ragged["n"] == 45 + 45 + 45 + 37
    square = nulls.get_slot(-0.34, 50, 4, 5)
    assert square["n"] == 4 * 45
    assert ragged["dc"]["abs_t_nw"]["median"] != square["dc"]["abs_t_nw"]["median"]
    # scalar and tuple spellings of the same shape are the same null
    assert nulls.get_slot(-0.34, (50, 50, 50, 50), 4, 5) is square
    # a shape that is all burn-in has no null at all
    assert nulls.get_slot(-0.34, (5, 5), 2, 5) is None
    assert nulls.get_slot(None, 50, 4, 5) is None


def test_ratio_interval_carries_the_null_monte_carlo_error():
    """finding: `ratio` was printed with the numerator's error only."""
    num = {"point": 1.9, "se": 0.029}
    den = {"point": 0.664, "se": 0.0}
    only_num = aca._ratio_with_error(num, den)
    both = aca._ratio_with_error(num, {"point": 0.664, "se": 0.015})
    assert both["ratio"] == pytest.approx(1.9 / 0.664)
    assert both["se"] > only_num["se"]
    width = both["ci95"][1] - both["ci95"][0]
    assert width > (only_num["ci95"][1] - only_num["ci95"][0])
    assert both["denominator_rel_se"] > both["numerator_rel_se"]
    assert aca._ratio_with_error(num, None) is None
    assert aca._ratio_with_error(num, {"point": 0.0, "se": 0.1}) is None


# ==========================================================================
# 6. End to end: synthetic sidecars through main()
# ==========================================================================

N_STEPS, T_REFRESH, K1, K2, BURN = 80, 20, 4, 4, 5


def _write_run(results: Path, run: str, lr: float, batch: int, seed: int, series):
    """One synthetic sidecar + its results JSON.

    ``series`` is (n_directions, N_STEPS); directions 0..K1-1 are `top`.
    """
    steps = list(range(N_STEPS))
    per_beta = {
        "0.9": {
            k: [0.0] * N_STEPS
            for k in ("step", "regime", "mu", "var", "rho", "t_stat",
                      "amplitude_ratio", "implied_eta_lambda", "ess",
                      "n_since_reset")
        }
    }
    dirs = []
    for j in range(series.shape[0]):
        dirs.append({
            "index": j,
            "kind": "top" if j < K1 else "bulk",
            "s": [float(v) for v in series[j]],
            "reset_steps": list(range(T_REFRESH, N_STEPS, T_REFRESH)),
            "refresh_alignment": [],
            "sigma": [],
            "lambda_hvp": [],
            "per_beta": per_beta,
        })
    log = {
        "instrumentation_schema_version": 2,
        "betas": [0.9],
        "hvp_enabled": False,
        "matrices": {
            "layer.weight": {
                "shape": [8, 8],
                "k1": K1,
                "k2": K2,
                "t_refresh": T_REFRESH,
                "steps": steps,
                "grad_fro_norm": [1.0] * N_STEPS,
                "top_sigma_m": [1.0] * N_STEPS,
                "refresh_steps": list(range(0, N_STEPS, T_REFRESH)),
                "directions": dirs,
            }
        },
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{run}{SIDECAR_SUFFIX}").write_text(
        json.dumps(log, indent=1, sort_keys=True) + "\n"
    )
    (results / f"{run}.json").write_text(json.dumps({
        "seed": seed,
        "config": {"contents": {"probe_overrides": {"lr": lr},
                                "train": {"batch_size": batch}}},
        "metrics": {"instrumentation_sidecar": f"{run}{SIDECAR_SUFFIX}"},
    }, indent=1, sort_keys=True) + "\n")


def _planted(rng, n_dirs, *, top_dc=0.0, alt=0.0):
    """AR(1) phi = -0.34 plus optional planted components.

    ``top_dc`` is a per-slot DC mean on `top` slots only (constant across
    refreshes, so it survives the slot-level pooling); ``alt`` is a period-2
    component on every slot, tied to the ABSOLUTE step.
    """
    x = ar1_streams(rng, -0.34, N_STEPS, n_dirs)
    if top_dc:
        x[:K1] += top_dc * rng.choice([-1.0, 1.0], size=(K1, 1))
    if alt:
        x += alt * np.where(np.arange(N_STEPS) % 2 == 0, 1.0, -1.0)[None, :]
    return x


def _run_main(tmp_path, extra=()):
    out_md = tmp_path / "out.md"
    out_json = tmp_path / "out.json"
    argv = [
        "--sidecars", str(tmp_path / "results"),
        "--out-md", str(out_md), "--out-json", str(out_json),
        "--burn-ins", "0,5", "--primary-burn-in", "5", "--min-n", "10",
        "--max-lag", "4", "--null-reps", "600", "--bootstrap-reps", "120",
        "--self-test-reps", "0", "--verify-blocks", "6",
        *extra,
    ]
    assert aca.main(argv) == 0
    return json.loads(out_json.read_text()), out_md.read_text()


@pytest.fixture(scope="module")
def planted_report(tmp_path_factory):
    """Frame gain planted on `top` only, at two batch sizes."""
    tmp = tmp_path_factory.mktemp("planted")
    rng = np.random.default_rng(SEED)
    for i, (lr, batch) in enumerate([(0.5, 500), (0.5, 1000), (0.9, 500)]):
        for rep in range(2):
            _write_run(
                tmp / "results",
                f"airbench_instrumented_seed{1400 + i * 2 + rep}_2026{i}{rep}",
                lr, batch, 1400 + i * 2 + rep,
                _planted(rng, K1 + K2, top_dc=0.55),
            )
    return _run_main(tmp)


def test_mirror_check_covers_top_slots_and_every_segment_length(planted_report):
    """The coverage the sampler exists to produce, read off the output."""
    report, _md = planted_report
    mc = report["diagnostics"]["mirror_check"]
    assert mc["n_checked"] == 6 and mc["n_slot_checked"] >= 1
    positions = mc["slot_positions_checked"]
    assert any(p < K1 for p in positions)  # a `top` slot
    assert any(p >= K1 for p in positions)  # a `bulk` slot
    assert mc["segment_lengths_checked"] == [T_REFRESH]
    for key, value in mc.items():
        if key.startswith("max_abs_dev_"):
            assert value < 1e-9, key


def test_end_to_end_recovers_the_planted_frame_gain(planted_report):
    """P2's shape: a DC mean on `top` only -> top ratio up, bulk ratio at 1."""
    report, md = planted_report
    top = report["strata"]["by_kind"]["top/lrALL/bALL"]["5"]
    bulk = report["strata"]["by_kind"]["bulk/lrALL/bALL"]["5"]
    assert top["channels"]["dc"]["ratio"] > 1.6
    assert bulk["channels"]["dc"]["ratio"] == pytest.approx(1.0, abs=0.25)
    assert top["channels"]["alt"]["ratio"] == pytest.approx(1.0, abs=0.25)
    assert "## 7. Slot-level read" in md
    assert top["n_runs"] == 6 and top["n_segments"] == 6 * 4 * K1


def test_end_to_end_slot_growth_sees_the_planted_persistence(planted_report):
    report, _md = planted_report
    slot = report["slot_strata"]["by_kind"]["top/lrALL/bALL"]["5"]["channels"]["dc"]
    assert slot["growth_calibrated"] > 1.5
    assert slot["median_segment_mean_coherence"] > 0.9
    bulk = report["slot_strata"]["by_kind"]["bulk/lrALL/bALL"]["5"]["channels"]["dc"]
    assert bulk["growth_calibrated"] == pytest.approx(1.0, abs=0.4)


def test_end_to_end_strata_and_populated_cell_count(planted_report):
    report, md = planted_report
    assert report["inputs"]["cells_populated"] == 3 * len(aca.KINDS)
    assert report["inputs"]["cells_possible"] == 2 * 2 * len(aca.KINDS)
    assert "cells populated 6 of 8" in md
    assert sorted(report["strata"]["by_kind_batch"]) == [
        "bulk/lrALL/b00500", "bulk/lrALL/b01000",
        "top/lrALL/b00500", "top/lrALL/b01000",
    ]


def test_tau_nw_is_the_bartlett_read_at_the_bandwidth_in_force(planted_report):
    """The lag-4 gloss was wrong on both counts (report §1 correction).

    (a) the bandwidth at the tracked window is 3, not 4, and (b) the
    estimator applies Bartlett weights, so the integrated time it implies is
    `1 + 2*sum_{j<=L}(1 - j/(L+1)) rho_j`, not the flat `1 + 2*sum_{k<=4}`.
    At the report's operating point the two differ by 30%.
    """
    report, _md = planted_report
    cell = report["strata"]["by_kind"]["bulk/lrALL/bALL"]["5"]
    L = cell["lag_truncation_at_median_n"]
    assert L == newey_west_bandwidth(int(cell["n_kept_median"]), 4) == 2
    rho = cell["rho_median"]
    expect = 1.0 + 2.0 * sum(
        (1.0 - j / (L + 1.0)) * rho[j - 1] for j in range(1, L + 1)
    )
    assert cell["tau_nw"] == pytest.approx(expect, rel=1e-12)
    assert cell["tau_nw"] != pytest.approx(cell["tau"]["4"], rel=1e-3)


def test_the_bandwidth_at_the_tracked_window_is_three_not_four():
    """The report's §1 gloss claimed "the lag-4 read the bandwidth uses".

    It is 3 at burn-in 5 and 2 at burn-in 25; L = 4 belongs to the frozen
    tier (n in {187, 195}), which is the tier the prereg's anchor table
    records it for.
    """
    assert newey_west_bandwidth(45, 8) == 3  # tracked, burn-in 5
    assert newey_west_bandwidth(35, 8) == 3  # tracked, burn-in 15
    assert newey_west_bandwidth(25, 8) == 2  # tracked, burn-in 25
    assert newey_west_bandwidth(195, 8) == 4  # frozen tier
    assert newey_west_bandwidth(187, 8) == 4


def test_bartlett_tau_and_flat_tau4_disagree_at_the_reported_ladder():
    """The estimator integrates lags 1..L with BARTLETT weights.

    A flat lag-4 sum is not the memory the estimator has. On the ladder the
    report actually prints for pooled `top` (-0.343, 0.071, -0.054, 0.029)
    the two read 0.529 and 0.405 -- 30% apart, and only the first is what
    `sigma_LR^2` integrates. This is arithmetic, so it is asserted exactly.
    """
    L = newey_west_bandwidth(45, 8)
    rho = [-0.343, 0.071, -0.054, 0.029]
    tau_nw = 1.0 + 2.0 * sum(
        (1.0 - j / (L + 1.0)) * rho[j - 1] for j in range(1, L + 1)
    )
    tau_flat = 1.0 + 2.0 * sum(rho)
    assert tau_nw == pytest.approx(0.5295, abs=5e-4)
    assert tau_flat == pytest.approx(0.406, abs=5e-4)
    assert abs(tau_nw - tau_flat) / tau_flat > 0.29
    bulk = [-0.172, 0.027, -0.029, 0.007]
    tau_nw_bulk = 1.0 + 2.0 * sum(
        (1.0 - j / (L + 1.0)) * bulk[j - 1] for j in range(1, L + 1)
    )
    assert tau_nw_bulk == pytest.approx(0.7545, abs=5e-4)
    assert 1.0 + 2.0 * sum(bulk) == pytest.approx(0.666, abs=5e-4)


def test_end_to_end_report_is_byte_identical_on_a_rerun(tmp_path):
    """Module contract: identical inputs -> identical outputs, no timestamps."""
    rng = np.random.default_rng(SEED)
    for i in range(2):
        _write_run(tmp_path / "results", f"airbench_instrumented_seed{1400 + i}_x",
                   0.5, 500, 1400 + i, _planted(rng, K1 + K2, top_dc=0.5))
    first_json, first_md = _run_main(tmp_path)
    second_json, second_md = _run_main(tmp_path)
    assert first_md == second_md
    assert json.dumps(first_json, sort_keys=True) == json.dumps(
        second_json, sort_keys=True
    )
    assert "20" not in first_md.split("\n")[0]  # no timestamp in the title


def test_phase_a_ledger_reports_the_obligation_as_open(planted_report):
    report, md = planted_report
    led = report["phase_a_reproduction"]
    assert led["status"] == "OPEN"
    assert "A5" in led["unreproduced"]
    a5 = [r for r in led["anchors"] if r["anchor"] == "A5"][0]
    assert a5["status"] == "NOT_REPRODUCIBLE"
    assert {r["anchor"] for r in led["anchors"]} == {"A1", "A2", "A3", "A4", "A5"}
    assert "Phase A reproduction obligation — **OPEN**" in md
    assert "obligation is OPEN" in md


def test_burn_in_zero_is_carried_so_the_a2_anchor_can_be_read(planted_report):
    report, _md = planted_report
    by_kind = report["strata"]["by_kind"]["top/lrALL/bALL"]
    assert "0" in by_kind and "5" in by_kind
    assert by_kind["0"]["n_kept_median"] == 20.0
    assert by_kind["5"]["n_kept_median"] == 15.0


# ==========================================================================
# 7. Refusals: mixed parity, tombstones, seed floor, run family
# ==========================================================================


def test_mixed_parity_segment_starts_are_refused(tmp_path):
    """A slot whose segments start on different parities cancels its own
    alternating signal when pooled, so the pooling must not run silently."""
    rng = np.random.default_rng(SEED)
    results = tmp_path / "results"
    _write_run(results, "airbench_instrumented_seed1400_a", 0.5, 500, 1400,
               _planted(rng, K1 + K2))
    # rewrite the matrix with an odd-length refresh cadence -> mixed parity
    path = results / f"airbench_instrumented_seed1400_a{SIDECAR_SUFFIX}"
    log = json.loads(path.read_text())
    log["matrices"]["layer.weight"]["refresh_steps"] = [0, 19, 38, 57]
    path.write_text(json.dumps(log, indent=1, sort_keys=True) + "\n")
    with pytest.raises(SystemExit, match="MIXED parity"):
        _run_main(tmp_path)


def test_tombstoned_runs_are_excluded(tmp_path):
    rng = np.random.default_rng(SEED)
    results = tmp_path / "results"
    for name, seed in (("airbench_instrumented_seed1400_a", 1400),
                       ("airbench_instrumented_seed1401_b", 1401)):
        _write_run(results, name, 0.5, 500, seed, _planted(rng, K1 + K2))
    (results / "INVALID_RUNS.json").write_text(json.dumps({
        "invalid": [{"file": "airbench_instrumented_seed1401_b.json",
                     "reason": "test tombstone", "recorded": "2026-08-31"}]
    }, indent=1, sort_keys=True) + "\n")
    assert aca.tombstoned_runs(results) == {"airbench_instrumented_seed1401_b"}
    report, _md = _run_main(tmp_path)
    sel = report["inputs"]["selection"]
    assert sel["excluded"]["invalid_runs"] == [
        f"airbench_instrumented_seed1401_b{SIDECAR_SUFFIX}"
    ]
    assert report["inputs"]["n_runs"] == 1


def test_evaluation_seeds_and_foreign_run_families_are_excluded(tmp_path):
    rng = np.random.default_rng(SEED)
    results = tmp_path / "results"
    _write_run(results, "airbench_instrumented_seed1400_a", 0.5, 500, 1400,
               _planted(rng, K1 + K2))
    _write_run(results, "airbench_instrumented_seed0042_eval", 0.5, 500, 42,
               _planted(rng, K1 + K2))
    _write_run(results, "nanogpt_seed1500_other", 0.5, 500, 1500,
               _planted(rng, K1 + K2))
    paths, sel = aca.select_sidecars(
        results, None, 1, run_prefix="airbench_instrumented_", min_seed=1000
    )
    assert [p.name for p in paths] == [
        f"airbench_instrumented_seed1400_a{SIDECAR_SUFFIX}"
    ]
    assert sel["excluded"]["min_seed"] == [
        f"airbench_instrumented_seed0042_eval{SIDECAR_SUFFIX}"
    ]
    assert sel["excluded"]["run_prefix"] == [
        f"nanogpt_seed1500_other{SIDECAR_SUFFIX}"
    ]
    assert sel["n_discovered"] == 3 and sel["n_selected"] == 1


def test_selection_refuses_when_everything_is_filtered_out(tmp_path):
    rng = np.random.default_rng(SEED)
    results = tmp_path / "results"
    _write_run(results, "airbench_instrumented_seed0042_eval", 0.5, 500, 42,
               _planted(rng, K1 + K2))
    with pytest.raises(SystemExit, match="filtered out"):
        aca.select_sidecars(results, None, 1, run_prefix=None, min_seed=1000)
