"""Tests for ``scripts/channel_audit_anchors.py`` (program #23 pre-launch anchors).

These run at reduced Monte-Carlo reps -- the registered anchors are 200000-rep
quantities and reproducing them here would cost minutes -- so they assert the
QUALITATIVE facts the pre-registration's repairs turn on, plus determinism and
the arithmetic identities, which are rep-independent:

* the published DC channel is far outside the AR(1) null it was calibrated
  against (repair R1) -- direction and order of magnitude, not the digit;
* median pooling of the 32-lag tau sum is biased low against mean pooling
  (repair R2), which is what made the previous DRAFT's decisive clause fire on
  white noise;
* tau(L = 4) = n/ESS on the published aggregate is < 1 for every probe
  (repair R3) -- pure arithmetic on a file on disk;
* the pure inflated null lands in P1's FAIL row under the band contrast while
  it passed both clauses under the previous DRAFT's statistic (repair R1);
* identical inputs produce byte-identical JSON.

No GPU, no network, no sidecar is read.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCRIPT = REPO_ROOT / "scripts" / "channel_audit_anchors.py"
FROZEN_PROBES = REPO_ROOT / "reports" / "frozen-probes.json"


def _load():
    spec = importlib.util.spec_from_file_location("channel_audit_anchors", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


anchors = _load()

# Cheap settings: enough reps to fix the sign and the order of magnitude of
# every asserted quantity, few enough to keep the file under a few seconds.
FAST = dict(
    null_reps=20_000,
    sensitivity_reps=40_000,
    solve_reps=1_500,
    power_reps=8_000,
    tau_probes=400,
)


def _args(tmp_path: Path, **over):
    import argparse

    ns = argparse.Namespace(
        frozen_probes=FROZEN_PROBES,
        out_json=tmp_path / "anchors.json",
        out_md=None,
        power_seed=99,
        **FAST,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return anchors.build_anchors(_args(tmp_path_factory.mktemp("anchors")))


def test_published_tau_is_below_one_for_every_probe(built):
    """tau(L = 4) = n/ESS, so min ESS > n means tau < 1 everywhere (repair R3)."""
    tau = built["published_aggregate"]["implied_tau_at_L4"]
    assert tau["max"] < 1.0
    assert tau["min"] < tau["median"] < tau["max"]
    # The prereg quotes these; they are arithmetic on a file, not Monte Carlo.
    assert tau["median"] == pytest.approx(0.5130, abs=5e-4)
    assert tau["max"] == pytest.approx(0.9194, abs=5e-4)
    assert built["published_aggregate"]["n_nw_floored"] == 0


def test_published_dc_channel_is_not_the_ar1_null(built):
    """Repair R1: a ~1.4x scale inflation and a ~100x tail excess."""
    ratio = built["instrument_null_contrast"]["observed_over_null_dc"]
    assert 1.25 < ratio["median"] < 1.5
    assert 1.15 < ratio["q25"] < 1.5
    assert 1.30 < ratio["q75"] < 1.7
    # frac_ge_4 is a 1e-4 null rate: resolvable at the registered 200000 reps
    # (132x), not at this file's reps, where the denominator may be 0 -> NaN.
    assert ratio["frac_ge_3"] > 5.0
    assert not (ratio["frac_ge_4"] < 20.0)  # NaN-safe: only a resolved LOW value fails


def test_phi_heterogeneity_does_not_explain_the_tail_excess(built):
    """The 6-matrix mixture null stays at the pooled null's tail, not the observed one."""
    c = built["instrument_null_contrast"]
    observed = built["published_aggregate"]["dc_final_abs_t"]["frac_ge_4"]
    assert c["mixture_null_6_matrices"]["dc"]["frac_ge_4"] < observed / 20.0
    assert -0.60 < min(c["per_matrix_phi_hat"].values()) < -0.45
    assert -0.35 < max(c["per_matrix_phi_hat"].values()) < -0.15


def test_previous_draft_p1_passes_on_a_pure_null(built):
    """The defect repair R1 fixes: both old clauses fire with zero planted signal."""
    prev = built["instrument_null_contrast"]["previous_draft_p1_on_a_pure_null"]
    assert prev["ratio_alt_scaled"] >= 1.3  # the old P1a bar
    assert prev["frac_alt_ge_4_scaled"] >= 0.010  # the old P1b absolute floor
    assert prev["frac_alt_ge_4_scaled"] >= 3.0 * prev["null_frac_alt_ge_4"]


def test_band_contrast_is_one_on_a_pure_inflated_null(built):
    """And rises monotonically with the planted alternating amplitude (repair R1)."""
    rows = built["p1_band_contrast_power"]["by_amplitude"]
    assert rows["0.00"]["band_contrast"] == pytest.approx(1.0, abs=0.10)
    assert rows["0.00"]["tail_contrast"] < 3.0
    assert rows["0.00"]["frac_alt"] < 0.010
    contrasts = [rows["%.2f" % a]["band_contrast"] for a in anchors.PLANTED_AMPLITUDES]
    assert contrasts == sorted(contrasts)
    assert rows["0.20"]["band_contrast"] > 2.0


def test_alt_channel_is_the_phi_sensitive_one(built):
    """Repair R7: the previous DRAFT disclosed the insensitive channel."""
    span = built["phi_sensitivity"]["_span"]
    assert span["dc_median_pct"] < 0.0 < span["alt_median_pct"]
    assert abs(span["alt_median_pct"]) > abs(span["dc_median_pct"])
    assert span["alt_frac_ge_4_ratio"] > 2.0


def test_median_pooled_tau_is_biased_low_and_mean_pooled_is_not(built):
    """Repair R2: the pooling choice is what made the decisive clause fire on white noise."""
    for n_key in ("tau_references_n195", "tau_references_n187"):
        for k in anchors.TAU_LAGS:
            e = built[n_key]["K%d" % k]
            assert e["tau_white_mean_pooled"] == pytest.approx(1.0, abs=0.06)
            assert e["tau_white_median_pooled"] <= e["tau_white_mean_pooled"]
    k32 = built["tau_references_n195"]["K%d" % anchors.TAU_PRIMARY_K]
    assert k32["tau_white_median_pooled"] < 0.96


def test_every_tau_branch_is_produced_by_the_right_stream(built):
    """White noise must NOT be decisive; phi > 0 must FAIL; phi < 0 must be decisive."""
    b = built["tau_branches"]
    assert b["white"]["registered_verdict"] == "UNDECIDED"
    assert b["ar1_phi_p0.20"]["registered_verdict"] == "FAIL"
    assert b["ar1_phi_p0.50"]["registered_verdict"] == "FAIL"
    assert b["ar1_phi_m0.34"]["registered_verdict"] == "DECISIVE"
    assert b["ar1_phi_m0.385"]["registered_verdict"] == "DECISIVE"
    for label in ("white", "ar1_phi_p0.20", "ar1_phi_p0.50", "ar1_phi_m0.385"):
        assert b[label]["k_stable"], label


def test_tau_reference_seed_is_not_the_base_seed(built):
    """A shared seed makes the white-noise control return 1.000 exactly (bbp A2)."""
    assert anchors.TAU_REF_SEED != anchors.BASE_SEED
    assert built["tau_branches"]["white"]["by_K"]["K32"]["tau_cal"] != 1.0


def test_k6_channels_have_the_same_calibrated_shape_under_the_null(built):
    """K6's diagnostic must not fire on the null it is meant to certify."""
    shape = built["k6_channel_shape"]
    assert shape["max_abs_pct_divergence"] < 15.0
    for q in ("q25", "q75", "q90"):
        assert shape["alt"][q] == pytest.approx(shape["dc"][q], rel=0.05)


def test_output_is_deterministic(tmp_path):
    # Determinism does not depend on rep count, so this builds twice at the
    # cheapest settings that still exercise every branch of build_anchors.
    cheap = dict(null_reps=1_000, sensitivity_reps=1_000, solve_reps=400,
                 power_reps=500, tau_probes=64)
    a = anchors.build_anchors(_args(tmp_path, out_json=tmp_path / "a.json", **cheap))
    b = anchors.build_anchors(_args(tmp_path, out_json=tmp_path / "b.json", **cheap))
    dumped = [json.dumps(x, indent=1, sort_keys=True) for x in (a, b)]
    assert dumped[0] == dumped[1]


def test_phi_solver_inverts_the_estimator(tmp_path):
    """Bisection on ESS/n must land back on the target it was given."""
    target = 1.95
    phi = anchors._solve_phi(target, reps=4000, seed=anchors.BASE_SEED)
    assert -0.60 < phi < -0.25
    got = anchors._ess_over_n(phi, anchors.N_OBS_PUBLISHED, 4000, anchors.BASE_SEED)
    assert got == pytest.approx(target, abs=0.05)


def test_markdown_states_no_verdict(built):
    """Descriptive output only (CLAUDE.md ground rule 1)."""
    md = anchors.to_markdown(built)
    lowered = md.lower()
    for word in ("pass/fail", "we conclude", "therefore the gate", "criterion is met"):
        assert word not in lowered
    assert "no criterion is evaluated" in lowered
    assert np.isfinite(built["p1_band_contrast_power"]["theta"])
