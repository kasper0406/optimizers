"""Program #22 analysis tests. Written against synthetic curves with known
answers, per the internal-review finding that every Wave-1-era analysis
script shipped untested."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

spec = importlib.util.spec_from_file_location(
    "analyze_bbp", REPO_ROOT / "scripts" / "analyze_bbp.py")
abbp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abbp)


def _curve(bs, a):
    return {"b_chunks": list(bs), "a_hat": list(a), "n_merges": [1] * len(bs)}


def test_interp_a_is_linear_in_log_b_and_clamps():
    c = _curve([1, 2, 4, 8], [0.1, 0.2, 0.4, 0.8])
    # exact grid points
    assert abbp.interp_a(c, 1) == pytest.approx(0.1)
    assert abbp.interp_a(c, 8) == pytest.approx(0.8)
    # midpoint in log space between b=2 and b=4 is b=2.828
    assert abbp.interp_a(c, 2 * (2 ** 0.5)) == pytest.approx(0.3, abs=1e-6)
    # clamped outside the measured grid
    assert abbp.interp_a(c, 0.01) == pytest.approx(0.1)
    assert abbp.interp_a(c, 1e6) == pytest.approx(0.8)


def test_saturated_curve_passes_S_and_rising_curve_fails():
    """A curve flat above the training batch must give sat ratio ~1; a curve
    still rising there must give a ratio well below the 0.9 bar."""
    bs = [2 ** k for k in range(9)]
    flat = _curve(bs, [0.2, 0.3, 0.45, 0.6, 0.72, 0.8, 0.8, 0.8, 0.8])
    rising = _curve(bs, [0.05, 0.08, 0.12, 0.18, 0.26, 0.36, 0.48, 0.62, 0.8])
    r_flat = abbp.analyze_probe({"curves": {"m": flat}})["median_sat_ratio"]
    r_rise = abbp.analyze_probe({"curves": {"m": rising}})["median_sat_ratio"]
    assert r_flat == pytest.approx(1.0, abs=1e-9), r_flat
    assert r_rise < 0.9, r_rise
    assert r_rise < r_flat


def test_momentum_correction_shifts_the_evaluation_point():
    """b_eff = 39b must actually be used: a curve that saturates only ABOVE
    the raw record batch still reads as saturated after the correction."""
    assert abbp.B_EFF_FACTOR == pytest.approx(39.0)
    bs = [2 ** k for k in range(9)]
    # rises until b=32, flat after: raw b=8 is unsaturated, b_eff=312 is not
    a = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.8, 0.8, 0.8]
    res = abbp.analyze_probe({"curves": {"m": _curve(bs, a)}})
    assert res["median_sat_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert res["per_matrix"]["m"]["a_at_record_b"] == pytest.approx(0.5)


def test_vacuity_guard_counts_matrices_with_real_dynamic_range():
    bs = [1, 2, 4]
    flat_line = _curve(bs, [0.50, 0.51, 0.52])       # range 0.02 -> vacuous
    real = _curve(bs, [0.10, 0.40, 0.75])            # range 0.65 -> informative
    res = abbp.analyze_probe({"curves": {"a": flat_line, "b": real, "c": real}})
    assert res["frac_matrices_range_ge_02"] == pytest.approx(2 / 3)


def test_criteria_refuse_off_grid_points_instead_of_clamping():
    """The guard behind prereg amendment A2: clamping made the original
    criterion identically 1.0. Criteria must read None off-grid."""
    c = _curve([1, 2, 4], [0.1, 0.3, 0.6])
    assert abbp.interp_a(c, 100, allow_clamp=False) is None
    assert abbp.interp_a(c, 0.1, allow_clamp=False) is None
    assert abbp.interp_a(c, 100) == pytest.approx(0.6)  # descriptive path clamps
    # a probe whose grid does not reach the criterion points yields no ratio,
    # rather than a spurious 1.0
    res = abbp.analyze_probe({"curves": {"m": c}})
    assert res["per_matrix"]["m"]["sat_ratio"] is None
    assert res["median_sat_ratio"] is None
