"""Unit tests for scripts/analyze_local_baseline.py."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "rm_analyze_local_baseline", REPO_ROOT / "scripts" / "analyze_local_baseline.py"
)
alb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alb)


def _write_run(d, seed, finals_by_step, tag="nanogpt_local_baseline", target_hit=False):
    f = d / f"nanogpt_seed{seed}_20260721T{seed}00.json"
    steps = sorted(finals_by_step)
    f.write_text(json.dumps({
        "seed": seed,
        "git_sha": "a" * 40,
        "git_dirty": False,
        "config": {"path": f"sweeps/{tag}/{tag}.yaml"},
        "metrics": {
            "final_val_loss": finals_by_step[steps[-1]],
            "steps_to_target": steps[-1] if target_hit else None,
            "val_curve": [
                {"step": s, "val_loss": v} for s, v in sorted(finals_by_step.items())
            ],
        },
    }))


def test_n_per_arm_formula():
    # n = ceil(2 * (sd/effect)^2 * z^2); sd=effect -> ceil(2 * 7.851) = 16
    assert alb.n_per_arm(0.00125, 0.00125) == 16
    assert alb.n_per_arm(0.00125, 0.005) == 2  # floor at 2


def test_build_report_stats_and_censoring(tmp_path):
    finals = [3.288, 3.289, 3.290, 3.2885, 3.2895, 3.2875, 3.2905, 3.2882,
              3.2898, 3.2891]
    for i, fv in enumerate(finals):
        _write_run(tmp_path, 1710 + i, {1625: fv + 0.02, 1750: fv})
    rep = alb.build_report(tmp_path, "nanogpt_local_baseline")
    assert rep["n_runs"] == 10
    import statistics as st
    assert rep["final_val_loss"]["mean"] == pytest.approx(st.mean(finals), abs=1e-5)
    assert rep["final_val_loss"]["sd"] == pytest.approx(st.stdev(finals), abs=1e-6)
    lo, hi = rep["final_val_loss"]["sd_ci95"]
    assert lo < rep["final_val_loss"]["sd"] < hi
    assert rep["steps_to_target_censored"] == "10/10"
    # slope: 0.02 loss over 125 steps
    assert rep["end_slope_loss_per_step"] == pytest.approx(0.02 / 125, rel=1e-6)
    md = alb.to_markdown(rep)
    assert "seeds/arm" in md and "censored: 10/10" in md


def test_sd_ci_only_for_df9(tmp_path):
    for i, fv in enumerate([3.288, 3.289, 3.290]):
        _write_run(tmp_path, 1710 + i, {1625: fv + 0.02, 1750: fv})
    rep = alb.build_report(tmp_path, "nanogpt_local_baseline")
    assert rep["final_val_loss"]["sd_ci95"] is None


def test_other_tags_excluded(tmp_path):
    _write_run(tmp_path, 1710, {1625: 3.30, 1750: 3.288})
    _write_run(tmp_path, 1701, {1625: 3.31, 1750: 3.285}, tag="wp02_nanogpt_repro")
    with pytest.raises(SystemExit):
        alb.build_report(tmp_path, "nanogpt_local_baseline")  # only 1 match
