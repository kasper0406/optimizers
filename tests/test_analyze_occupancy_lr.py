"""scripts/analyze_occupancy_lr.py tests: core statistics on synthetic data.

CPU-only, no sidecar files touched -- synthetic logs/point clouds only.
Seeds: dev seeds only (>= 1000) where seeds appear at all.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aol = _load_module("analyze_occupancy_lr", "scripts/analyze_occupancy_lr.py")


# ------------------------------------------------------------------ lr_at


def test_lr_at_matches_airbench_schedule_off_by_one():
    # sidecar step 1 = the update taken at loop step 0 = full lr0
    assert aol.lr_at(1, 0.24, 200) == pytest.approx(0.24)
    # sidecar step 200 = loop step 199
    assert aol.lr_at(200, 0.24, 200) == pytest.approx(0.24 * (1 - 199 / 200))
    assert aol.lr_at(101, 0.12, 200) == pytest.approx(0.06)


# ------------------------------------------------------------ occupancy


def _log(directions):
    return {"betas": ["0.9"], "matrices": {"m": {"directions": directions}}}


def _direction(steps, rhos, n_since_reset=None, var=1.0):
    n = len(steps)
    return {
        "kind": "top",
        "index": 0,
        "per_beta": {
            "0.9": {
                "step": list(steps),
                "rho": list(rhos),
                "var": [var] * n,
                "n_since_reset": list(n_since_reset or [100] * n),
            }
        },
    }


def test_occupancy_series_fraction_and_filters():
    # 4 directions at step 5: rhos -0.5, -0.3, 0.0, -0.1 -> 2/4 below -0.2
    dirs = [_direction([5], [r]) for r in (-0.5, -0.3, 0.0, -0.1)]
    out = aol.occupancy_series(_log(dirs), "0.9", burn_in=0, min_n=2)
    assert out == [(5, 0.5, 4)]


def test_occupancy_series_burn_in_and_var_filters_drop_snapshots():
    dirs = [
        _direction([5], [-0.5], n_since_reset=[3]),  # dropped at burn_in=10
        _direction([5], [-0.5], var=0.0),  # dropped: var <= 0
        _direction([5], [None]),  # dropped: rho None
        _direction([5], [-0.5]),
        _direction([5], [0.1]),
    ]
    out = aol.occupancy_series(_log(dirs), "0.9", burn_in=10, min_n=2)
    assert out == [(5, 0.5, 2)]
    # min_n drops the step entirely when too few snapshots survive
    assert aol.occupancy_series(_log(dirs), "0.9", burn_in=10, min_n=3) == []


# ------------------------------------------------------------ statistics


def test_isotonic_fit_perfect_on_monotone_data():
    lr = np.linspace(0.01, 0.24, 50)
    occ = 0.2 + 2.0 * lr  # monotone increasing
    predict, r2 = aol.isotonic_fit(lr, occ)
    assert r2 == pytest.approx(1.0, abs=1e-9)
    assert predict(np.array([0.1]))[0] == pytest.approx(0.4, abs=0.02)


def test_decile_decomposition_sums_and_config_share():
    rng = np.random.default_rng(1000)
    lr = rng.uniform(0.0, 0.24, 400)
    config = np.array(["baseline", "probe"] * 200)
    occ = 2.0 * lr + np.where(config == "probe", 0.1, 0.0)
    d = aol.decile_decomposition(lr, occ, config)
    # decomposition is exact: total = between-bins + within-bins
    assert d["ss_total"] == pytest.approx(
        d["ss_between_bins"] + d["ss_within_bins"], rel=1e-6
    )
    # lr explains most variance; the config offset owns most of the rest
    assert d["r2_between_bins"] > 0.8
    assert d["within_bin_config_share"] > 0.8


def _points(config, t, lr0, occ_fn):
    lr = np.array([aol.lr_at(s, l0, 200) for s, l0 in zip(t, lr0)])
    return config, t, lr, np.array([occ_fn(x) for x in lr])


def test_matched_lr_offsets_recover_planted_offset():
    steps = np.arange(5, 201, 5)
    config, t, lr, occ = [], [], [], []
    for name, cfg, lr0, offset in [
        ("base_a", "baseline", 0.24, 0.0),
        ("base_b", "baseline", 0.24, 0.0),
        ("probe_a", "lrhalf", 0.12, 0.1),
    ]:
        for s in steps:
            config.append(cfg)
            t.append(s)
            lr.append(aol.lr_at(int(s), lr0, 200))
            occ.append(2.0 * lr[-1] + offset)
        # run names: reuse config-unique names
        # (runs array built below in one go)
    n = len(steps)
    points = {
        "config": np.array(config),
        "run": np.array(["base_a"] * n + ["base_b"] * n + ["probe_a"] * n),
        "t": np.array(t),
        "lr": np.array(lr),
        "occ": np.array(occ),
        "n": np.full(len(t), 100),
    }
    out = aol.matched_lr_offsets(points)
    # occupancy is linear in lr, so interpolation is exact: planted +0.1
    assert out["lrhalf"]["mean_delta"] == pytest.approx(0.1, abs=1e-6)
    assert out["lrhalf"]["n_points"] > 0


def test_early_vs_late_matches_when_state_function_holds():
    steps = np.arange(5, 201, 5)
    rows = [("b%02d" % i, "baseline", 0.24) for i in range(3)]
    rows += [("q", "lrquarter", 0.06)]
    config, run, t, lr, occ = [], [], [], [], []
    for name, cfg, lr0 in rows:
        for s in steps:
            config.append(cfg)
            run.append(name)
            t.append(int(s))
            lr.append(aol.lr_at(int(s), lr0, 200))
            occ.append(2.0 * lr[-1])  # pure state function of lr
    points = {
        "config": np.array(config),
        "run": np.array(run),
        "t": np.array(t),
        "lr": np.array(lr),
        "occ": np.array(occ),
        "n": np.full(len(t), 100),
    }
    out = aol.early_vs_late(points)
    e = out["lrquarter"]
    # same lr window, very different mean t -- but occupancies agree
    assert e["probe"]["mean_t"] < 60
    assert e["baseline"]["mean_t"] > 140
    assert abs(e["probe_minus_baseline"]) < 0.02


# ------------------------------------------------------------ config id


def _results(optimizer_lr=0.24, probe_overrides=None, sampling=None, recipe=None, hvp=False):
    metrics = {"optimizer_lr": optimizer_lr}
    if probe_overrides:
        metrics["probe_overrides"] = probe_overrides
    if sampling:
        metrics["sampling"] = sampling
    return {
        "metrics": metrics,
        "config": {
            "contents": {
                "recipe": recipe or {},
                "instrumentation": {"hvp": hvp},
            }
        },
    }


def test_classify_config_from_results_json():
    assert aol.classify_config(_results()) == ("baseline", 0.24)
    assert aol.classify_config(
        _results(probe_overrides={"momentum": 0.0, "nesterov": False})
    ) == ("mom0", 0.24)
    assert aol.classify_config(
        _results(optimizer_lr=0.12, probe_overrides={"lr": 0.12})
    ) == ("lrhalf", 0.12)
    assert aol.classify_config(
        _results(optimizer_lr=0.06, probe_overrides={"lr": 0.06})
    ) == ("lrquarter", 0.06)
    assert aol.classify_config(_results(sampling="with_replacement")) == (
        "withrep",
        0.24,
    )
    assert aol.classify_config(
        _results(recipe={"compile": False}, hvp=True)
    ) == ("hvp_compileoff", 0.24)
