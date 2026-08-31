"""Tests for scripts/analyze_tempo_overhead.py on synthetic results JSONs."""

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_tempo_overhead import analyze_overhead, metric_value


def fake_run(seed, lr, opt="muon", kappa=0.0, scope="global",
             time_seconds=3.76, wall_time_s=None):
    d = {
        "seed": seed,
        "config": {
            "contents": {
                "optimizer": {"name": opt, "lr": lr, "kappa": kappa, "scope": scope}
            },
            "path": f"{opt}.yaml",
            "sha256": "0" * 64,
        },
        "metrics": {"time_seconds": time_seconds, "tta_val_acc": 0.94},
    }
    if wall_time_s is not None:
        d["wall_time_s"] = wall_time_s
    return d


def paired_arms(lr=0.24, seeds=(0, 1, 2, 3), base=4.00, delta=0.10, jitter=0.01):
    """Control + treatment at a fixed paired delta plus a per-seed jitter."""
    runs = []
    for i, s in enumerate(seeds):
        wobble = jitter * i
        runs.append(fake_run(s, lr, time_seconds=base + wobble))
        runs.append(
            fake_run(
                s, lr, opt="tempomuon", kappa=-0.25,
                time_seconds=base + wobble + delta + jitter * (i % 2),
            )
        )
    return runs


def test_paired_delta_and_percentage():
    out = analyze_overhead(paired_arms())
    row = out["per_metric"]["time_seconds"]["0.24"]["tempomuon-global"]
    assert row["n"] == 4
    assert row["mean"] == pytest.approx(0.105)
    assert row["base_mean"] == pytest.approx(4.015)
    assert row["arm_mean"] == pytest.approx(4.120)
    assert row["pct"] == pytest.approx(100.0 * 0.105 / 4.015)
    assert row["ci95_lo"] < row["mean"] < row["ci95_hi"]
    assert row["t"] > 0
    assert out["baseline"] == "muon"
    assert out["arms"] == ["muon", "tempomuon-global"]


def test_unpaired_seeds_are_dropped():
    runs = paired_arms(seeds=(0, 1, 2))
    runs.append(fake_run(9, 0.24, time_seconds=99.0))  # control-only seed
    out = analyze_overhead(runs)
    row = out["per_metric"]["time_seconds"]["0.24"]["tempomuon-global"]
    assert row["n"] == 3
    assert row["base_mean"] < 5.0  # the 99 s outlier is outside the pairing


def test_rungs_are_reported_separately():
    out = analyze_overhead(paired_arms(lr=0.24) + paired_arms(lr=0.96, delta=0.05))
    per_lr = out["per_metric"]["time_seconds"]
    assert sorted(per_lr) == ["0.24", "0.96"]
    assert per_lr["0.24"]["tempomuon-global"]["mean"] > (
        per_lr["0.96"]["tempomuon-global"]["mean"]
    )


def test_wall_time_read_from_run_root_and_optional():
    runs = paired_arms()
    for i, r in enumerate(runs):
        r["wall_time_s"] = 5.0 + 0.01 * i
    out = analyze_overhead(runs)
    assert "0.24" in out["per_metric"]["wall_time_s"]
    # metrics absent everywhere simply produce no rows, not a crash
    for r in runs:
        del r["wall_time_s"]
    out = analyze_overhead(runs)
    assert out["per_metric"]["wall_time_s"] == {}


def test_metric_value_prefers_metrics_dict():
    run = fake_run(0, 0.24, time_seconds=1.5, wall_time_s=2.5)
    assert metric_value(run, "time_seconds") == 1.5
    assert metric_value(run, "wall_time_s") == 2.5
    assert metric_value(run, "missing") is None
    run["metrics"]["time_seconds"] = None
    assert metric_value(run, "time_seconds") is None


def test_fewer_than_two_pairs_is_not_reported():
    out = analyze_overhead(paired_arms(seeds=(0,)))
    assert out["per_metric"]["time_seconds"] == {}


def test_missing_baseline_arm_raises():
    runs = [r for r in paired_arms() if r["config"]["contents"]["optimizer"]["name"]
            != "muon"]
    with pytest.raises(SystemExit):
        analyze_overhead(runs)


def test_report_is_deterministic_markdown():
    runs = paired_arms()
    first = analyze_overhead(runs)["report"]
    second = analyze_overhead(list(reversed(runs)))["report"]
    assert first == second
    assert first.startswith("# Program #8: seed-paired step-time overhead")
    assert not math.isnan(
        analyze_overhead(runs)["per_metric"]["time_seconds"]["0.24"][
            "tempomuon-global"
        ]["t"]
    )
