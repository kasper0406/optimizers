"""Tests for scripts/analyze_ema.py on synthetic seed-paired arm results."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "rm_analyze_ema", REPO_ROOT / "scripts" / "analyze_ema.py"
)
analyze_ema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_ema)


def _write(tmp_path, name, schedule, seed, val_accs, ema, started="2026-08-02T00:00:00"):
    payload = {
        "experiment": "airbench_ema",
        "seed": seed,
        "started_at": started,
        "gpu_type": "NVIDIA L40",
        "metrics": {
            "lr_schedule": schedule,
            "val_accs": val_accs,
            "val_acc": val_accs[-1],
            "tta_val_acc": val_accs[-1] + 0.01,
            "ema_val_accs": ema,
            "ema_tta_val_accs": {g: series[-1] + 0.01 for g, series in ema.items()},
        },
    }
    (tmp_path / name).write_text(json.dumps(payload))


def _populate(tmp_path):
    for seed in (1000, 1001):
        jitter = 0.001 * (seed - 1000)
        # linear arm: raw climbs to 0.94; its EMA reaches the final only at the end
        _write(
            tmp_path,
            f"lin_{seed}.json",
            "linear",
            seed,
            [0.80, 0.90, 0.92, 0.94 + jitter],
            {"0.9": [0.79, 0.89, 0.91, 0.94 + jitter]},
        )
        # constant arm: raw plateaus low; EMA crosses the linear final at epoch 3
        _write(
            tmp_path,
            f"con_{seed}.json",
            "constant",
            seed,
            [0.80, 0.90, 0.91, 0.92 + jitter],
            {"0.9": [0.82, 0.91, 0.945 + jitter, 0.946 + jitter]},
        )


def test_analyze_pairs_seeds_and_finds_harvest_epoch(tmp_path):
    _populate(tmp_path)
    result = analyze_ema.analyze(tmp_path)
    assert result["n_paired_seeds"] == 2
    g = result["per_gamma"]["0.9"]
    assert g["harvest_epoch"]["per_seed"] == {1000: 3, 1001: 3}
    assert g["harvest_epoch"]["median"] == 3
    # constant EMA final (0.946+j) - linear final (0.94+j) = +0.006 exactly, paired
    assert g["constant_ema_final_minus_linear_final"]["mean"] == pytest.approx(0.006)
    # raw constant final (0.92+j) - linear final (0.94+j) = -0.02
    assert result["constant_raw_final_minus_linear_final"]["mean"] == pytest.approx(
        -0.02
    )
    md = analyze_ema.to_markdown(result)
    assert "harvest epoch" in md and "0.9" in md


def test_unpaired_or_empty_dir_refused(tmp_path):
    with pytest.raises(SystemExit, match="no seed-paired"):
        analyze_ema.analyze(tmp_path)
    _write(tmp_path, "only_lin.json", "linear", 1000, [0.9], {"0.9": [0.9]})
    with pytest.raises(SystemExit, match="no seed-paired"):
        analyze_ema.analyze(tmp_path)


def test_mixed_gpu_types_refused(tmp_path):
    _populate(tmp_path)
    bad = json.loads((tmp_path / "con_1001.json").read_text())
    bad["gpu_type"] = "NVIDIA RTX A6000"
    (tmp_path / "con_1001.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="mixed GPU types"):
        analyze_ema.analyze(tmp_path)


def test_duplicate_seed_keeps_earliest(tmp_path):
    _populate(tmp_path)
    _write(
        tmp_path,
        "con_1000_rerun.json",
        "constant",
        1000,
        [0.5, 0.5, 0.5, 0.5],
        {"0.9": [0.5, 0.5, 0.5, 0.5]},
        started="2026-08-03T00:00:00",
    )
    result = analyze_ema.analyze(tmp_path)
    # the later rerun must not displace the original run's numbers
    assert result["per_gamma"]["0.9"]["harvest_epoch"]["per_seed"][1000] == 3


def test_crossing_epoch_edge_cases():
    assert analyze_ema.crossing_epoch([0.1, 0.2], 0.15) == 2
    assert analyze_ema.crossing_epoch([0.1, 0.2], 0.5) is None
    assert analyze_ema.crossing_epoch([0.5], 0.5) == 1
