"""Tests for the airbench_teleport experiment wiring and analyzer (round 2,
litreview j §6 item 4). The CUDA path is exercised on the cloud box."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.optim.airbench_zoo import _resolve_teleport, run_airbench_teleport

REPO_ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "rm_analyze_teleport", REPO_ROOT / "scripts" / "analyze_teleport.py"
)
analyze_teleport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_teleport)


def _tp_block(**over):
    block = {
        "enabled": True,
        "every": 25,
        "start_step": 10,
        "spread": 2.0,
        "ascend_iters": 20,
        "step_size": 0.3,
    }
    block.update(over)
    return {"teleport": block}


def test_resolve_teleport_accepts_valid():
    tp = _resolve_teleport(_tp_block())
    assert tp["enabled"] is True and tp["every"] == 25 and tp["spread"] == 2.0


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"teleport": {"enabled": True}},  # missing keys
        _tp_block(extra=1),
    ],
)
def test_resolve_teleport_rejects_bad_shapes(config):
    with pytest.raises(SystemExit):
        _resolve_teleport(config)


@pytest.mark.parametrize(
    "over",
    [
        {"every": 0},
        {"start_step": 0},
        {"spread": 1.0},
        {"spread": 0.5},
        {"ascend_iters": -1},
        {"step_size": 0.0},
    ],
)
def test_resolve_teleport_rejects_bad_values(over):
    with pytest.raises(SystemExit):
        _resolve_teleport(_tp_block(**over))


def test_teleport_refuses_optimizer_override():
    config = dict(_tp_block(), optimizer={"name": "muon"})
    with pytest.raises(SystemExit, match="does not accept an optimizer"):
        run_airbench_teleport(config, torch.device("cpu"))


def test_teleport_requires_block():
    with pytest.raises(SystemExit, match="needs a teleport: block"):
        run_airbench_teleport({"recipe": {}}, torch.device("cpu"))


@pytest.mark.parametrize(
    "name, enabled",
    [("wpj_teleport_on.yaml", True), ("wpj_teleport_off.yaml", False)],
)
def test_arm_configs(name, enabled):
    config = yaml.safe_load((REPO_ROOT / "configs" / name).read_text())
    assert config["experiment"] == "airbench_teleport"
    assert config["recipe"]["compile"] is True  # no higher-order autograd here
    assert config["sweep"]["seeds"]["policy"] == "dev"
    assert config["seed"] >= 1000
    assert "optimizer" not in config
    tp = _resolve_teleport(config)
    assert tp["enabled"] is enabled
    assert (tp["every"], tp["start_step"], tp["spread"]) == (25, 10, 2.0)


# ----------------------------------------------------------------- analyzer


def _write_run(tmp_path, name, enabled, seed, val_accs, started="2026-08-03T00:00:00"):
    ts = (
        [
            {"step": 10, "mean_ratio": 1.20, "max_ratio": 1.3, "min_ratio": 1.1},
            {"step": 35, "mean_ratio": 1.15, "max_ratio": 1.2, "min_ratio": 1.1},
            {"step": 60, "mean_ratio": 1.10, "max_ratio": 1.2, "min_ratio": 1.0},
        ]
        if enabled
        else []
    )
    payload = {
        "experiment": "airbench_teleport",
        "seed": seed,
        "started_at": started,
        "gpu_type": "NVIDIA L40",
        "metrics": {
            "teleport": dict(_tp_block()["teleport"], enabled=enabled),
            "val_accs": val_accs,
            "val_acc": val_accs[-1],
            "tta_val_acc": val_accs[-1] + 0.008,
            "n_teleports": len(ts),
            "teleport_timeseries": ts,
            "time_seconds": 40.0 + (2.0 if enabled else 0.0),
        },
    }
    (tmp_path / name).write_text(json.dumps(payload))


def test_analyzer_pairs_and_computes_deltas(tmp_path):
    for seed in (1000, 1001):
        j = 0.001 * (seed - 1000)
        _write_run(tmp_path, f"off_{seed}.json", False, seed, [0.90, 0.930 + j])
        _write_run(tmp_path, f"on_{seed}.json", True, seed, [0.91, 0.935 + j])
    result = analyze_teleport.analyze(tmp_path)
    assert result["n_paired_seeds"] == 2
    assert result["on_minus_off_val"]["mean"] == pytest.approx(0.005)
    assert result["on_minus_off_tta"]["mean"] == pytest.approx(0.005)
    assert result["per_epoch_on_minus_off"][0]["mean"] == pytest.approx(0.01)
    assert result["overhead_seconds"]["mean"] == pytest.approx(2.0)
    assert result["achieved_ratio"]["first"] == pytest.approx(1.20)
    md = analyze_teleport.to_markdown(result)
    assert "ON − OFF" in md and "1.2000" in md


def test_analyzer_refusals(tmp_path):
    with pytest.raises(SystemExit, match="no seed-paired"):
        analyze_teleport.analyze(tmp_path)
    _write_run(tmp_path, "on_1000.json", True, 1000, [0.9, 0.93])
    with pytest.raises(SystemExit, match="no seed-paired"):
        analyze_teleport.analyze(tmp_path)
    _write_run(tmp_path, "off_1000.json", False, 1000, [0.9, 0.93])
    bad = json.loads((tmp_path / "off_1000.json").read_text())
    bad["gpu_type"] = "NVIDIA RTX A6000"
    (tmp_path / "off_1000.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="mixed GPU types"):
        analyze_teleport.analyze(tmp_path)
