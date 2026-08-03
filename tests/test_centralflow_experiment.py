"""Tests for the airbench central-flow experiment wiring (litreview j §6
item 2): config validation, momentum-direction extraction, and the four arm
configs. The CUDA training path is exercised on the cloud box."""

from pathlib import Path

import pytest
import torch
import yaml

from src.optim.airbench_zoo import (
    _resolve_cf,
    momentum_topk_directions,
    run_airbench_centralflow,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cf_block(**over):
    block = {
        "lr_scale": 0.25,
        "enabled": True,
        "refresh_every": 10,
        "k_directions": 4,
        "beta_scale": 1.0,
    }
    block.update(over)
    return {"cf": block}


def test_resolve_cf_accepts_valid_block():
    cf = _resolve_cf(_cf_block())
    assert cf == {
        "lr_scale": 0.25,
        "enabled": True,
        "refresh_every": 10,
        "k_directions": 4,
        "beta_scale": 1.0,
    }


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"cf": None},
        {"cf": {"lr_scale": 0.25}},  # missing keys
        _cf_block(extra=1) | {},  # extra key merged below
    ],
)
def test_resolve_cf_rejects_bad_shapes(config):
    if "cf" in config and isinstance(config["cf"], dict) and "extra" in config["cf"]:
        pass  # extra-key case constructed via _cf_block(extra=1)
    with pytest.raises(SystemExit):
        _resolve_cf(config)


@pytest.mark.parametrize(
    "over",
    [
        {"lr_scale": 0.0},
        {"lr_scale": 1.5},
        {"refresh_every": 0},
        {"k_directions": 0},
        {"k_directions": 17},
        {"beta_scale": -0.1},
    ],
)
def test_resolve_cf_rejects_bad_values(over):
    with pytest.raises(SystemExit):
        _resolve_cf(_cf_block(**over))


def test_momentum_topk_directions_recovers_planted_direction():
    u = torch.tensor([3.0, 4.0]) / 5.0
    v = torch.tensor([1.0, 2.0, 2.0]) / 3.0
    m = 7.0 * torch.outer(u, v)  # rank-1: top direction is u v^T exactly
    (d,) = momentum_topk_directions(m, 1)
    assert d.shape == m.shape
    # unit Frobenius norm, aligned with u v^T up to sign
    assert float(d.norm()) == pytest.approx(1.0, rel=1e-6)
    assert abs(float((d * torch.outer(u, v)).sum())) == pytest.approx(1.0, rel=1e-5)


def test_momentum_topk_directions_orthogonal_and_capped():
    m = torch.diag(torch.tensor([5.0, 3.0, 1.0]))
    dirs = momentum_topk_directions(m, 16)  # capped at min(shape) = 3
    assert len(dirs) == 3
    for i, di in enumerate(dirs):
        for j, dj in enumerate(dirs):
            expected = 1.0 if i == j else 0.0
            assert float((di * dj).sum()) == pytest.approx(expected, abs=1e-6)


def test_momentum_topk_directions_conv_shape_roundtrip():
    m = torch.randn(8, 3, 2, 2)
    dirs = momentum_topk_directions(m, 2)
    assert all(d.shape == m.shape for d in dirs)


def test_centralflow_refuses_optimizer_override():
    config = dict(_cf_block(), optimizer={"name": "muon"})
    with pytest.raises(SystemExit, match="does not accept an optimizer"):
        run_airbench_centralflow(config, torch.device("cpu"))


def test_centralflow_requires_cf_block():
    with pytest.raises(SystemExit, match="needs a cf: block"):
        run_airbench_centralflow({"recipe": {}}, torch.device("cpu"))


ARMS = {
    "wpj_cf_stock.yaml": (1.0, False),
    "wpj_cf_cold.yaml": (0.25, False),
    "wpj_cf_cold_on.yaml": (0.25, True),
    "wpj_cf_stock_on.yaml": (1.0, True),
}


@pytest.mark.parametrize("name, expected", sorted(ARMS.items()))
def test_cf_arm_configs(name, expected):
    config = yaml.safe_load((REPO_ROOT / "configs" / name).read_text())
    assert config["experiment"] == "airbench_centralflow"
    assert config["recipe"]["compile"] is False  # third-order chain, no compile
    assert config["sweep"]["seeds"]["policy"] == "dev"
    assert config["seed"] >= 1000
    assert "optimizer" not in config
    cf = _resolve_cf(config)
    assert (cf["lr_scale"], cf["enabled"]) == expected
    # shared knobs identical across arms (only lr_scale/enabled vary)
    assert (cf["refresh_every"], cf["k_directions"], cf["beta_scale"]) == (10, 4, 1.0)


def test_momentum_topk_directions_skips_nonfinite_buffers():
    m = torch.randn(4, 6)
    m[1, 2] = float("inf")
    assert momentum_topk_directions(m, 2) == []
    m2 = torch.randn(4, 6)
    m2[0, 0] = float("nan")
    assert momentum_topk_directions(m2, 2) == []
