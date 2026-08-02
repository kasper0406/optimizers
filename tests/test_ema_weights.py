"""Unit tests for src/optim/ema_weights.py and the airbench_ema config
surface (T1 EMA-as-anytime-anneal, docs/litreview/j-theory-theorem-sweep.md §5).

CPU-only: WeightEMA is harness-agnostic; the airbench_ema CUDA path is
exercised on the cloud box. The refusal paths tested here all fire before any
CUDA work, matching the repo's config-validation convention.
"""

from pathlib import Path

import pytest
import torch
import yaml

from src.optim.ema_weights import WeightEMA, validate_gammas

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------ validate_gammas


def test_validate_gammas_accepts_valid_list():
    assert validate_gammas([0.9, 0.99]) == [0.9, 0.99]


@pytest.mark.parametrize(
    "bad",
    [
        [],
        None,
        "0.9",
        [0.0],
        [1.0],
        [-0.1],
        [1.5],
        [0.9, 0.9],
        [0.9, "x"],
        [True],
    ],
)
def test_validate_gammas_rejects_invalid(bad):
    with pytest.raises(SystemExit):
        validate_gammas(bad)


# ----------------------------------------------------------------- WeightEMA


def _named(*tensors):
    return [(f"t{i}", t) for i, t in enumerate(tensors)]


def test_shadows_initialize_to_live_values():
    t = torch.tensor([1.0, 2.0], dtype=torch.float16)
    ema = WeightEMA(_named(t), [0.5])
    with ema.applied(0.5):
        assert torch.equal(t, torch.tensor([1.0, 2.0], dtype=torch.float16))


def test_update_matches_hand_computed_recursion():
    t = torch.tensor([1.0])
    ema = WeightEMA(_named(t), [0.75])
    expected = 1.0
    for value in [3.0, -2.0, 0.5]:
        t.fill_(value)
        ema.update()
        expected = 0.75 * expected + 0.25 * value
    with ema.applied(0.75):
        assert t.item() == pytest.approx(expected, rel=1e-6)


def test_multiple_gammas_are_independent():
    t = torch.tensor([0.0])
    ema = WeightEMA(_named(t), [0.9, 0.5])
    t.fill_(10.0)
    ema.update()
    with ema.applied(0.9):
        assert t.item() == pytest.approx(1.0)
    with ema.applied(0.5):
        assert t.item() == pytest.approx(5.0)


def test_shadows_accumulate_in_fp32_for_fp16_tensors():
    # 1 + 1/1024 is not representable in fp16; a fp16 accumulator would stay
    # at 1.0 forever under tiny updates while fp32 shadows must move.
    t = torch.tensor([1.0], dtype=torch.float16)
    ema = WeightEMA(_named(t), [0.9])
    t.fill_(2.0)
    for _ in range(3):
        ema.update()
    with ema.applied(0.9):
        assert t.item() > 1.2  # fp32 recursion: 1.271; frozen fp16 would be 1.0


def test_applied_swaps_in_and_restores_bit_exactly():
    t = torch.tensor([1.0, 2.0], dtype=torch.float16)
    ema = WeightEMA(_named(t), [0.5])
    t.copy_(torch.tensor([5.0, 7.0], dtype=torch.float16))
    ema.update()  # shadows now [3, 4.5]
    live_before = t.clone()
    with ema.applied(0.5):
        assert t.tolist() == [3.0, 4.5]
    assert torch.equal(t, live_before)


def test_applied_restores_on_exception():
    t = torch.tensor([2.0])
    ema = WeightEMA(_named(t), [0.5])
    t.fill_(8.0)
    with pytest.raises(RuntimeError):
        with ema.applied(0.5):
            raise RuntimeError("eval blew up")
    assert t.item() == 8.0


def test_applied_unknown_gamma_refused():
    ema = WeightEMA(_named(torch.tensor([1.0])), [0.5])
    with pytest.raises(SystemExit):
        with ema.applied(0.9):
            pass


@pytest.mark.parametrize(
    "tensors",
    [
        [],
        [("a", torch.tensor([1.0])), ("a", torch.tensor([2.0]))],
        [("a", torch.tensor([1], dtype=torch.int64))],
    ],
)
def test_constructor_refuses_bad_tensor_sets(tensors):
    with pytest.raises(SystemExit):
        WeightEMA(tensors, [0.5])


# ------------------------------------------- airbench_ema config validation


def test_resolve_lr_schedule_default_and_values():
    from src.optim.airbench_zoo import _resolve_lr_schedule

    assert _resolve_lr_schedule({}) == "linear"
    assert _resolve_lr_schedule({"lr_schedule": "constant"}) == "constant"
    with pytest.raises(SystemExit):
        _resolve_lr_schedule({"lr_schedule": "cosine"})


def test_resolve_ema_shapes():
    from src.optim.airbench_zoo import _resolve_ema

    assert _resolve_ema({}) is None
    assert _resolve_ema({"ema": {"gammas": [0.9]}}) == [0.9]
    with pytest.raises(SystemExit):
        _resolve_ema({"ema": {"gammas": [0.9], "extra": 1}})
    with pytest.raises(SystemExit):
        _resolve_ema({"ema": [0.9]})


def test_airbench_ema_refuses_optimizer_override():
    from src.optim.airbench_zoo import run_airbench_ema

    config = {
        "optimizer": {"name": "muon"},
        "recipe": {"ema": {"gammas": [0.9]}},
    }
    with pytest.raises(SystemExit, match="does not accept an optimizer"):
        run_airbench_ema(config, torch.device("cpu"))


def test_airbench_ema_requires_ema_block():
    from src.optim.airbench_zoo import run_airbench_ema

    with pytest.raises(SystemExit, match="requires a recipe.ema block"):
        run_airbench_ema({"recipe": {}}, torch.device("cpu"))


def test_airbench_ema_pins_stock_recipe(monkeypatch):
    from src.optim import airbench_zoo

    captured = {}

    def fake_smoke(config, device):
        captured.update(config)
        return {"ok": True}

    monkeypatch.setattr(airbench_zoo, "run_airbench_smoke", fake_smoke)
    out = airbench_zoo.run_airbench_ema(
        {"recipe": {"ema": {"gammas": [0.9]}, "lr_schedule": "constant"}},
        torch.device("cpu"),
    )
    assert out == {"ok": True}
    assert captured["optimizer"] == dict(
        name="vendor_muon", lr=0.24, momentum=0.6, nesterov=True
    )
    assert captured["recipe"]["normalize_filter_weights"] is False
    assert captured["recipe"]["compile"] is True
    assert captured["recipe"]["lr_schedule"] == "constant"


# ------------------------------------------------------------- T1 configs


@pytest.mark.parametrize(
    "name, schedule",
    [("wpj_t1_ema_stock.yaml", "linear"), ("wpj_t1_ema_constant.yaml", "constant")],
)
def test_t1_configs_are_dev_seed_ema_arms(name, schedule):
    config = yaml.safe_load((REPO_ROOT / "configs" / name).read_text())
    assert config["experiment"] == "airbench_ema"
    assert config["recipe"]["lr_schedule"] == schedule
    validate_gammas(config["recipe"]["ema"]["gammas"])
    assert config["sweep"]["seeds"]["policy"] == "dev"
    assert config["seed"] >= 1000
    assert "optimizer" not in config
