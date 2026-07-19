"""Gate-1 amendment A4: mechanism-probe configs + the restricted
probe_overrides path of the instrumented airbench experiment.

CPU-only: config parsing / sweep expansion / validation guards. The actual
probe runs need CUDA and are launched via scripts/sweep.py on the GPU box.

All literal seeds are dev seeds (>= 1000).
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.optim.airbench_zoo import (  # noqa: E402
    AIRBENCH_STOCK_LR,
    PROBE_OVERRIDE_KEYS,
    _validate_probe_overrides,
    run_airbench_instrumented,
    run_airbench_smoke,
)


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_module("sweep_module_a4", "scripts/sweep.py")

CONFIG_DIR = REPO_ROOT / "configs" / "dev"
PROBE_CONFIGS = {
    "instrumented_airbench_mom0.yaml": (
        {"momentum": 0.0, "nesterov": False},
        [1300, 1301],
    ),
    "instrumented_airbench_lrhalf.yaml": ({"lr": 0.12}, [1302, 1303]),
    "instrumented_airbench_lrquarter.yaml": ({"lr": 0.06}, [1304, 1305]),
}


def load(name):
    with open(CONFIG_DIR / name) as fh:
        return yaml.safe_load(fh)


# ----------------------------------------------------------------- configs


@pytest.mark.parametrize("name", sorted(PROBE_CONFIGS))
def test_probe_config_parses_and_expands_to_two_dev_seeds(name):
    config = load(name)
    assert config["experiment"] == "airbench_instrumented"
    expected_probe, expected_seeds = PROBE_CONFIGS[name]
    assert config["probe_overrides"] == expected_probe
    plan = sweep.expand_sweep(config, CONFIG_DIR / name)
    assert plan["seed_policy"] == "explicit-dev"
    assert plan["seeds"] == expected_seeds
    assert len(plan["runs"]) == 2
    # No eval-seed literals anywhere (stricter launch gate).
    sweep.refuse_eval_seed_literals(config, name)


def test_probe_seed_sets_are_disjoint_from_all_other_instrumented_runs():
    used = set()
    for _, seeds in PROBE_CONFIGS.values():
        assert not (set(seeds) & used)
        used.update(seeds)
    # Disjoint from WP1.2 (1000-1019), with-replacement (1100-1102),
    # HVP (1200-1202).
    assert not (used & set(range(1000, 1020)))
    assert not (used & set(range(1100, 1103)))
    assert not (used & set(range(1200, 1203)))


@pytest.mark.parametrize("name", sorted(PROBE_CONFIGS))
def test_probe_config_matches_wp12_instrumentation_block(name):
    """Everything except probe_overrides / seeds is the WP1.2 recipe."""
    wp12 = yaml.safe_load(
        (REPO_ROOT / "configs" / "wp12_airbench_instrumented.yaml").read_text()
    )
    config = load(name)
    assert config["instrumentation"] == wp12["instrumentation"]
    assert config["train"] == wp12["train"]
    assert config["data"] == wp12["data"]
    assert config["recipe"] == wp12["recipe"]
    assert "optimizer" not in config  # overrides travel via probe_overrides


# ----------------------------------------------------- validation guards


def _instrumented_config(**extra):
    return {
        "experiment": "airbench_instrumented",
        "seed": 1300,
        "device": "cuda",
        "instrumentation": {"k1": 2, "k2": 2},
        **extra,
    }


def test_validate_probe_overrides_accepts_the_probe_keys():
    cfg = _instrumented_config(
        probe_overrides={"lr": 0.12, "momentum": 0.0, "nesterov": False}
    )
    assert _validate_probe_overrides(cfg) == {
        "lr": 0.12,
        "momentum": 0.0,
        "nesterov": False,
    }
    assert _validate_probe_overrides(_instrumented_config()) == {}


def test_validate_probe_overrides_rejects_unknown_keys():
    for bad in ({"ns_steps": 3}, {"lr": 0.12, "weight_decay": 0.1}):
        with pytest.raises(SystemExit, match="probe_overrides may only touch"):
            _validate_probe_overrides(_instrumented_config(probe_overrides=bad))
    assert "lr" in PROBE_OVERRIDE_KEYS  # the allowed set is lr/momentum/nesterov
    assert set(PROBE_OVERRIDE_KEYS) == {"lr", "momentum", "nesterov"}


def test_validate_probe_overrides_rejects_non_dict_or_empty():
    for bad in ({}, [0.12], "lr=0.12"):
        with pytest.raises(SystemExit, match="non-empty mapping"):
            _validate_probe_overrides(_instrumented_config(probe_overrides=bad))


def test_validate_probe_overrides_refuses_eval_seed_policies():
    for seeds in ("eval", {"policy": "eval"}):
        cfg = _instrumented_config(
            probe_overrides={"lr": 0.12}, sweep={"seeds": seeds}
        )
        with pytest.raises(SystemExit, match="never use the eval seed policy"):
            _validate_probe_overrides(cfg)
    # Dev policies pass.
    cfg = _instrumented_config(
        probe_overrides={"lr": 0.12}, sweep={"seeds": [1302, 1303]}
    )
    assert _validate_probe_overrides(cfg) == {"lr": 0.12}


def test_validate_probe_overrides_refuses_sub_dev_seed():
    cfg = _instrumented_config(probe_overrides={"lr": 0.12})
    cfg["seed"] = 42
    with pytest.raises(SystemExit, match="dev-seed only"):
        _validate_probe_overrides(cfg)


# ------------------------------------------------- experiment-level wiring


def test_instrumented_still_rejects_optimizer_block():
    cfg = _instrumented_config(optimizer={"name": "muon"})
    with pytest.raises(SystemExit, match="probe_overrides"):
        run_airbench_instrumented(cfg, torch.device("cpu"))


def test_instrumented_validates_probe_before_needing_cuda():
    """A bad probe block fails with its own message (not the CUDA guard) on
    CPU; a valid one proceeds up to the CUDA requirement."""
    bad = _instrumented_config(probe_overrides={"weight_decay": 0.1})
    with pytest.raises(SystemExit, match="probe_overrides may only touch"):
        run_airbench_instrumented(bad, torch.device("cpu"))
    if torch.cuda.is_available():  # pragma: no cover - dev box is CPU
        pytest.skip("CUDA present; the CPU-guard assertion does not apply")
    good = _instrumented_config(probe_overrides={"momentum": 0.0, "nesterov": False})
    good["recipe"] = {"compile": False}
    with pytest.raises(SystemExit, match="CUDA"):
        run_airbench_instrumented(good, torch.device("cpu"))


def test_smoke_harness_never_reads_probe_overrides():
    """probe_overrides is honored by the instrumented experiment ONLY; the
    zoo smoke harness ignores/never consumes it (its optimizer comes from
    config['optimizer'])."""
    cfg = {
        "probe_overrides": {"lr": 999.0},
        "optimizer": {"name": "muon", "lr": 0.1},
        "recipe": {},
    }
    if torch.cuda.is_available():  # pragma: no cover - dev box is CPU
        pytest.skip("CPU-guard assertion does not apply")
    with pytest.raises(SystemExit, match="CUDA"):
        # Reaches the CUDA guard without ever validating probe_overrides:
        # an (invalid) lr=999 probe block is simply inert here.
        run_airbench_smoke(cfg, torch.device("cpu"))
    assert AIRBENCH_STOCK_LR == 0.24  # record value untouched by A4
