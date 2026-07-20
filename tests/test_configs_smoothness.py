"""Contract tests for the two stability-law measurement configs.

configs/dev/instrumented_airbench_smoothness.yaml and its LR-ladder companion
must: parse, expand through scripts/sweep.py, carry dev seeds only, keep
compile off (required by both the HVP double-backward and the functional_call
smoothness probe), enable HVP + smoothness + frozen probes on the SAME runs,
and -- for the ladder -- vary only the learning rate across rungs.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest
import yaml

from src.instrument.tracker import _frozen_probe_settings

BASE = REPO_ROOT / "configs" / "dev" / "instrumented_airbench_smoothness.yaml"
LADDER = REPO_ROOT / "configs" / "dev" / "instrumented_airbench_smoothness_lr_ladder.yaml"
RECORD_LR = 0.24


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_module("sweep_module_smoothness", "scripts/sweep.py")


@pytest.fixture(params=[BASE, LADDER], ids=["base", "ladder"])
def config_path(request):
    return request.param


def test_configs_parse_and_declare_the_instrumented_experiment(config_path):
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["experiment"] == "airbench_instrumented"
    assert cfg["seed"] >= 1000
    # compile MUST be off: double-backward / functional_call constraint.
    assert cfg["recipe"]["compile"] is False
    instr = cfg["instrumentation"]
    assert instr["hvp"] is True  # Euclidean eta*lambda on the same runs
    assert instr["smoothness"]["enabled"] is True
    assert instr["smoothness"]["t_meas"] == 5
    assert instr["smoothness"]["grad_source"] in ("recompute", "training")
    frozen = _frozen_probe_settings(instr["frozen_probes"])
    assert frozen["k3"] == 16


def test_sweeps_expand_to_dev_seed_runs(config_path):
    cfg = yaml.safe_load(config_path.read_text())
    plan = sweep.expand_sweep(cfg, config_path)
    assert plan["seed_policy"] == "explicit-dev"
    assert all(s >= 1000 for s in plan["seeds"])
    for variant in plan["variants"]:
        assert "seed" not in variant["config"]
        sweep.refuse_eval_seed_literals(variant["config"], variant["name"])


def test_base_config_is_a_single_variant_three_seed_sweep():
    plan = sweep.expand_sweep(yaml.safe_load(BASE.read_text()), BASE)
    assert len(plan["variants"]) == 1
    assert plan["seeds"] == [1300, 1301, 1302]


def test_ladder_varies_only_the_learning_rate_across_three_rungs():
    cfg = yaml.safe_load(LADDER.read_text())
    plan = sweep.expand_sweep(cfg, LADDER)
    lrs = sorted(v["overrides"]["probe_overrides.lr"] for v in plan["variants"])
    assert lrs == [0.5 * RECORD_LR, RECORD_LR, 2.0 * RECORD_LR]
    assert len(plan["runs"]) == 6  # 3 rungs x 2 seeds
    # Rungs are otherwise identical: strip lr and everything must match.
    stripped = []
    for variant in plan["variants"]:
        c = dict(variant["config"])
        c.pop("probe_overrides")
        stripped.append(c)
    assert all(c == stripped[0] for c in stripped)


def test_probe_overrides_touch_only_the_sanctioned_keys():
    from src.optim.airbench_zoo import PROBE_OVERRIDE_KEYS, _validate_probe_overrides

    cfg = yaml.safe_load(LADDER.read_text())
    probe = _validate_probe_overrides(cfg)
    assert set(probe) <= set(PROBE_OVERRIDE_KEYS)
    assert probe["lr"] == RECORD_LR


def test_instrumented_experiment_threads_the_smoothness_probe_end_to_end(monkeypatch):
    """The GPU path's wiring, exercised on CPU: run_airbench_instrumented must
    build the probe, feed it the batch, call it around optimizer.step(), and
    fold its log into the instrumentation sidecar payload."""
    import torch

    from src.optim import airbench_zoo

    calls = {"pre": [], "post": []}

    def fake_smoke(config, device, _hub_factory=None, _batch_hook=None,
                   _pre_step_hook=None, _post_step_hook=None):
        model = torch.nn.Sequential(torch.nn.Linear(6, 5, bias=False))
        params = [model[0].weight]
        hub = _hub_factory(model, torch.optim.SGD(params, lr=0.1), params)
        gen = torch.Generator().manual_seed(1600)
        for step in range(1, 5):
            x = torch.randn(8, 6, generator=gen)
            y = torch.randint(0, 5, (8,), generator=gen)
            _batch_hook(x, y)
            torch.nn.functional.cross_entropy(model(x), y, reduction="sum").backward()
            hub.capture_grads()
            _pre_step_hook(step)
            calls["pre"].append(step)
            with torch.no_grad():
                model[0].weight.add_(0.01)
            _post_step_hook(step, 0.24)
            calls["post"].append(step)
            hub.after_step()
            model.zero_grad(set_to_none=True)
        return {"steps": 4}

    monkeypatch.setattr(airbench_zoo, "run_airbench_smoke", fake_smoke)
    cfg = yaml.safe_load(BASE.read_text())
    cfg["instrumentation"].update(
        {"k1": 2, "k2": 1, "t_refresh": 2, "min_dim": 2, "hvp": False,
         "smoothness": {"enabled": True, "t_meas": 2}}
    )
    cfg["instrumentation"]["frozen_probes"] = {"enabled": True, "k3": 2}
    metrics = airbench_zoo.run_airbench_instrumented(cfg, torch.device("cpu"))

    assert calls["pre"] == calls["post"] == [1, 2, 3, 4]
    log = metrics["_instrumentation_log"]
    assert log["frozen_probes_enabled"] is True
    sm = log["smoothness"]
    assert sm["n_measured_steps"] == 2  # t_meas=2 over 4 steps
    assert sm["matrices"]["0.weight"]["lr"] == [0.24, 0.24]
    assert metrics["smoothness_forward_passes"] == 4  # 2 steps x (base + 1 matrix)
    assert metrics["smoothness_backward_passes"] == 2
    assert metrics["optimizer_lr"] == RECORD_LR


def test_smoothness_and_hvp_both_refuse_compile_on():
    from src.optim.airbench_zoo import run_airbench_instrumented
    import torch

    cfg = yaml.safe_load(BASE.read_text())
    cfg["recipe"]["compile"] = True
    with pytest.raises(SystemExit, match="compile"):
        run_airbench_instrumented(cfg, torch.device("cpu"))
