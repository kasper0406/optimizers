"""WP1.2 wiring tests: the instrumented-airbench experiment registration,
the pre-registration launch gate in scripts/run.py, the sweep expansion of
configs/wp12_airbench_instrumented.yaml (dev policy -> exactly seeds
1000-1019), the instrumentation sidecar written next to the results JSON,
and the directory mode of src.instrument.plots.

criteria/phase1_preregistration.md is HUMAN-AUTHORED; these tests never
create it in the repo -- the "present" case is exercised by monkeypatching
the gate's path to a temp file.

All literal seeds are dev seeds (>= 1000).
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch
import yaml

from src.instrument import InstrumentationHub
from src.instrument.plots import make_all_plots
from src.instrument.schema import (
    SIDECAR_SUFFIX,
    load_instrumentation,
    write_sidecar,
)
from src.optim import Muon

CONFIG_PATH = REPO_ROOT / "configs" / "wp12_airbench_instrumented.yaml"
CLASSIFIER_KWARGS = dict(tau_sig=4.0, tau_noise=2.0, rho_osc=0.5, n_min=30)


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_module("sweep_module_wp12", "scripts/sweep.py")
run_mod = _load_module("run_module_wp12", "scripts/run.py")


# ---------------------------------------------------------------- the config


def test_wp12_config_expands_to_exactly_20_dev_seeds():
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    assert config["experiment"] == "airbench_instrumented"
    plan = sweep.expand_sweep(config, CONFIG_PATH)
    assert plan["seed_policy"] == "dev"
    assert plan["seeds"] == list(range(1000, 1020))
    assert len(plan["runs"]) == 20
    assert len(plan["variants"]) == 1
    # Materialized configs carry no seed key; seeds travel via --seed only.
    assert "seed" not in plan["variants"][0]["config"]


def test_wp12_config_has_no_eval_seed_literals_and_dev_seeds_only():
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    sweep.refuse_eval_seed_literals(config, str(CONFIG_PATH))  # must not raise
    assert config["seed"] >= 1000
    assert config["instrumentation"]["seed"] >= 1000
    # The stock recipe is pinned by the experiment; no optimizer override.
    assert "optimizer" not in config
    # HVP stays off for airbench (validation-only, plan section 1.1).
    assert config["instrumentation"]["hvp"] is False


# ----------------------------------------------- pre-registration launch gate


def test_run_refuses_instrumented_without_preregistration(tmp_path, monkeypatch):
    """criteria/phase1_preregistration.md is absent (agent never creates it):
    run.py must refuse before doing any work."""
    missing = tmp_path / "phase1_preregistration.md"
    assert not missing.exists()
    monkeypatch.setattr(run_mod, "PHASE1_PREREG", missing)
    with pytest.raises(SystemExit, match="phase1_preregistration"):
        run_mod.main([str(CONFIG_PATH), "--seed", "1000", "--out-dir", str(tmp_path)])


def test_run_gate_is_wired_to_the_repo_criteria_path():
    assert run_mod.PHASE1_PREREG == REPO_ROOT / "criteria" / "phase1_preregistration.md"
    assert "airbench_instrumented" in run_mod.PREREG_GATED_EXPERIMENTS
    assert "airbench_instrumented" in run_mod.EXPERIMENT_REGISTRY


def test_run_proceeds_past_gate_when_preregistration_exists(tmp_path, monkeypatch):
    """With the criteria file present (temp stand-in, never the repo path)
    the gate passes and main() reaches the experiment function -- proven by
    a sentinel replacing the (CUDA-only) experiment."""
    prereg = tmp_path / "phase1_preregistration.md"
    prereg.write_text("# stand-in for the human-authored criteria (test only)\n")
    monkeypatch.setattr(run_mod, "PHASE1_PREREG", prereg)

    def sentinel(config, device):
        raise RuntimeError("gate-passed sentinel")

    monkeypatch.setitem(
        run_mod.EXPERIMENT_REGISTRY, "airbench_instrumented", sentinel
    )
    with pytest.raises(RuntimeError, match="gate-passed sentinel"):
        run_mod.main([str(CONFIG_PATH), "--seed", "1000", "--out-dir", str(tmp_path)])


# ------------------------------------------------------- sidecar in run.py


def _tiny_hub_log(seed: int):
    """A real (CPU) instrumentation log: tiny MLP + Muon + hub, few steps."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(12, 16, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 4, bias=False),
    )
    optimizer = Muon(
        [p for p in model.parameters() if p.ndim >= 2],
        lr=0.05, momentum=0.9, nesterov=True, ns_steps=3, ns_dtype=torch.float32,
    )
    hub = InstrumentationHub(
        list(model.named_parameters()),
        optimizer,
        k1=3, k2=2, t_refresh=10, betas=(0.9, 0.99),
        classifier_kwargs=CLASSIFIER_KWARGS,
        snapshot_every=5, seed=seed,
    )
    gen = torch.Generator().manual_seed(seed + 1)
    loss_fn = torch.nn.MSELoss()
    for _ in range(25):
        x = torch.randn(8, 12, generator=gen)
        y = torch.randn(8, 4, generator=gen)
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        hub.capture_grads()  # exercise the capture path end to end
        optimizer.step()
        hub.after_step()
    return hub.to_log()


def test_run_writes_instrumentation_sidecar_next_to_results(tmp_path):
    """Any experiment returning metrics['_instrumentation_log'] gets a
    schema-validated sidecar next to its results JSON, referenced from
    metrics (the airbench_instrumented flow, minus the CUDA harness)."""
    log = _tiny_hub_log(seed=1900)

    def fake_experiment(config, device):
        return {"ok": True, "_instrumentation_log": log}

    cfg = {"experiment": "fake_instrumented_wp12", "seed": 1900, "device": "cpu"}
    cfg_path = tmp_path / "fake_instrumented_wp12.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    run_mod.EXPERIMENT_REGISTRY["fake_instrumented_wp12"] = fake_experiment
    try:
        rc = run_mod.main([str(cfg_path), "--out-dir", str(tmp_path)])
    finally:
        del run_mod.EXPERIMENT_REGISTRY["fake_instrumented_wp12"]
    assert rc == 0

    results = list(tmp_path.glob("fake_instrumented_wp12_seed1900_*.json"))
    results = [p for p in results if not p.name.endswith(SIDECAR_SUFFIX)]
    assert len(results) == 1
    with open(results[0]) as fh:
        result = json.load(fh)
    sidecar_name = result["metrics"]["instrumentation_sidecar"]
    assert "_instrumentation_log" not in result["metrics"]
    sidecar = results[0].with_name(sidecar_name)
    assert sidecar.exists()
    assert sidecar.name == results[0].name[: -len(".json")] + SIDECAR_SUFFIX
    loaded = load_instrumentation(sidecar)  # validates against the schema
    assert loaded["betas"] == ["0.9", "0.99"]


# ------------------------------------------------------ plots directory mode


def test_plots_consume_a_directory_of_sidecars(tmp_path):
    """make_all_plots on a directory of CPU-generated sidecars produces the
    three plan-section-1.2 plots, pooling directions across runs; lr is
    recovered from the sibling results JSONs."""
    for seed in (1910, 1911):
        results_json = tmp_path / f"airbench_instrumented_seed{seed}_t.json"
        write_sidecar(_tiny_hub_log(seed), results_json)
        # Minimal sibling results JSON carrying the pinned lr (as
        # airbench_instrumented records it in metrics.optimizer_lr).
        results_json.write_text(json.dumps({"metrics": {"optimizer_lr": 0.24}}))

    out_dir = tmp_path / "plots"
    paths = make_all_plots(tmp_path, out_dir)  # no --lr: sibling recovery
    assert [p.name for p in paths] == [
        "regime_scatter.png",
        "regime_occupancy.png",
        "eta_lambda_calibration.png",
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
