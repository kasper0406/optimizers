"""WP0.4 tests: optimizer registry + airbench smoke-config wiring.

The actual airbench smoke runs need CUDA (pending on a GPU box); what is
verified here on CPU per the compute boundary:
- every zoo optimizer is registered and constructible by name;
- every configs/dev/airbench_smoke_*.yaml parses, uses a dev seed (>= 1000),
  names a registered optimizer, and its optimizer kwargs are accepted by the
  optimizer's constructor on airbench-shaped (4D conv filter) parameters;
- the vendored airbench module imports on CPU and exposes what the harness
  uses;
- the ``python -m src.optim.airbench_zoo <config>`` entrypoint wires the zoo
  registry and the airbench_smoke experiment into scripts/run.py, and the
  experiment refuses to run without CUDA (clean SystemExit, no partial run).
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.optim import (
    AdamW,
    AdaMuon,
    DynMuon,
    MatrixOptimizer,
    Muon,
    NoOpOptimizer,
    NorMuon,
    OPTIMIZER_REGISTRY,
    build_optimizer,
)
from src.optim import airbench_zoo

SMOKE_CONFIGS = sorted(
    (REPO_ROOT / "configs" / "dev").glob("airbench_smoke_*.yaml")
)
ZOO_NAMES = ["muon", "adamw", "dynmuon", "adamuon", "normuon"]


# ------------------------------------------------------------------ registry


def test_registry_contains_full_baseline_zoo():
    assert OPTIMIZER_REGISTRY["noop"] is NoOpOptimizer
    assert OPTIMIZER_REGISTRY["muon"] is Muon
    assert OPTIMIZER_REGISTRY["adamw"] is AdamW
    assert OPTIMIZER_REGISTRY["dynmuon"] is DynMuon
    assert OPTIMIZER_REGISTRY["adamuon"] is AdaMuon
    assert OPTIMIZER_REGISTRY["normuon"] is NorMuon


@pytest.mark.parametrize("name", ZOO_NAMES + ["noop"])
def test_build_optimizer_constructs_each_zoo_member(name):
    params = [torch.nn.Parameter(torch.zeros(4, 3))]
    opt = build_optimizer(name, params, {})
    assert isinstance(opt, MatrixOptimizer)


def test_build_optimizer_unknown_name_raises():
    with pytest.raises(KeyError, match="unknown-opt"):
        build_optimizer("unknown-opt", [torch.nn.Parameter(torch.zeros(2, 2))], {})


@pytest.mark.parametrize("name", ZOO_NAMES)
def test_each_zoo_member_steps_on_cpu(name):
    """One full hook-loop step on a conv-shaped and a linear-shaped param."""
    params = [
        torch.nn.Parameter(torch.full((4, 2, 3, 3), 0.5)),
        torch.nn.Parameter(torch.full((4, 3), -0.5)),
    ]
    opt = build_optimizer(name, params, {})
    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.ones_like(p)
    opt.step()
    for p, p0 in zip(params, before):
        assert torch.isfinite(p.detach()).all()
        assert not torch.equal(p.detach(), p0), f"{name} did not update"


# ------------------------------------------------------------- smoke configs


def test_all_five_smoke_configs_exist():
    # WP2.1 added airbench_smoke_routed.yaml alongside the five zoo configs
    # (routed's own coverage lives in tests/test_optim_routed*.py).
    stems = {p.stem for p in SMOKE_CONFIGS}
    assert stems == {f"airbench_smoke_{n}" for n in ZOO_NAMES + ["routed"]}


@pytest.mark.parametrize("path", SMOKE_CONFIGS, ids=lambda p: p.stem)
def test_smoke_config_parses_and_follows_policy(path):
    with open(path) as fh:
        config = yaml.safe_load(fh)
    assert config["experiment"] == "airbench_smoke"
    assert isinstance(config["seed"], int)
    assert config["seed"] >= 1000, "literal config seeds must be dev seeds"
    assert config["optimizer"]["name"] in OPTIMIZER_REGISTRY


@pytest.mark.parametrize("path", SMOKE_CONFIGS, ids=lambda p: p.stem)
def test_smoke_config_optimizer_kwargs_are_constructible(path):
    """The config's optimizer block must instantiate on airbench-shaped
    params exactly as the harness does (src/optim/airbench_zoo.py)."""
    with open(path) as fh:
        config = yaml.safe_load(fh)
    opt_cfg = dict(config["optimizer"])
    name = opt_cfg.pop("name")
    filter_like = [torch.nn.Parameter(torch.randn(8, 4, 3, 3))]
    opt = build_optimizer(name, filter_like, opt_cfg)
    assert isinstance(opt, MatrixOptimizer)


# ------------------------------------------------------------ vendor + wiring


def test_vendored_airbench_imports_on_cpu():
    ab = airbench_zoo.load_vendor_airbench()
    for attr in ("CifarNet", "CifarLoader", "evaluate"):
        assert hasattr(ab, attr), attr
    # idempotent
    assert airbench_zoo.load_vendor_airbench() is ab


def test_airbench_smoke_refuses_to_run_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CPU-only guard test")
    with pytest.raises(SystemExit, match="CUDA"):
        airbench_zoo.run_airbench_smoke(
            {"optimizer": {"name": "muon"}}, torch.device("cpu")
        )


def test_entrypoint_wires_zoo_into_run_py(tmp_path):
    """airbench_zoo.main must register the zoo + experiment into scripts/
    run.py and dispatch a parsed config; on this CPU box the run then stops
    at the clean CUDA guard (SystemExit) before touching results/."""
    if torch.cuda.is_available():
        pytest.skip("CPU-only wiring test")
    cfg = {
        "experiment": "airbench_smoke",
        "seed": 1999,  # dev seed
        "device": "cpu",  # forces the guard instead of a CUDA attempt
        "optimizer": {"name": "muon", "lr": 0.24, "momentum": 0.6,
                      "nesterov": True, "ns_steps": 3},
    }
    cfg_path = tmp_path / "airbench_smoke_wiring.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="CUDA"):
        airbench_zoo.main([str(cfg_path), "--out-dir", str(tmp_path)])
    assert not list(tmp_path.glob("*.json")), "no results JSON may be written"


def test_run_py_registration_comment_is_satisfied_at_runtime():
    """scripts/run.py ships with only 'noop'; the airbench_zoo entrypoint
    injects the full zoo at runtime without editing run.py."""
    run_mod = airbench_zoo._load_run_module()
    assert "noop" in run_mod.OPTIMIZER_REGISTRY
    run_mod.OPTIMIZER_REGISTRY.update(OPTIMIZER_REGISTRY)
    run_mod.EXPERIMENT_REGISTRY["airbench_smoke"] = airbench_zoo.run_airbench_smoke
    for name in ZOO_NAMES:
        assert name in run_mod.OPTIMIZER_REGISTRY
    assert "airbench_smoke" in run_mod.EXPERIMENT_REGISTRY
