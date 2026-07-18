"""Tests for scripts/sweep.py: expansion counts, seed policy resolution,
eval-seed-literal refusal, and no-eval-seeds-on-disk invariants.

All literal seeds in this file are dev seeds (>= 1000) per CLAUDE.md rule 2;
eval seeds appear only as the *output* of the launch-time policy resolution,
which is exactly what these tests pin down.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_module("sweep_module", "scripts/sweep.py")


def base_sweep_config(seeds="dev", grid=None):
    config = {
        "experiment": "smoke",
        "device": "cpu",
        "model": {"in_dim": 8, "hidden_dim": 8, "out_dim": 2},
        "data": {"batch_size": 4},
        "train": {"steps": 2},
        "optimizer": {"name": "noop", "lr": 0.0},
        "sweep": {"seeds": seeds},
    }
    if grid is not None:
        config["sweep"]["grid"] = grid
    return config


def write_config(tmp_path, config, name="devsweep.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config))
    return path


# ------------------------------------------------------ seed policy resolution


def test_eval_policy_resolves_to_full_eval_set():
    policy, seeds = sweep.resolve_seed_policy("eval")
    assert policy == "eval"
    assert seeds == list(range(100))
    assert len(seeds) == 100


def test_eval_policy_refuses_partial_set():
    with pytest.raises(sweep.SweepConfigError):
        sweep.resolve_seed_policy({"policy": "eval", "num_seeds": 50})


def test_dev_policy_resolves_to_documented_range():
    policy, seeds = sweep.resolve_seed_policy("dev")
    assert policy == "dev"
    assert seeds[0] == sweep.DEV_SEED_BASE == 1000
    assert len(seeds) == sweep.DEV_SEED_DEFAULT_COUNT
    assert all(s >= 1000 for s in seeds)

    _, five = sweep.resolve_seed_policy({"policy": "dev", "num_seeds": 5})
    assert five == [1000, 1001, 1002, 1003, 1004]


def test_explicit_seed_list_must_be_dev_seeds():
    policy, seeds = sweep.resolve_seed_policy([1000, 2026])
    assert policy == "explicit-dev"
    assert seeds == [1000, 2026]
    with pytest.raises(sweep.SweepConfigError):
        sweep.resolve_seed_policy([1000, 7])  # 7 is an eval seed


def test_unknown_policy_rejected():
    with pytest.raises(sweep.SweepConfigError):
        sweep.resolve_seed_policy("prod")


# ------------------------------------------------------------ expansion counts


def test_grid_expansion_counts(tmp_path):
    config = base_sweep_config(
        seeds={"policy": "dev", "num_seeds": 5},
        grid={"optimizer.lr": [0.0, 0.1], "train.steps": [2, 3, 4]},
    )
    plan = sweep.expand_sweep(config, tmp_path / "devsweep.yaml")
    assert len(plan["variants"]) == 6  # 2 x 3
    assert len(plan["runs"]) == 30  # 6 variants x 5 seeds
    lrs = {v["config"]["optimizer"]["lr"] for v in plan["variants"]}
    steps = {v["config"]["train"]["steps"] for v in plan["variants"]}
    assert lrs == {0.0, 0.1} and steps == {2, 3, 4}
    # Variant names unique
    names = [v["name"] for v in plan["variants"]]
    assert len(set(names)) == len(names)


def test_no_grid_single_variant(tmp_path):
    plan = sweep.expand_sweep(base_sweep_config(seeds="dev"), tmp_path / "d.yaml")
    assert len(plan["variants"]) == 1
    assert len(plan["runs"]) == sweep.DEV_SEED_DEFAULT_COUNT


def test_eval_policy_expansion_seed_count(tmp_path):
    plan = sweep.expand_sweep(base_sweep_config(seeds="eval"), tmp_path / "e.yaml")
    assert plan["seed_policy"] == "eval"
    assert plan["seeds"] == list(range(100))
    assert len(plan["runs"]) == 100


# ------------------------------------------------- eval-seed-literal refusal


def test_refuses_top_level_eval_seed(tmp_path):
    config = base_sweep_config()
    config["seed"] = 5  # literal eval seed
    with pytest.raises(sweep.SweepConfigError, match="eval seed"):
        sweep.expand_sweep(config, tmp_path / "bad.yaml")


def test_refuses_nested_eval_seed(tmp_path):
    config = base_sweep_config()
    config["data"]["shuffle_seed"] = 42
    with pytest.raises(sweep.SweepConfigError, match="eval seed"):
        sweep.expand_sweep(config, tmp_path / "bad.yaml")


def test_refuses_eval_seed_in_seed_list(tmp_path):
    config = base_sweep_config(seeds=[0, 1, 2])
    with pytest.raises(sweep.SweepConfigError):
        sweep.expand_sweep(config, tmp_path / "bad.yaml")


def test_allows_dev_seed_literals_and_seed_counts(tmp_path):
    config = base_sweep_config(seeds={"policy": "dev", "num_seeds": 3})
    config["seed"] = 1234  # base dev seed is fine (and gets dropped)
    plan = sweep.expand_sweep(config, tmp_path / "ok.yaml")
    assert len(plan["runs"]) == 3
    # num_seeds: 3 must not be misread as an eval-seed literal
    assert sweep.find_eval_seed_literals(config["sweep"]) == []


def test_find_eval_seed_literals_ignores_non_seed_keys():
    assert sweep.find_eval_seed_literals({"train": {"steps": 10}}) == []
    assert sweep.find_eval_seed_literals({"seed": 1234}) == []
    assert sweep.find_eval_seed_literals({"seed": 99}) == ["seed = 99"]


# ------------------------------------------------------- materialized outputs


def test_materialized_configs_have_no_seed_keys_and_no_eval_literals(tmp_path):
    config_path = write_config(
        tmp_path,
        base_sweep_config(seeds="eval", grid={"optimizer.lr": [0.0, 0.1]}),
    )
    out_dir = tmp_path / "out"
    with open(config_path) as fh:
        plan = sweep.expand_sweep(yaml.safe_load(fh), config_path)
    manifest = sweep.write_plan(plan, out_dir)

    yaml_files = sorted(out_dir.glob("*.yaml"))
    assert len(yaml_files) == 2
    for path in yaml_files:
        with open(path) as fh:
            materialized = yaml.safe_load(fh)
        assert "seed" not in materialized
        assert "sweep" not in materialized
        assert sweep.find_eval_seed_literals(materialized) == []

    # Seeds travel only on the run.py command line.
    assert manifest["n_runs"] == 200
    for run in manifest["runs"]:
        assert "--seed" in run["command"]
        assert run["command"][run["command"].index("--seed") + 1] == str(run["seed"])

    # manifest + runnable script exist
    assert json.loads((out_dir / "manifest.json").read_text())["n_runs"] == 200
    run_all = (out_dir / "run_all.sh").read_text()
    assert run_all.count("scripts/run.py") == 200


def test_write_plan_refuses_configs_dir(tmp_path):
    plan = sweep.expand_sweep(base_sweep_config(), tmp_path / "d.yaml")
    with pytest.raises(sweep.SweepConfigError, match="configs/"):
        sweep.write_plan(plan, REPO_ROOT / "configs" / "generated")
    assert not (REPO_ROOT / "configs" / "generated").exists()


def test_write_plan_refuses_existing_manifest(tmp_path):
    plan = sweep.expand_sweep(base_sweep_config(), tmp_path / "d.yaml")
    out = tmp_path / "out"
    sweep.write_plan(plan, out)
    with pytest.raises(sweep.SweepConfigError, match="manifest"):
        sweep.write_plan(plan, out)


# ------------------------------------------------------------------ main/CLI


def test_main_dry_run_writes_nothing(tmp_path, capsys):
    config_path = write_config(tmp_path, base_sweep_config(seeds="dev"))
    out_dir = tmp_path / "never_created"
    rc = sweep.main([str(config_path), "--dry-run", "--out-dir", str(out_dir)])
    assert rc == 0
    assert not out_dir.exists()
    captured = capsys.readouterr()
    assert "dry run" in captured.out


def test_main_rejects_eval_literal_config(tmp_path, capsys):
    config = base_sweep_config()
    config["seed"] = 3
    config_path = write_config(tmp_path, config, "bad.yaml")
    rc = sweep.main([str(config_path), "--dry-run"])
    assert rc == 2
    assert "eval seed" in capsys.readouterr().err


def test_main_writes_expansion(tmp_path):
    config_path = write_config(
        tmp_path, base_sweep_config(seeds={"policy": "dev", "num_seeds": 2})
    )
    out_dir = tmp_path / "out"
    rc = sweep.main([str(config_path), "--out-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "run_all.sh").exists()


# ----------------------------------------------------- repo config invariants


def test_repo_sweep_configs_expand_cleanly_without_eval_literals():
    """Every checked-in sweep config must parse, contain no eval-seed
    literals, and expand (dry logic only — nothing written)."""
    sweep_configs = sorted((REPO_ROOT / "configs").glob("wp*_*.yaml"))
    assert sweep_configs, "expected wp* sweep configs under configs/"
    for path in sweep_configs:
        with open(path) as fh:
            config = yaml.safe_load(fh)
        assert sweep.find_eval_seed_literals(config) == [], path
        if isinstance(config.get("sweep"), dict):
            plan = sweep.expand_sweep(config, path)
            for variant in plan["variants"]:
                assert "seed" not in variant["config"]


def test_wp01_eval_config_expands_to_100_eval_runs():
    path = REPO_ROOT / "configs" / "wp01_airbench_eval.yaml"
    with open(path) as fh:
        plan = sweep.expand_sweep(yaml.safe_load(fh), path)
    assert plan["seed_policy"] == "eval"
    assert len(plan["runs"]) == 100
    assert sorted(r["seed"] for r in plan["runs"]) == list(range(100))
