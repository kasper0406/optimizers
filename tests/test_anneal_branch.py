"""Tests for the anneal-dissection experiment (`airbench_anneal_branch`,
docs/litreview/j-theory-theorem-sweep.md §6, flow-first program item 1).

Three surfaces, all CPU-only:

  * ``src/optim/train_snapshot.py`` — the bit-exact snapshot/restore the
    experiment branches on: roundtrip, divergence detection, determinism of a
    continuation replayed from two restores, and the mismatch refusals;
  * ``src.optim.airbench_zoo._resolve_branch`` / ``run_airbench_anneal_branch``
    config validation (every refusal fires before any CUDA work), plus
    ``configs/wpj_flow_anneal_branch.yaml``;
  * ``scripts/analyze_anneal_branch.py`` on a synthetic seed-paired results
    directory (pairing, hand-computable paired deltas, k*, steps_saved,
    and the SystemExit refusals), mirroring tests/test_analyze_ema.py.

The CUDA training path itself is exercised on the cloud box.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.optim.train_snapshot import (
    restore_training_state,
    snapshot_equal,
    snapshot_training_state,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "rm_analyze_anneal_branch", REPO_ROOT / "scripts" / "analyze_anneal_branch.py"
)
analyze_anneal_branch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_anneal_branch)


# --------------------------------------------------------- train_snapshot

# A stack with floating buffers (BatchNorm running stats) AND an integer one
# (num_batches_tracked, which the snapshot deliberately skips).
def _make_model(width: int = 8, seed: int = 1234):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(4, width),
        torch.nn.BatchNorm1d(width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 3),
    )


def _make_optimizers(model):
    """Two optimizers over disjoint param groups, as the experiment uses:
    momentum SGD (stateful) on the head, plain SGD on the rest."""
    head = list(model[3].parameters())
    head_ids = {id(p) for p in head}
    rest = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        torch.optim.SGD(head, lr=0.1, momentum=0.9, nesterov=True),
        torch.optim.SGD(rest, lr=0.05),
    ]


def _batches(n: int, seed: int = 7):
    gen = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(6, 4, generator=gen),
            torch.randint(0, 3, (6,), generator=gen),
        )
        for _ in range(n)
    ]


def _train_step(model, optimizers, batch):
    inputs, labels = batch
    loss = torch.nn.functional.cross_entropy(model(inputs), labels)
    loss.backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)


def _params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _params_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_snapshot_roundtrip_and_deterministic_continuation():
    model = _make_model()
    model.train()
    optimizers = _make_optimizers(model)
    batches = _batches(8)

    for batch in batches[:3]:
        _train_step(model, optimizers, batch)

    snap = snapshot_training_state(model, optimizers)
    assert snapshot_equal(model, optimizers, snap)
    # momentum buffers exist, so the snapshot carries real optimizer state
    assert snap["optimizers"][0]["state"]
    # BatchNorm running stats were captured; num_batches_tracked was not
    assert "1.running_mean" in snap["buffers"]
    assert "1.num_batches_tracked" not in snap["buffers"]

    # continuation taken right after the snapshot (the reference)
    _train_step(model, optimizers, batches[3])
    reference = _params(model)
    assert not snapshot_equal(model, optimizers, snap)

    # diverge further: params, buffers and optimizer state all move on
    for batch in batches[4:7]:
        _train_step(model, optimizers, batch)
    assert not snapshot_equal(model, optimizers, snap)

    restore_training_state(model, optimizers, snap)
    assert snapshot_equal(model, optimizers, snap)

    # replay the identical continuation step from the restored state
    _train_step(model, optimizers, batches[3])
    replay_a = _params(model)
    assert _params_equal(replay_a, reference)

    # ... and again from a second restore of the SAME snapshot object
    restore_training_state(model, optimizers, snap)
    assert snapshot_equal(model, optimizers, snap)
    _train_step(model, optimizers, batches[3])
    replay_b = _params(model)
    assert _params_equal(replay_b, replay_a)


def test_snapshot_detects_buffer_only_divergence():
    """Running stats move in train mode even when no optimizer steps."""
    model = _make_model()
    model.train()
    optimizers = _make_optimizers(model)
    snap = snapshot_training_state(model, optimizers)
    model(_batches(1)[0][0])  # forward only: updates BatchNorm running stats
    assert not snapshot_equal(model, optimizers, snap)
    restore_training_state(model, optimizers, snap)
    assert snapshot_equal(model, optimizers, snap)


def test_restore_refuses_mismatched_parameter_names():
    model = _make_model()
    optimizers = _make_optimizers(model)
    snap = snapshot_training_state(model, optimizers)

    other = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.BatchNorm1d(8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
        torch.nn.Linear(3, 3),  # extra layer -> extra parameter names
    )
    with pytest.raises(SystemExit, match="parameter names do not match"):
        restore_training_state(other, _make_optimizers(model), snap)


def test_restore_refuses_mismatched_buffers():
    model = _make_model()
    optimizers = _make_optimizers(model)
    snap = snapshot_training_state(model, optimizers)
    # identical parameter names/shapes, but LayerNorm keeps no running stats
    plain = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.LayerNorm(8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    )
    with pytest.raises(SystemExit, match="buffer names do not match"):
        restore_training_state(plain, optimizers, snap)


def test_restore_refuses_optimizer_count_mismatch():
    model = _make_model()
    optimizers = _make_optimizers(model)
    snap = snapshot_training_state(model, optimizers)
    with pytest.raises(SystemExit, match="optimizer count mismatch"):
        restore_training_state(model, optimizers[:1], snap)


def test_restore_refuses_mismatched_shapes():
    """Same names, different widths: restore must not silently broadcast.

    (Deviation from a pure-SystemExit expectation: a shape clash is caught by
    ``Tensor.copy_`` itself, which raises RuntimeError.)"""
    model = _make_model(width=8)
    optimizers = _make_optimizers(model)
    snap = snapshot_training_state(model, optimizers)
    wider = _make_model(width=16)
    with pytest.raises((SystemExit, RuntimeError)):
        restore_training_state(wider, _make_optimizers(wider), snap)


# ------------------------------------------------------------ _resolve_branch


def _branch_config(steps, lengths):
    return {"branch": {"branch_steps": steps, "anneal_lengths": lengths}}


def test_resolve_branch_accepts_config_shape():
    from src.optim.airbench_zoo import _resolve_branch

    steps, lengths = _resolve_branch(
        _branch_config([100, 150, 200], [0, 5, 10, 25, 50])
    )
    assert steps == [100, 150, 200]
    assert lengths == [0, 5, 10, 25, 50]
    # the bound check passes when the last branch point is inside the run
    assert _resolve_branch(_branch_config([100, 200], [0, 5]), 200)[0] == [100, 200]


@pytest.mark.parametrize(
    "config",
    [
        {},  # missing block entirely
        {"branch": None},
        {"branch": [100, 150]},  # not a mapping
        {"branch": {"branch_steps": [100]}},  # missing anneal_lengths
        {  # extra key
            "branch": {
                "branch_steps": [100],
                "anneal_lengths": [0],
                "extra": 1,
            }
        },
        _branch_config([], [0, 5]),  # empty list
        _branch_config([100], []),
        _branch_config([100, "150"], [0, 5]),  # non-int entry
        _branch_config([100], [0, 5.0]),
        _branch_config([True], [0]),  # booleans are not step counts
        _branch_config([100], [True, 5]),
        _branch_config([150, 100], [0, 5]),  # unsorted
        _branch_config([100], [5, 0]),
        _branch_config([100, 100], [0, 5]),  # duplicates
        _branch_config([100], [0, 0]),
        _branch_config([0, 100], [0, 5]),  # branch_steps[0] < 1
        _branch_config([-10, 100], [0, 5]),
        _branch_config([100], [-1, 0]),  # negative anneal length
    ],
)
def test_resolve_branch_rejects_invalid(config):
    from src.optim.airbench_zoo import _resolve_branch

    with pytest.raises(SystemExit):
        _resolve_branch(config)


def test_resolve_branch_rejects_branch_step_beyond_run():
    from src.optim.airbench_zoo import _resolve_branch

    config = _branch_config([100, 150, 300], [0, 5])
    _resolve_branch(config)  # fine without the bound
    with pytest.raises(SystemExit, match="beyond the run"):
        _resolve_branch(config, 200)


# ------------------------------------------- run_airbench_anneal_branch guards


def test_anneal_branch_refuses_optimizer_override():
    from src.optim.airbench_zoo import run_airbench_anneal_branch

    config = dict(
        _branch_config([100], [0, 5]),
        optimizer={"name": "muon"},
    )
    with pytest.raises(SystemExit, match="does not accept an optimizer"):
        run_airbench_anneal_branch(config, torch.device("cpu"))


def test_anneal_branch_requires_branch_block():
    from src.optim.airbench_zoo import run_airbench_anneal_branch

    with pytest.raises(SystemExit, match="needs a branch: block"):
        run_airbench_anneal_branch({"recipe": {}}, torch.device("cpu"))


# ------------------------------------------------------------------- config


def test_flow_anneal_branch_config_is_a_dev_seed_branch_run():
    from src.optim.airbench_zoo import _resolve_branch

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "wpj_flow_anneal_branch.yaml").read_text()
    )
    assert config["experiment"] == "airbench_anneal_branch"
    assert config["sweep"]["seeds"]["policy"] == "dev"
    assert config["seed"] >= 1000
    assert "optimizer" not in config
    steps, lengths = _resolve_branch(config)
    assert steps == [100, 150, 200]
    assert lengths == [0, 5, 10, 25, 50]


# ----------------------------------------------------------------- analyzer

# Synthetic design: per-seed branch accuracy = that seed's stock final + a
# planted delta, so every paired delta is exact by construction. Deltas carry
# a +0.0002 per-seed tilt, so the two-seed mean is (planted + 0.0001) and the
# CI is non-degenerate. TTA readouts sit +0.012 above raw on the branch runs
# and +0.010 on the stock runs, so the TTA delta is the val delta + 0.002.
BRANCH_STEPS = [100, 150, 200]
ANNEAL_LENGTHS = [0, 5, 10, 25, 50]
STOCK_FINALS = {1000: 0.9300, 1001: 0.9320}
PLANTED = {
    # k* = 10 (the middle length): the first k within 0.002 of the stock final
    100: {0: -0.0500, 5: -0.0100, 10: -0.0010, 25: -0.0005, 50: 0.0000},
    # k* = None: even k=50 stays 0.0029 below the stock final
    150: {0: -0.0400, 5: -0.0200, 10: -0.0100, 25: -0.0050, 50: -0.0030},
    # k* = 0: branching at the end of the budget already matches
    200: {0: -0.0015, 5: -0.0010, 10: -0.0005, 25: 0.0000, 50: 0.0005},
}
BASE_VAL = {100: 0.8700, 150: 0.8900, 200: 0.9000}


def _write_branch_run(tmp_path, name, seed, started="2026-08-02T00:00:00", **over):
    stock = STOCK_FINALS[seed]
    tilt = 0.0002 * (seed - 1000)
    payload = {
        "experiment": "airbench_anneal_branch",
        "seed": seed,
        "started_at": started,
        "gpu_type": "NVIDIA L40",
        "metrics": {
            "lr_schedule": "constant",
            "steps": 200,
            "branch_steps": BRANCH_STEPS,
            "anneal_lengths": ANNEAL_LENGTHS,
            "base_val_accs": {str(t): BASE_VAL[t] for t in BRANCH_STEPS},
            "branches": {
                str(t): {
                    str(k): {
                        "val_acc": stock + PLANTED[t][k] + tilt,
                        "tta_val_acc": stock + PLANTED[t][k] + tilt + 0.012,
                    }
                    for k in ANNEAL_LENGTHS
                }
                for t in BRANCH_STEPS
            },
            "final_val_acc": 0.9000 + tilt,
            "final_tta_val_acc": 0.9120 + tilt,
        },
    }
    payload.update(over)
    (tmp_path / name).write_text(json.dumps(payload))


def _write_stock_run(
    tmp_path, name, seed, schedule="linear", started="2026-08-02T00:00:00"
):
    val = STOCK_FINALS[seed] if schedule == "linear" else 0.9000
    payload = {
        "experiment": "airbench_ema",
        "seed": seed,
        "started_at": started,
        "gpu_type": "NVIDIA L40",
        "metrics": {
            "lr_schedule": schedule,
            "steps": 200,
            "val_accs": [0.80, 0.90, val],
            "val_acc": val,
            "tta_val_acc": val + 0.010,
            "ema_val_accs": {"0.9": [0.79, 0.89, val]},
            "ema_tta_val_accs": {"0.9": val + 0.010},
        },
    }
    (tmp_path / name).write_text(json.dumps(payload))


def _populate(tmp_path):
    for seed in (1000, 1001):
        _write_branch_run(tmp_path, f"branch_{seed}.json", seed)
        _write_stock_run(tmp_path, f"ema_lin_{seed}.json", seed)
        # the constant EMA arm must be ignored as a baseline
        _write_stock_run(tmp_path, f"ema_con_{seed}.json", seed, schedule="constant")


def test_analyze_pairs_seeds_and_computes_paired_deltas(tmp_path):
    _populate(tmp_path)
    result = analyze_anneal_branch.analyze(tmp_path)

    assert result["n_paired_seeds"] == 2
    assert result["seeds"] == [1000, 1001]
    assert result["gpu_type"] == "NVIDIA L40"
    assert result["total_steps"] == 200
    assert result["branch_steps"] == BRANCH_STEPS
    assert result["anneal_lengths"] == ANNEAL_LENGTHS
    # stock finals: mean of 0.9300 and 0.9320; TTA is +0.010 on both
    assert result["stock_final_val_mean"] == pytest.approx(0.9310, abs=1e-12)
    assert result["stock_final_tta_mean"] == pytest.approx(0.9410, abs=1e-12)
    # constant-arm final at the full budget: 0.9000 and 0.9002
    assert result["constant_final_val_mean"] == pytest.approx(0.9001, abs=1e-12)
    assert result["constant_final_tta_mean"] == pytest.approx(0.9121, abs=1e-12)

    block = result["per_branch"]["100"]
    assert block["base_val_mean"] == pytest.approx(0.8700, abs=1e-12)
    # planted -0.0010 with a +0.0002 tilt on seed 1001 -> mean -0.0009
    val_delta = block["per_k"]["10"]["val_delta"]
    assert val_delta["n"] == 2
    assert val_delta["mean"] == pytest.approx(-0.0009, abs=1e-12)
    # ci95 = t(df=1) * sd / sqrt(2), sd = 0.0002/sqrt(2) over the two seeds
    assert val_delta["ci95"] == pytest.approx(12.706 * 0.0001, abs=1e-9)
    # TTA readouts sit 0.002 higher on the branch runs than on the stock runs
    assert block["per_k"]["10"]["tta_delta"]["mean"] == pytest.approx(
        0.0011, abs=1e-12
    )
    assert block["per_k"]["0"]["val_delta"]["mean"] == pytest.approx(
        -0.0499, abs=1e-12
    )


def test_analyze_picks_k_star_and_steps_saved(tmp_path):
    _populate(tmp_path)
    result = analyze_anneal_branch.analyze(tmp_path)
    assert result["k_star_tol"] == -0.002

    # branch 100: k=5 is -0.0099 (too far), k=10 is -0.0009 -> the middle k
    assert result["per_branch"]["100"]["k_star"] == 10
    assert result["per_branch"]["100"]["steps_saved"] == 200 - (100 + 10)
    # branch 150: even k=50 is -0.0029, below the tolerance -> no k qualifies
    assert result["per_branch"]["150"]["k_star"] is None
    assert result["per_branch"]["150"]["steps_saved"] is None
    # branch 200: k=0 already qualifies; nothing is saved over the budget
    assert result["per_branch"]["200"]["k_star"] == 0
    assert result["per_branch"]["200"]["steps_saved"] == 0

    md = analyze_anneal_branch.to_markdown(result)
    assert "Branch step 100" in md
    assert "k* = 10" in md and "steps saved 90" in md
    assert "k* = none" in md


def test_analyze_main_writes_outputs(tmp_path):
    _populate(tmp_path)
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    code = analyze_anneal_branch.main(
        [str(tmp_path), "--json", str(out_json), "--md", str(out_md)]
    )
    assert code == 0
    assert json.loads(out_json.read_text())["per_branch"]["100"]["k_star"] == 10
    assert "Anneal dissection" in out_md.read_text()


def test_unpaired_or_empty_dir_refused(tmp_path):
    with pytest.raises(SystemExit, match="no seed-paired"):
        analyze_anneal_branch.analyze(tmp_path)
    # branch runs alone are not enough: the stock baseline must be present
    _write_branch_run(tmp_path, "branch_1000.json", 1000)
    with pytest.raises(SystemExit, match="no seed-paired"):
        analyze_anneal_branch.analyze(tmp_path)
    # ... and neither is a stock run for a different seed
    _write_stock_run(tmp_path, "ema_lin_1001.json", 1001)
    with pytest.raises(SystemExit, match="no seed-paired"):
        analyze_anneal_branch.analyze(tmp_path)


def test_mixed_gpu_types_refused(tmp_path):
    _populate(tmp_path)
    bad = json.loads((tmp_path / "branch_1001.json").read_text())
    bad["gpu_type"] = "NVIDIA RTX A6000"
    (tmp_path / "branch_1001.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="mixed GPU types"):
        analyze_anneal_branch.analyze(tmp_path)


def test_duplicate_seed_keeps_earliest(tmp_path):
    _populate(tmp_path)
    payload = json.loads((tmp_path / "branch_1000.json").read_text())
    for t in BRANCH_STEPS:
        for k in ANNEAL_LENGTHS:
            payload["metrics"]["branches"][str(t)][str(k)]["val_acc"] = 0.5
    payload["started_at"] = "2026-08-03T00:00:00"
    (tmp_path / "branch_1000_rerun.json").write_text(json.dumps(payload))
    result = analyze_anneal_branch.analyze(tmp_path)
    # the later rerun must not displace the original run's numbers
    assert result["per_branch"]["100"]["per_k"]["10"]["val_delta"][
        "mean"
    ] == pytest.approx(-0.0009, abs=1e-12)
