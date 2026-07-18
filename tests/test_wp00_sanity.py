"""WP0.0 sanity tests: imports, results schema, smoke config, NoOpOptimizer.

All seeds here are dev seeds (>= 1000) per CLAUDE.md ground rule 2.
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import results_io
from src.optim import MatrixOptimizer, NoOpOptimizer


# ------------------------------------------------------------------- imports


def test_packages_import():
    import src
    import src.instrument
    import src.optim
    import src.optim.interface
    import src.stats

    assert src is not None


# ------------------------------------------------------------- results schema


def sample_result():
    return {
        "schema_version": results_io.SCHEMA_VERSION,
        "experiment": "smoke",
        "config": {
            "path": "configs/smoke.yaml",
            "sha256": "0" * 64,
            "contents": {"experiment": "smoke", "seed": 1234},
        },
        "git_sha": "f" * 40,
        "git_dirty": False,
        "seed": 1234,
        "gpu_type": "cpu",
        "wall_time_s": 1.5,
        "cost_usd": None,
        "started_at": "2026-07-18T00:00:00+00:00",
        "finished_at": "2026-07-18T00:00:02+00:00",
        "metrics": {"loss_last": 1.0},
    }


def test_validate_accepts_sample():
    assert results_io.validate(sample_result()) is not None


@pytest.mark.parametrize(
    "missing_key",
    sorted(results_io.REQUIRED_KEYS),
)
def test_validate_rejects_missing_key(missing_key):
    result = sample_result()
    del result[missing_key]
    with pytest.raises(results_io.ResultsValidationError):
        results_io.validate(result)


def test_validate_rejects_bad_types():
    result = sample_result()
    result["seed"] = "1234"  # str, not int
    with pytest.raises(results_io.ResultsValidationError):
        results_io.validate(result)

    result = sample_result()
    result["config"] = {"path": "x"}  # missing sha256/contents
    with pytest.raises(results_io.ResultsValidationError):
        results_io.validate(result)

    result = sample_result()
    result["started_at"] = "yesterday"
    with pytest.raises(results_io.ResultsValidationError):
        results_io.validate(result)


def test_write_result_is_append_only(tmp_path):
    out = tmp_path / "r.json"
    results_io.write_result(sample_result(), out)
    assert out.exists()
    with pytest.raises(FileExistsError):
        results_io.write_result(sample_result(), out)
    assert results_io.load_result(out)["seed"] == 1234


# --------------------------------------------------------------- smoke config


def test_smoke_config_parses_and_uses_dev_seed():
    config_path = REPO_ROOT / "configs" / "smoke.yaml"
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    assert config["experiment"] == "smoke"
    assert isinstance(config["seed"], int)
    assert config["seed"] >= 1000, "literal config seeds must be dev seeds"
    assert config["optimizer"]["name"] == "noop"
    assert config["train"]["steps"] == 10


def test_no_eval_seeds_in_configs():
    """No literal eval seed (0-99) in any 'seed:' line under configs/."""
    for path in (REPO_ROOT / "configs").rglob("*.yaml"):
        with open(path) as fh:
            config = yaml.safe_load(fh)
        seed = config.get("seed") if isinstance(config, dict) else None
        if isinstance(seed, int):
            assert not (0 <= seed <= 99), f"{path} uses eval seed {seed}"


# ------------------------------------------------------ optimizer interface


def test_noop_optimizer_changes_nothing():
    torch.manual_seed(2026)
    model = torch.nn.Linear(8, 3)
    before = [p.detach().clone() for p in model.parameters()]

    opt = NoOpOptimizer(model.parameters(), lr=0.0)
    assert isinstance(opt, MatrixOptimizer)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = model(torch.randn(4, 8)).sum()
        loss.backward()
        opt.step()

    for p, p0 in zip(model.parameters(), before):
        assert torch.equal(p.detach(), p0)
    # Hook loop actually ran:
    assert all(opt.state[p]["step"] == 3 for p in model.parameters())


def test_default_post_step_applies_lr_and_wd():
    """A minimal identity-shaping optimizer must reduce to decoupled-WD SGD."""

    class IdentityOpt(MatrixOptimizer):
        def pre_step(self, G, state, group):
            return G

        def shape_spectrum(self, O, state, group):
            return O

    torch.manual_seed(3000)
    p = torch.nn.Parameter(torch.randn(5, 4))
    lr, wd = 0.1, 0.01
    opt = IdentityOpt([p], lr=lr, weight_decay=wd)
    p0 = p.detach().clone()
    grad = torch.randn(5, 4)
    p.grad = grad.clone()
    opt.step()
    expected = p0 * (1 - lr * wd) - lr * grad
    assert torch.allclose(p.detach(), expected, atol=1e-7)
