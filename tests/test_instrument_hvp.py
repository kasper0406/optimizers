"""AirbenchHvpProbe tests (Phase-1 disambiguation, Task 2). CPU-only.

Covers: analytic correctness on a quadratic (H known exactly), fp32-safety
on a half-precision model, read-only guarantee, per-batch graph caching, the
hub wiring end to end (lambda_hvp populated once per pair per refresh,
sidecar schema-valid), the eta-lambda calibration write path
(collect_calibration_points non-empty from a log with HVP records and
oscillating-classified directions), and the config/experiment gating.

Seeds: dev seeds only (>= 1000).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import importlib.util

import numpy as np
import pytest
import torch
import yaml

from src.instrument import hub_from_config, write_sidecar
from src.instrument.hvp import AirbenchHvpProbe
from src.instrument.plots import collect_calibration_points
from src.instrument.schema import INSTRUMENTATION_SCHEMA_VERSION
from src.instrument.tracker import MatrixTracker
from src.optim import Muon
from src.optim import airbench_zoo

HVP_CONFIG = REPO_ROOT / "configs" / "dev" / "instrumented_airbench_hvp.yaml"
CLASSIFIER_KWARGS = dict(tau_sig=4.0, tau_noise=2.0, rho_osc=0.5, n_min=40)


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_module("sweep_module_hvp", "scripts/sweep.py")


def _quadratic_setup(dtype=torch.float32, seed=1600):
    """f(W) = 0.5 * sum_b ||W x_b||^2 -- the Hessian contraction along D is
    exactly sum_b ||D x_b||^2 = ||X D^T||_F^2, no approximation."""
    gen = torch.Generator().manual_seed(seed)
    model = torch.nn.Linear(5, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.randn(3, 5, generator=gen))
    model = model.to(dtype)
    x = torch.randn(7, 5, generator=gen).to(dtype)
    labels = torch.zeros(7, dtype=torch.long)  # unused by the loss below

    def loss_fn(outputs, labels):
        return 0.5 * (outputs ** 2).sum()

    probe = AirbenchHvpProbe(model, [model.weight], loss_fn=loss_fn)
    probe.set_batch(x, labels)
    return model, x, probe, gen


def _analytic(x, D):
    return float((x.float() @ D.float().t()).pow(2).sum())


# ------------------------------------------------------ analytic correctness


def test_hvp_matches_analytic_quadratic_fp32():
    model, x, probe, gen = _quadratic_setup()
    for _ in range(4):
        D = torch.randn(3, 5, generator=gen)
        D = D / D.norm()
        lam = probe(model.weight, D)
        assert lam == pytest.approx(_analytic(x, D), rel=1e-5)


def test_hvp_flattened_direction_is_reshaped():
    """The tracker hands D shaped like the (flattened) matrix; the probe must
    reshape onto the parameter."""
    model, x, probe, gen = _quadratic_setup()
    D = torch.randn(3, 5, generator=gen)
    D = D / D.norm()
    assert probe(model.weight, D.reshape(-1)) == pytest.approx(
        _analytic(x, D), rel=1e-5
    )


def test_hvp_fp16_model_computed_in_fp32():
    """On a half-precision model the probe must recompute in fp32: the result
    matches the fp32 analytic value to fp16-input accuracy, far tighter than
    an fp16 double-backward could guarantee -- and the training model is left
    byte-identical (params, dtypes, grads)."""
    model, x, probe, gen = _quadratic_setup(dtype=torch.float16)
    w_before = model.weight.detach().clone()
    D32 = torch.randn(3, 5, generator=gen)
    D32 = D32 / D32.norm()
    D16 = D32.to(torch.float16)  # the tracker casts D to param dtype
    lam = probe(model.weight, D16)
    # Ground truth from the fp32 values the fp16 tensors actually hold.
    expected = _analytic(x, D16)
    assert lam == pytest.approx(expected, rel=1e-3)
    # Read-only guarantee.
    assert model.weight.dtype == torch.float16
    assert torch.equal(model.weight.detach(), w_before)
    assert model.weight.grad is None
    assert model.weight.requires_grad


def test_hvp_positive_for_this_convex_loss():
    model, x, probe, gen = _quadratic_setup()
    D = torch.randn(3, 5, generator=gen)
    assert probe(model.weight, D / D.norm()) > 0


# ------------------------------------------------------------------- caching


def test_graph_built_once_per_batch_and_invalidated_on_set_batch():
    gen = torch.Generator().manual_seed(1601)
    model = torch.nn.Linear(4, 2, bias=False)
    calls = {"n": 0}

    def loss_fn(outputs, labels):
        calls["n"] += 1
        return (outputs ** 2).sum()

    probe = AirbenchHvpProbe(model, [model.weight], loss_fn=loss_fn)
    probe.set_batch(torch.randn(3, 4, generator=gen), torch.zeros(3))
    D = torch.eye(2, 4)
    probe(model.weight, D)
    probe(model.weight, D)
    assert calls["n"] == 1 and probe.n_graph_builds == 1
    probe.set_batch(torch.randn(3, 4, generator=gen), torch.zeros(3))
    probe(model.weight, D)
    assert calls["n"] == 2 and probe.n_graph_builds == 2


def test_probe_errors():
    gen = torch.Generator().manual_seed(1602)
    model = torch.nn.Linear(4, 2, bias=False)
    probe = AirbenchHvpProbe(model, [model.weight])
    with pytest.raises(RuntimeError, match="set_batch"):
        probe(model.weight, torch.zeros(2, 4))
    probe.set_batch(torch.randn(3, 4, generator=gen), torch.zeros(3, dtype=torch.long))
    other = torch.nn.Linear(4, 2, bias=False)
    with pytest.raises(KeyError, match="tracked"):
        probe(other.weight, torch.zeros(2, 4))


def test_default_loss_is_label_smoothed_sum_ce():
    """Default loss must equal the airbench training loss (fp32)."""
    gen = torch.Generator().manual_seed(1603)
    model = torch.nn.Linear(6, 4, bias=False)
    x = torch.randn(5, 6, generator=gen)
    y = torch.randint(0, 4, (5,), generator=gen)
    probe = AirbenchHvpProbe(model, [model.weight], label_smoothing=0.2)
    with torch.no_grad():
        out = model(x)
        expected = torch.nn.functional.cross_entropy(
            out, y, label_smoothing=0.2, reduction="sum"
        )
    assert float(probe.loss_fn(out, y)) == pytest.approx(float(expected))


# ------------------------------------------------------------------ hub wiring


def test_hub_populates_lambda_hvp_once_per_pair_per_refresh(tmp_path):
    """Real training loop (tiny fp32 MLP + Muon) with the REAL probe through
    hub_from_config(hvp: true): every tracked direction gets one finite
    lambda per refresh; the log validates as a sidecar with
    hvp_enabled=true."""
    torch.manual_seed(1610)
    model = torch.nn.Sequential(
        torch.nn.Linear(12, 16, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 4, bias=False),
    )
    params = [p for p in model.parameters() if p.ndim >= 2]
    optimizer = Muon(
        params, lr=0.05, momentum=0.9, nesterov=True, ns_steps=3,
        ns_dtype=torch.float32,
    )
    probe = AirbenchHvpProbe(
        model,
        params,
        loss_fn=lambda out, y: torch.nn.functional.mse_loss(
            out, y, reduction="sum"
        ),
    )
    instr = dict(
        k1=3, k2=2, t_refresh=10, betas=[0.9, 0.99], snapshot_every=5,
        seed=1610, hvp=True, classifier=CLASSIFIER_KWARGS,
    )
    hub = hub_from_config(
        instr, list(model.named_parameters()), optimizer, hvp_fn=probe
    )
    gen = torch.Generator().manual_seed(1611)
    for _ in range(25):
        x = torch.randn(8, 12, generator=gen)
        y = torch.randn(8, 4, generator=gen)
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(x), y, reduction="sum").backward()
        probe.set_batch(x, y)
        hub.capture_grads()
        optimizer.step()
        hub.after_step()

    log = hub.to_log()
    assert log["hvp_enabled"] is True
    n_refreshes = 3  # steps 1, 11, 21 for t_refresh=10 over 25 steps
    for mat in log["matrices"].values():
        for d in mat["directions"]:
            assert d["lambda_hvp"]["step"] == [1, 11, 21]
            assert len(d["lambda_hvp"]["value"]) == n_refreshes
            assert all(np.isfinite(v) for v in d["lambda_hvp"]["value"])
    assert probe.n_graph_builds == n_refreshes  # cached across pairs/matrices
    sidecar = write_sidecar(log, tmp_path / "hvp_run_seed1610.json")
    assert sidecar.exists()


def test_hub_from_config_hvp_false_ignores_provided_probe():
    torch.manual_seed(1612)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))
    instr = dict(
        k1=2, k2=1, t_refresh=5, betas=[0.9], seed=1612, hvp=False,
        classifier=CLASSIFIER_KWARGS,
    )
    hub = hub_from_config(
        instr, list(model.named_parameters()), None, hvp_fn=lambda p, D: 1.0
    )
    assert hub.hvp_fn is None


# --------------------------------------------- calibration plot write path


def test_calibration_points_populate_from_hvp_records():
    """A log with an oscillating direction and per-refresh HVP records must
    yield non-empty eta-lambda calibration points (the WP1.2 plot was empty
    solely because lambda_hvp was empty)."""
    gen = torch.Generator().manual_seed(1620)
    rng = np.random.default_rng(1620)
    m, n = 20, 14
    U0, _ = torch.linalg.qr(torch.randn(m, 3, generator=gen))
    V0, _ = torch.linalg.qr(torch.randn(n, 3, generator=gen))
    M = (
        15.0 * torch.outer(U0[:, 0], V0[:, 0])
        + 7.0 * torch.outer(U0[:, 1], V0[:, 1])
        + 3.0 * torch.outer(U0[:, 2], V0[:, 2])
    )
    tracker = MatrixTracker(
        "layer0", (m, n), k1=3, k2=2, t_refresh=40, betas=(0.9, 0.99),
        classifier_kwargs=CLASSIFIER_KWARGS, snapshot_every=5, generator=gen,
    )
    param = torch.zeros(m, n)
    hvp_fn = lambda p, D: 20.0  # lr * lambda = 2.0 ~ implied (1 + r) at r=1
    for t in range(200):
        c_sig = 1.0 + 0.1 * rng.standard_normal()
        c_osc = 3.0 * ((-1.0) ** t) * (1.0 + 0.02 * rng.standard_normal())
        c_noise = 0.5 * rng.standard_normal()
        G = (
            c_sig * torch.outer(U0[:, 0], V0[:, 0])
            + c_osc * torch.outer(U0[:, 1], V0[:, 1])
            + c_noise * torch.outer(U0[:, 2], V0[:, 2])
        )
        G = G + 0.02 * torch.from_numpy(
            rng.standard_normal((m, n)).astype(np.float32)
        )
        tracker.observe(G, M, hvp_fn=hvp_fn, param=param)
    log = {
        "instrumentation_schema_version": INSTRUMENTATION_SCHEMA_VERSION,
        "betas": ["0.9", "0.99"],
        "hvp_enabled": True,
        "matrices": {"layer0": tracker.to_log()},
    }
    points = collect_calibration_points(log, lr=0.1)
    for beta in ("0.9", "0.99"):
        xs, ys = points[beta]
        assert len(xs) > 0, f"no calibration points for beta {beta}"
        assert all(x == pytest.approx(0.1 * 20.0) for x in xs)
        assert all(np.isfinite(y) for y in ys)
    # The oscillating direction's implied eta*lambda ~ 1 + r = 2.
    xs, ys = points["0.9"]
    assert np.median(ys) == pytest.approx(2.0, abs=0.5)


# --------------------------------------------------------- config + gating


def test_hvp_config_parses_and_expands_to_seeds_1200_1202():
    with open(HVP_CONFIG) as fh:
        config = yaml.safe_load(fh)
    assert config["experiment"] == "airbench_instrumented"
    assert config["instrumentation"]["hvp"] is True
    assert config["recipe"]["compile"] is False
    # Identical instrumentation settings to the WP1.2 baseline, hvp aside.
    with open(REPO_ROOT / "configs" / "wp12_airbench_instrumented.yaml") as fh:
        wp12 = yaml.safe_load(fh)
    ours = dict(config["instrumentation"])
    theirs = dict(wp12["instrumentation"])
    assert ours.pop("hvp") is True and theirs.pop("hvp") is False
    assert ours == theirs
    assert config["train"] == wp12["train"]

    sweep.refuse_eval_seed_literals(config, str(HVP_CONFIG))  # must not raise
    plan = sweep.expand_sweep(config, HVP_CONFIG)
    assert plan["seed_policy"] == "explicit-dev"
    assert plan["seeds"] == [1200, 1201, 1202]
    assert len(plan["runs"]) == 3


def test_instrumented_experiment_refuses_hvp_with_compile():
    """hvp: true without recipe.compile: false must be refused (no
    double-backward through torch.compile) -- checked before any CUDA work."""
    with open(HVP_CONFIG) as fh:
        config = yaml.safe_load(fh)
    config.pop("sweep", None)
    config["recipe"].pop("compile")  # stock default: compile on
    with pytest.raises(SystemExit, match="compile"):
        airbench_zoo.run_airbench_instrumented(config, torch.device("cpu"))


def test_instrumented_experiment_hvp_with_compile_off_reaches_cuda_guard():
    with open(HVP_CONFIG) as fh:
        config = yaml.safe_load(fh)
    config.pop("sweep", None)
    with pytest.raises(SystemExit, match="CUDA"):
        airbench_zoo.run_airbench_instrumented(config, torch.device("cpu"))
