"""WP1.1 hub tests: InstrumentationHub attached to a real Muon training loop
(CPU), config-driven construction, parameter filtering, and the overhead
benchmark script in synthetic CPU mode.

Seeds: dev seeds only (>= 1000).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from src import results_io
from src.instrument import InstrumentationHub, hub_from_config
from src.optim import Muon

CLASSIFIER_KWARGS = dict(tau_sig=4.0, tau_noise=2.0, rho_osc=0.5, n_min=30)


def _train(model, optimizer, hub, steps, gen):
    loss_fn = torch.nn.MSELoss()
    for _ in range(steps):
        x = torch.randn(8, 12, generator=gen)
        y = torch.randn(8, 4, generator=gen)
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        optimizer.step()
        hub.after_step()


def _model():
    torch.manual_seed(1700)
    return torch.nn.Sequential(
        torch.nn.Linear(12, 16, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 4, bias=True),  # bias must be filtered out
    )


def test_hub_with_muon_on_real_training_loop():
    model = _model()
    optimizer = Muon(
        [p for p in model.parameters() if p.ndim >= 2],  # Muon: matrices only
        lr=0.05, momentum=0.9, nesterov=True,
        ns_steps=3, ns_dtype=torch.float32,
    )
    hub = InstrumentationHub(
        list(model.named_parameters()),
        optimizer,
        k1=3,
        k2=2,
        t_refresh=10,
        betas=(0.9, 0.99),
        classifier_kwargs=CLASSIFIER_KWARGS,
        snapshot_every=5,
        seed=1700,
    )
    # Only the two weight matrices tracked; the 1-D bias is skipped.
    assert set(hub.trackers) == {"0.weight", "2.weight"}

    gen = torch.Generator().manual_seed(1701)
    _train(model, optimizer, hub, steps=25, gen=gen)

    log = hub.to_log()
    # 0.weight is 16x12: full k1=3 + k2=2. 2.weight is 4x16: the tracked
    # block shrinks to fit min(m, n) = 4 (k1=3, k2=1).
    expected_dirs = {"0.weight": 5, "2.weight": 4}
    for name in ("0.weight", "2.weight"):
        mat = log["matrices"][name]
        assert mat["steps"] == list(range(1, 26))
        assert mat["refresh_steps"] == [1, 11, 21]
        assert len(mat["directions"]) == expected_dirs[name]
        for d in mat["directions"]:
            assert len(d["s"]) == 25
            # Momentum-based tracking: sigma estimates recorded per refresh.
            assert len(d["sigma"]["step"]) == 3
        # ||G||_F logged every step and strictly positive.
        assert all(v > 0 for v in mat["grad_fro_norm"])


def test_hub_reads_momentum_buffer_not_gradient():
    model = _model()
    optimizer = Muon(
        [p for p in model.parameters() if p.ndim >= 2],  # Muon: matrices only
        lr=0.05, momentum=0.9, nesterov=True,
        ns_steps=3, ns_dtype=torch.float32,
    )
    hub = InstrumentationHub(
        list(model.named_parameters()),
        optimizer,
        k1=2,
        k2=1,
        t_refresh=5,
        betas=(0.9,),
        classifier_kwargs=CLASSIFIER_KWARGS,
        seed=1700,
    )
    gen = torch.Generator().manual_seed(1702)
    _train(model, optimizer, hub, steps=12, gen=gen)
    p = dict(model.named_parameters())["0.weight"]
    buf = optimizer.state[p]["momentum_buffer"]
    tracker = hub.trackers["0.weight"]
    # The tracker's per-step top sigma must match |u^T buf v| for its own
    # tracked top pair -- i.e. the subspace follows the momentum matrix.
    expected = float(tracker.subspace.project(buf.float())[: tracker.subspace.k1].abs().max())
    assert abs(tracker.top_sigma_m[-1] - expected) < 1e-5


def test_hub_from_config_matches_yaml_block():
    cfg_path = REPO_ROOT / "configs/dev/instrumented_mlp_smoke.yaml"
    with open(cfg_path) as fh:
        config = yaml.safe_load(fh)
    assert config["seed"] >= 1000  # seed discipline
    instr = config["instrumentation"]

    model = _model()
    optimizer = Muon(
        [p for p in model.parameters() if p.ndim >= 2],
        lr=0.05, momentum=0.6, ns_dtype=torch.float32,
    )
    instr = dict(instr, min_dim=2)  # the test model is smaller than airbench
    hub = hub_from_config(instr, list(model.named_parameters()), optimizer)
    tracker = next(iter(hub.trackers.values()))
    assert tracker.t_refresh == instr["t_refresh"]
    assert hub.betas == [0.9, 0.99]

    # Missing classifier block must be rejected (no scientific defaults).
    with pytest.raises(ValueError, match="classifier"):
        hub_from_config({}, list(model.named_parameters()), optimizer)
    # hvp: true without a callback must be rejected.
    with pytest.raises(ValueError, match="hvp"):
        hub_from_config(
            dict(instr, hvp=True), list(model.named_parameters()), optimizer
        )


def test_hub_requires_matrix_params():
    bias_only = [("b", torch.nn.Parameter(torch.zeros(7)))]
    with pytest.raises(ValueError, match="no matrix parameters"):
        InstrumentationHub(
            bias_only, None, classifier_kwargs=CLASSIFIER_KWARGS, seed=1700
        )


def test_bench_overhead_synthetic_cpu(tmp_path):
    """scripts/bench_overhead.py runs on CPU and writes a valid results JSON."""
    cfg = {
        "experiment": "instrumented_mlp_smoke",
        "seed": 1600,
        "device": "cpu",
        "model": {"matrix_shapes": [[16, 12], [12, 10]]},
        "train": {"steps": 5, "batch_size": 8},
        "optimizer": {
            "name": "muon", "lr": 0.1, "momentum": 0.6,
            "nesterov": True, "ns_steps": 3, "weight_decay": 0.0,
        },
        "instrumentation": {
            "k1": 2, "k2": 2, "t_refresh": 4, "subspace_iters": 2,
            "betas": [0.9, 0.99], "align_min": 0.9, "snapshot_every": 2,
            "seed": 4242, "min_dim": 2,
            "classifier": dict(CLASSIFIER_KWARGS),
        },
    }
    cfg_path = tmp_path / "bench_cfg_seed1600.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out_path = tmp_path / "bench_result.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/bench_overhead.py"),
            "--config", str(cfg_path),
            "--mode", "synthetic",
            "--device", "cpu",
            "--steps", "4",
            "--warmup", "2",
            "--out", str(out_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    result = results_io.load_result(out_path)
    metrics = result["metrics"]
    assert metrics["mode"] == "synthetic"
    assert metrics["n_timed_steps"] == 4
    assert metrics["stock_step_ms_median"] > 0
    assert metrics["instrumented_step_ms_median"] > 0
    assert "overhead_frac_median" in metrics
