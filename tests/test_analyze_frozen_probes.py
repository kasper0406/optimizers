"""CPU end-to-end for the two new measurement tiers + the analysis script.

The airbench harness needs CUDA, so the end-to-end path exercised here is the
same wiring on a small CPU MLP trained with the zoo Muon: an
``InstrumentationHub`` built from a config block with ``frozen_probes``
enabled, a ``SmoothnessProbe`` driven with the same pre/post-step hook order
the airbench loop uses, one sidecar written through the real schema writer,
and ``scripts/analyze_frozen_probes.py`` run over it.

Dev seeds (>= 1000) throughout.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from src.instrument import hub_from_config
from src.instrument.schema import SIDECAR_SUFFIX, load_instrumentation, write_sidecar
from src.instrument.smoothness import SmoothnessProbe
from src.optim.registry import build_optimizer

INSTR_CFG = {
    "k1": 3,
    "k2": 2,
    "t_refresh": 10,
    "subspace_iters": 2,
    "betas": [0.9],
    "align_min": 0.9,
    "snapshot_every": 2,
    "seed": 4242,
    "min_dim": 4,
    "frozen_probes": {"enabled": True, "k3": 4, "max_lag": 4, "decimate": 1},
    "classifier": {"tau_sig": 4.0, "tau_noise": 2.0, "rho_osc": 0.5, "n_min": 10},
}


def _load_script():
    path = REPO_ROOT / "scripts" / "analyze_frozen_probes.py"
    spec = importlib.util.spec_from_file_location("analyze_frozen_probes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(steps=40, seed=1500, lr=0.05):
    """A tiny instrumented training run with both new tiers on (CPU)."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(12, 8, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 4, bias=False),
    )
    named = [(n, p) for n, p in model.named_parameters()]
    opt = build_optimizer("muon", [p for _, p in named], dict(lr=lr, momentum=0.9))
    hub = hub_from_config(INSTR_CFG, named, opt)
    probe = SmoothnessProbe(
        model,
        named,
        t_meas=5,
        loss_fn=lambda o, y: torch.nn.functional.cross_entropy(o, y, reduction="sum"),
        loss_reduction="sum",
    )
    gen = torch.Generator().manual_seed(seed + 1)
    for step in range(1, steps + 1):
        x = torch.randn(16, 12, generator=gen)
        y = torch.randint(0, 4, (16,), generator=gen)
        probe.set_batch(x, y)
        loss = torch.nn.functional.cross_entropy(model(x), y, reduction="sum")
        loss.backward()
        hub.capture_grads()
        probe.before_step(step)
        opt.step()
        probe.after_step(step, lr)
        hub.after_step()
        model.zero_grad(set_to_none=True)
    log = hub.to_log()
    log["smoothness"] = probe.to_log()
    return log


def test_end_to_end_log_carries_both_tiers_and_validates(tmp_path):
    log = _run()
    results_json = tmp_path / "airbench_instrumented_seed1500_x.json"
    sidecar = write_sidecar(log, results_json)
    assert sidecar.name.endswith(SIDECAR_SUFFIX)

    reloaded = load_instrumentation(sidecar)
    assert reloaded["frozen_probes_enabled"] is True
    for name, mat in reloaded["matrices"].items():
        fp = mat["frozen_probes"]
        assert fp["k3"] == 4 and len(fp["probes"]) == 4
        assert fp["n_observations"] == 40
        # Cumulative statistics snapshot on the tracker's snapshot cadence.
        assert len(fp["snapshot_steps"]) == len(fp["probes"][0]["t_nw"])
        assert all(np.isfinite(fp["probes"][0]["t_nw"]))
    sm = reloaded["smoothness"]
    assert sm["t_meas"] == 5 and sm["grad_source"] == "recompute"
    assert sm["n_measured_steps"] == 8  # steps 1, 6, ..., 36
    for series in sm["matrices"].values():
        assert len(series["d_smooth_spectral"]) == 8
        assert all(np.isfinite(series["lr_times_d_smooth_spectral"]))
        # ||D||_2 <= ||D||_F always, so |d_spectral| >= |d_frobenius| and the
        # two share the sign of the remainder.
        for spec, fro, ds, df, rem in zip(
            series["spec_norm_D"],
            series["fro_norm_D"],
            series["d_smooth_spectral"],
            series["d_smooth_frobenius"],
            series["remainder"],
        ):
            assert spec <= fro + 1e-9
            assert abs(ds) >= abs(df) - 1e-9
            assert np.sign(ds) == np.sign(df) == np.sign(rem)


def test_analysis_script_runs_and_is_deterministic(tmp_path):
    mod = _load_script()
    side_dir = tmp_path / "sidecars"
    side_dir.mkdir()
    for seed in (1500, 1501):
        write_sidecar(_run(seed=seed), side_dir / f"run_seed{seed}.json")

    outputs = []
    for i in (0, 1):
        md = tmp_path / f"out{i}.md"
        js = tmp_path / f"out{i}.json"
        png = tmp_path / f"out{i}.png"
        assert (
            mod.main(
                [
                    "--sidecars", str(side_dir),
                    "--out-md", str(md),
                    "--out-json", str(js),
                    "--out-plot", str(png),
                ]
            )
            == 0
        )
        outputs.append((md.read_text(), js.read_text()))
        assert png.exists() and png.stat().st_size > 0
    assert outputs[0] == outputs[1]  # byte-identical: deterministic

    report = json.loads(outputs[0][1])
    assert report["n_runs"] == 2
    assert report["n_frozen_probes"] == 2 * 2 * 4  # runs x matrices x k3
    for est in ("t_naive", "t_nw"):
        assert report["pooled"][est]["final_abs_t"]["n"] == 8 * 2
        assert 0.0 <= report["pooled"][est]["frac_crossing"]["4"] <= 1.0
    assert report["tracked_tier_final_abs_t"]["0.9"]["all"]["n"] > 0
    md_text = outputs[0][0]
    assert "Frozen-probe tier" in md_text and "pooled" in md_text


def test_growth_slope_recovers_the_planted_exponents():
    mod = _load_script()
    steps = np.arange(10, 1000, 10, dtype=float)
    assert mod.growth_slope(steps, np.sqrt(steps)) == pytest.approx(0.5, abs=1e-6)
    assert mod.growth_slope(steps, np.full_like(steps, 1.7)) == pytest.approx(
        0.0, abs=1e-6
    )
    assert mod.growth_slope(steps, -np.sqrt(steps)) == pytest.approx(0.5, abs=1e-6)
    assert mod.growth_slope([1.0, 2.0], [1.0, 2.0]) is None  # too few points
    assert mod.growth_slope(steps, np.zeros_like(steps)) is None  # degenerate


def test_analysis_refuses_sidecars_without_the_frozen_block(tmp_path):
    mod = _load_script()
    side_dir = tmp_path / "sidecars"
    side_dir.mkdir()
    log = _run(steps=20)
    for mat in log["matrices"].values():
        mat.pop("frozen_probes")
    log["frozen_probes_enabled"] = False
    write_sidecar(log, side_dir / "run.json")
    with pytest.raises(SystemExit):
        mod.main(["--sidecars", str(side_dir)])
    with pytest.raises(SystemExit):
        mod.main(["--sidecars", str(tmp_path / "empty")])
