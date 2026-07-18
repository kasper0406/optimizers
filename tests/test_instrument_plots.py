"""WP1.1 plotting tests: the three Phase-1 plots generate from a synthetic
instrumentation JSON alone (no live training), deterministically.

Seeds: dev seeds only (>= 1000).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


import numpy as np
import pytest
import torch

from src.instrument import plots
from src.instrument.schema import INSTRUMENTATION_SCHEMA_VERSION, write_sidecar
from src.instrument.tracker import MatrixTracker

CLASSIFIER_KWARGS = dict(tau_sig=4.0, tau_noise=2.0, rho_osc=0.5, n_min=40)


@pytest.fixture(scope="module")
def synthetic_log_path(tmp_path_factory):
    """A log with all three regimes present and HVP records, as a sidecar."""
    gen = torch.Generator().manual_seed(1500)
    rng = np.random.default_rng(1500)
    m, n = 20, 14
    U0, _ = torch.linalg.qr(torch.randn(m, 3, generator=gen))
    V0, _ = torch.linalg.qr(torch.randn(n, 3, generator=gen))
    M = (
        15.0 * torch.outer(U0[:, 0], V0[:, 0])
        + 7.0 * torch.outer(U0[:, 1], V0[:, 1])
        + 3.0 * torch.outer(U0[:, 2], V0[:, 2])
    )
    tracker = MatrixTracker(
        "layer0",
        (m, n),
        k1=3,
        k2=2,
        t_refresh=40,
        betas=(0.9, 0.99),
        classifier_kwargs=CLASSIFIER_KWARGS,
        snapshot_every=5,
        generator=gen,
    )
    param = torch.zeros(m, n)
    # HVP lambda chosen so lr * lambda ~ implied eta*lambda ( = 1 + r = 2 )
    # at lr = 0.1: lambda = 20.
    hvp_fn = lambda p, D: 20.0
    for t in range(200):
        c_sig = 1.0 + 0.1 * rng.standard_normal()
        c_osc = 3.0 * ((-1.0) ** t) * (1.0 + 0.02 * rng.standard_normal())
        c_noise = 0.5 * rng.standard_normal()
        G = (
            c_sig * torch.outer(U0[:, 0], V0[:, 0])
            + c_osc * torch.outer(U0[:, 1], V0[:, 1])
            + c_noise * torch.outer(U0[:, 2], V0[:, 2])
        )
        G = G + 0.02 * torch.from_numpy(rng.standard_normal((m, n)).astype(np.float32))
        tracker.observe(G, M, hvp_fn=hvp_fn, param=param)
    log = {
        "instrumentation_schema_version": INSTRUMENTATION_SCHEMA_VERSION,
        "betas": ["0.9", "0.99"],
        "hvp_enabled": True,
        "matrices": {"layer0": tracker.to_log()},
    }
    out_dir = tmp_path_factory.mktemp("logs")
    return write_sidecar(log, out_dir / "run_seed1500.json")


def test_make_all_plots_writes_three_pngs(synthetic_log_path, tmp_path):
    out = tmp_path / "plots"
    paths = plots.make_all_plots(synthetic_log_path, out, lr=0.1)
    assert [p.name for p in paths] == [
        "regime_scatter.png",
        "regime_occupancy.png",
        "eta_lambda_calibration.png",
    ]
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 5_000  # a real rendered figure, not a stub


def test_plots_are_deterministic(synthetic_log_path, tmp_path):
    a = plots.make_all_plots(synthetic_log_path, tmp_path / "a", lr=0.1)
    b = plots.make_all_plots(synthetic_log_path, tmp_path / "b", lr=0.1)
    for pa, pb in zip(a, b):
        assert pa.read_bytes() == pb.read_bytes(), f"{pa.name} not deterministic"


def test_lr_required_without_results_config(synthetic_log_path, tmp_path):
    with pytest.raises(ValueError, match="lr"):
        plots.make_all_plots(synthetic_log_path, tmp_path / "c", lr=None)


def test_cli_entrypoint(synthetic_log_path, tmp_path):
    out = tmp_path / "cli"
    rc = plots.main([str(synthetic_log_path), str(out), "--lr", "0.1"])
    assert rc == 0
    assert (out / "regime_scatter.png").exists()
