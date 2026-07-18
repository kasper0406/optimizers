"""WP1.1 log-schema tests: hub log validation, sidecar round-trip,
append-only refusal, embedded-results reading, direction iteration.

Seeds: dev seeds only (>= 1000).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


import json

import numpy as np
import pytest
import torch

from src.instrument.schema import (
    INSTRUMENTATION_SCHEMA_VERSION,
    InstrumentationValidationError,
    iter_directions,
    load_instrumentation,
    sidecar_path,
    validate_instrumentation,
    write_sidecar,
)
from src.instrument.tracker import MatrixTracker

CLASSIFIER_KWARGS = dict(tau_sig=4.0, tau_noise=2.0, rho_osc=0.5, n_min=30)


def _small_log(steps=60, hvp=False):
    """Build a real log via MatrixTracker on a planted stream."""
    gen = torch.Generator().manual_seed(1400)
    rng = np.random.default_rng(1400)
    m, n = 12, 8
    U0, _ = torch.linalg.qr(torch.randn(m, 2, generator=gen))
    V0, _ = torch.linalg.qr(torch.randn(n, 2, generator=gen))
    M = 10.0 * torch.outer(U0[:, 0], V0[:, 0]) + 4.0 * torch.outer(U0[:, 1], V0[:, 1])
    tracker = MatrixTracker(
        "w0",
        (m, n),
        k1=2,
        k2=1,
        t_refresh=20,
        betas=(0.9, 0.99),
        classifier_kwargs=CLASSIFIER_KWARGS,
        snapshot_every=5,
        generator=gen,
    )
    param = torch.zeros(m, n)
    hvp_fn = (lambda p, D: 3.0) if hvp else None
    for t in range(steps):
        G = (1.0 + 0.1 * rng.standard_normal()) * torch.outer(U0[:, 0], V0[:, 0])
        G = G + 0.3 * rng.standard_normal() * torch.outer(U0[:, 1], V0[:, 1])
        G = G + 0.01 * torch.from_numpy(rng.standard_normal((m, n)).astype(np.float32))
        tracker.observe(G, M, hvp_fn=hvp_fn, param=param if hvp else None)
    return {
        "instrumentation_schema_version": INSTRUMENTATION_SCHEMA_VERSION,
        "betas": ["0.9", "0.99"],
        "hvp_enabled": hvp,
        "matrices": {"w0": tracker.to_log()},
    }


def test_validate_accepts_real_log():
    log = _small_log()
    assert validate_instrumentation(log) is log


def test_validate_rejects_missing_keys():
    log = _small_log()
    del log["matrices"]["w0"]["directions"][0]["per_beta"]
    with pytest.raises(InstrumentationValidationError):
        validate_instrumentation(log)
    with pytest.raises(InstrumentationValidationError):
        validate_instrumentation({"betas": []})


def test_validate_rejects_length_mismatch():
    log = _small_log()
    log["matrices"]["w0"]["directions"][0]["s"].pop()
    with pytest.raises(InstrumentationValidationError):
        validate_instrumentation(log)


def test_sidecar_roundtrip_and_append_only(tmp_path):
    log = _small_log(hvp=True)
    results_path = tmp_path / "run_seed1400.json"
    out = write_sidecar(log, results_path)
    assert out == sidecar_path(results_path)
    assert out.name == "run_seed1400.instrumentation.json"

    loaded = load_instrumentation(out)
    # Byte-level round trip (canonical JSON form).
    assert json.dumps(loaded, sort_keys=True) == json.dumps(log, sort_keys=True)

    with pytest.raises(FileExistsError):
        write_sidecar(log, results_path)


def test_load_from_embedded_results_json(tmp_path):
    log = _small_log()
    results = {"metrics": {"instrumentation": log}, "seed": 1400}
    path = tmp_path / "embedded.json"
    path.write_text(json.dumps(results))
    loaded = load_instrumentation(path)
    assert loaded["matrices"].keys() == log["matrices"].keys()


def test_load_rejects_non_instrumentation_json(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"metrics": {}}))
    with pytest.raises(InstrumentationValidationError):
        load_instrumentation(path)


def test_iter_directions_order_and_count():
    log = _small_log()
    rows = list(iter_directions(log))
    assert len(rows) == 3  # k1=2 + k2=1
    names = [name for name, _, _ in rows]
    assert names == sorted(names)
    kinds = [d["kind"] for _, _, d in rows]
    assert kinds == ["top", "top", "bulk"]
    for _, mat, d in rows:
        assert len(d["s"]) == len(mat["steps"]) == 60
        for beta_key in ("0.9", "0.99"):
            series = d["per_beta"][beta_key]
            assert len(series["step"]) == len(series["regime"])
            assert series["step"][-1] == 60  # snapshot_every=5 divides 60
