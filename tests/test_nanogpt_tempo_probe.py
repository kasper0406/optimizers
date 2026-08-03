"""PORT CHANGE P6 tests: passive tempo probe in the nanogpt Muon step.

CPU-only; drives src.nanogpt.optim.Muon directly with planted gradient
streams through test_nanogpt_port._SingleRankDist (the established exact
single-rank collective stubs — gloo lacks ReduceOp.AVG reduce_scatter).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from test_nanogpt_port import _SingleRankDist

from src.nanogpt.optim import Muon
from src.nanogpt.tempo_probe import TempoProbe

SHAPE = (16, 24)


@pytest.fixture(autouse=True)
def _single_rank_dist(monkeypatch):
    _SingleRankDist.install(monkeypatch)
    # The @torch.compile'd Newton-Schulz probes dist state under the stubbed
    # is_initialized; run it eagerly here (identical math, CPU test only).
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    yield


def ar1_stream(rho: float, n_steps: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(SHAPE)
    out = []
    for _ in range(n_steps):
        eps = rng.standard_normal(SHAPE)
        x = rho * x + math.sqrt(1.0 - rho * rho) * eps
        out.append(torch.tensor(x, dtype=torch.float32))
    return out


def make_muon(params, probe=None):
    opt = Muon(params, lr=0.01, momentum=0.95, weight_decay=0.0, rank=0, world_size=1)
    if probe is not None:
        opt.tempo_probe = probe
    return opt


def drive(opt, params, streams):
    for grads in zip(*streams):
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()


def mean_cos(rows, key, matrix, lo=5):
    v = [r[key] for r in rows if r["matrix"] == matrix and r[key] is not None
         and r["step"] >= lo]
    return sum(v) / len(v)


def test_probe_recovers_planted_serial_sign():
    p_neg = torch.nn.Parameter(torch.randn(SHAPE))
    p_pos = torch.nn.Parameter(torch.randn(SHAPE))
    probe = TempoProbe(subset=1, flush_every=4)
    opt = make_muon([p_neg, p_pos], probe)
    drive(opt, [p_neg, p_pos],
          [ar1_stream(-0.8, 60, seed=1), ar1_stream(+0.8, 60, seed=2)])
    log = probe.to_log()
    assert mean_cos(log["rows"], "cos_gg", 0) == pytest.approx(-0.8, abs=0.12)
    assert mean_cos(log["rows"], "cos_gg", 1) == pytest.approx(+0.8, abs=0.12)
    # gm alignment: positive-serial stream aligns with its momentum buffer;
    # negative-serial stream anti-aligns.
    assert mean_cos(log["rows"], "cos_gm", 1) > 0.3
    assert mean_cos(log["rows"], "cos_gm", 0) < -0.1


def test_probe_leaves_update_path_untouched():
    grads = ar1_stream(-0.5, 25, seed=3)
    p_a = torch.nn.Parameter(torch.randn(SHAPE))
    p_b = torch.nn.Parameter(p_a.data.clone())
    plain = make_muon([p_a])
    probed = make_muon([p_b], TempoProbe(subset=1))
    drive(plain, [p_a], [grads])
    drive(probed, [p_b], [grads])
    assert torch.equal(p_a.data, p_b.data)


def test_subset_skips_matrices():
    params = [torch.nn.Parameter(torch.randn(SHAPE)) for _ in range(4)]
    probe = TempoProbe(subset=2)
    opt = make_muon(params, probe)
    drive(opt, params, [ar1_stream(0.0, 10, seed=s) for s in range(4)])
    log = probe.to_log()
    observed = {r["matrix"] for r in log["rows"]}
    assert observed == {0, 2}
    assert set(log["matrices"]) == {"0", "2"}
