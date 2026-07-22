"""Program #9 unit tests: FIRMuon process-matched momentum filter.

Synthetic-signal discipline: planted AR(1) streams with known serial
structure; assertions on tap synthesis, estimator-variance wins, warm-up
equivalence with stock Muon, and the force_rho placebo path.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.optim.firmuon import FIRMuon, synthesize_taps
from src.optim.muon import Muon

SHAPE = (8, 12)


def make_param(seed: int = 0) -> torch.nn.Parameter:
    g = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.randn(*SHAPE, generator=g))


def drive(opt, params, grad_seq):
    for grads in grad_seq:
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()


def ar1_stream(rho: float, n_steps: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(SHAPE)
    out = []
    for _ in range(n_steps):
        eps = rng.standard_normal(SHAPE)
        x = rho * x + math.sqrt(1.0 - rho * rho) * eps
        out.append(torch.tensor(x, dtype=torch.float32))
    return out


def fir(param, **kw):
    defaults = dict(
        lr=0.01, momentum=0.6, nesterov=True, ns_steps=3,
        n_taps=8, tau=1.5, ridge=0.05, rho_beta=0.9, warmup_steps=10,
    )
    defaults.update(kw)
    return FIRMuon([param], **defaults)


# ------------------------------------------------------------ tap synthesis


def test_taps_satisfy_constraints():
    for rho, tau in ((-0.4, 1.5), (0.0, 0.8), (0.3, 3.0)):
        w = synthesize_taps(rho, tau, 12, 0.05)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
        assert float((w * torch.arange(12)).sum()) == pytest.approx(tau, abs=1e-5)


def test_synthesized_taps_beat_nesterov_kernel_on_anticorrelated_noise():
    """The offline decomposition (2026-07-22 kill-test) found the gain is
    dominated by the kernel FAMILY: both white-optimal and rho-matched taps
    beat the Nesterov-EMA kernel by >2x variance at matched mean lag, while
    rho-matching adds only ~1-6% on top. Encode both facts."""
    rng = np.random.default_rng(7)
    rho = -0.35
    n = 20000
    eps = rng.standard_normal(n)
    s = np.empty(n)
    s[0] = eps[0]
    c = math.sqrt(1 - rho * rho)
    for t in range(1, n):
        s[t] = rho * s[t - 1] + c * eps[t]
    L = 8
    beta = 0.6  # airbench record momentum
    w_nest = beta ** (np.arange(L) + 1.0)
    w_nest[0] += 1.0
    w_nest /= w_nest.sum()
    tau = float(np.dot(w_nest, np.arange(L)))
    w_matched = synthesize_taps(rho, tau, L, 0.0).numpy()
    w_white = synthesize_taps(0.0, tau, L, 0.0).numpy()
    v = {k: np.convolve(s, w, mode="valid").var()
         for k, w in (("nest", w_nest), ("white", w_white), ("matched", w_matched))}
    assert v["white"] < 0.5 * v["nest"]      # kernel-family effect (large)
    assert v["matched"] <= v["white"] * 1.01  # rho-matching: small, non-negative


# ---------------------------------------------------------- optimizer paths


def test_warmup_is_bitwise_stock_muon():
    grads = ar1_stream(-0.4, 9, seed=3)  # < warmup_steps and < n_taps
    p1, p2 = make_param(1), make_param(1)
    stock = Muon([p1], lr=0.01, momentum=0.6, nesterov=True, ns_steps=3)
    filt = fir(p2, warmup_steps=50)
    drive(stock, [p1], [[g] for g in grads])
    drive(filt, [p2], [[g] for g in grads])
    assert torch.equal(p1.data, p2.data)
    assert not filt.tempo_stats()["final"][f"matrix0_{SHAPE[0]}x{SHAPE[1]}"]["active"]


def test_filter_activates_after_warmup_and_diverges_from_stock():
    grads = ar1_stream(-0.4, 40, seed=5)
    p1, p2 = make_param(2), make_param(2)
    stock = Muon([p1], lr=0.01, momentum=0.6, nesterov=True, ns_steps=3)
    filt = fir(p2)
    drive(stock, [p1], [[g] for g in grads])
    drive(filt, [p2], [[g] for g in grads])
    st = filt.tempo_stats()["final"][f"matrix0_{SHAPE[0]}x{SHAPE[1]}"]
    assert st["active"]
    assert st["rho"] == pytest.approx(-0.4, abs=0.15)
    assert st["taps"] is not None and len(st["taps"]) == 8
    assert not torch.equal(p1.data, p2.data)


def test_force_rho_zero_gives_white_noise_taps():
    grads = ar1_stream(-0.8, 30, seed=9)  # strongly anti-correlated stream
    p = make_param(4)
    placebo = fir(p, force_rho=0.0)
    drive(placebo, [p], [[g] for g in grads])
    taps = placebo.tempo_stats()["final"][f"matrix0_{SHAPE[0]}x{SHAPE[1]}"]["taps"]
    expected = synthesize_taps(0.0, 1.5, 8, 0.05)
    assert taps == pytest.approx(expected.tolist(), abs=1e-5)


def test_fp16_gradients_do_not_poison_rho():
    p = torch.nn.Parameter(torch.randn(SHAPE, dtype=torch.float16))
    opt = fir(p, warmup_steps=3)
    big = [
        (300.0 * torch.randn(SHAPE, generator=torch.Generator().manual_seed(s))).half()
        for s in range(20)
    ]
    assert torch.isinf(big[0] * big[0]).any()
    drive(opt, [p], [[g] for g in big])
    st = opt.tempo_stats()["final"][f"matrix0_{SHAPE[0]}x{SHAPE[1]}"]
    assert math.isfinite(st["rho"])
    assert torch.isfinite(p.data).all()


def test_state_dict_roundtrip_resumes_trajectory():
    grads = ar1_stream(-0.5, 50, seed=11)
    p_full = make_param(6)
    full = fir(p_full)
    drive(full, [p_full], [[g] for g in grads])

    p_a = make_param(6)
    first = fir(p_a)
    drive(first, [p_a], [[g] for g in grads[:25]])
    sd = first.state_dict()

    p_b = make_param(6)
    p_b.data.copy_(p_a.data)
    resumed = fir(p_b)
    resumed.load_state_dict(sd)
    drive(resumed, [p_b], [[g] for g in grads[25:]])
    assert torch.allclose(p_full.data, p_b.data, atol=1e-6)
