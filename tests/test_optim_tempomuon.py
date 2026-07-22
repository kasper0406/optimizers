"""Program #8 unit tests: TempoMuon temporal trust ratio.

Synthetic-signal discipline (WP0.5 style): planted gradient streams with
known serial structure drive the optimizer directly (no training, no GPU);
tests assert the measured rho_hat and the gain response.

Key discrimination test: equal-variance AR(+0.8) vs AR(-0.8) streams must
produce opposite gain responses — the separation that norm/noise-magnitude
trust ratios (OrScale/NAMO/LANTON) cannot make by construction.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.optim.muon import Muon
from src.optim.tempomuon import TempoMuon

SHAPE = (8, 12)


def make_param(seed: int = 0) -> torch.nn.Parameter:
    g = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.randn(*SHAPE, generator=g))


def drive(opt: torch.optim.Optimizer, params, grad_seq) -> None:
    """Feed a list of per-step gradients (one tensor per param) and step."""
    for grads in grad_seq:
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()


def ar1_stream(rho: float, n_steps: int, seed: int, scale: float = 1.0):
    """Matrix-valued AR(1): X_t = rho * X_{t-1} + sqrt(1-rho^2) * eps_t.

    Stationary variance is scale^2 regardless of rho, so streams with
    opposite rho are indistinguishable to any magnitude-based signal.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(SHAPE)
    out = []
    for _ in range(n_steps):
        eps = rng.standard_normal(SHAPE)
        x = rho * x + math.sqrt(1.0 - rho * rho) * eps
        out.append(torch.tensor(x * scale, dtype=torch.float32))
    return out


def tempo(param, **kw) -> TempoMuon:
    defaults = dict(
        lr=0.01,
        momentum=0.6,
        nesterov=True,
        ns_steps=3,
        rho_beta=0.9,
        warmup_steps=10,
        gain_min=0.25,
        gain_max=1.0,
    )
    defaults.update(kw)
    return TempoMuon([param], **defaults)


# --------------------------------------------------------------- rho recovery


@pytest.mark.parametrize("rho", [-0.8, 0.0, 0.8])
def test_rho_hat_recovers_planted_autocorrelation(rho):
    p = make_param()
    opt = tempo(p, kappa=0.0)
    drive(opt, [p], [[g] for g in ar1_stream(rho, 120, seed=7)])
    measured = opt.tempo_stats()["final"][f"matrix0_{SHAPE[0]}x{SHAPE[1]}"]["rho"]
    # cos(G_t, G_{t-1}) of a matrix AR(1) concentrates near rho (96 dims);
    # EMA adds jitter — 0.1 tolerance mirrors the WP0.5 oscillation bound.
    assert measured == pytest.approx(rho, abs=0.1)


def test_passive_mode_is_bitwise_stock_muon():
    grads = ar1_stream(-0.5, 30, seed=3)
    p1, p2 = make_param(1), make_param(1)
    stock = Muon([p1], lr=0.01, momentum=0.6, nesterov=True, ns_steps=3)
    passive = tempo(p2, kappa=0.0)
    drive(stock, [p1], [[g] for g in grads])
    drive(passive, [p2], [[g] for g in grads])
    assert torch.equal(p1.data, p2.data)


# ------------------------------------------------------------- gain dynamics


def test_oscillating_stream_drives_gain_to_floor():
    p = make_param()
    opt = tempo(p, kappa=0.3, rho_star=-0.2)
    drive(opt, [p], [[g] for g in ar1_stream(-0.8, 100, seed=11)])
    stats = opt.tempo_stats()["final"][f"matrix0_{SHAPE[0]}x{SHAPE[1]}"]
    assert stats["gain"] == pytest.approx(0.25, abs=1e-9)


def test_persistent_stream_keeps_gain_at_cap():
    p = make_param()
    opt = tempo(p, kappa=0.3, rho_star=-0.2)
    drive(opt, [p], [[g] for g in ar1_stream(0.8, 100, seed=11)])
    stats = opt.tempo_stats()["final"][f"matrix0_{SHAPE[0]}x{SHAPE[1]}"]
    assert stats["gain"] == pytest.approx(1.0, abs=1e-9)


def test_equal_variance_ar_streams_get_opposite_treatment():
    """The killer separation vs the magnitude family: same variance, only the
    serial structure differs; gains must split to opposite bounds."""
    p_neg, p_pos = make_param(2), make_param(2)
    opt = TempoMuon(
        [p_neg, p_pos],
        lr=0.01,
        momentum=0.6,
        ns_steps=3,
        kappa=0.3,
        rho_star=-0.2,
        warmup_steps=10,
    )
    neg = ar1_stream(-0.8, 100, seed=5)
    pos = ar1_stream(+0.8, 100, seed=5)
    drive(opt, [p_neg, p_pos], [[a, b] for a, b in zip(neg, pos)])
    rows = opt.tempo_stats()["final"]
    gains = sorted(r["gain"] for r in rows.values())
    assert gains[0] == pytest.approx(0.25, abs=1e-9)
    assert gains[1] == pytest.approx(1.0, abs=1e-9)


def test_gain_frozen_during_warmup():
    p = make_param()
    opt = tempo(p, kappa=1.0, rho_star=0.5, warmup_steps=50)
    drive(opt, [p], [[g] for g in ar1_stream(-0.8, 40, seed=9)])
    stats = opt.tempo_stats()["final"][f"matrix0_{SHAPE[0]}x{SHAPE[1]}"]
    assert stats["gain"] == 1.0
    assert stats["obs"] == 39  # first step has no prev-grad pair


def test_gain_scales_the_applied_update():
    """With kappa large and rho_star=+1 (unreachable), gain pins to the floor;
    the parameter step must shrink by exactly gain_min vs stock Muon."""
    grads = ar1_stream(0.0, 40, seed=13)
    p_stock, p_gained = make_param(4), make_param(4)
    stock = Muon([p_stock], lr=0.01, momentum=0.6, nesterov=True, ns_steps=3)
    gained = tempo(p_gained, kappa=50.0, rho_star=1.0, warmup_steps=1, gain_min=0.5)
    w_prev_stock = p_stock.data.clone()
    w_prev_gained = p_gained.data.clone()
    for g in grads[:-1]:
        p_stock.grad = g.clone()
        p_gained.grad = g.clone()
        stock.step()
        gained.step()
        w_prev_stock = p_stock.data.clone()
        w_prev_gained = p_gained.data.clone()
    p_stock.grad = grads[-1].clone()
    p_gained.grad = grads[-1].clone()
    stock.step()
    gained.step()
    step_stock = p_stock.data - w_prev_stock
    step_gained = p_gained.data - w_prev_gained
    # Momentum buffers saw identical gradients, so the NS outputs match and
    # the applied steps differ by exactly the gain factor.
    assert torch.allclose(step_gained, 0.5 * step_stock, atol=1e-6)


# ------------------------------------------------------------------- scoping


def test_global_scope_shares_one_gain_across_matrices():
    p_neg, p_pos = make_param(2), make_param(2)
    opt = TempoMuon(
        [p_neg, p_pos],
        lr=0.01,
        momentum=0.6,
        ns_steps=3,
        kappa=0.3,
        rho_star=-0.2,
        warmup_steps=10,
        scope="global",
    )
    neg = ar1_stream(-0.8, 100, seed=5)
    pos = ar1_stream(+0.8, 100, seed=5)
    drive(opt, [p_neg, p_pos], [[a, b] for a, b in zip(neg, pos)])
    stats = opt.tempo_stats()
    # Pooled mean of (-0.8, +0.8) cosines ~ 0 > rho_star -> gain stays at cap.
    assert stats["final_global"]["rho"] == pytest.approx(0.0, abs=0.1)
    assert stats["final_global"]["gain"] == pytest.approx(1.0, abs=1e-9)


def test_state_dict_roundtrip_resumes_trajectory():
    grads = ar1_stream(-0.6, 60, seed=17)
    p_full = make_param(6)
    full = tempo(p_full, kappa=0.3)
    drive(full, [p_full], [[g] for g in grads])

    p_a = make_param(6)
    first = tempo(p_a, kappa=0.3)
    drive(first, [p_a], [[g] for g in grads[:30]])
    sd = first.state_dict()

    p_b = make_param(6)
    p_b.data.copy_(p_a.data)
    resumed = tempo(p_b, kappa=0.3)
    resumed.load_state_dict(sd)
    drive(resumed, [p_b], [[g] for g in grads[30:]])
    assert torch.allclose(p_full.data, p_b.data, atol=1e-6)


def test_negative_kappa_shrinks_gain_when_rho_above_setpoint():
    """The Phase-A rescue law: kappa < 0, setpoint deep-negative; a
    decorrelated (too-hot-like, rho ~ 0) stream must drive the gain down."""
    p = make_param()
    opt = tempo(p, kappa=-0.25, rho_star=-0.48)
    drive(opt, [p], [[g] for g in ar1_stream(0.0, 100, seed=21)])
    stats = opt.tempo_stats()["final"][f"matrix0_{SHAPE[0]}x{SHAPE[1]}"]
    assert stats["gain"] == pytest.approx(0.25, abs=1e-9)


def test_negative_kappa_idles_at_cap_for_coherent_oscillation():
    """Same law: a healthy-like deep-negative stream (rho below setpoint)
    must leave the gain pinned at the cap (stock behavior)."""
    p = make_param()
    opt = tempo(p, kappa=-0.25, rho_star=-0.48)
    drive(opt, [p], [[g] for g in ar1_stream(-0.8, 100, seed=21)])
    stats = opt.tempo_stats()["final"][f"matrix0_{SHAPE[0]}x{SHAPE[1]}"]
    assert stats["gain"] == pytest.approx(1.0, abs=1e-9)


def test_ctrl_end_step_freezes_gain():
    p1, p2 = make_param(8), make_param(8)
    grads = ar1_stream(0.0, 80, seed=23)
    frozen = tempo(p1, kappa=-0.1, rho_star=-0.48, warmup_steps=5, ctrl_end_step=20)
    free = tempo(p2, kappa=-0.1, rho_star=-0.48, warmup_steps=5)
    drive(frozen, [p1], [[g] for g in grads])
    drive(free, [p2], [[g] for g in grads])
    label = f"matrix0_{SHAPE[0]}x{SHAPE[1]}"
    g_frozen = frozen.tempo_stats()["final"][label]["gain"]
    g_free = free.tempo_stats()["final"][label]["gain"]
    hist = frozen.tempo_stats()["history"]
    gain_at_20 = hist[19]["gain"][label]
    assert g_frozen == pytest.approx(gain_at_20, abs=1e-12)  # frozen after window
    assert g_free < g_frozen  # kept falling without the window


def test_fp16_large_gradients_do_not_poison_rho_or_gain():
    """Regression (program #8 Phase-A collapse): fp16 gradients whose
    elementwise products overflow fp16 (max 65504) must not produce NaN in
    rho_hat or the gain — the cosine is computed in fp32."""
    p = torch.nn.Parameter(torch.randn(SHAPE, dtype=torch.float16))
    opt = tempo(p, kappa=0.3, warmup_steps=2)
    big = [
        (300.0 * torch.randn(SHAPE, generator=torch.Generator().manual_seed(s))).half()
        for s in range(20)
    ]
    # sanity: the naive fp16 product does overflow for these magnitudes
    assert torch.isinf(big[0] * big[0]).any()
    drive(opt, [p], [[g] for g in big])
    stats = opt.tempo_stats()["final"][f"matrix0_{SHAPE[0]}x{SHAPE[1]}"]
    assert math.isfinite(stats["rho"])
    assert math.isfinite(stats["gain"])
    assert torch.isfinite(p.data).all()


def test_gain_schedule_replay_ignores_feedback():
    """Placebo mode: the applied gain follows the supplied schedule exactly,
    regardless of what the stream's serial structure says."""
    grads = ar1_stream(0.0, 6, seed=31)
    sched = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    p_replay, p_stock = make_param(9), make_param(9)
    replay = tempo(p_replay, kappa=-5.0, rho_star=-0.48, warmup_steps=1,
                   gain_schedule=sched)
    stock = Muon([p_stock], lr=0.01, momentum=0.6, nesterov=True, ns_steps=3)
    # steps 1-2: schedule gain 1.0 -> identical to stock
    for g in grads[:2]:
        p_replay.grad = g.clone(); p_stock.grad = g.clone()
        replay.step(); stock.step()
    assert torch.equal(p_replay.data, p_stock.data)
    # step 3: gain 0.5 -> step is half of stock's
    w_r, w_s = p_replay.data.clone(), p_stock.data.clone()
    p_replay.grad = grads[2].clone(); p_stock.grad = grads[2].clone()
    replay.step(); stock.step()
    assert torch.allclose(p_replay.data - w_r, 0.5 * (p_stock.data - w_s), atol=1e-6)


def test_history_and_telemetry_shapes():
    p = make_param()
    opt = tempo(p, kappa=0.3, history_every=5)
    drive(opt, [p], [[g] for g in ar1_stream(-0.4, 20, seed=1)])
    stats = opt.tempo_stats()
    assert stats["scope"] == "per_matrix"
    assert len(stats["history"]) == 4
    assert stats["history"][-1]["step"] == 20
    label = f"matrix0_{SHAPE[0]}x{SHAPE[1]}"
    assert label in stats["history"][-1]["rho"]
    assert stats["config"]["kappa"] == 0.3
