"""WP0.4 unit tests: DynMuon on a fixed tiny matrix, hand-computed.

Expectations are computed inside the tests from the written update rule
(vendor/DynMuon/dynmuon/dynmuon.py):

    buf_t = mu * buf_{t-1} + G_t                      (dynmuon.py:585-587)
    U_t   = mu * buf_t + G_t  if nesterov else buf_t  (dynmuon.py:589-593)
    p_t   = Logistic_Scheduler(step_t)                (dynmuon.py:44-55, 1-based
                                                       step, dynmuon.py:186-192)
    O_t   = identity        if p >= 0.25              (dynmuon.py:32-39)
          | NewtonSchulz    if 0 <= p < 0.25
          | sigma -> sigma^p if p < 0
    W_t   = W_{t-1}*(1 - lr*wd) - adjusted_lr * O_t   (dynmuon.py:629-635)

using the verbatim vendor NS copy (tests/test_optim_refs.py:
dynmuon_ns_reference) for the NS branch and numpy float64 SVD for the exact
p < 0 branch. The regime is pinned per test by setting p_max == p_min, which
makes the logistic schedule constant.

Fixed literal 4x3 matrix; >= 2 steps everywhere; tight tolerances.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from test_optim_refs import dynmuon_ns_reference

from src.optim.dynmuon import DynMuon, LogisticPScheduler

W0 = torch.tensor(
    [
        [0.5, -0.2, 0.1],
        [0.3, 0.8, -0.5],
        [-0.7, 0.4, 0.2],
        [0.1, -0.6, 0.9],
    ]
)
GRADS = [
    torch.tensor(
        [
            [0.1, -0.3, 0.2],
            [0.4, 0.0, -0.1],
            [-0.2, 0.5, 0.3],
            [0.0, -0.4, 0.1],
        ]
    ),
    torch.tensor(
        [
            [-0.2, 0.1, 0.4],
            [0.3, -0.5, 0.0],
            [0.1, 0.2, -0.3],
            [-0.4, 0.0, 0.2],
        ]
    ),
]


def run_dynmuon(w0, grads, **kwargs):
    param = torch.nn.Parameter(w0.clone())
    opt = DynMuon([param], **kwargs)
    for g in grads:
        param.grad = g.clone()
        opt.step()
    return param.detach(), opt


def momentum_sequence(grads, mu, nesterov):
    """Independent momentum recurrence (dynmuon.py:585-593)."""
    buf = torch.zeros_like(grads[0])
    out = []
    for g in grads:
        buf = mu * buf + g
        out.append(mu * buf + g if nesterov else buf.clone())
    return out


# ------------------------------------------------------------- p-schedule


def test_logistic_scheduler_matches_written_formula():
    """p(step) = p_min + (p_max-p_min) / (1 + exp((step/total - tau)/width)),
    recomputed independently in numpy (dynmuon.py:44-55)."""
    sched = LogisticPScheduler(p_max=1.0, p_min=-0.25, tau_ratio=0.02,
                               width_ratio=0.08)
    total = 20000
    for step in [0, 1, 100, 400, 1000, 5000, 20000]:
        q = step / total
        expected = -0.25 + (1.0 - (-0.25)) / (1.0 + np.exp((q - 0.02) / 0.08))
        assert math.isclose(sched.get_p(step, total), expected, rel_tol=1e-12)


def test_p_schedule_endpoints():
    """Early steps sit at ~p_max (identity regime with the defaults), the
    final step reaches p_min, and the schedule is monotone decreasing."""
    sched = LogisticPScheduler()  # defaults p_max=1.0, p_min=-0.25
    total = 20000
    # sharp-transition variant pins the endpoints tightly:
    steep = LogisticPScheduler(p_max=1.0, p_min=-0.25, tau_ratio=0.5,
                               width_ratio=0.01)
    assert math.isclose(steep.get_p(0, total), 1.0, abs_tol=1e-6)
    assert math.isclose(steep.get_p(total, total), -0.25, abs_tol=1e-6)
    # defaults: end of training is fully annealed to p_min
    assert math.isclose(sched.get_p(total, total), -0.25, abs_tol=1e-4)
    ps = [sched.get_p(s, total) for s in range(0, total + 1, 100)]
    assert all(a > b for a, b in zip(ps, ps[1:]))
    # the DynMuon story: positive p early, negative p late
    assert sched.get_p(1, total) > 0.25
    assert sched.get_p(total, total) < 0.0


def test_default_hyperparameters_match_reference():
    """Defaults mirror DynMuon.__init__ (dynmuon.py:61-81)."""
    param = torch.nn.Parameter(W0.clone())
    opt = DynMuon([param])
    g = opt.param_groups[0]
    assert g["lr"] == 0.01
    assert g["momentum"] == 0.95
    assert g["weight_decay"] == 0.01
    assert g["nesterov"] is False
    assert g["adjust_lr"] == "spectral_norm"
    assert opt.total_steps == 20000
    assert opt.scheduler.p_max == 1.0
    assert opt.scheduler.p_min == -0.25
    assert opt.scheduler.tau_ratio == 0.02
    assert opt.scheduler.width_ratio == 0.08


# --------------------------------------------------- regime-pinned two-steppers

COMMON = dict(lr=0.1, weight_decay=0.0, momentum=0.6, nesterov=False,
              total_steps=100, tau_ratio=0.02, width_ratio=0.08,
              adjust_lr=None)


def test_identity_regime_two_steps_hand_computed():
    """p pinned at 1.0 (>= 0.25): the update is the raw momentum
    (dynmuon.py:33-34), passed through the NS working dtype."""
    got, opt = run_dynmuon(
        W0, GRADS, p_max=1.0, p_min=1.0, ns_dtype=torch.float32, **COMMON
    )
    W = W0.clone()
    for U in momentum_sequence(GRADS, mu=0.6, nesterov=False):
        W = W - 0.1 * U
    torch.testing.assert_close(got, W, atol=1e-6, rtol=1e-6)
    param = opt.param_groups[0]["params"][0]
    assert opt.state[param]["p"] == 1.0


def test_identity_regime_default_dtype_is_bfloat16_momentum():
    """With the reference's bfloat16 working dtype (dynmuon.py:595-597), the
    identity branch applies the bf16-rounded momentum."""
    got, _ = run_dynmuon(W0, GRADS, p_max=1.0, p_min=1.0, **COMMON)
    W = W0.clone()
    for U in momentum_sequence(GRADS, mu=0.6, nesterov=False):
        W = W - 0.1 * U.to(torch.bfloat16).to(torch.float32)
    torch.testing.assert_close(got, W, atol=1e-6, rtol=1e-6)


def test_ns_regime_two_steps_hand_computed():
    """p pinned at 0.1 (in [0, 0.25)): Newton-Schulz orthogonalization with
    DynMuon's coefficient schedule, via the verbatim vendor copy."""
    got, opt = run_dynmuon(W0, GRADS, p_max=0.1, p_min=0.1, **COMMON)
    W = W0.clone()
    for U in momentum_sequence(GRADS, mu=0.6, nesterov=False):
        O = dynmuon_ns_reference(U).to(torch.float32)
        W = W - 0.1 * O
    torch.testing.assert_close(got, W, atol=1e-6, rtol=1e-6)
    param = opt.param_groups[0]["params"][0]
    assert abs(opt.state[param]["p"] - 0.1) < 1e-12


def test_spectral_regime_svd_two_steps_hand_computed_numpy():
    """p pinned at -0.5 (< 0) with spectral_impl='svd': exact
    sigma -> sigma^p, expectation via numpy float64 SVD."""
    p = -0.5
    got, _ = run_dynmuon(
        W0, GRADS, p_max=p, p_min=p, spectral_impl="svd", **COMMON
    )
    W = W0.numpy().astype(np.float64)
    buf = np.zeros_like(W)
    for g_t in GRADS:
        g = g_t.numpy().astype(np.float64)
        buf = 0.6 * buf + g
        u, s, vt = np.linalg.svd(buf, full_matrices=False)
        O = (u * s**p) @ vt
        W = W - 0.1 * O
    np.testing.assert_allclose(got.numpy(), W, atol=1e-4, rtol=1e-4)


def test_nesterov_and_wd_and_adjust_lr_two_steps_hand_computed():
    """Full convention check in the identity regime: nesterov momentum,
    decoupled decay at base lr, step at lr*sqrt(fan_out/fan_in)
    (dynmuon.py:589-593, 629-635, 671-680)."""
    lr, wd = 0.1, 0.5
    got, _ = run_dynmuon(
        W0, GRADS, lr=lr, weight_decay=wd, momentum=0.6, nesterov=True,
        total_steps=100, p_max=1.0, p_min=1.0, tau_ratio=0.02,
        width_ratio=0.08, adjust_lr="spectral_norm", ns_dtype=torch.float32,
    )
    alpha = lr * math.sqrt(4.0 / 3.0)
    W = W0.clone()
    for U in momentum_sequence(GRADS, mu=0.6, nesterov=True):
        W = W * (1.0 - lr * wd) - alpha * U
    torch.testing.assert_close(got, W, atol=1e-6, rtol=1e-6)


def test_schedule_advances_with_optimizer_steps():
    """state['step'] is the 1-based step count the reference feeds to the
    scheduler (dynmuon.py:170-192); a run crossing the logistic transition
    must traverse identity -> NS -> spectral regimes."""
    param = torch.nn.Parameter(W0.clone())
    opt = DynMuon(
        [param], lr=0.01, weight_decay=0.0, momentum=0.6, total_steps=10,
        p_max=1.0, p_min=-0.5, tau_ratio=0.5, width_ratio=0.15,
        adjust_lr=None,
    )
    ps = []
    for i in range(10):
        param.grad = GRADS[i % 2].clone()
        opt.step()
        ps.append(opt.state[param]["p"])
    assert opt.state[param]["step"] == 10
    expected = [opt.scheduler.get_p(t, 10) for t in range(1, 11)]
    assert ps == pytest.approx(expected, rel=1e-12)
    assert ps[0] >= 0.25          # early: identity regime
    assert any(0.0 <= p < 0.25 for p in ps)  # transition: NS regime
    assert ps[-1] < 0.0           # late: negative-p spectral regime
