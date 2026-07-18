"""WP0.4 unit tests: AdaMuon on a fixed tiny matrix, hand-computed.

Expectation computed inside the test from Algorithm 1 of the AdaMuon paper
(arXiv:2507.11005v3), with the orthogonalization supplied by the verbatim
vendor NS copy (tests/test_optim_refs.py:airbench_ns_reference -- the paper
uses the same quintic coefficients, Sec. 2.1) and everything else in numpy
float64:

    M_t = beta * M_{t-1} + G_t
    O_t = NewtonSchulz(Sign(M_t), T)
    V_t = beta * V_{t-1} + (1 - beta) * O_t (.) O_t     (no bias correction)
    O^_t = O_t (/) (sqrt(V_t) + eps)
    gamma_t = 0.2 * sqrt(m*n) / ||O^_t||_F
    W_t = W_{t-1} - lr * (gamma_t * O^_t + wd * W_{t-1})

Plus the reduction property: with sign stabilization, the second moment and
the RMS alignment all disabled, AdaMuon must reproduce reference Muon
(heavy-ball, nesterov=False) exactly.

Fixed literal 4x3 matrix; 3 steps; tight tolerances.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from test_optim_refs import airbench_ns_reference

from src.optim.adamuon import AdaMuon
from src.optim.muon import Muon

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
    torch.tensor(
        [
            [0.3, 0.2, -0.1],
            [-0.1, 0.4, 0.5],
            [0.0, -0.3, 0.2],
            [0.2, 0.1, -0.4],
        ]
    ),
]


def test_adamuon_three_steps_hand_computed():
    lr, wd, beta, eps, T = 0.05, 0.1, 0.95, 1e-8, 5

    param = torch.nn.Parameter(W0.clone())
    opt = AdaMuon([param], lr=lr, weight_decay=wd, momentum=beta, eps=eps,
                  ns_steps=T)
    for g in GRADS:
        param.grad = g.clone()
        opt.step()

    # Independent expectation (numpy float64 + verbatim vendor NS)
    W = W0.numpy().astype(np.float64)
    buf = np.zeros_like(W)
    V = np.zeros_like(W)
    for g_t in GRADS:
        g = g_t.numpy().astype(np.float64)
        buf = beta * buf + g                                   # M_t
        S = np.sign(buf)                                       # Sign(M_t)
        O = airbench_ns_reference(
            torch.from_numpy(S).to(torch.float32), steps=T
        ).to(torch.float64).numpy()                            # NS(Sign(M_t))
        V = beta * V + (1 - beta) * O * O                      # V_t
        Ohat = O / (np.sqrt(V) + eps)                          # O^_t
        gamma = 0.2 * math.sqrt(12) / np.linalg.norm(Ohat)     # gamma_t
        W = W * (1 - lr * wd) - lr * gamma * Ohat              # update

    np.testing.assert_allclose(param.detach().numpy(), W, atol=1e-5, rtol=1e-5)


def test_adamuon_second_moment_state_shape_and_no_bias_correction():
    """V_t is element-wise (same shape as the matrix) and uses the plain EMA
    with no 1/(1-beta^t) correction: after one step from V_0 = 0,
    V_1 = (1-beta) * O_1^2 exactly."""
    beta, eps, T = 0.95, 1e-8, 5
    param = torch.nn.Parameter(W0.clone())
    opt = AdaMuon([param], lr=0.05, weight_decay=0.0, momentum=beta, eps=eps,
                  ns_steps=T)
    param.grad = GRADS[0].clone()
    opt.step()

    V = opt.state[param]["second_moment"]
    assert V.shape == (4, 3)
    O = airbench_ns_reference(torch.sign(GRADS[0]), steps=T).to(torch.float32)
    torch.testing.assert_close(V, (1 - beta) * O * O, atol=1e-6, rtol=1e-6)


def test_adamuon_reduces_to_muon_when_extras_disabled():
    """sign_stabilize=False, adaptive=False, rms_align=False must reproduce
    reference Muon (heavy-ball momentum, no nesterov) step for step."""
    kwargs = dict(lr=0.1, weight_decay=0.3, momentum=0.6, ns_steps=5,
                  ns_dtype=torch.float32)

    param_a = torch.nn.Parameter(W0.clone())
    ada = AdaMuon([param_a], sign_stabilize=False, adaptive=False,
                  rms_align=False, **kwargs)
    param_m = torch.nn.Parameter(W0.clone())
    muon = Muon([param_m], nesterov=False, **kwargs)

    for g in GRADS:
        param_a.grad = g.clone()
        param_m.grad = g.clone()
        ada.step()
        muon.step()
        torch.testing.assert_close(param_a.detach(), param_m.detach(),
                                   atol=1e-7, rtol=1e-7)
    assert "second_moment" not in ada.state[param_a]


def test_adamuon_rms_alignment_sets_update_frobenius_norm():
    """After the gamma_t rescaling the applied direction has Frobenius norm
    exactly 0.2*sqrt(m*n) (RMS 0.2, the paper's Adam-alignment constant)."""
    param = torch.nn.Parameter(W0.clone())
    opt = AdaMuon([param], lr=1.0, weight_decay=0.0, momentum=0.95)
    param.grad = GRADS[0].clone()

    state = opt.state[param]
    state["step"] = 1
    M = opt.pre_step(param.grad, state, opt.param_groups[0])
    out = opt.shape_spectrum(M, state, opt.param_groups[0])
    assert math.isclose(float(out.norm()), 0.2 * math.sqrt(12), rel_tol=1e-5)
