"""WP0.4 unit tests: NorMuon on a fixed tiny matrix, hand-computed.

Expectation computed inside the test from Algorithm 1 of the NorMuon paper
(arXiv:2510.05491), with the orthogonalization supplied by the verbatim
vendor NS copy (tests/test_optim_refs.py:airbench_ns_reference) and
everything else in numpy float64:

    M_t = beta1 * M_{t-1} + (1 - beta1) * G_t
    O_t = NS5(M_t)
    v_t = beta2 * v_{t-1} + (1 - beta2) * mean_cols(O_t (.) O_t)   # v in R^m
    O^_t = O_t (/) (sqrt(v_t) + eps)
    eta^ = 0.2 * eta * sqrt(m*n) / ||O^_t||_F
    W_t = W_{t-1} - eta * lambda * W_{t-1} - eta^ * O^_t

Plus the reduction property: with normalization and RMS alignment disabled,
NorMuon must reproduce reference Muon (heavy-ball) -- the (1-beta1) EMA
factor is a pure scale, which Newton-Schulz's input normalization removes.

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

from src.optim.muon import Muon
from src.optim.normuon import NorMuon

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


def test_normuon_three_steps_hand_computed():
    lr, wd, b1, b2, eps, T = 0.05, 0.01, 0.95, 0.95, 1e-8, 5

    param = torch.nn.Parameter(W0.clone())
    opt = NorMuon([param], lr=lr, weight_decay=wd, beta1=b1, beta2=b2,
                  eps=eps, ns_steps=T)
    for g in GRADS:
        param.grad = g.clone()
        opt.step()

    # Independent expectation (numpy float64 + verbatim vendor NS)
    W = W0.numpy().astype(np.float64)
    buf = np.zeros_like(W)
    v = np.zeros((4, 1))
    for g_t in GRADS:
        g = g_t.numpy().astype(np.float64)
        buf = b1 * buf + (1 - b1) * g                          # M_t (EMA)
        O = airbench_ns_reference(
            torch.from_numpy(buf).to(torch.float32), steps=T
        ).to(torch.float64).numpy()                            # NS5(M_t)
        v = b2 * v + (1 - b2) * (O * O).mean(axis=1, keepdims=True)  # v_t
        Ohat = O / (np.sqrt(v) + eps)                          # O^_t
        eta_hat = 0.2 * lr * math.sqrt(12) / np.linalg.norm(Ohat)
        W = W - lr * wd * W - eta_hat * Ohat                   # line 11

    np.testing.assert_allclose(param.detach().numpy(), W, atol=1e-5, rtol=1e-5)


def test_normuon_second_moment_is_neuron_wise():
    """v_t holds one scalar per output row (m values for an m x n matrix),
    reduced over columns; after one step v_1 = (1-beta2)*mean_cols(O_1^2)."""
    b1, b2, T = 0.95, 0.95, 5
    param = torch.nn.Parameter(W0.clone())
    opt = NorMuon([param], lr=0.05, weight_decay=0.0, beta1=b1, beta2=b2,
                  ns_steps=T)
    param.grad = GRADS[0].clone()
    opt.step()

    v = opt.state[param]["neuron_second_moment"]
    assert v.shape == (4, 1)
    O = airbench_ns_reference(
        (1 - b1) * GRADS[0], steps=T
    ).to(torch.float32)
    torch.testing.assert_close(
        v, (1 - b2) * O.square().mean(dim=1, keepdim=True),
        atol=1e-6, rtol=1e-6,
    )


def test_normuon_reduces_to_muon_when_extras_disabled():
    """normalize=False, rms_align=False must reproduce reference Muon
    (heavy-ball, same beta): the momentum buffers differ by the constant
    factor (1-beta1), which NS's input normalization cancels (up to the eps
    in the norm)."""
    param_n = torch.nn.Parameter(W0.clone())
    nor = NorMuon([param_n], lr=0.1, weight_decay=0.3, beta1=0.95, beta2=0.95,
                  ns_steps=5, normalize=False, rms_align=False,
                  ns_dtype=torch.float32)
    param_m = torch.nn.Parameter(W0.clone())
    muon = Muon([param_m], lr=0.1, weight_decay=0.3, momentum=0.95,
                nesterov=False, ns_steps=5, ns_dtype=torch.float32)

    for g in GRADS:
        param_n.grad = g.clone()
        param_m.grad = g.clone()
        nor.step()
        muon.step()
        torch.testing.assert_close(param_n.detach(), param_m.detach(),
                                   atol=1e-5, rtol=1e-5)
    assert "neuron_second_moment" not in nor.state[param_n]


def test_normuon_rms_align_sets_applied_step_norm():
    """With rms_align, the applied step -Delta W (wd=0) has Frobenius norm
    exactly 0.2*lr*sqrt(m*n) regardless of the raw update magnitude."""
    lr = 0.05
    param = torch.nn.Parameter(W0.clone())
    opt = NorMuon([param], lr=lr, weight_decay=0.0)
    before = param.detach().clone()
    param.grad = GRADS[0].clone()
    opt.step()
    delta = param.detach() - before
    assert math.isclose(float(delta.norm()), 0.2 * lr * math.sqrt(12),
                        rel_tol=1e-4)
