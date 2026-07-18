"""WP0.4 unit tests: AdamW wrapper on a fixed tiny matrix, hand-computed.

Two independent expectations:
1. A numpy float64 implementation of the written AdamW rule (Loshchilov &
   Hutter 2019, decoupled decay), computed inside the test.
2. ``torch.optim.AdamW`` itself, stepped on a clone with identical gradients
   (the wrapper's docstring promises bit-level-equivalent math for float32
   params).

Fixed literal 4x3 matrix; 3 steps; tight tolerances.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.optim.adamw import AdamW

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

LR, BETAS, EPS, WD = 0.1, (0.9, 0.999), 1e-8, 0.01


def run_ours(w0, grads):
    param = torch.nn.Parameter(w0.clone())
    opt = AdamW([param], lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    for g in grads:
        param.grad = g.clone()
        opt.step()
    return param.detach()


def test_adamw_three_steps_match_numpy_hand_computed():
    W = W0.numpy().astype(np.float64)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    b1, b2 = BETAS
    for t, g_t in enumerate(GRADS, start=1):
        g = g_t.numpy().astype(np.float64)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1**t)
        denom = np.sqrt(v) / np.sqrt(1 - b2**t) + EPS
        W = W * (1 - LR * WD)          # decoupled decay
        W = W - LR * m_hat / denom     # Adam step

    got = run_ours(W0, GRADS).numpy()
    np.testing.assert_allclose(got, W, atol=1e-6, rtol=1e-6)


def test_adamw_matches_torch_optim_adamw_bitlevel():
    param_ref = torch.nn.Parameter(W0.clone())
    ref = torch.optim.AdamW(
        [param_ref], lr=LR, betas=BETAS, eps=EPS, weight_decay=WD
    )
    for g in GRADS:
        param_ref.grad = g.clone()
        ref.step()

    got = run_ours(W0, GRADS)
    torch.testing.assert_close(got, param_ref.detach(), atol=1e-7, rtol=1e-7)


def test_adamw_matches_torch_over_longer_run_with_zero_wd():
    torch.manual_seed(4321)  # dev-seeded random grads; expectation is torch itself
    grads = [torch.randn(4, 3) for _ in range(20)]

    param_ref = torch.nn.Parameter(W0.clone())
    ref = torch.optim.AdamW(
        [param_ref], lr=0.02, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )
    param_got = torch.nn.Parameter(W0.clone())
    ours = AdamW([param_got], lr=0.02, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    for g in grads:
        param_ref.grad = g.clone()
        param_got.grad = g.clone()
        ref.step()
        ours.step()
    torch.testing.assert_close(param_got.detach(), param_ref.detach(),
                               atol=1e-6, rtol=1e-6)


def test_adamw_shape_spectrum_is_identity():
    param = torch.nn.Parameter(W0.clone())
    opt = AdamW([param], lr=LR)
    x = torch.tensor([[1.0, -2.0, 3.0]])
    out = opt.shape_spectrum(x, {}, opt.param_groups[0])
    assert out is x
