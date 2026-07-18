"""WP0.4 unit tests: reference Muon on a fixed tiny matrix, hand-computed.

The expected trajectory is computed inside each test from the written update
rule (vendor/airbench/airbench94_muon.py:77-85):

    buf_t = momentum * buf_{t-1} + G_t
    M_t   = G_t + momentum * buf_t   (nesterov)  |  buf_t (heavy-ball)
    O_t   = NewtonSchulz5(M_t)
    W_t   = W_{t-1} * (1 - lr*wd) - alpha * O_t

with the orthogonalization supplied by the *verbatim vendor NS copy*
(tests/test_optim_refs.py:airbench_ns_reference) and everything else done
with plain elementwise arithmetic, independent of src/optim internals.

All matrices are fixed literals; >= 2 steps everywhere; tight tolerances.
"""

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from test_optim_refs import airbench_ns_reference

from src.optim.muon import Muon, adjusted_lr_for_shape

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


def run_muon(w0, grads, **kwargs):
    param = torch.nn.Parameter(w0.clone())
    opt = Muon([param], **kwargs)
    for g in grads:
        param.grad = g.clone()
        opt.step()
    return param.detach()


def expected_muon(w0, grads, lr, momentum, nesterov, ns_steps, wd=0.0, alpha=None):
    """Independent trajectory from the written rule + verbatim vendor NS."""
    alpha = lr if alpha is None else alpha
    W = w0.clone()
    buf = torch.zeros_like(w0)
    for g in grads:
        buf = momentum * buf + g
        M = g + momentum * buf if nesterov else buf
        O = airbench_ns_reference(M, steps=ns_steps).to(torch.float32)
        W = W * (1.0 - lr * wd) - alpha * O
    return W


def test_muon_three_steps_nesterov_matches_hand_computed():
    """Airbench record hyperparameters (lr 0.24, momentum 0.6, nesterov,
    ns_steps 3) over 3 fixed steps."""
    got = run_muon(
        W0, GRADS, lr=0.24, momentum=0.6, nesterov=True, ns_steps=3
    )
    want = expected_muon(W0, GRADS, lr=0.24, momentum=0.6, nesterov=True, ns_steps=3)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def test_muon_three_steps_heavy_ball_matches_hand_computed():
    got = run_muon(
        W0, GRADS, lr=0.1, momentum=0.95, nesterov=False, ns_steps=5
    )
    want = expected_muon(W0, GRADS, lr=0.1, momentum=0.95, nesterov=False, ns_steps=5)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def test_muon_weight_decay_is_decoupled():
    got = run_muon(
        W0, GRADS[:2], lr=0.1, momentum=0.6, nesterov=True, ns_steps=3,
        weight_decay=0.5,
    )
    want = expected_muon(
        W0, GRADS[:2], lr=0.1, momentum=0.6, nesterov=True, ns_steps=3, wd=0.5
    )
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def test_muon_adjust_lr_spectral_norm():
    """adjust_lr='spectral_norm' steps at lr*sqrt(fan_out/fan_in); decoupled
    decay stays at base lr (DynMuon convention, dynmuon.py:629-680)."""
    lr = 0.1
    got = run_muon(
        W0, GRADS[:2], lr=lr, momentum=0.6, nesterov=True, ns_steps=3,
        weight_decay=0.5, adjust_lr="spectral_norm",
    )
    want = expected_muon(
        W0, GRADS[:2], lr=lr, momentum=0.6, nesterov=True, ns_steps=3, wd=0.5,
        alpha=lr * math.sqrt(4.0 / 3.0),
    )
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def test_adjusted_lr_for_shape_matches_vendor_formulas():
    # dynmuon.py:671-680 (spectral_norm) and :658-668 (rms_norm), flatten=True
    shape = torch.Size([8, 4, 3, 3])
    fan_out, fan_in = 8, 36
    assert adjusted_lr_for_shape(0.1, shape, None) == 0.1
    assert math.isclose(
        adjusted_lr_for_shape(0.1, shape, "spectral_norm"),
        0.1 * math.sqrt(fan_out / fan_in),
    )
    assert math.isclose(
        adjusted_lr_for_shape(0.1, shape, "rms_norm"),
        0.1 * 0.2 * math.sqrt(max(fan_out, fan_in)),
    )


def test_muon_flattens_conv_filters_like_reference():
    """4D params are reshaped to (out_channels, -1) before NS and restored
    (airbench94_muon.py:84). Expectation flattens independently."""
    w0 = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) - 11.5) / 10.0
    g1 = torch.sin(torch.arange(24, dtype=torch.float32)).reshape(2, 3, 2, 2)
    g2 = torch.cos(torch.arange(24, dtype=torch.float32)).reshape(2, 3, 2, 2)

    param = torch.nn.Parameter(w0.clone())
    opt = Muon([param], lr=0.2, momentum=0.6, nesterov=True, ns_steps=3)
    for g in (g1, g2):
        param.grad = g.clone()
        opt.step()

    W = w0.reshape(2, 12).clone()
    buf = torch.zeros(2, 12)
    for g in (g1, g2):
        gf = g.reshape(2, 12)
        buf = 0.6 * buf + gf
        M = gf + 0.6 * buf
        O = airbench_ns_reference(M, steps=3).to(torch.float32)
        W = W - 0.2 * O
    torch.testing.assert_close(param.detach(), W.reshape(2, 3, 2, 2),
                               atol=1e-6, rtol=1e-6)


def test_muon_state_dict_roundtrip_resumes_identically():
    """Checkpoint resumability (plan: every experiment resumable).

    The state dict is deep-copied before loading, as a torch.save/torch.load
    cycle would do; torch's ``state_dict()`` references the live state tensors,
    so loading it uncopied into a second live optimizer would alias buffers.
    """
    import copy

    param_a = torch.nn.Parameter(W0.clone())
    opt_a = Muon([param_a], lr=0.1, momentum=0.6, nesterov=True, ns_steps=3)
    param_a.grad = GRADS[0].clone()
    opt_a.step()

    param_b = torch.nn.Parameter(param_a.detach().clone())
    opt_b = Muon([param_b], lr=0.1, momentum=0.6, nesterov=True, ns_steps=3)
    opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))

    for g in GRADS[1:]:
        param_a.grad = g.clone()
        param_b.grad = g.clone()
        opt_a.step()
        opt_b.step()
    torch.testing.assert_close(param_a.detach(), param_b.detach(),
                               atol=0.0, rtol=0.0)
