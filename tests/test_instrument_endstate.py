"""Program #13 tooling tests: power-iteration top Hessian eigenvalue.

Pre-registered obligation (endstate-prereg.md §8): the new lambda1 loop is
unit-tested against a fixed quadratic before use.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.instrument.endstate import endpoint_lambda1, power_iteration_top_eig


def test_power_iteration_recovers_top_eigenvalue_of_fixed_matrix():
    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(12, 12, generator=g, dtype=torch.float64))
    eigs = torch.tensor([9.0, 4.0, 1.0] + [0.1] * 9, dtype=torch.float64)
    A = (Q * eigs) @ Q.T
    lam, v = power_iteration_top_eig(
        lambda x: A.float() @ x, dim=12, iters=60
    )
    assert lam == pytest.approx(9.0, rel=1e-3)
    assert float(torch.dot(v.double(), Q[:, 0]).abs()) == pytest.approx(1.0, abs=1e-2)


def test_power_iteration_dominant_negative_eigenvalue_keeps_sign():
    A = torch.diag(torch.tensor([-5.0, 2.0, 1.0]))
    lam, _ = power_iteration_top_eig(lambda x: A @ x, dim=3, iters=60)
    assert lam == pytest.approx(-5.0, rel=1e-3)


def test_endpoint_lambda1_matches_exact_hessian_on_tiny_model():
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(),
                                torch.nn.Linear(8, 3))
    x = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))
    lam = endpoint_lambda1(model, x, y, iters=80)

    # exact reference: dense Hessian of the identical fp32 loss
    params = list(model.parameters())
    names = [n for n, _ in model.named_parameters()]

    def loss_of_flat(flat):
        vs, off = {}, 0
        for n, p in zip(names, params):
            vs[n] = flat[off:off + p.numel()].reshape(p.shape)
            off += p.numel()
        out = torch.func.functional_call(model, vs, (x,))
        return torch.nn.functional.cross_entropy(
            out, y, label_smoothing=0.2, reduction="sum"
        )

    flat0 = torch.cat([p.detach().reshape(-1) for p in params])
    H = torch.autograd.functional.hessian(loss_of_flat, flat0)
    exact = torch.linalg.eigvalsh(H).max().item()
    assert lam == pytest.approx(exact, rel=0.02)


def test_endpoint_lambda1_deterministic_given_seed():
    torch.manual_seed(5)
    model = torch.nn.Linear(6, 3)
    x = torch.randn(10, 6)
    y = torch.randint(0, 3, (10,))
    a = endpoint_lambda1(model, x, y, iters=30, seed=11)
    b = endpoint_lambda1(model, x, y, iters=30, seed=11)
    assert a == b
