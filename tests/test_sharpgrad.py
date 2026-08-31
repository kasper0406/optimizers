"""Analytic tests for src/optim/sharpgrad.py (central-flow curvature gradient).

CPU-only, float64 throughout: every case has a closed-form lambda = v^T H v
and grad_w lambda, asserted to 1e-8.  Covers the quartic (nonzero third
derivative), a separable loss where the curvature depends on a *different*
parameter than the direction touches, the quadratic case (constant Hessian =>
the third-order term must vanish exactly), the allow_unused path, the
documented internal unit-normalization of directions, and agreement between
the measurement-only and full entry points.
"""

import pytest
import torch

from src.optim.sharpgrad import directional_curvature, directional_curvature_and_grad

TOL = 1e-8


def _p(value):
    return torch.tensor([value], dtype=torch.float64, requires_grad=True)


def _ones_like(p):
    return torch.ones_like(p.detach())


def _zeros_like(p):
    return torch.zeros_like(p.detach())


# --------------------------------------------------------------- quartic 1-D


def test_quartic_curvature_and_grad():
    # L(x) = a x^4  =>  lambda = L'' = 12 a x^2,  d lambda / dx = 24 a x
    a, x0 = 0.7, 1.3
    x = _p(x0)
    curvatures, grads = directional_curvature_and_grad(
        lambda: a * (x**4).sum(), [x], [[_ones_like(x)]]
    )
    assert curvatures[0] == pytest.approx(12 * a * x0**2, abs=TOL)
    assert float(grads[0]) == pytest.approx(24 * a * x0, abs=TOL)


def test_quartic_weight_scales_penalty_grad_but_not_curvature():
    a, x0, w = 0.7, 1.3, 2.5
    x = _p(x0)
    curvatures, grads = directional_curvature_and_grad(
        lambda: a * (x**4).sum(), [x], [[_ones_like(x)]], weights=[w]
    )
    assert curvatures[0] == pytest.approx(12 * a * x0**2, abs=TOL)
    assert float(grads[0]) == pytest.approx(w * 24 * a * x0, abs=TOL)


# ------------------------------------------------------- 2-param separable


def test_separable_curvature_depends_on_other_param():
    # L(x, y) = 1/2 e^y x^2, v = e_x  =>  lambda = e^y,
    # d lambda / dx = 0,  d lambda / dy = e^y
    x0, y0 = 0.37, -0.4
    x, y = _p(x0), _p(y0)
    loss = lambda: (0.5 * torch.exp(y) * x**2).sum()  # noqa: E731
    curvatures, grads = directional_curvature_and_grad(
        loss, [x, y], [[_ones_like(x), _zeros_like(y)]]
    )
    import math

    assert curvatures[0] == pytest.approx(math.exp(y0), abs=TOL)
    assert float(grads[0]) == pytest.approx(0.0, abs=TOL)
    assert float(grads[1]) == pytest.approx(math.exp(y0), abs=TOL)


def test_separable_direction_along_y_sees_the_x_curvature_block():
    # same loss, v = e_y  =>  lambda = d2L/dy2 = 1/2 e^y x^2
    x0, y0 = 0.37, -0.4
    x, y = _p(x0), _p(y0)
    loss = lambda: (0.5 * torch.exp(y) * x**2).sum()  # noqa: E731
    curvatures, grads = directional_curvature_and_grad(
        loss, [x, y], [[_zeros_like(x), _ones_like(y)]]
    )
    import math

    lam = 0.5 * math.exp(y0) * x0**2
    assert curvatures[0] == pytest.approx(lam, abs=TOL)
    assert float(grads[0]) == pytest.approx(math.exp(y0) * x0, abs=TOL)
    assert float(grads[1]) == pytest.approx(lam, abs=TOL)


# ------------------------------------------------- quadratic: no third order


def test_two_directions_with_weights_on_a_quadratic():
    # L = 1/2 (3 x^2 + 5 y^2): H constant => lambdas exact, penalty grad zero.
    x, y = _p(0.9), _p(-1.7)
    loss = lambda: (0.5 * (3 * x**2 + 5 * y**2)).sum()  # noqa: E731
    directions = [
        [_ones_like(x), _zeros_like(y)],
        [_zeros_like(x), _ones_like(y)],
    ]
    curvatures, grads = directional_curvature_and_grad(
        loss, [x, y], directions, weights=[2.0, 0.5]
    )
    assert curvatures[0] == pytest.approx(3.0, abs=TOL)
    assert curvatures[1] == pytest.approx(5.0, abs=TOL)
    assert float(grads[0]) == pytest.approx(0.0, abs=TOL)
    assert float(grads[1]) == pytest.approx(0.0, abs=TOL)
    assert [g.shape for g in grads] == [x.shape, y.shape]
    assert not any(g.requires_grad for g in grads)


def test_mixed_quadratic_off_diagonal_direction():
    # L = 1/2 (x^2 + 4 y^2) + x y, v = (1, 1)/sqrt(2):
    # H = [[1, 1], [1, 4]]  =>  lambda = (1 + 2 + 4)/2 = 3.5
    x, y = _p(0.3), _p(0.2)
    loss = lambda: (0.5 * (x**2 + 4 * y**2) + x * y).sum()  # noqa: E731
    curvatures, grads = directional_curvature_and_grad(
        loss, [x, y], [[_ones_like(x), _ones_like(y)]]
    )
    assert curvatures[0] == pytest.approx(3.5, abs=TOL)
    assert float(grads[0]) == pytest.approx(0.0, abs=TOL)
    assert float(grads[1]) == pytest.approx(0.0, abs=TOL)


# ----------------------------------------------------------- allow_unused


def test_direction_not_touching_a_param_gives_zero_entry():
    # L = a x^4 + b y^4 (fully separable), v = e_x with a None block for y:
    # the y branch never enters the lambda graph (allow_unused path).
    a, b, x0, y0 = 0.5, 2.0, 1.1, -0.8
    x, y = _p(x0), _p(y0)
    loss = lambda: (a * x**4).sum() + (b * y**4).sum()  # noqa: E731
    curvatures, grads = directional_curvature_and_grad(loss, [x, y], [[_ones_like(x), None]])
    assert curvatures[0] == pytest.approx(12 * a * x0**2, abs=TOL)
    assert float(grads[0]) == pytest.approx(24 * a * x0, abs=TOL)
    assert float(grads[1]) == 0.0


def test_param_absent_from_the_loss_gets_zero_penalty_grad():
    a, x0 = 0.5, 1.1
    x, spectator = _p(x0), _p(3.0)
    curvatures, grads = directional_curvature_and_grad(
        lambda: (a * x**4).sum(), [x, spectator], [[_ones_like(x), None]]
    )
    assert curvatures[0] == pytest.approx(12 * a * x0**2, abs=TOL)
    assert float(grads[1]) == 0.0


# ------------------------------------------------------- unit normalization


def test_directions_are_normalized_internally():
    # Documented choice: directions are scaled to unit global L2 norm inside
    # the function, so lambda is the Rayleigh quotient and is invariant to the
    # scale of the direction the caller passes.
    a, x0 = 0.7, 1.3
    x = _p(x0)
    loss = lambda: a * (x**4).sum()  # noqa: E731
    unit, gunit = directional_curvature_and_grad(loss, [x], [[_ones_like(x)]])
    scaled, gscaled = directional_curvature_and_grad(loss, [x], [[5.0 * _ones_like(x)]])
    tiny, gtiny = directional_curvature_and_grad(loss, [x], [[1e-3 * _ones_like(x)]])
    assert scaled[0] == pytest.approx(unit[0], abs=TOL)
    assert tiny[0] == pytest.approx(unit[0], abs=TOL)
    assert float(gscaled[0]) == pytest.approx(float(gunit[0]), abs=TOL)
    assert float(gtiny[0]) == pytest.approx(float(gunit[0]), abs=TOL)


def test_multi_param_direction_normalized_over_the_global_norm():
    # L = 1/2 (3 x^2 + 5 y^2), v = (1, 1) -> unit (1, 1)/sqrt(2):
    # lambda = (3 + 5) / 2 = 4
    x, y = _p(0.9), _p(-1.7)
    loss = lambda: (0.5 * (3 * x**2 + 5 * y**2)).sum()  # noqa: E731
    curvatures, _ = directional_curvature_and_grad(
        loss, [x, y], [[_ones_like(x), _ones_like(y)]]
    )
    assert curvatures[0] == pytest.approx(4.0, abs=TOL)


def test_zero_direction_refused():
    x = _p(1.0)
    with pytest.raises(ValueError, match="zero norm"):
        directional_curvature_and_grad(lambda: (x**4).sum(), [x], [[_zeros_like(x)]])
    with pytest.raises(ValueError, match="zero norm"):
        directional_curvature_and_grad(lambda: (x**4).sum(), [x], [[None]])


# --------------------------------------------------------------- validation


def test_param_without_requires_grad_refused():
    x = _p(1.0)
    frozen = torch.tensor([2.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="requires_grad=False"):
        directional_curvature_and_grad(
            lambda: (x**4).sum(), [x, frozen], [[_ones_like(x), None]]
        )


def test_direction_length_and_shape_mismatch_refused():
    x, y = _p(1.0), _p(2.0)
    loss = lambda: (x**4).sum() + (y**4).sum()  # noqa: E731
    with pytest.raises(ValueError, match="entries but there are 2 params"):
        directional_curvature_and_grad(loss, [x, y], [[_ones_like(x)]])
    bad = torch.ones(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="expected"):
        directional_curvature_and_grad(loss, [x, y], [[bad, None]])


def test_weights_length_mismatch_refused():
    x = _p(1.0)
    with pytest.raises(ValueError, match="weights has"):
        directional_curvature_and_grad(
            lambda: (x**4).sum(), [x], [[_ones_like(x)]], weights=[1.0, 2.0]
        )


def test_empty_params_and_directions_refused():
    x = _p(1.0)
    with pytest.raises(ValueError, match="at least one parameter"):
        directional_curvature_and_grad(lambda: (x**4).sum(), [], [])
    with pytest.raises(ValueError, match="at least one direction"):
        directional_curvature_and_grad(lambda: (x**4).sum(), [x], [])


def test_non_scalar_loss_refused():
    x = _p(1.0)
    with pytest.raises(ValueError, match="scalar tensor"):
        directional_curvature_and_grad(lambda: x**4, [x], [[_ones_like(x)]])


def test_directions_are_detached_constants():
    # A direction carrying requires_grad must not leak into the graph: the
    # result is identical to the same direction detached.
    a, x0 = 0.7, 1.3
    x = _p(x0)
    loss = lambda: a * (x**4).sum()  # noqa: E731
    live = torch.ones_like(x.detach()).requires_grad_(True)
    curvatures, grads = directional_curvature_and_grad(loss, [x], [[live]])
    assert curvatures[0] == pytest.approx(12 * a * x0**2, abs=TOL)
    assert float(grads[0]) == pytest.approx(24 * a * x0, abs=TOL)
    assert live.grad is None


# --------------------------------------------------- measurement-only path


def test_measurement_only_matches_full_curvatures():
    x, y = _p(0.37), _p(-0.4)
    loss = lambda: (0.5 * torch.exp(y) * x**2).sum() + 0.25 * (y**4).sum()  # noqa: E731
    directions = [
        [_ones_like(x), _zeros_like(y)],
        [_zeros_like(x), _ones_like(y)],
        [_ones_like(x), _ones_like(y)],
    ]
    full, grads = directional_curvature_and_grad(loss, [x, y], directions)
    cheap = directional_curvature(loss, [x, y], directions)
    assert cheap == pytest.approx(full, abs=TOL)
    assert grads is not None
    _, none_grads = directional_curvature_and_grad(
        loss, [x, y], directions, create_graph_chain=False
    )
    assert none_grads is None


def test_multiple_directions_share_one_forward_pass():
    calls = {"n": 0}
    x = _p(1.1)

    def loss():
        calls["n"] += 1
        return 0.7 * (x**4).sum()

    directions = [[_ones_like(x)] for _ in range(4)]
    curvatures, _ = directional_curvature_and_grad(loss, [x], directions)
    assert calls["n"] == 1
    assert curvatures == pytest.approx([12 * 0.7 * 1.1**2] * 4, abs=TOL)


# -------------------------------------------------------- multi-element case


def test_vector_param_quartic_sum():
    # L(w) = a sum_k w_k^4, v = e_1: lambda = 12 a w_1^2, grad = 24 a w_1 e_1
    a = 0.3
    w = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    curvatures, grads = directional_curvature_and_grad(
        lambda: a * (w**4).sum(), [w], [[v]]
    )
    assert curvatures[0] == pytest.approx(12 * a * 1.5**2, abs=TOL)
    expected = torch.tensor([24 * a * 1.5, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(grads[0], expected, atol=TOL)
