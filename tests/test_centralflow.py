"""Tests for src/optim/centralflow.py — bookkeeping plus the edge-of-stability
toy that checks the *mechanism*.

CPU-only, float64, fully deterministic (the toy has no stochasticity; the
seed is set anyway so any future noise addition stays reproducible).

The toy (Cohen et al., arXiv:2410.24206, self-stabilization):

    L(x, y) = 1/2 e^y x^2 + rho y,   rho > 0 small

The x-curvature is h(y) = e^y, so y controls the sharpness.  Plain GD moves y
down through two channels: the rho term (slow, oscillation-independent) and
the 1/2 e^y x^2 term (only when x is away from 0, i.e. only under
oscillation).  With eta e^{y0} > 2 the x-iteration is unstable, x oscillates
with growing amplitude, and that second channel drives y down until the
oscillation dies -- self-stabilization.  At small eta there is no
oscillation and y barely moves.

Central-flow prediction: the time-averaged trajectory of the large-eta run is
gradient flow plus -(sigma^2/2) grad_y lambda with lambda = v^T H v = e^y for
v = e_x and sigma^2 the mean-square x.  Run (c) feeds exactly that term to a
small-eta run through :class:`CentralFlowTerm` and must reproduce run (a)'s
y-trajectory, which run (b) (same small eta, no term) does not.
"""

import math

import pytest
import torch

from src.optim.centralflow import CentralFlowTerm

# ------------------------------------------------------------- toy constants
RHO = 1e-4  # slow oscillation-independent drift in y
X0, Y0 = 0.01, 1.0  # eta_a * e^{Y0} = 2.45 > 2  => x unstable in run (a)
ETA_LARGE = 0.9
ETA_SMALL = 0.15
STEPS_LARGE = 100
K = int(round(ETA_LARGE / ETA_SMALL))  # small-eta steps per large-eta step
STEPS_SMALL = STEPS_LARGE * K  # matched flow time eta * steps
SIGMA_WINDOW = 5  # trailing window (large-eta steps) for <x^2>


def _leaf(value):
    return torch.tensor([value], dtype=torch.float64, requires_grad=True)


def _toy_loss(x, y, rho=RHO):
    return (0.5 * torch.exp(y) * x**2 + rho * y).sum()


def _gd(eta, steps):
    """Plain GD on the toy; returns per-step (x, y) before each update."""
    x, y = _leaf(X0), _leaf(Y0)
    xs, ys = [], []
    for _ in range(steps):
        xs.append(float(x.detach()))
        ys.append(float(y.detach()))
        gx, gy = torch.autograd.grad(_toy_loss(x, y), [x, y])
        with torch.no_grad():
            x.sub_(gx, alpha=eta)
            y.sub_(gy, alpha=eta)
    xs.append(float(x.detach()))
    ys.append(float(y.detach()))
    return xs, ys


def _trailing_mean_square(xs, window):
    out = []
    for t in range(len(xs)):
        lo = max(0, t - window + 1)
        seg = xs[lo : t + 1]
        out.append(sum(v * v for v in seg) / len(seg))
    return out


def _gd_with_central_flow(eta, steps, sigma2_schedule, term):
    """Small-eta GD plus the explicit central-flow drift along v = e_x."""
    x, y = _leaf(X0), _leaf(Y0)
    params = [x, y]
    ex = [torch.ones_like(x.detach()), torch.zeros_like(y.detach())]
    ys = [float(y.detach())]
    for step in range(steps):
        gx, gy = torch.autograd.grad(_toy_loss(x, y), params)
        with torch.no_grad():
            x.sub_(gx, alpha=eta)
            y.sub_(gy, alpha=eta)
        # w_i = sigma_i^2 / 2 ; beta = eta (the flow-time increment per step)
        term.refresh(
            lambda: _toy_loss(x, y),
            params,
            [ex],
            weights=[0.5 * sigma2_schedule[step]],
            step=step,
        )
        term.apply(params, beta=eta)
        ys.append(float(y.detach()))
    return ys


# --------------------------------------------------------------- the toy test


def test_central_flow_term_reproduces_eos_self_stabilization():
    torch.manual_seed(1234)  # dev seed (>= 1000); the toy is deterministic

    # (a) large eta: x oscillates (eta * e^{y0} > 2), y self-stabilizes down.
    xs_a, ys_a = _gd(ETA_LARGE, STEPS_LARGE)
    assert ETA_LARGE * math.exp(Y0) > 2.0
    assert max(abs(v) for v in xs_a) > 10 * X0  # the oscillation did grow
    # sign flips: the hallmark period-2 EoS oscillation
    flips = sum(1 for a, b in zip(xs_a, xs_a[1:]) if a * b < 0)
    assert flips > 10

    # (b) small eta, matched flow time: no oscillation, y barely moves.
    xs_b, ys_b = _gd(ETA_SMALL, STEPS_SMALL)
    assert ETA_SMALL * math.exp(Y0) < 2.0
    assert max(abs(v) for v in xs_b) <= X0

    # (c) same small eta + explicit central-flow term, fed run (a)'s measured
    # mean-square x on the matching flow-time schedule.
    ms_a = _trailing_mean_square(xs_a[:STEPS_LARGE], SIGMA_WINDOW)
    schedule = [ms_a[min(k // K, STEPS_LARGE - 1)] for k in range(STEPS_SMALL)]
    term = CentralFlowTerm()
    ys_c = _gd_with_central_flow(ETA_SMALL, STEPS_SMALL, schedule, term)

    # y_a: the time-averaged run-(a) value over the final 10% of the run.
    tail = max(1, STEPS_LARGE // 10)
    y_a = sum(ys_a[-tail:]) / tail
    y_b, y_c = ys_b[-1], ys_c[-1]

    # Direction of the mechanism: oscillation drives sharpness down.
    assert y_a < Y0 - 0.3  # run (a) self-stabilized
    assert y_b > Y0 - 0.05  # run (b) barely moved
    # Run (c) tracks run (a) far better than run (b) does.
    assert abs(y_c - y_a) < 0.2 * abs(y_b - y_a)

    # The term stayed in the regime it claims: lambda = e^y along v = e_x.
    stats = term.stats()
    assert stats["n_directions"] == 1
    assert stats["curvatures"][0] == pytest.approx(math.exp(y_c), rel=1e-6)
    assert stats["refresh_step"] == STEPS_SMALL - 1


def test_toy_run_c_curvature_ends_near_run_a_curvature():
    """The same comparison expressed in sharpness (e^y) instead of y."""
    xs_a, ys_a = _gd(ETA_LARGE, STEPS_LARGE)
    _, ys_b = _gd(ETA_SMALL, STEPS_SMALL)
    ms_a = _trailing_mean_square(xs_a[:STEPS_LARGE], SIGMA_WINDOW)
    schedule = [ms_a[min(k // K, STEPS_LARGE - 1)] for k in range(STEPS_SMALL)]
    ys_c = _gd_with_central_flow(ETA_SMALL, STEPS_SMALL, schedule, CentralFlowTerm())
    tail = max(1, STEPS_LARGE // 10)
    h_a = math.exp(sum(ys_a[-tail:]) / tail)
    h_b, h_c = math.exp(ys_b[-1]), math.exp(ys_c[-1])
    assert abs(h_c - h_a) < 0.2 * abs(h_b - h_a)


# ------------------------------------------------------------- bookkeeping


def test_apply_before_refresh_raises():
    x = _leaf(1.0)
    term = CentralFlowTerm()
    with pytest.raises(RuntimeError, match="before refresh"):
        term.apply([x], beta=0.1)


def test_apply_subtracts_beta_times_penalty_grad_exactly():
    # L = a x^4, v = e_x: lambda = 12 a x^2, grad lambda = 24 a x.
    # apply(beta) must set x <- x - beta * w * 24 a x.
    a, x0, w, beta = 0.7, 1.3, 0.5, 0.05
    x = _leaf(x0)
    term = CentralFlowTerm()
    term.refresh(lambda: a * (x**4).sum(), [x], [[torch.ones_like(x.detach())]], [w])
    assert term.curvatures[0] == pytest.approx(12 * a * x0**2, abs=1e-8)
    assert float(term.penalty_grads[0]) == pytest.approx(w * 24 * a * x0, abs=1e-8)
    term.apply([x], beta=beta)
    assert float(x.detach()) == pytest.approx(x0 - beta * w * 24 * a * x0, abs=1e-12)
    # The cached gradient is reusable: applying again subtracts the SAME
    # (stale) vector, which is the whole point of the refresh-every-M contract.
    term.apply([x], beta=beta)
    assert float(x.detach()) == pytest.approx(x0 - 2 * beta * w * 24 * a * x0, abs=1e-12)


def test_refresh_default_weights_are_one():
    a, x0 = 0.7, 1.3
    x = _leaf(x0)
    term = CentralFlowTerm()
    term.refresh(lambda: a * (x**4).sum(), [x], [[torch.ones_like(x.detach())]])
    assert float(term.penalty_grads[0]) == pytest.approx(24 * a * x0, abs=1e-8)


def test_stats_shape_and_values():
    x, y = _leaf(1.1), _leaf(-0.3)
    term = CentralFlowTerm()
    assert term.stats() == {
        "n_directions": 0,
        "curvatures": [],
        "penalty_grad_norm": 0.0,
        "refresh_step": None,
    }
    directions = [
        [torch.ones_like(x.detach()), torch.zeros_like(y.detach())],
        [torch.zeros_like(x.detach()), torch.ones_like(y.detach())],
    ]
    term.refresh(
        lambda: (0.5 * torch.exp(y) * x**2).sum(),
        [x, y],
        directions,
        weights=[1.0, 1.0],
        step=42,
    )
    stats = term.stats()
    assert set(stats) == {
        "n_directions",
        "curvatures",
        "penalty_grad_norm",
        "refresh_step",
    }
    assert stats["n_directions"] == 2
    assert len(stats["curvatures"]) == 2
    assert stats["refresh_step"] == 42
    expected = math.sqrt(
        sum(float((g**2).sum()) for g in term.penalty_grads)
    )
    assert stats["penalty_grad_norm"] == pytest.approx(expected, rel=1e-12)
    assert stats["penalty_grad_norm"] > 0.0


def test_refresh_step_defaults_to_refresh_counter():
    x = _leaf(1.0)
    term = CentralFlowTerm()
    for expected in range(3):
        term.refresh(lambda: (x**4).sum(), [x], [[torch.ones_like(x.detach())]])
        assert term.step_of_refresh == expected
    assert term.n_refreshes == 3


def test_apply_refuses_mismatched_params():
    x, y = _leaf(1.0), _leaf(2.0)
    term = CentralFlowTerm()
    term.refresh(lambda: (x**4).sum(), [x], [[torch.ones_like(x.detach())]])
    with pytest.raises(ValueError, match="entries"):
        term.apply([x, y], beta=0.1)
    wrong_shape = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="refreshed with"):
        term.apply([wrong_shape], beta=0.1)


def test_apply_does_not_build_graph_or_touch_grads():
    x = _leaf(1.0)
    term = CentralFlowTerm()
    term.refresh(lambda: (x**4).sum(), [x], [[torch.ones_like(x.detach())]])
    term.apply([x], beta=0.1)
    assert x.grad is None
    assert x.is_leaf and x.requires_grad
