"""Tests for src/optim/teleport.py (teleport moves inside Muon, WP-J round 2).

CPU-only, float32/float64.  The load-bearing test is the FINITE-DIFFERENCE
check: the closed-form log-scale gradient ``dF/dt_c = -(A o U V^T) row-c sum``
is what the whole teleport rests on, and it is checked against central
differences of ``F(t) = ||diag(e^-t) G||_*`` coordinate by coordinate.

The last test closes the loop against the real thing: predict the nuclear-norm
gain from the base-point gradient alone, apply the move to the WEIGHTS with
``orbit.apply_channel_scales``, then re-run a fresh backward on the same batch
and check the realized gain equals the prediction.
"""

import math

import pytest
import torch
from torch import nn

from src.instrument.orbit import apply_channel_scales, find_conv_bn_pairs
from src.optim.teleport import (
    _nuclear_value_and_grad,
    nuclear_ascent,
    teleport_alphas,
    transport_gradlike,
)


# ------------------------------------------------------------------ helpers


def _nuclear(A: torch.Tensor) -> float:
    return float(torch.linalg.svdvals(A).sum().item())


def _ratio(G: torch.Tensor, alphas: torch.Tensor) -> float:
    """``||diag(1/alpha) G||_* / ||G||_*``, recomputed from scratch (the
    definition of ``achieved_ratio``, independent of the ascent's bookkeeping)."""
    scaled = (1.0 / alphas.to(G.dtype)).unsqueeze(1) * G
    return _nuclear(scaled) / _nuclear(G)


def _unbalanced(seed: int = 0) -> torch.Tensor:
    """A gradient matrix with very unbalanced row norms (rows scaled 1, 1, 10)."""
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(3, 5, generator=g, dtype=torch.float64)
    G[2] *= 10.0
    return G


def _make_net(seed: int = 1234) -> nn.Sequential:
    """Conv->BN->ReLU->Conv->BN->Flatten->Linear, float32, CPU (the
    tests/test_orbit.py fixture)."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 4, kernel_size=3, padding=1),
        nn.BatchNorm2d(4),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 5),
    )


def _batch(n: int = 16, seed: int = 7):
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, 3, 8, 8, generator=gen),
        torch.randint(0, 5, (n,), generator=gen),
    )


def _loss(model, batch) -> torch.Tensor:
    inputs, labels = batch
    return nn.functional.cross_entropy(model(inputs), labels, reduction="sum")


def _grad_mats(pairs):
    return [
        conv.weight.grad.detach().reshape(conv.weight.shape[0], -1).clone()
        for conv, _bn, _name in pairs
    ]


# ------------------------------------------------- the gradient of the objective


def test_analytic_t_gradient_matches_central_differences():
    """``dF/dt_c`` from the SVD subgradient vs central differences of
    ``F(t) = ||diag(e^-t) G||_*`` at t = 0, float64, h = 1e-6."""
    gen = torch.Generator().manual_seed(11)
    G = torch.randn(6, 10, generator=gen, dtype=torch.float64)

    value, analytic = _nuclear_value_and_grad(G)
    assert value == pytest.approx(_nuclear(G), rel=1e-12)

    h = 1e-6
    for c in range(G.shape[0]):
        e = torch.zeros(G.shape[0], dtype=torch.float64)
        e[c] = h
        plus = _nuclear(torch.exp(-e).unsqueeze(1) * G)
        minus = _nuclear(torch.exp(e).unsqueeze(1) * G)
        fd = (plus - minus) / (2 * h)
        assert float(analytic[c]) == pytest.approx(fd, abs=1e-5)

    # sanity on the sign/structure: sum_c dF/dt_c = -<A, UV^T> = -||G||_*
    assert float(analytic.sum()) == pytest.approx(-_nuclear(G), rel=1e-10)


def test_analytic_gradient_holds_away_from_the_base_point():
    """Same check at a non-zero t (the ascent evaluates the formula there too)."""
    gen = torch.Generator().manual_seed(12)
    G = torch.randn(4, 7, generator=gen, dtype=torch.float64)
    t0 = torch.tensor([0.3, -0.2, 0.1, -0.45], dtype=torch.float64)

    def F(t):
        return _nuclear(torch.exp(-t).unsqueeze(1) * G)

    _, analytic = _nuclear_value_and_grad(torch.exp(-t0).unsqueeze(1) * G)
    h = 1e-6
    for c in range(G.shape[0]):
        e = torch.zeros(4, dtype=torch.float64)
        e[c] = h
        fd = (F(t0 + e) - F(t0 - e)) / (2 * h)
        assert float(analytic[c]) == pytest.approx(fd, abs=1e-5)


# ------------------------------------------------------------------- ascent


def test_ascent_improves_the_nuclear_norm_on_unbalanced_rows():
    G = _unbalanced()
    alphas, ratio = nuclear_ascent(G, spread=2.0, iters=20, step_size=0.3)

    assert alphas.shape == (3,)
    assert ratio > 1.02
    # the reported ratio is the definition, recomputed independently
    assert _ratio(G, alphas) == pytest.approx(ratio, rel=1e-9)
    assert bool((alphas >= 0.5 - 1e-12).all()) and bool((alphas <= 2.0 + 1e-12).all())


def test_returned_alphas_stay_inside_the_spread_box():
    gen = torch.Generator().manual_seed(3)
    for spread in (1.1, 1.5, 4.0):
        G = torch.randn(9, 20, generator=gen)
        alphas, _ = nuclear_ascent(G, spread=spread, iters=30, step_size=1.0)
        assert bool((alphas >= 1.0 / spread - 1e-6).all())
        assert bool((alphas <= spread + 1e-6).all())


def test_ratio_is_never_below_one_best_seen_semantics():
    """t = 0 is the first candidate and the argmax is returned, so a teleport
    can never be worse than not teleporting -- including with steps far too
    large for the objective (fixed-step projected ascent is not monotone)."""
    gen = torch.Generator().manual_seed(5)
    for step_size in (0.0, 0.01, 0.5, 5.0, 50.0):
        for shape in ((3, 5), (8, 2), (1, 6), (16, 16)):
            G = torch.randn(*shape, generator=gen)
            alphas, ratio = nuclear_ascent(G, 2.0, iters=7, step_size=step_size)
            assert ratio >= 1.0
            assert _ratio(G, alphas) == pytest.approx(ratio, rel=1e-5)


def test_ratio_factors_into_gauge_and_relative_parts():
    """Module GAUGE NOTE, pinned: ``ratio = (1/geomean(alpha)) * ratio_rel``.

    The uniform component of the move is gauge-trivial for Muon
    (``polar(cG) = polar(G)``) and is removed by the recipe's per-step weight
    renormalization; only ``ratio_rel`` is descent potential a Muon step can
    spend.  The ascent direction always carries that uniform component
    (``sum_c dF/dt_c = -||A||_*``), so on a gradient with near-uniform row
    norms most of the reported ratio is gauge.
    """
    G = _unbalanced()
    alphas, ratio = nuclear_ascent(G, spread=2.0, iters=20, step_size=0.3)
    geomean = float(alphas.log().mean().exp())
    relative = _ratio(G, alphas / geomean)
    assert ratio == pytest.approx(relative / geomean, rel=1e-9)
    assert relative > 1.05  # unbalanced rows leave real, non-gauge headroom

    # Gauge-fixed ascent (round-2 decision): geomean(alpha) ~= 1 -- the clamp
    # budget is spent on the relative pattern, so the reported ratio IS the
    # Muon-spendable ratio, and even an i.i.d. matrix has real relative
    # headroom (the pre-fix ascent wasted its budget on the uniform gauge
    # direction and got < 1.01 here).
    assert abs(math.log(geomean)) < 0.05
    gen = torch.Generator().manual_seed(0)
    iid = torch.randn(64, 288, generator=gen)
    a_iid, r_iid = nuclear_ascent(iid, spread=2.0, iters=20, step_size=0.3)
    g_iid = float(a_iid.log().mean().exp())
    assert abs(math.log(g_iid)) < 0.05
    assert r_iid == pytest.approx(_ratio(iid, a_iid / g_iid) / g_iid, rel=1e-5)
    assert r_iid > 1.05  # real, spendable gain even on i.i.d. structure


def test_ascent_is_deterministic():
    G = _unbalanced(seed=2)
    first = nuclear_ascent(G, 2.0, 15, 0.25)
    second = nuclear_ascent(G, 2.0, 15, 0.25)
    assert torch.equal(first[0], second[0])
    assert first[1] == second[1]


def test_half_precision_gradients_are_promoted_to_float32():
    G = _unbalanced().float()
    alphas, ratio = nuclear_ascent(G.half(), 2.0, 20, 0.3)
    assert alphas.dtype is torch.float32
    reference, ref_ratio = nuclear_ascent(G, 2.0, 20, 0.3)
    assert torch.allclose(alphas, reference, atol=1e-3)
    assert ratio == pytest.approx(ref_ratio, rel=1e-3)


def test_float64_gradients_keep_their_precision():
    alphas, _ = nuclear_ascent(_unbalanced(), 2.0, 5, 0.3)
    assert alphas.dtype is torch.float64


# --------------------------------------------------------------- edge cases


def test_zero_gradient_is_the_identity_move():
    alphas, ratio = nuclear_ascent(torch.zeros(4, 6), 2.0, 20, 0.3)
    assert torch.equal(alphas, torch.ones(4))
    assert ratio == 1.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_gradient_is_the_identity_move(bad):
    G = _unbalanced().float()
    G[1, 2] = bad
    alphas, ratio = nuclear_ascent(G, 2.0, 20, 0.3)
    assert torch.equal(alphas, torch.ones(3))
    assert ratio == 1.0


def test_zero_iterations_is_the_identity_move():
    alphas, ratio = nuclear_ascent(_unbalanced(), 2.0, iters=0, step_size=0.3)
    assert torch.equal(alphas, torch.ones(3, dtype=torch.float64))
    assert ratio == 1.0


def test_nuclear_ascent_rejects_bad_arguments():
    G = _unbalanced()
    with pytest.raises(ValueError, match="2-D"):
        nuclear_ascent(G.reshape(1, 3, 5), 2.0, 5, 0.3)
    with pytest.raises(ValueError, match="2-D"):
        nuclear_ascent(G[0], 2.0, 5, 0.3)
    with pytest.raises(ValueError, match="spread"):
        nuclear_ascent(G, 1.0, 5, 0.3)
    with pytest.raises(ValueError, match="spread"):
        nuclear_ascent(G, 0.5, 5, 0.3)
    with pytest.raises(ValueError, match="iters"):
        nuclear_ascent(G, 2.0, -1, 0.3)
    with pytest.raises(ValueError, match="step_size"):
        nuclear_ascent(G, 2.0, 5, -0.3)


# ------------------------------------------------------------ transport


def test_transport_scales_conv_channels_by_one_over_alpha():
    base = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    tensor = base.clone()
    transport_gradlike(tensor, torch.tensor([2.0, 0.5]))
    assert torch.equal(tensor[0], base[0] * 0.5)  # alpha = 2 halves the row
    assert torch.equal(tensor[1], base[1] * 2.0)


def test_transport_handles_bias_shaped_tensors():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    transport_gradlike(tensor, torch.tensor([2.0, 4.0, 0.5]))
    assert tensor.tolist() == pytest.approx([0.5, 0.5, 6.0])


def test_transport_is_in_place_on_the_same_storage():
    tensor = torch.ones(3, 4)
    before = tensor.data_ptr()
    assert transport_gradlike(tensor, torch.full((3,), 2.0)) is None
    assert tensor.data_ptr() == before
    assert torch.equal(tensor, torch.full((3, 4), 0.5))


def test_transport_computes_the_reciprocal_in_float32_for_half_tensors():
    tensor = torch.tensor([[4.0, -8.0], [1.0, 3.0]], dtype=torch.float16)
    transport_gradlike(tensor, torch.tensor([0.25, 2.0], dtype=torch.float32))
    assert tensor.dtype is torch.float16
    assert torch.equal(
        tensor, torch.tensor([[16.0, -32.0], [0.5, 1.5]], dtype=torch.float16)
    )

    # the reciprocal is taken in fp32 and rounded ONCE to fp16; taking it in
    # fp16 (as ``1 / alphas.half()``) rounds twice and gives different bits
    gen = torch.Generator().manual_seed(21)
    alphas = torch.tensor([2.035400629043579, 0.18707044422626495, 1.0 / 3.0])
    # premise: for these alphas the two reciprocals are genuinely different bits
    assert not torch.equal((1.0 / alphas).half(), 1.0 / alphas.half())

    values = torch.randn(3, 5, generator=gen)
    half = values.half()
    transport_gradlike(half, alphas)
    assert torch.equal(half, values.half() * (1.0 / alphas).half().unsqueeze(1))


def test_transport_of_alpha_then_one_over_alpha_round_trips():
    gen = torch.Generator().manual_seed(23)
    base = torch.randn(4, 6, generator=gen, dtype=torch.float64)
    tensor = base.clone()
    alphas = torch.tensor([0.5, 2.0, 1.25, 0.8], dtype=torch.float64)
    transport_gradlike(tensor, alphas)
    transport_gradlike(tensor, 1.0 / alphas)
    assert torch.allclose(tensor, base, atol=1e-12)


def test_transport_rejects_bad_arguments():
    tensor = torch.ones(3, 4)
    with pytest.raises(ValueError, match="1-D of length 3"):
        transport_gradlike(tensor, torch.ones(2))
    with pytest.raises(ValueError, match="1-D of length 3"):
        transport_gradlike(tensor, torch.ones(3, 1))
    with pytest.raises(ValueError, match="strictly positive"):
        transport_gradlike(tensor, torch.tensor([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        transport_gradlike(tensor, torch.tensor([1.0, -1.0, 2.0]))
    with pytest.raises(ValueError, match="float tensor"):
        transport_gradlike(torch.ones(3, dtype=torch.int64), torch.ones(3))
    assert torch.equal(tensor, torch.ones(3, 4))  # no partial mutation


# --------------------------------------------------------- teleport_alphas


def test_teleport_alphas_matches_per_pair_nuclear_ascent():
    pairs = [object(), object()]
    grads = [_unbalanced(seed=1), _unbalanced(seed=2)]
    alphas, ratios = teleport_alphas(pairs, grads, spread=2.0, iters=12, step_size=0.2)

    assert len(alphas) == len(ratios) == 2
    for grad, alpha, ratio in zip(grads, alphas, ratios):
        ref_alpha, ref_ratio = nuclear_ascent(grad, 2.0, 12, 0.2)
        assert torch.equal(alpha, ref_alpha)
        assert ratio == ref_ratio
        assert _ratio(grad, alpha) == pytest.approx(ratio, rel=1e-9)


def test_teleport_alphas_requires_one_gradient_matrix_per_pair():
    grads = [_unbalanced(seed=1), _unbalanced(seed=2)]
    with pytest.raises(ValueError, match="expected 3 gradient matrices"):
        teleport_alphas([object()] * 3, grads, 2.0, 5, 0.2)
    with pytest.raises(ValueError, match="expected 1 gradient matrices"):
        teleport_alphas([object()], grads, 2.0, 5, 0.2)


def test_teleport_alphas_pairs_are_independent():
    """Each pair's objective sees only its own gradient, so permuting the
    inputs permutes the outputs exactly."""
    pairs = [object(), object(), object()]
    grads = [_unbalanced(seed=s) for s in (1, 2, 3)]
    alphas, ratios = teleport_alphas(pairs, grads, 2.0, 12, 0.2)

    order = [2, 0, 1]
    permuted_alphas, permuted_ratios = teleport_alphas(
        pairs, [grads[i] for i in order], 2.0, 12, 0.2
    )
    for out_index, in_index in enumerate(order):
        assert torch.equal(permuted_alphas[out_index], alphas[in_index])
        assert permuted_ratios[out_index] == ratios[in_index]


# ------------------------------------------ composition with the orbit module


def test_teleport_move_realizes_the_predicted_nuclear_gain():
    """Close the loop: predicted transformation == realized transformation.

    The prediction uses only the base-point gradient (no forward pass).  The
    move is then applied to the WEIGHTS via ``orbit.apply_channel_scales`` and
    the gradient is recomputed by a fresh backward on the SAME batch in train
    mode; the realized nuclear norms must match the predicted ratios.
    """
    model = _make_net()
    model.train()
    pairs = find_conv_bn_pairs(model)
    batch = _batch()

    base_loss = _loss(model, batch)
    base_loss.backward()
    base_value = float(base_loss.detach())
    base_grads = _grad_mats(pairs)
    base_nuclear = [_nuclear(g) for g in base_grads]
    model.zero_grad(set_to_none=True)

    alphas, ratios = teleport_alphas(
        pairs, base_grads, spread=2.0, iters=5, step_size=0.15
    )
    # the move must have real per-channel structure, not just a uniform rescale
    for alpha in alphas:
        assert float(alpha.max() / alpha.min()) > 1.02

    apply_channel_scales(pairs, alphas)

    moved_loss = _loss(model, batch)
    moved_loss.backward()
    moved_nuclear = [_nuclear(g) for g in _grad_mats(pairs)]

    # the move stayed on the level set (BN eps only) ...
    assert abs(float(moved_loss.detach()) - base_value) / abs(base_value) < 1e-4
    # ... and every pair realized the gain predicted from the base gradient
    for realized, base, predicted in zip(moved_nuclear, base_nuclear, ratios):
        assert realized / base == pytest.approx(predicted, rel=0.02)
        assert predicted > 1.0


def test_transport_gradlike_reproduces_the_freshly_computed_gradient():
    """The other half of the contract: transporting the OLD gradient by
    ``1/alpha`` gives the gradient the new orbit point actually produces, so
    the optimizer's momentum buffer can be transported instead of recomputed."""
    model = _make_net(seed=99)
    model.train()
    pairs = find_conv_bn_pairs(model)
    batch = _batch(seed=13)

    _loss(model, batch).backward()
    predicted = [conv.weight.grad.detach().clone() for conv, _bn, _name in pairs]
    grads = [p.reshape(p.shape[0], -1) for p in predicted]
    model.zero_grad(set_to_none=True)

    alphas, _ = teleport_alphas(pairs, grads, 2.0, iters=5, step_size=0.15)
    assert all(float(a.max() / a.min()) > 1.02 for a in alphas)
    for tensor, alpha in zip(predicted, alphas):
        transport_gradlike(tensor, alpha)

    apply_channel_scales(pairs, alphas)
    _loss(model, batch).backward()

    for (conv, _bn, name), expected in zip(pairs, predicted):
        realized = conv.weight.grad.detach()
        scale = float(expected.abs().max())
        assert torch.allclose(realized, expected, atol=1e-3 * scale), name


def test_teleport_leaves_the_loss_on_the_level_set_over_repeated_moves():
    """Three teleports in a row: each is loss-invariant and each is predicted
    correctly from that step's gradient (the training-loop pattern)."""
    model = _make_net(seed=5)
    model.train()
    pairs = find_conv_bn_pairs(model)
    batch = _batch(seed=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

    for _ in range(3):
        loss_before = float(_loss(model, batch).detach())
        _loss(model, batch).backward()
        grads = _grad_mats(pairs)
        base_nuclear = [_nuclear(g) for g in grads]
        alphas, ratios = teleport_alphas(pairs, grads, 2.0, iters=5, step_size=0.15)

        apply_channel_scales(pairs, alphas)
        for (conv, _bn, _name), alpha in zip(pairs, alphas):
            transport_gradlike(conv.weight.grad, alpha)
            if conv.bias is not None and conv.bias.grad is not None:
                transport_gradlike(conv.bias.grad, alpha)
            buf = optimizer.state.get(conv.weight, {}).get("momentum_buffer")
            if buf is not None:
                transport_gradlike(buf, alpha)

        loss_after = float(_loss(model, batch).detach())
        assert abs(loss_after - loss_before) / abs(loss_before) < 1e-4

        model.zero_grad(set_to_none=True)
        _loss(model, batch).backward()
        for realized_mat, base, predicted in zip(_grad_mats(pairs), base_nuclear, ratios):
            assert _nuclear(realized_mat) / base == pytest.approx(predicted, rel=0.02)

        optimizer.step()
        model.zero_grad(set_to_none=True)

    assert math.isfinite(float(_loss(model, batch).detach()))
