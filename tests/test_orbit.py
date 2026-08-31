"""Tests for the teleportation go/no-go gate (`airbench_teleport_gate`,
docs/litreview/j-theory-theorem-sweep.md §6 item 4 / T6).

Three surfaces, all CPU-only:

  * ``src/instrument/orbit.py`` — conv->BatchNorm pair discovery, the
    loss-invariance of a channel rescaling in TRAIN mode, the 1/alpha_c
    gradient transformation it induces, the Frobenius/nuclear norm readout on a
    hand-checkable matrix, log-uniform sampling bounds/determinism, and the
    constrained random-search refiner;
  * ``src.optim.airbench_zoo._resolve_gate`` / ``run_airbench_teleport_gate``
    config validation (every refusal fires before any CUDA work);
  * ``configs/wpj_teleport_gate.yaml``.

The CUDA training path itself is exercised on the cloud box.
"""

import math
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from src.instrument.orbit import (
    apply_channel_scales,
    ascend_scales,
    find_conv_bn_pairs,
    grad_norms,
    invert_channel_scales,
    pair_names,
    refine_scales,
    sample_log_uniform_scales,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ fixtures


def _make_net(seed: int = 1234) -> nn.Sequential:
    """Conv->BN->ReLU->Conv->BN->Flatten->Linear, float32, CPU."""
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


# ------------------------------------------------------- find_conv_bn_pairs


def test_finds_exactly_the_two_conv_bn_pairs():
    model = _make_net()
    pairs = find_conv_bn_pairs(model)
    assert len(pairs) == 2
    assert pair_names(pairs) == ["0", "3"]
    assert [conv for conv, _bn, _n in pairs] == [model[0], model[3]]
    assert [bn for _c, bn, _n in pairs] == [model[1], model[4]]


def test_conv_without_a_following_batchnorm_is_excluded():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),  # breaks the symmetry: no pair
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 4, 3, padding=1),
        nn.BatchNorm2d(4),
    )
    assert pair_names(find_conv_bn_pairs(model)) == ["3"]
    # a trailing conv with nothing after it is likewise excluded
    assert find_conv_bn_pairs(nn.Sequential(nn.Conv2d(3, 8, 3))) == []
    # ... as is a BatchNorm whose channel count does not match the conv
    mismatched = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(4))
    assert find_conv_bn_pairs(mismatched) == []


def test_pooling_between_conv_and_batchnorm_still_pairs():
    """MaxPool commutes with a positive channel scale (the CifarNet pattern:
    ConvGroup runs conv1 -> pool -> norm1)."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(8),
        nn.GELU(),
    )
    assert pair_names(find_conv_bn_pairs(model)) == ["0"]


def test_pairs_are_found_in_nested_containers_with_qualified_names():
    class Group(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.norm1 = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
            self.norm2 = nn.BatchNorm2d(8)
            self.activ = nn.GELU()

    model = nn.Sequential(nn.GELU(), Group(), Group())
    assert pair_names(find_conv_bn_pairs(model)) == [
        "1.conv1",
        "1.conv2",
        "2.conv1",
        "2.conv2",
    ]


# ------------------------------------------------------------- invariance


def test_train_mode_loss_is_invariant_along_the_orbit():
    model = _make_net()
    model.train()
    pairs = find_conv_bn_pairs(model)
    batch = _batch()

    before = float(_loss(model, batch).item())
    gen = torch.Generator().manual_seed(11)
    scales = sample_log_uniform_scales(pairs, 2.0, gen)
    apply_channel_scales(pairs, scales)
    after = float(_loss(model, batch).item())

    assert abs(after - before) / abs(before) < 1e-4  # BN eps effect only


def test_invert_channel_scales_round_trips_the_weights():
    model = _make_net()
    pairs = find_conv_bn_pairs(model)
    original = [pair[0].weight.detach().clone() for pair in pairs]
    gen = torch.Generator().manual_seed(3)
    scales = sample_log_uniform_scales(pairs, 2.0, gen)
    apply_channel_scales(pairs, scales)
    assert not torch.allclose(pairs[0][0].weight, original[0])
    invert_channel_scales(pairs, scales)
    for (conv, _bn, _n), w0 in zip(pairs, original):
        assert torch.allclose(conv.weight, w0, atol=1e-6)


def test_apply_channel_scales_rejects_bad_scale_vectors():
    model = _make_net()
    pairs = find_conv_bn_pairs(model)
    with pytest.raises(ValueError):  # wrong number of vectors
        apply_channel_scales(pairs, [torch.ones(8)])
    with pytest.raises(ValueError):  # wrong length
        apply_channel_scales(pairs, [torch.ones(7), torch.ones(4)])
    with pytest.raises(ValueError):  # non-positive scale
        apply_channel_scales(pairs, [torch.ones(8), torch.zeros(4)])
    with pytest.raises(ValueError):  # not 1-D
        apply_channel_scales(pairs, [torch.ones(8, 1), torch.ones(4)])


def test_conv_bias_is_scaled_with_its_channel():
    conv = nn.Conv2d(3, 2, 3, padding=1, bias=True)
    with torch.no_grad():
        conv.bias.copy_(torch.tensor([1.0, 2.0]))
    model = nn.Sequential(conv, nn.BatchNorm2d(2))
    pairs = find_conv_bn_pairs(model)
    apply_channel_scales(pairs, [torch.tensor([3.0, 0.5])])
    assert conv.bias.tolist() == pytest.approx([3.0, 1.0])


# ------------------------------------------------- gradient transformation


def test_channel_gradient_scales_as_one_over_alpha():
    model = _make_net()
    model.train()
    pairs = find_conv_bn_pairs(model)
    conv = pairs[0][0]
    batch = _batch()

    _loss(model, batch).backward()
    base_grad = conv.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    # scale ONLY channel 2 of the first pair, by alpha = 2
    alpha = torch.ones(conv.weight.shape[0])
    alpha[2] = 2.0
    apply_channel_scales(pairs, [alpha, torch.ones(pairs[1][0].weight.shape[0])])

    _loss(model, batch).backward()  # batch stats recomputed in train mode
    new_grad = conv.weight.grad.detach().clone()

    scale = base_grad.abs().max()
    # the rescaled channel's gradient row halves ...
    assert torch.allclose(new_grad[2], 0.5 * base_grad[2], atol=1e-3 * float(scale))
    # ... and every other row is unchanged
    others = [c for c in range(conv.weight.shape[0]) if c != 2]
    assert torch.allclose(
        new_grad[others], base_grad[others], atol=1e-3 * float(scale)
    )


def test_orbit_move_changes_gradient_size_at_fixed_loss():
    """The gate's premise, in miniature: same loss, different ||grad||."""
    model = _make_net()
    model.train()
    pairs = find_conv_bn_pairs(model)
    batch = _batch()

    loss_before = float(_loss(model, batch).item())
    model.zero_grad(set_to_none=True)
    _loss(model, batch).backward()
    base = grad_norms(model, pairs)
    model.zero_grad(set_to_none=True)

    apply_channel_scales(
        pairs,
        [
            torch.full((pairs[0][0].weight.shape[0],), 2.0),
            torch.full((pairs[1][0].weight.shape[0],), 2.0),
        ],
    )
    loss_after = float(_loss(model, batch).item())
    model.zero_grad(set_to_none=True)
    _loss(model, batch).backward()
    moved = grad_norms(model, pairs)

    assert abs(loss_after - loss_before) / abs(loss_before) < 1e-4
    # every conv-weight gradient row halved -> both norms halve
    for i in range(len(pairs)):
        assert moved["fro"][i] == pytest.approx(0.5 * base["fro"][i], rel=1e-2)
        assert moved["nuclear"][i] == pytest.approx(0.5 * base["nuclear"][i], rel=1e-2)


# ------------------------------------------------------------- grad_norms


def test_grad_norms_on_a_hand_checkable_2x2_case():
    """A conv with a [2, 1, 1, 2] weight reshapes to a 2x2 gradient matrix."""
    conv = nn.Conv2d(1, 2, kernel_size=(1, 2), bias=False)
    model = nn.Sequential(conv, nn.BatchNorm2d(2))
    pairs = find_conv_bn_pairs(model)
    assert len(pairs) == 1

    # G = [[3, 0], [0, -4]]: singular values 4 and 3 -> nuclear 7, Frobenius 5
    conv.weight.grad = torch.tensor([3.0, 0.0, 0.0, -4.0]).reshape(2, 1, 1, 2)
    norms = grad_norms(model, pairs)
    assert norms["fro"] == pytest.approx([5.0])
    assert norms["nuclear"] == pytest.approx([7.0])
    assert norms["fro_sum"] == pytest.approx(5.0)
    assert norms["nuclear_sum"] == pytest.approx(7.0)
    # only this parameter carries a gradient, so the model total is its own norm
    assert norms["total"] == pytest.approx(5.0)

    # G = [[1, 1], [1, -1]]: both singular values sqrt(2) -> nuclear 2*sqrt(2)
    conv.weight.grad = torch.tensor([1.0, 1.0, 1.0, -1.0]).reshape(2, 1, 1, 2)
    norms = grad_norms(model, pairs)
    assert norms["fro"] == pytest.approx([2.0])
    assert norms["nuclear"] == pytest.approx([2.0 * math.sqrt(2.0)])


def test_grad_norms_total_covers_all_parameters_with_grads():
    conv = nn.Conv2d(1, 2, kernel_size=(1, 2), bias=False)
    head = nn.Linear(2, 1, bias=False)
    model = nn.Sequential(conv, nn.BatchNorm2d(2), nn.Flatten(), head)
    pairs = find_conv_bn_pairs(model)
    conv.weight.grad = torch.tensor([3.0, 0.0, 0.0, -4.0]).reshape(2, 1, 1, 2)
    head.weight.grad = torch.tensor([[0.0, 12.0]])
    # sqrt(25 + 144) = 13, while the pair-level norms only see the conv
    assert grad_norms(model, pairs)["total"] == pytest.approx(13.0)
    assert grad_norms(model, pairs)["fro"] == pytest.approx([5.0])


def test_grad_norms_reports_zero_for_a_pair_without_a_gradient():
    model = _make_net()
    pairs = find_conv_bn_pairs(model)
    norms = grad_norms(model, pairs)
    assert norms["fro"] == [0.0, 0.0]
    assert norms["nuclear"] == [0.0, 0.0]
    assert norms["total"] == 0.0


def test_grad_norms_casts_half_precision_grads_to_float32():
    conv = nn.Conv2d(1, 2, kernel_size=(1, 2), bias=False).half()
    model = nn.Sequential(conv, nn.BatchNorm2d(2))
    pairs = find_conv_bn_pairs(model)
    conv.weight.grad = torch.tensor([3.0, 0.0, 0.0, -4.0]).reshape(2, 1, 1, 2).half()
    norms = grad_norms(model, pairs)
    assert norms["fro"] == pytest.approx([5.0], rel=1e-3)
    assert norms["nuclear"] == pytest.approx([7.0], rel=1e-3)


# ------------------------------------------------ sample_log_uniform_scales


def test_sampled_scales_respect_the_spread_bounds_and_shapes():
    model = _make_net()
    pairs = find_conv_bn_pairs(model)
    gen = torch.Generator().manual_seed(0)
    spread = 2.0
    for _ in range(20):
        scales = sample_log_uniform_scales(pairs, spread, gen)
        assert [s.shape[0] for s in scales] == [8, 4]
        for s in scales:
            assert s.dtype is torch.float32
            assert bool((s >= 1.0 / spread - 1e-6).all())
            assert bool((s <= spread + 1e-6).all())


def test_sampled_scales_are_deterministic_given_the_generator():
    model = _make_net()
    pairs = find_conv_bn_pairs(model)
    a = sample_log_uniform_scales(pairs, 2.0, torch.Generator().manual_seed(5))
    b = sample_log_uniform_scales(pairs, 2.0, torch.Generator().manual_seed(5))
    c = sample_log_uniform_scales(pairs, 2.0, torch.Generator().manual_seed(6))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))


def test_sample_log_uniform_scales_rejects_degenerate_spread():
    pairs = find_conv_bn_pairs(_make_net())
    with pytest.raises(ValueError):
        sample_log_uniform_scales(pairs, 1.0, torch.Generator().manual_seed(0))


# --------------------------------------------------------------- refinement


def _quadratic_objective(target: float = 0.4):
    """Toy surrogate: maximized where every log-scale equals ``target``."""

    def objective(scales):
        return -float(
            sum(((s.log() - target) ** 2).sum().item() for s in scales)
        )

    return objective


def test_refiner_improves_or_equals_its_starting_objective():
    pairs = find_conv_bn_pairs(_make_net())
    objective = _quadratic_objective()
    init = [torch.ones(int(c.weight.shape[0])) for c, _b, _n in pairs]
    start = objective(init)

    out = refine_scales(
        objective,
        init,
        iters=200,
        step_size=0.2,
        spread_limit=2.0,
        generator=torch.Generator().manual_seed(1),
    )
    assert out["value"] >= start
    assert out["value"] > start  # the surrogate is easy; the search must move
    assert out["n_accepted"] >= 1
    assert out["n_infeasible"] == 0
    assert objective(out["scales"]) == pytest.approx(out["value"], rel=1e-6)
    assert [s.shape[0] for s in out["scales"]] == [8, 4]


def test_refiner_with_zero_iterations_returns_its_start():
    pairs = find_conv_bn_pairs(_make_net())
    objective = _quadratic_objective()
    init = [torch.full((int(c.weight.shape[0]),), 1.5) for c, _b, _n in pairs]
    out = refine_scales(
        objective, init, iters=0, step_size=0.2, spread_limit=2.0, generator=None
    )
    assert out["value"] == pytest.approx(objective(init))
    assert out["n_accepted"] == 0


def test_refiner_is_deterministic_and_stays_inside_the_spread_box():
    pairs = find_conv_bn_pairs(_make_net())
    objective = _quadratic_objective(target=5.0)  # pulls hard past the clamp
    init = [torch.ones(int(c.weight.shape[0])) for c, _b, _n in pairs]
    runs = [
        refine_scales(
            objective,
            init,
            iters=50,
            step_size=0.5,
            spread_limit=2.0,
            generator=torch.Generator().manual_seed(9),
        )
        for _ in range(2)
    ]
    assert runs[0]["value"] == runs[1]["value"]
    assert all(
        torch.equal(a, b) for a, b in zip(runs[0]["scales"], runs[1]["scales"])
    )
    for s in runs[0]["scales"]:
        assert bool((s <= 2.0 + 1e-5).all()) and bool((s >= 0.5 - 1e-5).all())


def test_refiner_discards_infeasible_proposals():
    pairs = find_conv_bn_pairs(_make_net())
    calls = {"n": 0}

    def objective(scales):
        calls["n"] += 1
        if calls["n"] == 1:
            return 0.0  # the start is feasible
        return None  # every proposal is off the level set

    init = [torch.ones(int(c.weight.shape[0])) for c, _b, _n in pairs]
    out = refine_scales(
        objective,
        init,
        iters=7,
        step_size=0.3,
        spread_limit=2.0,
        generator=torch.Generator().manual_seed(2),
    )
    assert out["value"] == 0.0
    assert out["n_accepted"] == 0
    assert out["n_infeasible"] == 7
    assert all(torch.equal(s, torch.ones_like(s)) for s in out["scales"])


def test_refiner_reports_minus_inf_when_nothing_is_feasible():
    pairs = find_conv_bn_pairs(_make_net())
    init = [torch.ones(int(c.weight.shape[0])) for c, _b, _n in pairs]
    out = refine_scales(
        objective=lambda scales: None,
        init_scales=init,
        iters=3,
        step_size=0.3,
        spread_limit=2.0,
        generator=torch.Generator().manual_seed(2),
    )
    assert out["value"] == float("-inf")
    assert out["n_infeasible"] == 3


def test_refiner_rejects_bad_arguments():
    pairs = find_conv_bn_pairs(_make_net())
    init = [torch.ones(int(c.weight.shape[0])) for c, _b, _n in pairs]
    objective = _quadratic_objective()
    with pytest.raises(ValueError):
        refine_scales(objective, init, iters=-1, step_size=0.1, spread_limit=2.0)
    with pytest.raises(ValueError):
        refine_scales(objective, init, iters=1, step_size=0.1, spread_limit=1.0)


def test_ascend_scales_starts_from_unit_scales():
    pairs = find_conv_bn_pairs(_make_net())
    seen = []

    def objective(scales):
        seen.append([s.clone() for s in scales])
        return -float(sum(((s.log()) ** 2).sum().item() for s in scales))

    out = ascend_scales(
        objective,
        pairs,
        steps=5,
        lr=0.2,
        spread_limit=2.0,
        generator=torch.Generator().manual_seed(4),
    )
    assert all(torch.allclose(s, torch.ones_like(s)) for s in seen[0])
    assert out["n_iters"] == 5
    assert out["value"] == pytest.approx(0.0, abs=1e-6)  # alpha = 1 is optimal


# --------------------------------------------- probe loop composition (CPU)


def test_probe_loop_composes_and_leaves_the_state_bit_exact():
    """The gate's inner loop, in miniature: snapshot -> base -> orbit draws ->
    constrained refinement -> restore, on the small CPU net.

    Mirrors ``run_airbench_teleport_gate``'s ``teleport_probe`` composition
    (the airbench-specific wiring itself only runs on the cloud box) and pins
    the two properties the experiment depends on: repeated in-place orbit moves
    are fully undone by ``train_snapshot``, and the constrained search only
    ever reports feasible points.
    """
    from src.optim.train_snapshot import (
        restore_training_state,
        snapshot_equal,
        snapshot_training_state,
    )

    tol = 1e-3
    model = _make_net()
    model.train()
    pairs = find_conv_bn_pairs(model)
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)]
    batch = _batch()

    # a couple of real steps so the optimizer carries momentum state
    for _ in range(2):
        _loss(model, batch).backward()
        optimizers[0].step()
        model.zero_grad(set_to_none=True)

    snap = snapshot_training_state(model, optimizers)

    def measure():
        loss = _loss(model, batch)
        loss.backward()
        norms = grad_norms(model, pairs)
        value = float(loss.item())
        model.zero_grad(set_to_none=True)
        return value, norms

    base_loss, base = measure()
    restore_training_state(model, optimizers, snap)

    def evaluate(scales):
        apply_channel_scales(pairs, scales)
        value, norms = measure()
        restore_training_state(model, optimizers, snap)
        return {
            "rel_dloss": (value - base_loss) / abs(base_loss),
            "total_grad_ratio": norms["total"] / base["total"],
            "nuclear_sum_ratio": norms["nuclear_sum"] / base["nuclear_sum"],
        }

    gen = torch.Generator().manual_seed(17)
    draws = [
        (s, evaluate(s))
        for s in (sample_log_uniform_scales(pairs, 2.0, gen) for _ in range(12))
    ]
    assert snapshot_equal(model, optimizers, snap)  # every draw was undone

    # loss invariance held for every draw; gradient size did not
    assert max(abs(r["rel_dloss"]) for _s, r in draws) < tol
    ratios = [r["nuclear_sum_ratio"] for _s, r in draws]
    assert max(ratios) - min(ratios) > 0.05

    feasible = [(s, r) for s, r in draws if abs(r["rel_dloss"]) < tol]
    init, best_random = max(feasible, key=lambda sr: sr[1]["nuclear_sum_ratio"])

    best = {"value": float("-inf"), "record": None}

    def objective(scales):
        record = evaluate(scales)
        if abs(record["rel_dloss"]) >= tol:
            return None
        value = record["nuclear_sum_ratio"]
        if value > best["value"]:
            best.update(value=value, record=record)
        return value

    search = refine_scales(
        objective, init, iters=25, step_size=0.25, spread_limit=2.0, generator=gen
    )
    restore_training_state(model, optimizers, snap)
    assert snapshot_equal(model, optimizers, snap)

    # the refinement starts at the best random draw, so it can only improve
    assert search["value"] >= best_random["nuclear_sum_ratio"]
    assert best["record"]["nuclear_sum_ratio"] == pytest.approx(search["value"])
    assert abs(best["record"]["rel_dloss"]) < tol


# ------------------------------------------------------------- _resolve_gate


def _gate_config(**over):
    gate = {
        "snapshot_steps": [50, 100, 150],
        "n_samples": 64,
        "spread": 2.0,
        "refine_iters": 64,
        "probe_batches": 2,
    }
    gate.update(over)
    return {"gate": gate}


def test_resolve_gate_accepts_the_config_shape():
    from src.optim.airbench_zoo import _resolve_gate

    gate = _resolve_gate(_gate_config())
    assert gate == {
        "snapshot_steps": [50, 100, 150],
        "n_samples": 64,
        "spread": 2.0,
        "refine_iters": 64,
        "probe_batches": 2,
    }
    assert isinstance(gate["spread"], float)
    # an integer spread is accepted and normalized to float
    assert _resolve_gate(_gate_config(spread=3))["spread"] == 3.0
    # refine_iters may be zero (random draws only)
    assert _resolve_gate(_gate_config(refine_iters=0))["refine_iters"] == 0
    # the bound check passes when the last snapshot is inside the run
    assert _resolve_gate(_gate_config(), 150)["snapshot_steps"] == [50, 100, 150]


@pytest.mark.parametrize(
    "config",
    [
        {},  # missing block entirely
        {"gate": None},
        {"gate": [50, 100]},  # not a mapping
        {"gate": {"snapshot_steps": [50]}},  # missing keys
        _gate_config(extra=1),  # extra key
        _gate_config(snapshot_steps=[]),  # empty
        _gate_config(snapshot_steps=[100, 50]),  # unsorted
        _gate_config(snapshot_steps=[50, 50]),  # duplicates
        _gate_config(snapshot_steps=[0, 50]),  # < 1
        _gate_config(snapshot_steps=[-5]),
        _gate_config(snapshot_steps=[50.0]),  # non-int
        _gate_config(snapshot_steps=[True]),  # booleans are not step counts
        _gate_config(snapshot_steps=50),  # not a list
        _gate_config(n_samples=0),
        _gate_config(n_samples=1.5),
        _gate_config(n_samples=True),
        _gate_config(refine_iters=-1),
        _gate_config(probe_batches=0),
        _gate_config(probe_batches="2"),
        _gate_config(spread=1.0),  # a degenerate orbit box
        _gate_config(spread=0.5),
        _gate_config(spread="2.0"),
        _gate_config(spread=True),
    ],
)
def test_resolve_gate_rejects_invalid(config):
    from src.optim.airbench_zoo import _resolve_gate

    with pytest.raises(SystemExit):
        _resolve_gate(config)


def test_resolve_gate_rejects_snapshot_beyond_the_run():
    from src.optim.airbench_zoo import _resolve_gate

    config = _gate_config(snapshot_steps=[50, 300])
    _resolve_gate(config)  # fine without the bound
    with pytest.raises(SystemExit, match="beyond the run"):
        _resolve_gate(config, 200)


# ------------------------------------------ run_airbench_teleport_gate guards


def test_teleport_gate_refuses_optimizer_override():
    from src.optim.airbench_zoo import run_airbench_teleport_gate

    config = dict(_gate_config(), optimizer={"name": "muon"})
    with pytest.raises(SystemExit, match="does not accept an optimizer"):
        run_airbench_teleport_gate(config, torch.device("cpu"))


def test_teleport_gate_requires_gate_block():
    from src.optim.airbench_zoo import run_airbench_teleport_gate

    with pytest.raises(SystemExit, match="needs a gate: block"):
        run_airbench_teleport_gate({"recipe": {}}, torch.device("cpu"))


def test_teleport_gate_refuses_non_vendored_sampling():
    from src.optim.airbench_zoo import run_airbench_teleport_gate

    config = dict(_gate_config(), recipe={"sampling": "with_replacement"})
    with pytest.raises(SystemExit, match="vendored sampling only"):
        run_airbench_teleport_gate(config, torch.device("cpu"))


def test_teleport_gate_refuses_cpu_device():
    from src.optim.airbench_zoo import run_airbench_teleport_gate

    with pytest.raises(SystemExit, match="requires a CUDA device"):
        run_airbench_teleport_gate(_gate_config(), torch.device("cpu"))


def test_teleport_gate_is_registered_in_the_runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rm_run_for_teleport_gate", REPO_ROOT / "scripts" / "run.py"
    )
    run_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_module)
    from src.optim.airbench_zoo import run_airbench_teleport_gate

    assert (
        run_module.EXPERIMENT_REGISTRY["airbench_teleport_gate"]
        is run_airbench_teleport_gate
    )


# ------------------------------------------------------------------- config


def test_teleport_gate_config_is_a_dev_seed_measurement_run():
    from src.optim.airbench_zoo import _resolve_gate

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "wpj_teleport_gate.yaml").read_text()
    )
    assert config["experiment"] == "airbench_teleport_gate"
    assert config["sweep"]["seeds"]["policy"] == "dev"
    assert config["seed"] >= 1000
    assert "optimizer" not in config
    assert config["recipe"] == {"compile": True, "tta_level": 0}
    assert config["train"] == {"epochs": 8, "batch_size": 2000}

    gate = _resolve_gate(config)
    assert gate == {
        "snapshot_steps": [50, 100, 150],
        "n_samples": 64,
        "spread": 2.0,
        "refine_iters": 64,
        "probe_batches": 2,
    }
    # 8 epochs x 25 batches = 200 steps; every snapshot lies inside the run
    _resolve_gate(config, 200)
