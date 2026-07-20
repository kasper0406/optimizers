"""Directional-smoothness probe: exact analytic checks on a quadratic loss.

The measured quantity is a second-order REMAINDER, so the test model is chosen
so that the remainder is known in closed form: for a bias-free linear model
W (m x n) and the sum-of-squares loss

    L(W) = 0.5 * sum_i ||W x_i - y_i||^2 ,

the loss is exactly quadratic in W and

    L(W + D) - L(W) - <grad L(W), D> = 0.5 * ||D X||_F^2

with X the (n x B) batch matrix.  Hence, exactly,

    D_smooth_frobenius = ||D X||_F^2 / ||D||_F^2 ,
    D_smooth_spectral  = ||D X||_F^2 / ||D||_2^2 ,

which is what these tests assert -- no tolerance-shopping, no training run,
CPU only.  Dev seeds (>= 1000) throughout.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch

from src.instrument.schema import validate_instrumentation
from src.instrument.smoothness import SmoothnessProbe, smoothness_from_config

M_OUT, N_IN, BATCH = 6, 5, 8


def _sq_loss(outputs, targets):
    return 0.5 * ((outputs - targets) ** 2).sum()


def _model(seed=1234):
    torch.manual_seed(seed)
    return torch.nn.Linear(N_IN, M_OUT, bias=False)


def _batch(seed=1235):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH, N_IN, generator=gen)
    y = torch.randn(BATCH, M_OUT, generator=gen)
    return x, y


def _expected(D, x):
    """(remainder, d_frobenius, d_spectral) for the quadratic loss above."""
    DX = D @ x.T  # (m, B)
    remainder = 0.5 * float((DX**2).sum())
    fro = float(torch.linalg.norm(D))
    spec = float(torch.linalg.matrix_norm(D, ord=2))
    return remainder, 2 * remainder / fro**2, 2 * remainder / spec**2


def _measure(model, probe, x, y, delta, step=1, lr=0.1):
    probe.set_batch(x, y)
    out = model(x)
    _sq_loss(out, y).backward()
    probe.before_step(step)
    with torch.no_grad():
        model.weight.add_(delta)
    rec = probe.after_step(step, lr)
    model.zero_grad(set_to_none=True)
    return rec


def test_quadratic_remainder_and_both_norms_exact():
    model = _model()
    x, y = _batch()
    probe = SmoothnessProbe(model, [("weight", model.weight)], t_meas=1, loss_fn=_sq_loss)
    gen = torch.Generator().manual_seed(1001)
    delta = 0.05 * torch.randn(M_OUT, N_IN, generator=gen)

    rec = _measure(model, probe, x, y, delta, lr=0.3)["weight"]
    exp_rem, exp_fro, exp_spec = _expected(delta, x)

    assert rec["remainder"] == pytest.approx(exp_rem, rel=1e-4, abs=1e-6)
    assert rec["d_smooth_frobenius"] == pytest.approx(exp_fro, rel=1e-4)
    assert rec["d_smooth_spectral"] == pytest.approx(exp_spec, rel=1e-4)
    # The spectral denominator is never larger than the Frobenius one, so the
    # spectral smoothness is never smaller -- an invariant of the definitions.
    assert rec["d_smooth_spectral"] >= rec["d_smooth_frobenius"] - 1e-9
    assert rec["spec_norm_D"] == pytest.approx(
        float(torch.linalg.matrix_norm(delta, ord=2)), rel=1e-5
    )
    assert rec["fro_norm_D"] == pytest.approx(float(torch.linalg.norm(delta)), rel=1e-5)
    assert rec["lr_times_d_smooth_spectral"] == pytest.approx(0.3 * exp_spec, rel=1e-4)
    assert rec["lr_times_d_smooth_frobenius"] == pytest.approx(0.3 * exp_fro, rel=1e-4)


def test_delta_is_the_actual_applied_update_not_minus_lr_times_direction():
    """D is measured from the weights, so an update the optimizer modifies
    after the fact (here: a renormalization, as the vendored Muon does) is
    still captured exactly."""
    model = _model()
    x, y = _batch()
    probe = SmoothnessProbe(model, [("weight", model.weight)], t_meas=1, loss_fn=_sq_loss)
    gen = torch.Generator().manual_seed(1002)
    raw = 0.05 * torch.randn(M_OUT, N_IN, generator=gen)

    probe.set_batch(x, y)
    _sq_loss(model(x), y).backward()
    probe.before_step(1)
    before = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(raw)
        model.weight.mul_(1.03)  # post-hoc rescale, as a renormalization would
    applied = model.weight.detach() - before
    rec = probe.after_step(1, 0.1)["weight"]

    assert rec["fro_norm_D"] == pytest.approx(float(torch.linalg.norm(applied)), rel=1e-5)
    exp_rem, exp_fro, exp_spec = _expected(applied, x)
    assert rec["remainder"] == pytest.approx(exp_rem, rel=1e-4, abs=1e-6)
    assert rec["d_smooth_spectral"] == pytest.approx(exp_spec, rel=1e-4)


def test_cadence_t_meas_only_measures_due_steps():
    model = _model()
    x, y = _batch()
    probe = SmoothnessProbe(model, [("weight", model.weight)], t_meas=3, loss_fn=_sq_loss)
    gen = torch.Generator().manual_seed(1003)
    for step in range(1, 11):
        delta = 0.01 * torch.randn(M_OUT, N_IN, generator=gen)
        rec = _measure(model, probe, x, y, delta, step=step)
        assert (rec is not None) == ((step - 1) % 3 == 0), step
    assert probe.records["weight"]["step"] == [1.0, 4.0, 7.0, 10.0]
    assert probe.n_measured_steps == 4
    # Cost accounting: one fwd+bwd for L(W) plus one fwd per matrix per step.
    assert probe.n_backward == 4
    assert probe.n_forward == 8


def test_grad_source_training_matches_recompute_on_a_fp32_model():
    """On an fp32 model the training gradient and the recomputed fp32 gradient
    are the same object mathematically, so both grad_source modes must agree
    (the modes differ only on the fp16 training graph, where the shortcut's
    inconsistency is the documented cost)."""
    x, y = _batch()
    gen = torch.Generator().manual_seed(1004)
    delta = 0.05 * torch.randn(M_OUT, N_IN, generator=gen)
    results = {}
    for source in ("recompute", "training"):
        model = _model()
        probe = SmoothnessProbe(
            model,
            [("weight", model.weight)],
            t_meas=1,
            loss_fn=_sq_loss,
            grad_source=source,
        )
        results[source] = _measure(model, probe, x, y, delta)["weight"]
        if source == "training":
            assert probe.n_backward == 0  # the whole point of the shortcut
    assert results["training"]["d_smooth_spectral"] == pytest.approx(
        results["recompute"]["d_smooth_spectral"], rel=1e-5
    )


def test_probe_never_mutates_training_state():
    model = _model()
    x, y = _batch()
    probe = SmoothnessProbe(model, [("weight", model.weight)], t_meas=1, loss_fn=_sq_loss)
    gen = torch.Generator().manual_seed(1005)
    delta = 0.05 * torch.randn(M_OUT, N_IN, generator=gen)

    probe.set_batch(x, y)
    _sq_loss(model(x), y).backward()
    probe.before_step(1)
    with torch.no_grad():
        model.weight.add_(delta)
    weight_before_probe = model.weight.detach().clone()
    grad_before_probe = model.weight.grad.detach().clone()
    probe.after_step(1, 0.1)

    assert torch.equal(model.weight.detach(), weight_before_probe)
    assert torch.equal(model.weight.grad.detach(), grad_before_probe)
    assert model.weight.dtype == torch.float32


def test_batchnorm_buffers_do_not_drift_between_the_two_forwards():
    """Fresh fp32 overrides per forward: a BatchNorm in train mode must not
    let the L(W) forward's running-stat update shift the L(W+D) surface."""
    torch.manual_seed(1006)
    model = torch.nn.Sequential(torch.nn.Linear(N_IN, M_OUT, bias=False), torch.nn.BatchNorm1d(M_OUT))
    model.train()
    x, y = _batch()
    name = "0.weight"
    param = dict(model.named_parameters())[name]
    probe = SmoothnessProbe(model, [(name, param)], t_meas=1, loss_fn=_sq_loss)

    running_mean = model[1].running_mean.detach().clone()
    probe.set_batch(x, y)
    _sq_loss(model(x), y).backward()
    probe.before_step(1)
    train_running_mean = model[1].running_mean.detach().clone()
    with torch.no_grad():
        param.add_(0.01)
    probe.after_step(1, 0.1)
    # The probe's forwards ran in train mode on COPIES; the model's buffer is
    # exactly what the training forward left it at.
    assert torch.equal(model[1].running_mean, train_running_mean)
    assert not torch.equal(train_running_mean, running_mean)  # training did update it

    # With a zero update the remainder must be exactly zero; any buffer drift
    # between the two forwards would show up here as a nonzero loss difference.
    probe2 = SmoothnessProbe(model, [(name, param)], t_meas=1, loss_fn=_sq_loss)
    probe2.set_batch(x, y)
    _sq_loss(model(x), y).backward()
    probe2.before_step(2)
    rec = probe2.after_step(2, 0.1)[name]
    assert rec["loss_perturbed"] == pytest.approx(rec["loss_base"], rel=0, abs=1e-6)


def test_log_round_trips_through_the_schema_validator():
    model = _model()
    x, y = _batch()
    probe = SmoothnessProbe(model, [("weight", model.weight)], t_meas=2, loss_fn=_sq_loss)
    gen = torch.Generator().manual_seed(1007)
    for step in (1, 2, 3):
        _measure(model, probe, x, y, 0.01 * torch.randn(M_OUT, N_IN, generator=gen), step=step)
    log = probe.to_log()
    assert log["t_meas"] == 2 and log["grad_source"] == "recompute"
    assert log["loss_reduction"] == "sum" and log["batch_size"] == BATCH
    assert len(log["matrices"]["weight"]["d_smooth_spectral"]) == 2
    validate_instrumentation(
        {
            "instrumentation_schema_version": 2,
            "betas": ["0.9"],
            "hvp_enabled": False,
            "matrices": {},
            "smoothness": log,
        }
    )


def test_config_factory_defaults_off_and_validates():
    model = _model()
    named = [("weight", model.weight)]
    assert smoothness_from_config({}, model, named) is None
    assert smoothness_from_config({"smoothness": False}, model, named) is None
    assert (
        smoothness_from_config({"smoothness": {"enabled": False}}, model, named) is None
    )
    probe = smoothness_from_config(
        {"smoothness": {"enabled": True, "t_meas": 7, "grad_source": "training"}},
        model,
        named,
    )
    assert probe.t_meas == 7 and probe.grad_source == "training"
    with pytest.raises(ValueError):
        SmoothnessProbe(model, named, t_meas=0)
    with pytest.raises(ValueError):
        SmoothnessProbe(model, named, grad_source="nope")
    with pytest.raises(ValueError):
        SmoothnessProbe(model, [("not_a_param", model.weight)])
