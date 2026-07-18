"""WP1.1 end-to-end tests: planted signal/noise/oscillating directions in a
synthetic gradient stream -> projections + WP0.5 stats -> correct regime
labels; innovation resets on subspace rotation; HVP hook cadence and value.

The statistics/classification layer is the WP0.5-tested src.stats code; these
tests exercise the instrumentation plumbing end to end on top of it.

Seeds: dev seeds only (>= 1000).  Classifier thresholds below are synthetic-
scenario values (as in the WP0.5 suite), not scientific defaults.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


import numpy as np
import torch

from src.instrument.tracker import MatrixTracker
from src.stats import Regime

M_ROWS, N_COLS = 24, 16
CLASSIFIER_KWARGS = dict(tau_sig=4.0, tau_noise=2.0, rho_osc=0.5, n_min=60)


def _planted_basis(gen):
    U0, _ = torch.linalg.qr(torch.randn(M_ROWS, 3, generator=gen))
    V0, _ = torch.linalg.qr(torch.randn(N_COLS, 3, generator=gen))
    return U0, V0


def _momentum_matrix(U0, V0, sigmas=(20.0, 8.0, 4.0)):
    M = torch.zeros(M_ROWS, N_COLS)
    for i, s in enumerate(sigmas):
        M += s * torch.outer(U0[:, i], V0[:, i])
    return M


def _grad(U0, V0, t, rng):
    """Slot 0 (sigma 20): signal. Slot 1 (sigma 8): oscillation.
    Slot 2 (sigma 4): noise. Plus small isotropic background."""
    c_sig = 1.0 + 0.1 * rng.standard_normal()
    c_osc = 3.0 * ((-1.0) ** t) * (1.0 + 0.02 * rng.standard_normal())
    c_noise = 0.5 * rng.standard_normal()
    G = (
        c_sig * torch.outer(U0[:, 0], V0[:, 0])
        + c_osc * torch.outer(U0[:, 1], V0[:, 1])
        + c_noise * torch.outer(U0[:, 2], V0[:, 2])
    )
    G += 0.02 * torch.from_numpy(
        rng.standard_normal((M_ROWS, N_COLS)).astype(np.float32)
    )
    return G


def _make_tracker(gen, **overrides):
    kwargs = dict(
        k1=3,
        k2=2,
        t_refresh=50,
        subspace_iters=2,
        betas=(0.9, 0.99),
        classifier_kwargs=CLASSIFIER_KWARGS,
        align_min=0.9,
        snapshot_every=10,
        generator=gen,
    )
    kwargs.update(overrides)
    return MatrixTracker("test", (M_ROWS, N_COLS), **kwargs)


def _run(tracker, U0, V0, steps, rng, hvp_fn=None, param=None):
    M = _momentum_matrix(U0, V0)
    for t in range(steps):
        tracker.observe(_grad(U0, V0, t, rng), M, hvp_fn=hvp_fn, param=param)


def test_planted_regimes_recovered_end_to_end():
    gen = torch.Generator().manual_seed(1300)
    rng = np.random.default_rng(1300)
    U0, V0 = _planted_basis(gen)
    tracker = _make_tracker(gen)
    _run(tracker, U0, V0, steps=320, rng=rng)

    # No spurious innovation resets on a static subspace.
    for track in tracker.directions:
        assert track.reset_steps == []

    for beta in (0.9, 0.99):
        regimes = [t.classifiers[beta].regime for t in tracker.directions]
        assert regimes[0] is Regime.SIGNAL, f"beta={beta}: {regimes}"
        assert regimes[1] is Regime.OSCILLATING, f"beta={beta}: {regimes}"
        assert regimes[2] is Regime.NOISE, f"beta={beta}: {regimes}"
        # Bulk probes see only the isotropic background -> noise.
        assert regimes[3] is Regime.NOISE, f"beta={beta}: {regimes}"
        assert regimes[4] is Regime.NOISE, f"beta={beta}: {regimes}"

    # Oscillating direction: implied eta*lambda = 1 + amplitude ratio ~ 2.
    stats_osc = tracker.directions[1].classifiers[0.9].stats
    assert abs(float(stats_osc.implied_eta_lambda) - 2.0) < 0.1

    # Per-matrix per-step series populated with the right lengths.
    assert len(tracker.steps) == 320
    assert len(tracker.grad_fro_norm) == 320
    assert len(tracker.top_sigma_m) == 320
    # Top sigma estimate tracks the planted sigma_1 = 20 of M.
    assert abs(tracker.top_sigma_m[-1] - 20.0) < 0.5
    # Refreshes every t_refresh steps starting at step 1.
    assert tracker.refresh_steps == [1, 51, 101, 151, 201, 251, 301]


def test_subspace_rotation_triggers_reset_and_reclassification():
    gen = torch.Generator().manual_seed(1301)
    rng = np.random.default_rng(1301)
    U0, V0 = _planted_basis(gen)
    tracker = _make_tracker(gen)
    _run(tracker, U0, V0, steps=150, rng=rng)

    # Rotate slot 1: replace its pair with fresh directions orthogonal to
    # everything planted so far; keep sigma so the slot ordering is stable.
    def _fresh(dim, block):
        w = torch.randn(dim, generator=gen)
        w -= block @ (block.T @ w)
        return w / w.norm()

    u_new = _fresh(M_ROWS, U0)
    v_new = _fresh(N_COLS, V0)
    U1, V1 = U0.clone(), V0.clone()
    U1[:, 1], V1[:, 1] = u_new, v_new

    M2 = _momentum_matrix(U1, V1)
    for t in range(150, 320):
        tracker.observe(_grad(U1, V1, t, rng), M2)

    rotated = tracker.directions[1]
    stable = tracker.directions[0]
    assert rotated.reset_steps, "rotated direction must reset on refresh"
    assert 150 < rotated.reset_steps[0] <= 201  # first refresh after rotation
    assert stable.reset_steps == []
    # After the reset the direction re-earns its oscillating label.
    for beta in (0.9, 0.99):
        assert rotated.classifiers[beta].regime is Regime.OSCILLATING
    # The reset actually cleared history: fewer observations than the run.
    assert rotated.classifiers[0.9].stats.n_obs < 320 - 140


def test_hvp_called_once_per_pair_per_refresh():
    gen = torch.Generator().manual_seed(1302)
    rng = np.random.default_rng(1302)
    U0, V0 = _planted_basis(gen)
    tracker = _make_tracker(gen)
    param = torch.zeros(M_ROWS, N_COLS)
    calls = []

    def hvp_fn(p, D):
        assert p.shape == param.shape
        assert abs(float(D.norm()) - 1.0) < 1e-4  # D = u v^T has unit Frobenius norm
        calls.append(1)
        return 7.5

    _run(tracker, U0, V0, steps=120, rng=rng, hvp_fn=hvp_fn, param=param)
    # Refreshes at steps 1, 51, 101 -> 3 refreshes x (k1 + k2) = 5 pairs.
    assert len(calls) == 3 * 5
    for track in tracker.directions:
        assert len(track.lambda_hvp) == 3
        assert all(v == 7.5 for _, v in track.lambda_hvp)


def test_hvp_correctness_through_autograd_on_quadratic():
    """lambda = vec(uv^T)^T H vec(uv^T) via double backward matches the
    analytic value for a diagonal quadratic L = 0.5 sum h_ij W_ij^2."""
    gen = torch.Generator().manual_seed(1303)
    h = torch.rand(6, 5, generator=gen) + 0.5
    W = torch.randn(6, 5, generator=gen, requires_grad=True)

    def hvp_fn(param, D):
        with torch.enable_grad():
            loss = 0.5 * (h * param**2).sum()
            (g,) = torch.autograd.grad(loss, param, create_graph=True)
            (hv,) = torch.autograd.grad((g * D).sum(), param)
        return float((hv * D).sum())

    u = torch.randn(6, generator=gen)
    u /= u.norm()
    v = torch.randn(5, generator=gen)
    v /= v.norm()
    D = torch.outer(u, v)
    analytic = float((h * D**2).sum())
    assert abs(hvp_fn(W, D) - analytic) < 1e-5


def test_small_matrix_shrinks_tracked_blocks():
    gen = torch.Generator().manual_seed(1304)
    tracker = MatrixTracker(
        "tiny",
        (10, 6),
        k1=16,
        k2=16,
        t_refresh=10,
        betas=(0.9,),
        classifier_kwargs=CLASSIFIER_KWARGS,
        generator=gen,
    )
    assert tracker.subspace.k1 + tracker.subspace.k2 <= 6
    assert tracker.subspace.k1 >= 1
