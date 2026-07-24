"""Program #21 Stage-1 harness tests (prereg reports/central-flow-prereg.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.theory.muon_flow_sim import (
    MatrixQuadratic, integrate_naive_flow, msign_exact, simulate_muon,
    trajectory_error,
)


def test_msign_exact_is_orthogonal_factor():
    g = torch.Generator().manual_seed(0)
    M = torch.randn(8, 5, generator=g)
    S = msign_exact(M)
    assert torch.allclose(S.T @ S, torch.eye(5), atol=1e-5)
    # polar factor maximizes <M, T> over spectral-ball T: check vs random orthogonals
    for _ in range(5):
        Q, _ = torch.linalg.qr(torch.randn(8, 5, generator=g))
        assert (M * S).sum() >= (M * Q).sum() - 1e-4


def test_quadratic_spectrum_and_descent():
    prob = MatrixQuadratic.make(cond_a=10, cond_b=100, seed=1)
    assert prob.lam_max == pytest.approx(1.0, abs=1e-5)  # both spectra top out at 1
    # fixed-norm updates move eta*sqrt(rank)/step; the ~55-unit start distance
    # needs ~1600 steps at eta=0.01 -- 3000 reaches the oscillation floor
    sim = simulate_muon(prob, eta=0.01, sigma=0.0, steps=3000, seed=1)
    assert sim["losses"][-1] < min(1.0, sim["losses"][0] * 0.05)
    assert torch.isfinite(sim["traj_ema"]).all()


def test_muon_oscillation_bounded_far_beyond_gd_eos():
    """The lab's headline stability fact reproduces on the synthetic family:
    at eta*lam ~ 40 (GD diverges at 2/lam) Muon stays bounded with step norm
    saturating at ~eta*sqrt(rank)."""
    prob = MatrixQuadratic.make(cond_a=10, cond_b=100, seed=2)
    eta = 40.0 / prob.lam_max
    sim = simulate_muon(prob, eta=eta, sigma=0.0, steps=1500, seed=2)
    late = sim["step_norms"][-300:]
    assert torch.isfinite(sim["losses"]).all()
    expected = eta * (48 ** 0.5)  # msign of a full-rank matrix has ||.||_F = sqrt(rank)
    assert late.mean() == pytest.approx(expected, rel=0.15)


def test_naive_flow_matches_subeos_momentumless_dynamics():
    """Machinery consistency check: with beta=0 (no momentum transient) and
    far sub-EoS eta, the naive flow must track the discrete dynamics within
    the registered 10%. (With beta=0.95 the momentum warmup alone breaks
    this -- which is part of what the derived flow must model.)"""
    prob = MatrixQuadratic.make(seed=3)
    eta = 0.02 / prob.lam_max  # far sub-EoS
    # compare the TRANSIT phase only: at step norm eta*sqrt(48)=0.14 the
    # ~55-unit start distance takes ~400 steps; 300 steps stays outside the
    # near-optimum oscillation band (where flow-vs-discrete divergence is the
    # EoS physics the derived flow exists to capture, not a machinery bug)
    sim = simulate_muon(prob, eta=eta, sigma=0.0, steps=300, seed=3, beta=0.0,
                        ema_half_life=1e-6)  # no averaging needed sub-EoS
    g = torch.Generator().manual_seed(3)
    start = torch.randn(prob.Wstar.shape, generator=g)
    flow = integrate_naive_flow(prob, eta=eta, steps=300, W0=start)
    err = trajectory_error(sim["traj_ema"], flow)
    assert err < 0.10, f"naive flow should match momentumless sub-EoS, err={err:.3f}"
