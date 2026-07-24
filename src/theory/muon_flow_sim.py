"""Program #21 Stage-1 simulation harness (prereg reports/central-flow-prereg.md).

Direct simulation of Muon dynamics on matrix quadratics
L(W) = 1/2 tr((W-W*)^T A (W-W*) B), gradient A(W-W*)B, with i.i.d. Gaussian
gradient noise — the registered synthetic family any candidate central flow
must match. Provides:

- ``MatrixQuadratic``: controlled-spectrum problem generator.
- ``simulate_muon``: the exact discrete dynamics (momentum beta, msign via
  exact SVD or the record's 5-step Newton-Schulz), returning the trajectory
  and its EMA time-average.
- ``integrate_naive_flow``: dW/dt = -eta*msign(grad(W)) reference flow (the
  bar the derived flow must beat where this one fails).
- ``trajectory_error``: the registered comparison metric (time-averaged
  Frobenius error / trajectory norm over the first T steps).

Everything is torch, CPU-fast at the registered 64x48 size, seeded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
from torch import Tensor


def _spd_with_spectrum(n: int, cond: float, g: torch.Generator) -> Tensor:
    """SPD matrix with log-uniform spectrum in [1/cond, 1] and Haar basis."""
    lam = torch.logspace(-math.log10(cond), 0, n)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    return (Q * lam) @ Q.T


@dataclass
class MatrixQuadratic:
    A: Tensor  # (n, n) SPD
    B: Tensor  # (m, m) SPD
    Wstar: Tensor  # (n, m)

    @classmethod
    def make(cls, n=64, m=48, cond_a=10.0, cond_b=100.0, scale=1.0, seed=0) -> "MatrixQuadratic":
        g = torch.Generator().manual_seed(seed)
        return cls(A=scale * _spd_with_spectrum(n, cond_a, g),
                   B=_spd_with_spectrum(m, cond_b, g),
                   Wstar=torch.randn(n, m, generator=g))

    def grad(self, W: Tensor) -> Tensor:
        return self.A @ (W - self.Wstar) @ self.B

    def loss(self, W: Tensor) -> Tensor:
        D = W - self.Wstar
        return 0.5 * torch.trace(D.T @ self.A @ D @ self.B)

    @property
    def lam_max(self) -> float:
        """Largest curvature eigenvalue lambda_i(A)*mu_j(B)."""
        return float(torch.linalg.eigvalsh(self.A)[-1] * torch.linalg.eigvalsh(self.B)[-1])

    def curvature_of_pair(self, i: int, j: int) -> float:
        return float(torch.linalg.eigvalsh(self.A)[-1 - i] * torch.linalg.eigvalsh(self.B)[-1 - j])


def msign_exact(M: Tensor) -> Tensor:
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


def msign_ns5(M: Tensor) -> Tensor:
    from src.nanogpt.optim import zeropower_via_newtonschulz5

    return zeropower_via_newtonschulz5(M.float(), 5).to(M.dtype)


def simulate_muon(
    prob: MatrixQuadratic,
    eta: float,
    sigma: float,
    steps: int,
    beta: float = 0.95,
    variant: Literal["svd", "ns5"] = "svd",
    seed: int = 0,
    ema_half_life: float = 20.0,
    W0: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Discrete Muon dynamics; returns per-step losses, EMA-averaged
    trajectory samples, and oscillation statistics."""
    g = torch.Generator().manual_seed(seed)
    W = prob.Wstar.clone() + (
        W0 if W0 is not None else torch.randn(prob.Wstar.shape, generator=g)
    )
    M = torch.zeros_like(W)
    sign = msign_exact if variant == "svd" else msign_ns5
    ema = W.clone()
    rho = 0.5 ** (1.0 / ema_half_life)
    traj_ema, losses, amp = [], [], []
    prev = W.clone()
    for t in range(steps):
        gnoise = sigma * torch.randn(W.shape, generator=g)
        M = beta * M + prob.grad(W) + gnoise
        W = W - eta * sign(M)
        ema = rho * ema + (1 - rho) * W
        traj_ema.append(ema.clone())
        losses.append(float(prob.loss(W)))
        amp.append(float((W - prev).norm()))
        prev = W.clone()
    return {
        "traj_ema": torch.stack(traj_ema),
        "losses": torch.tensor(losses),
        "step_norms": torch.tensor(amp),
        "W_final": W,
    }


def integrate_naive_flow(
    prob: MatrixQuadratic, eta: float, steps: int, W0: Tensor, substeps: int = 4
) -> Tensor:
    """dW/dt = -eta*msign(grad); Euler with substeps, sampled per unit step."""
    W = prob.Wstar.clone() + W0
    out = []
    h = 1.0 / substeps
    for _ in range(steps):
        for _ in range(substeps):
            G = prob.grad(W)
            if G.norm() < 1e-9:
                break
            W = W - eta * h * msign_exact(G)
        out.append(W.clone())
    return torch.stack(out)


def trajectory_error(traj_a: Tensor, traj_b: Tensor) -> float:
    """Registered metric: time-averaged Frobenius error over trajectory norm
    (both trajectories measured relative to the shared start point)."""
    T = min(len(traj_a), len(traj_b))
    a, b = traj_a[:T], traj_b[:T]
    err = (a - b).flatten(1).norm(dim=1).mean()
    scale = (a - a[0]).flatten(1).norm(dim=1).mean().clamp_min(1e-9)
    return float(err / scale)


def stability_and_floor_vs_curvature(
    eta: float = 0.5, sigma: float = 0.0, steps: int = 4000, seed: int = 0,
    scales=(0.13, 4.0, 130.0),
) -> Dict[str, list]:
    """Registered first Stage-1 measurement. For msign dynamics the per-step
    norm is definitionally eta*sqrt(rank); the non-trivial regime structure is
    (i) boundedness at eta*lam far beyond GD's 2 (the lab measured stability
    at eta*lam ~ 65 on airbench) and (ii) how the late-time loss floor scales
    with curvature at fixed eta — a quantity any candidate central flow must
    reproduce. Scales chosen so eta*lam spans ~0.065 to ~65."""
    rows = {"lam_max": [], "eta_lam": [], "bounded": [], "late_loss": [],
            "floor_ratio": []}
    for k, s in enumerate(scales):
        prob = MatrixQuadratic.make(scale=s, seed=seed + k)
        sim = simulate_muon(prob, eta=eta, sigma=sigma, steps=steps, seed=seed + k)
        lam = prob.lam_max
        late = float(sim["losses"][-500:].mean())
        band = eta * (min(prob.Wstar.shape) ** 0.5)
        rows["lam_max"].append(lam)
        rows["eta_lam"].append(eta * lam)
        rows["bounded"].append(bool(torch.isfinite(sim["losses"]).all())
                               and late < 10 * float(sim["losses"][0]))
        rows["late_loss"].append(late)
        # floor relative to the naive band prediction 0.5*lam*(eta*sqrt(r))^2
        rows["floor_ratio"].append(late / (0.5 * lam * band ** 2))
    return rows
